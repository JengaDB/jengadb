"""Local JSON result cache. No camera, model, network or robot dependencies."""
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from .artifact_store import ArtifactStore


class CacheMiss(KeyError):
    """No usable saved result; callers choose whether to process, queue or wait."""


@dataclass(frozen=True)
class CacheOutput:
    value: object
    artifacts: tuple = ()


@dataclass(frozen=True)
class CacheEntry:
    key: str
    value: object
    artifacts: tuple


def canonical(value):
    def check_keys(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                check_keys(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check_keys(child)
    check_keys(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(operation, inputs, model_version, settings):
    """Inputs are ordered byte strings; include every input affecting the result."""
    if not isinstance(operation, str) or not operation.strip() or not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("Operation and model version are required")
    if not isinstance(inputs, (list, tuple)) or not inputs or any(not isinstance(item, bytes) for item in inputs):
        raise ValueError("Provide an ordered, nonempty list of byte inputs")
    if not isinstance(settings, dict):
        raise ValueError("Settings must be a JSON object")
    request = {"schema": 1, "operation": operation, "model_version": model_version,
               "input_hashes": [hashlib.sha256(item).hexdigest() for item in inputs],
               "settings": settings}
    encoded = canonical(request)
    return hashlib.sha256(encoded.encode()).hexdigest(), encoded


class ResultCache:
    def __init__(self, path, *, artifact_directory=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts = ArtifactStore(artifact_directory or self.path.parent / "artifacts")
        with closing(self.connect()) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, "
                       "request TEXT NOT NULL, result TEXT NOT NULL, created_at TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS result_artifacts ("
                       "key TEXT PRIMARY KEY REFERENCES results(key) ON DELETE CASCADE, refs TEXT NOT NULL)")

    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def _lookup(self, key):
        with closing(self.connect()) as db:
            row = db.execute("SELECT r.result, r.created_at, a.refs FROM results r "
                             "LEFT JOIN result_artifacts a ON a.key = r.key WHERE r.key = ?", (key,)).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(row[0])
            canonical(value)  # Reject invalid JSON constants such as NaN.
            refs = json.loads(row[2]) if row[2] is not None else []
            if not isinstance(refs, list) or not all(self.artifacts.valid(ref) for ref in refs):
                raise ValueError("Missing or damaged artifacts")
            return CacheEntry(key, value, tuple(refs))
        except (ValueError, TypeError):
            # Only invalidate the exact entry read, not a concurrent replacement.
            with closing(self.connect()) as db, db:
                db.execute("DELETE FROM results WHERE key = ? AND created_at = ? AND result = ?",
                           (key, row[1], row[0]))
            return None

    def lookup(self, *, operation, inputs, model_version, settings):
        """Return an entry or None. Never invokes processing or any network API."""
        key, _ = fingerprint(operation, inputs, model_version, settings)
        return self._lookup(key)

    def _store(self, key, request, output):
        result = canonical(output.value)
        refs = canonical(list(output.artifacts))
        if not all(self.artifacts.valid(ref) for ref in output.artifacts):
            raise ValueError("Cannot cache missing or damaged artifacts")
        with closing(self.connect()) as db, db:
            inserted = db.execute("INSERT OR IGNORE INTO results VALUES (?, ?, ?, ?)",
                                  (key, request, result, datetime.now(timezone.utc).isoformat()))
            if inserted.rowcount:
                db.execute("INSERT INTO result_artifacts VALUES (?, ?)", (key, refs))
        entry = self._lookup(key)
        if entry is None:
            raise CacheMiss("Result artifacts became unavailable")
        return entry

    def store(self, *, operation, inputs, model_version, settings, value, artifacts=()):
        """Save a finished result; a valid earlier result for the same key wins."""
        key, request = fingerprint(operation, inputs, model_version, settings)
        previous = self._lookup(key)
        if previous is not None:
            return previous
        return self._store(key, request, CacheOutput(value, tuple(artifacts)))

    def get_or_compute(self, *, operation, inputs, model_version, settings, compute=None):
        key, request = fingerprint(operation, inputs, model_version, settings)
        entry = self._lookup(key)
        if entry is not None:
            return entry.value, True, key
        if compute is None:
            raise CacheMiss(key)
        # Processing happens outside any database connection/transaction.
        # Concurrent misses may compute twice, but only one result is retained.
        output = compute()
        if not isinstance(output, CacheOutput):
            output = CacheOutput(output)
        entry = self._store(key, request, output)
        return entry.value, False, key
