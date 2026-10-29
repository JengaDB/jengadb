"""Media upload ownership protocol (backend-agnostic; GridFS and in-memory backends implement MediaBackend).

Every upload is an attempt with a unique id that owns its file. A SHA-keyed, insert-only manifest decides
which attempt is the canonical copy. Losers delete only their own file. Cleanup is bounded by attempt age
and state, so an in-progress upload from another writer can never be removed.
"""
from datetime import datetime, timedelta, timezone
from typing import Protocol


class MediaBackend(Protocol):
    def now(self) -> datetime: ...
    def insert_attempt(self, sha256: str, source_alias: str, started_at: datetime) -> str: ...
    def set_attempt_state(self, attempt_id: str, state: str) -> None: ...
    def get_attempt(self, attempt_id: str) -> dict | None: ...
    def write_file(self, attempt_id: str, data: bytes, metadata: dict) -> None: ...
    def file_length(self, file_id: str) -> int | None: ...
    def download(self, file_id: str) -> bytes: ...
    def delete_file(self, file_id: str) -> None: ...
    def find_manifest(self, sha256: str) -> dict | None: ...
    def insert_manifest(self, doc: dict) -> bool: ...
    def list_attempts(self, states, older_than: datetime) -> list: ...


class MediaUploader:
    def __init__(self, backend, max_age=timedelta(hours=24), hasher=None):
        import hashlib
        self.backend = backend
        self.max_age = max_age
        self.hasher = hasher or (lambda data: hashlib.sha256(data).hexdigest())

    def verified_manifest(self, sha256, expected_bytes=None):
        """Return the manifest entry when its file exists and hashes correctly; otherwise None."""
        entry = self.backend.find_manifest(sha256)
        if entry is None:
            return None
        length = self.backend.file_length(entry["gridfs_id"])
        if length is None or (expected_bytes is not None and length != expected_bytes):
            return None
        if self.hasher(self.backend.download(entry["gridfs_id"])) != sha256:
            return None
        return entry

    def start_attempt(self, sha256, data, media_type, kind, source_alias):
        attempt_id = self.backend.insert_attempt(sha256, source_alias, self.backend.now())
        self.backend.write_file(attempt_id, data, {"sha256": sha256, "attempt_id": attempt_id,
                                                   "media_type": media_type, "kind": kind})
        return attempt_id

    def commit_attempt(self, attempt_id, sha256, data, source_alias):
        attempt = self.backend.get_attempt(attempt_id)
        if attempt is None:
            raise RuntimeError("Unknown upload attempt")
        if self.backend.now() - attempt["started_at"] > self.max_age:
            self.backend.set_attempt_state(attempt_id, "abandoned")
            self.backend.delete_file(attempt_id)
            raise RuntimeError("Upload attempt exceeded the abandonment bound; retry with a fresh attempt")
        if self.backend.file_length(attempt_id) != len(data) or self.hasher(self.backend.download(attempt_id)) != sha256:
            self.backend.set_attempt_state(attempt_id, "abandoned")
            self.backend.delete_file(attempt_id)
            raise RuntimeError("Uploaded media failed verification")
        doc = {"_id": sha256, "gridfs_id": attempt_id, "bytes": len(data), "verified_at": self.backend.now(),
               "source_alias": source_alias}
        if self.backend.insert_manifest(doc):
            self.backend.set_attempt_state(attempt_id, "committed")
            return attempt_id
        # Another writer committed the same content first. Keep theirs, drop only ours.
        self.backend.set_attempt_state(attempt_id, "superseded")
        self.backend.delete_file(attempt_id)
        winner = self.backend.find_manifest(sha256)
        if winner is None:
            raise RuntimeError("Manifest insert rejected but no winner found")
        return winner["gridfs_id"]

    def ensure_present(self, sha256, data, media_type, kind, source_alias):
        if self.hasher(data) != sha256:
            raise ValueError("Local media does not match its recorded sha256")
        entry = self.verified_manifest(sha256, len(data))
        if entry is not None:
            return entry["gridfs_id"]
        attempt_id = self.start_attempt(sha256, data, media_type, kind, source_alias)
        return self.commit_attempt(attempt_id, sha256, data, source_alias)

    def orphan_attempts(self):
        """Attempts past the age bound that never became the canonical copy."""
        cutoff = self.backend.now() - self.max_age
        return [a for a in self.backend.list_attempts(("writing", "superseded", "abandoned"), cutoff)
                if (self.backend.find_manifest(a["sha256"]) or {}).get("gridfs_id") != a["_id"]]

    def clean_orphans(self):
        removed = 0
        for attempt in self.orphan_attempts():
            self.backend.delete_file(attempt["_id"])
            self.backend.set_attempt_state(attempt["_id"], "cleaned")
            removed += 1
        return removed


def utc(dt=None):
    return (dt or datetime.now(timezone.utc)).astimezone(timezone.utc)
