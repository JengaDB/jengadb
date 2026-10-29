"""Single-laptop JSONL store. Atomic replacement, process lock, immutable IDs."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import json
import os
import tempfile
from datetime import datetime


def validate(record):
    required = {"schema_version", "attempt_id", "run_id", "timestamp", "mode", "machine_alias",
                "object_label", "action_type", "strategy", "pose", "observed_outcome",
                "duration_ms", "error_class", "confidence", "artifact_refs"}
    if not required.issubset(record) or record["schema_version"] != 1:
        raise ValueError("Invalid attempt schema")
    if record["mode"] not in ("mock", "live") or record["observed_outcome"] not in ("success", "failure", "unknown"):
        raise ValueError("Invalid mode or observed outcome")
    for key in ("attempt_id", "run_id", "machine_alias", "object_label", "action_type", "strategy"):
        if not isinstance(record[key], str) or not record[key] or len(record[key]) > 120:
            raise ValueError("Invalid attempt label")
    stamp = datetime.fromisoformat(record["timestamp"])
    if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError("Timestamp must be UTC")
    if not isinstance(record["duration_ms"], (int, float)) or record["duration_ms"] < 0:
        raise ValueError("Invalid duration_ms")
    if record["confidence"] is not None and not 0 <= record["confidence"] <= 1:
        raise ValueError("Invalid confidence")
    if not isinstance(record["artifact_refs"], list) or any(not isinstance(x, str) or x.startswith("data:") for x in record["artifact_refs"]):
        raise ValueError("Artifact references must be strings, not embedded data")
    if record["error_class"] not in (None, "unobserved", "timeout", "authentication", "remote_error"):
        raise ValueError("Use a sanitized error category")
    if record["pose"] is not None:
        pose = record["pose"]
        if not isinstance(pose, dict) or not pose.get("frame") or pose.get("position_unit") != "mm" or pose.get("angle_unit") != "deg":
            raise ValueError("Pose needs a frame and explicit mm/deg units")


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


class LocalStore:
    def __init__(self, directory, mode="mock"):
        self.directory = Path(directory) / mode
        self.mode = mode
        self.path = self.directory / "attempts.jsonl"

    @contextmanager
    def lock(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def read(self):
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        ids = set()
        for row in rows:
            validate(row)
            if row["mode"] != self.mode or row["attempt_id"] in ids:
                raise ValueError("Mixed modes or duplicate IDs in local store")
            ids.add(row["attempt_id"])
        return rows

    def put_many(self, records):
        with self.lock():
            rows = {r["attempt_id"]: r for r in self.read()}
            for record in records:
                validate(record)
                if record["mode"] != self.mode:
                    raise ValueError("Cannot mix mock and live records")
                previous = rows.get(record["attempt_id"])
                if previous is not None and previous != record:
                    raise ValueError("Attempt ID already exists with different content")
                rows[record["attempt_id"]] = record
            atomic_write(self.path, "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows.values()))
            return len(rows)

    def flush(self, remote, target):
        """A timeout may commit remotely. Only record an ack after confirmed delivery.

        target is a hash of endpoint + database + collection; different targets need delivery.
        """
        with self.lock():
            ack_path = self.directory / f"acks-{target}.json"
            acked = set(json.loads(ack_path.read_text())) if ack_path.exists() else set()
            sent = 0
            records = self.read()
            for record in records:
                if record["attempt_id"] in acked:
                    continue
                try:
                    remote.upsert(record)
                except Exception:
                    return {"sent": sent, "pending": sum(r["attempt_id"] not in acked for r in records), "error": "remote_error"}
                acked.add(record["attempt_id"])
                atomic_write(ack_path, json.dumps(sorted(acked)))
                sent += 1
            return {"sent": sent, "pending": 0, "error": None}
