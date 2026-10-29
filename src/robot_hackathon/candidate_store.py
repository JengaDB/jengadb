"""Local candidate JSONL store, per-target acknowledgments and the idempotent sync to a remote.

Acks are split into media (sha256 -> gridfs id, after verified presence), observations and candidates
(after read-back equality). Local availability of media is reported but never downgrades remote state.
"""
import json
from contextlib import contextmanager
from pathlib import Path
import fcntl
from .local_store import atomic_write
from . import rgbd, pipeline

REQUIRED = {"schema_version", "candidate_id", "observation_id", "episode_id", "mode", "setup_id", "phase", "mask_index",
            "bbox", "crop_transform", "exposure", "pickability", "segmentation", "cutout_version", "encoder_version",
            "embedding_dimension", "embedding_ref", "files", "processing_version", "calibration_version", "created_at"}


def validate_candidate(record):
    if not isinstance(record, dict) or not REQUIRED.issubset(record) or record["schema_version"] != 1:
        raise ValueError("Invalid candidate schema")
    if record["mode"] not in ("mock", "live") or record["phase"] not in ("pick", "place"):
        raise ValueError("Invalid candidate mode or phase")
    if record["exposure"].get("status") not in ("exposed", "occluded", "unknown"):
        raise ValueError("exposure.status must be exposed, occluded or unknown")
    if record["pickability"].get("status") != "unknown":
        raise ValueError("Pickability is not evaluated in this pipeline; status must stay unknown")
    if "embedding" in record:
        raise ValueError("Store vectors in checkpoint files, not in the local JSONL")
    return record


class CandidateLocalStore:
    def __init__(self, data_dir, mode):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.directory = self.data_dir / mode
        self.path = self.directory / "candidates.jsonl"

    @contextmanager
    def lock(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".rgbd.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def read(self):
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        seen = set()
        for row in rows:
            validate_candidate(row)
            if row["mode"] != self.mode or row["candidate_id"] in seen:
                raise ValueError("Mixed modes or duplicate candidate IDs in local store")
            seen.add(row["candidate_id"])
        return rows

    def put_many(self, records):
        with self.lock():
            rows = {r["candidate_id"]: r for r in self.read()}
            for record in records:
                validate_candidate(record)
                if record["mode"] != self.mode:
                    raise ValueError("Cannot mix mock and live candidates")
                previous = rows.get(record["candidate_id"])
                if previous is not None and previous != record:
                    raise ValueError("Candidate ID already exists with different content")
                rows[record["candidate_id"]] = record
            atomic_write(self.path, "".join(rgbd.canonical_json(r) + "\n" for r in rows.values()))
            return len(rows)


class Acks:
    def __init__(self, directory, target):
        self.path = Path(directory) / f"acks-rgbd-{target}.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.data.setdefault("media", {})
        self.data.setdefault("observations", [])
        self.data.setdefault("candidates", [])

    def save(self):
        atomic_write(self.path, json.dumps(self.data, sort_keys=True))


def sync(data_dir, mode, remote, source_alias="laptop"):
    """Media -> observation -> artifact_sync -> candidates, per observation. Stops at the first remote error.

    Returns a summary with per-artifact local/remote status. A timeout may have committed remotely; acks are
    recorded only after verified presence or read-back, so the next run reconciles instead of duplicating.
    """
    obs_store = rgbd.ObservationStore(data_dir, mode)
    cand_store = CandidateLocalStore(data_dir, mode)
    summary = {"media_uploaded": 0, "observations_sent": 0, "candidates_sent": 0, "artifacts": [], "error": None}
    with cand_store.lock():
        acks = Acks(cand_store.directory, remote.target)
        candidates = cand_store.read()
        by_obs = {}
        for c in candidates:
            by_obs.setdefault(c["observation_id"], []).append(c)
        try:
            for obs_id in obs_store.list_ids():
                obs = obs_store.load(obs_id, verify=False)
                local = obs_store.local_artifact_status(obs)
                obs_dir = obs_store.directory(obs_id)
                for art in obs["artifacts"]:
                    sha = art["sha256"]
                    line = {"observation_id": obs_id, "kind": art["kind"], "sha256": sha, "local_status": local[art["kind"]]}
                    if sha in acks.data["media"]:
                        line["remote_status"] = "synced"
                        if local[art["kind"]] != "present":
                            line["note"] = "local copy missing, remote verified"
                        summary["artifacts"].append(line)
                        continue
                    if local[art["kind"]] != "present":
                        line["remote_status"] = "pending"
                        line["note"] = "not uploadable from this laptop"
                        remote.set_artifact_sync(sha, obs_id, "pending", None, art["bytes"],
                                                 {"source_alias": source_alias, "status": local[art["kind"]]})
                        summary["artifacts"].append(line)
                        continue
                    data = (obs_dir / art["path"]).read_bytes()
                    gridfs_id = remote.media.ensure_present(sha, data, art["media_type"], art["kind"], source_alias)
                    acks.data["media"][sha] = str(gridfs_id)
                    acks.save()
                    summary["media_uploaded"] += 1
                    remote.set_artifact_sync(sha, obs_id, "synced", str(gridfs_id), art["bytes"],
                                             {"source_alias": source_alias, "status": "present"})
                    line["remote_status"] = "synced"
                    summary["artifacts"].append(line)
                if obs_id not in acks.data["observations"]:
                    remote.upsert_observation(obs)
                    acks.data["observations"].append(obs_id)
                    acks.save()
                    summary["observations_sent"] += 1
                for cand in by_obs.get(obs_id, []):
                    if cand["candidate_id"] in acks.data["candidates"]:
                        continue
                    remote.upsert_candidate(remote_candidate(data_dir, cand))
                    acks.data["candidates"].append(cand["candidate_id"])
                    acks.save()
                    summary["candidates_sent"] += 1
        except Exception as exc:  # noqa: BLE001 - sanitized; remote errors may echo configuration
            summary["error"] = type(exc).__name__
    summary["pending_candidates"] = sum(c["candidate_id"] not in acks.data["candidates"] for c in candidates)
    return summary


def remote_candidate(data_dir, cand):
    doc = dict(cand)
    doc["embedding"] = pipeline.load_embedding(data_dir, cand)
    for name in ("cutout_png", "context_png", "mask_png", "depth_crop_png"):
        doc[name] = pipeline.load_candidate_file(data_dir, cand, name)
    return doc
