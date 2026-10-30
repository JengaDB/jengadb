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

    def prune_stale(self, observation_id, keep_processing_version):
        """Drop this observation's candidates from other processing versions, except ones already uploaded.

        Their checkpoints and vectors remain under processing/<old_version>/ on disk; only the active list changes.
        """
        uploaded = set()
        if self.directory.exists():
            for path in self.directory.glob("acks-rgbd-*.json"):
                uploaded |= set(json.loads(path.read_text()).get("candidates", []))
        with self.lock():
            rows = self.read()
            kept, removed = [], 0
            for r in rows:
                stale = r["observation_id"] == observation_id and r["processing_version"] != keep_processing_version
                if stale and r["candidate_id"] not in uploaded:
                    removed += 1
                    continue
                kept.append(r)
            if removed:
                atomic_write(self.path, "".join(rgbd.canonical_json(r) + "\n" for r in kept))
            return removed


class Acks:
    def __init__(self, directory, target):
        self.path = Path(directory) / f"acks-rgbd-{target}.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.data.setdefault("media", {})
        self.data.setdefault("observations", [])
        self.data.setdefault("candidates", [])

    def save(self):
        atomic_write(self.path, json.dumps(self.data, sort_keys=True))


class Review:
    """Operator curation per observation: which detections are ignored (never uploaded). Separate from the
    immutable candidate records, so reprocessing stays deterministic and the decision is auditable."""

    def __init__(self, data_dir, mode):
        self.root = Path(data_dir) / mode / "review"

    def path(self, observation_id):
        if not observation_id or "/" in observation_id or observation_id.startswith("."):
            raise ValueError("Invalid observation_id")
        return self.root / f"{observation_id}.json"

    DEFAULT_PICKABILITY = "pickable"
    PICKABILITY_LABELS = ("pickable", "not_pickable")

    def load(self, observation_id):
        p = self.path(observation_id)
        doc = json.loads(p.read_text()) if p.exists() else {"observation_id": observation_id, "ignored": [], "reviewed_by": None, "updated_at": None}
        doc.setdefault("labels", {})
        return doc

    def ignored(self, observation_id):
        return set(self.load(observation_id)["ignored"])

    def _write(self, doc):
        doc["updated_at"] = rgbd.utc_now()
        atomic_write(self.path(doc["observation_id"]), rgbd.canonical_json(doc))
        return doc

    def set_ignored(self, observation_id, candidate_ids, reviewed_by=None):
        doc = self.load(observation_id)
        doc["ignored"] = sorted(set(candidate_ids))
        if reviewed_by:
            doc["reviewed_by"] = reviewed_by
        return self._write(doc)

    def set_labels(self, observation_id, labels, reviewed_by=None):
        """labels: {candidate_id: pickable|not_pickable}. Explicit human verdicts; unset ids use the default."""
        for cid, status in labels.items():
            if status not in self.PICKABILITY_LABELS:
                raise ValueError(f"Invalid pickability label {status!r}")
        doc = self.load(observation_id)
        doc["labels"].update(labels)
        if reviewed_by:
            doc["reviewed_by"] = reviewed_by
        return self._write(doc)

    @staticmethod
    def pickability_label(review_doc, candidate_id):
        """Human pickability label with its provenance; the heuristic field on the record stays untouched."""
        explicit = review_doc.get("labels", {}).get(candidate_id)
        return {"status": explicit or Review.DEFAULT_PICKABILITY,
                "label_source": "human" if explicit else "human_default",
                "reviewed_by": review_doc.get("reviewed_by"), "reviewed_at": review_doc.get("updated_at")}

    def remove(self, observation_id):
        p = self.path(observation_id)
        if p.exists():
            p.unlink()


def upload_state(data_dir, mode, observation_id):
    """What any destination has acknowledged for this observation (observation, included candidates, media)."""
    store = CandidateLocalStore(data_dir, mode)
    obs_store = rgbd.ObservationStore(data_dir, mode)
    try:
        obs = obs_store.load(observation_id, verify=False)
        shas = {a["sha256"] for a in obs["artifacts"]}
    except (FileNotFoundError, ValueError):
        shas = set()
    ignored = Review(data_dir, mode).ignored(observation_id)
    cand_ids = {c["candidate_id"] for c in store.read() if c["observation_id"] == observation_id} - ignored
    state = {"observation": False, "candidates_acked": 0, "candidates_total": len(cand_ids), "media_acked": 0, "targets": []}
    if store.directory.exists():
        for path in store.directory.glob("acks-rgbd-*.json"):
            d = json.loads(path.read_text())
            hit = observation_id in d.get("observations", [])
            c_ack = len(cand_ids & set(d.get("candidates", [])))
            m_ack = len(shas & set(d.get("media", {})))
            # Media is content-addressed and may be shared by identical bytes; only the observation record or its
            # candidates tie this capture to a destination.
            if hit or c_ack:
                state["targets"].append(path.stem.replace("acks-rgbd-", ""))
            state["observation"] |= hit
            state["candidates_acked"] = max(state["candidates_acked"], c_ack)
            state["media_acked"] = max(state["media_acked"], m_ack)
    state["any"] = bool(state["targets"])
    state["complete"] = state["observation"] and state["candidates_acked"] == state["candidates_total"] and state["media_acked"] == len(shas) and len(shas) > 0
    return state


def delete_observation(data_dir, mode, observation_id):
    """Remove a never-uploaded capture and everything derived from it. Refuses once any part was acknowledged."""
    import shutil
    store = CandidateLocalStore(data_dir, mode)
    obs_store = rgbd.ObservationStore(data_dir, mode)
    directory = obs_store.directory(observation_id)
    with store.lock():
        state = upload_state(data_dir, mode, observation_id)
        if state["any"]:
            raise PermissionError(f"Observation {observation_id} was already uploaded to {state['targets']}; remote records are immutable")
        rows = store.read()
        kept = [r for r in rows if r["observation_id"] != observation_id]
        if len(kept) != len(rows):
            atomic_write(store.path, "".join(rgbd.canonical_json(r) + "\n" for r in kept))
        existed = directory.exists()
        if not existed and len(kept) == len(rows):
            raise FileNotFoundError(f"No local capture {observation_id}")
        if existed:
            shutil.rmtree(directory)
        Review(data_dir, mode).remove(observation_id)
    return {"observation_id": observation_id, "removed_candidates": len(rows) - len(kept), "removed_directory": existed}


def sync(data_dir, mode, remote, source_alias="laptop", observation_ids=None):
    """Media -> observation -> artifact_sync -> candidates, for the selected observations only.

    observation_ids is required to be explicit: nothing is uploaded unless named. Stops at the first remote error.
    A timeout may have committed remotely; acks are recorded only after verified presence or read-back, so the
    next run reconciles instead of duplicating.
    """
    if not observation_ids:
        raise ValueError("sync requires an explicit list of observation IDs to upload")
    obs_store = rgbd.ObservationStore(data_dir, mode)
    cand_store = CandidateLocalStore(data_dir, mode)
    summary = {"media_uploaded": 0, "observations_sent": 0, "candidates_sent": 0, "artifacts": [], "error": None,
               "observation_ids": list(observation_ids)}
    review = Review(data_dir, mode)
    with cand_store.lock():
        acks = Acks(cand_store.directory, remote.target)
        candidates = []
        skipped = 0
        for c in cand_store.read():
            if c["observation_id"] not in set(observation_ids):
                continue
            if c["candidate_id"] in review.ignored(c["observation_id"]):
                skipped += 1
                continue
            candidates.append(c)
        summary["candidates_ignored"] = skipped
        by_obs = {}
        for c in candidates:
            by_obs.setdefault(c["observation_id"], []).append(c)
        try:
            for obs_id in observation_ids:
                if not (obs_store.directory(obs_id) / "observation.json").exists():
                    raise FileNotFoundError(obs_id)
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
                rev = review.load(obs_id)
                for cand in by_obs.get(obs_id, []):
                    if cand["candidate_id"] in acks.data["candidates"]:
                        continue
                    remote.upsert_candidate(remote_candidate(data_dir, cand, rev))
                    acks.data["candidates"].append(cand["candidate_id"])
                    acks.save()
                    summary["candidates_sent"] += 1
        except Exception as exc:  # noqa: BLE001 - sanitized; remote errors may echo configuration
            summary["error"] = type(exc).__name__
    summary["pending_candidates"] = sum(c["candidate_id"] not in acks.data["candidates"] for c in candidates)
    return summary


def remote_candidate(data_dir, cand, review_doc=None):
    doc = dict(cand)
    review_doc = review_doc or {"labels": {}}
    doc["review"] = {"status": "included", "reviewed_by": review_doc.get("reviewed_by"),
                     "reviewed_at": review_doc.get("updated_at")}
    # Human verdict, separate from the heuristic `pickability` (which stays unknown in this pipeline).
    doc["pickability_label"] = Review.pickability_label(review_doc, cand["candidate_id"])
    doc["pickable"] = doc["pickability_label"]["status"] == "pickable"
    doc["embedding"] = pipeline.load_embedding(data_dir, cand)
    for name in ("cutout_png", "context_png", "mask_png", "depth_crop_png"):
        doc[name] = pipeline.load_candidate_file(data_dir, cand, name)
    return doc
