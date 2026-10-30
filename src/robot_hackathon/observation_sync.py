"""Upload captured observations: media -> observation record -> artifact_sync.

Reused from the SKY-66 candidate sync with the candidate and review stages removed. Acks are kept in the same
`acks-rgbd-<target>.json` file layout so a later candidate stage can resume without re-uploading media.
Nothing is uploaded unless its observation ID is named explicitly. Stops at the first remote error.
"""
import fcntl
import json
from contextlib import contextmanager
from pathlib import Path
from .local_store import atomic_write
from . import rgbd


class Acks:
    def __init__(self, directory, target):
        self.path = Path(directory) / f"acks-rgbd-{target}.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.data.setdefault("media", {})
        self.data.setdefault("observations", [])
        self.data.setdefault("candidates", [])

    def save(self):
        atomic_write(self.path, json.dumps(self.data, sort_keys=True))


@contextmanager
def lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".rgbd.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def sync_observations(data_dir, mode, remote, source_alias="laptop", observation_ids=None):
    """A timeout may have committed remotely; acks are recorded only after verified presence or read-back,
    so the next run reconciles instead of duplicating."""
    if not observation_ids:
        raise ValueError("sync requires an explicit list of observation IDs to upload")
    obs_store = rgbd.ObservationStore(data_dir, mode)
    directory = Path(data_dir) / mode
    summary = {"media_uploaded": 0, "observations_sent": 0, "artifacts": [], "error": None,
               "observation_ids": list(observation_ids)}
    with lock(directory):
        acks = Acks(directory, remote.target)
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
        except Exception as exc:  # noqa: BLE001 - sanitized; remote errors may echo configuration
            summary["error"] = type(exc).__name__
    summary["pending_observations"] = sum(i not in acks.data["observations"] for i in observation_ids)
    return summary
