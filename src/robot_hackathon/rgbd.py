"""RGB-D observation schema v1, image IO, calibration registry and the local observation store.

numpy and Pillow are imported lazily so the dependency-free mock attempt logger keeps working.
Dimension equality between RGB and depth is recorded as a fact; it is never treated as registration.
"""
import hashlib
import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .local_store import atomic_write

SCHEMA_VERSION = 1
DEPTH_MAGIC = b"DEPTHMAP"
ALIGNMENT_STATES = ("verified", "unverified", "unknown")


def _np():
    import numpy
    return numpy


def _image():
    from PIL import Image
    return Image


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_path(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------- image encoding

def encode_rgb_png(array):
    buffer = io.BytesIO()
    _image().fromarray(_np().ascontiguousarray(array, dtype=_np().uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def encode_depth16_png(array):
    np = _np()
    buffer = io.BytesIO()
    _image().fromarray(np.ascontiguousarray(array, dtype=np.uint16)).save(buffer, format="PNG")
    return buffer.getvalue()


def encode_mask_png(mask):
    np = _np()
    buffer = io.BytesIO()
    _image().fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8)).convert("1").save(buffer, format="PNG")
    return buffer.getvalue()


def decode_png(data):
    """Return (array, kind) where kind is rgb, depth16 or mask."""
    np = _np()
    image = _image().open(io.BytesIO(data))
    if image.mode in ("I;16", "I;16B", "I"):
        return np.asarray(image, dtype=np.uint16), "depth16"
    if image.mode == "1":
        return np.asarray(image.convert("L")) > 0, "mask"
    return np.asarray(image.convert("RGB"), dtype=np.uint8), "rgb"


def decode_viam_depth(data):
    """Decode Viam's image/vnd.viam.dep: 8-byte magic, big-endian uint64 width/height, big-endian uint16 rows."""
    np = _np()
    if len(data) < 24 or data[:8] != DEPTH_MAGIC:
        raise ValueError("Not a Viam depth map")
    width = int.from_bytes(data[8:16], "big")
    height = int.from_bytes(data[16:24], "big")
    expected = 24 + width * height * 2
    if len(data) != expected:
        raise ValueError("Depth payload length mismatch")
    return np.frombuffer(data, dtype=">u2", offset=24).reshape(height, width).astype(np.uint16)


def encode_viam_depth(array):
    np = _np()
    array = np.asarray(array, dtype=np.uint16)
    height, width = array.shape
    return DEPTH_MAGIC + width.to_bytes(8, "big") + height.to_bytes(8, "big") + array.astype(">u2").tobytes()


# ---------------------------------------------------------------- observation schema

def validate_observation(obs):
    required = {"schema_version", "observation_id", "episode_id", "mode", "setup_id", "camera", "captured_at",
                "timing", "intrinsics", "depth", "alignment", "calibration_version", "artifacts", "flags"}
    if not isinstance(obs, dict) or not required.issubset(obs) or obs["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Invalid observation schema")
    if obs["mode"] not in ("mock", "live"):
        raise ValueError("Invalid observation mode")
    for key in ("observation_id", "episode_id", "setup_id", "calibration_version"):
        if not isinstance(obs[key], str) or not obs[key] or len(obs[key]) > 120:
            raise ValueError(f"Invalid observation field {key}")
    stamp = datetime.fromisoformat(obs["captured_at"])
    if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError("captured_at must be UTC")
    alignment = obs["alignment"]
    if not isinstance(alignment, dict) or alignment.get("status") not in ALIGNMENT_STATES:
        raise ValueError("alignment.status must be verified, unverified or unknown")
    depth = obs["depth"]
    if depth.get("unit") != "mm" or depth.get("dtype") != "uint16" or "invalid_value" not in depth:
        raise ValueError("Depth metadata needs unit=mm, dtype=uint16 and invalid_value")
    kinds = set()
    for artifact in obs["artifacts"]:
        if artifact.get("kind") not in ("rgb", "depth") or not artifact.get("sha256") or not artifact.get("path"):
            raise ValueError("Artifacts need kind, path and sha256")
        if Path(artifact["path"]).is_absolute() or ".." in Path(artifact["path"]).parts:
            raise ValueError("Artifact paths must be relative to the observation directory")
        kinds.add(artifact["kind"])
    if kinds != {"rgb", "depth"}:
        raise ValueError("An observation needs exactly one rgb and one depth artifact")
    return obs


def artifact(obs, kind):
    return next(a for a in obs["artifacts"] if a["kind"] == kind)


# ---------------------------------------------------------------- calibration registry

class CalibrationRegistry:
    """Operator-recorded calibration/alignment verdicts per setup. Never inferred by software."""

    def __init__(self, data_dir, mode):
        self.root = Path(data_dir) / mode / "calibration"

    def _dir(self, setup_id):
        if not setup_id or "/" in setup_id or setup_id.startswith("."):
            raise ValueError("Invalid setup_id")
        return self.root / setup_id

    def record(self, setup_id, version, alignment_status, method, checked_by, viewing_pose="unknown", notes="",
               block_dimensions_mm=None, make_current=True):
        if alignment_status not in ALIGNMENT_STATES:
            raise ValueError("alignment_status must be verified, unverified or unknown")
        entry = {"schema_version": 1, "setup_id": setup_id, "calibration_version": version,
                 "alignment": {"status": alignment_status, "method": method, "checked_by": checked_by,
                               "checked_at": utc_now()},
                 "viewing_pose": viewing_pose, "block_dimensions_mm": block_dimensions_mm, "notes": notes}
        path = self._dir(setup_id) / f"{version}.json"
        if path.exists() and json.loads(path.read_text()) != entry:
            existing = json.loads(path.read_text())
            existing_core = {k: v for k, v in existing.items() if k != "alignment"} | {"alignment": {k: v for k, v in existing["alignment"].items() if k != "checked_at"}}
            new_core = {k: v for k, v in entry.items() if k != "alignment"} | {"alignment": {k: v for k, v in entry["alignment"].items() if k != "checked_at"}}
            if existing_core != new_core:
                raise ValueError("Calibration version already recorded with different content; use a new version")
            return existing
        atomic_write(path, canonical_json(entry))
        if make_current:
            atomic_write(self._dir(setup_id) / "current", version)
        return entry

    def current(self, setup_id):
        path = self._dir(setup_id) / "current"
        return path.read_text().strip() if path.exists() else None

    def load(self, setup_id, version):
        path = self._dir(setup_id) / f"{version}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def content_sha256(self, setup_id, version):
        path = self._dir(setup_id) / f"{version}.json"
        return sha256_path(path) if path.exists() else None


# ---------------------------------------------------------------- observation store

class ObservationStore:
    def __init__(self, data_dir, mode):
        self.mode = mode
        self.root = Path(data_dir) / mode / "observations"

    def directory(self, observation_id):
        if not observation_id or "/" in observation_id or observation_id.startswith("."):
            raise ValueError("Invalid observation_id")
        return self.root / observation_id

    def save(self, obs, rgb, depth):
        """Write rgb.png, depth.png and observation.json. Fills artifact hashes and dimensions."""
        np = _np()
        rgb = np.asarray(rgb)
        depth = np.asarray(depth)
        if obs["mode"] != self.mode:
            raise ValueError("Cannot mix mock and live observations")
        rgb_png, depth_png = encode_rgb_png(rgb), encode_depth16_png(depth)
        obs = dict(obs)
        obs["artifacts"] = [
            {"kind": "rgb", "path": "rgb.png", "media_type": "image/png", "sha256": sha256_bytes(rgb_png),
             "bytes": len(rgb_png), "width": int(rgb.shape[1]), "height": int(rgb.shape[0])},
            {"kind": "depth", "path": "depth.png", "media_type": "image/png", "sha256": sha256_bytes(depth_png),
             "bytes": len(depth_png), "width": int(depth.shape[1]), "height": int(depth.shape[0])}]
        obs.setdefault("flags", [])
        if rgb.shape[:2] != depth.shape[:2]:
            obs["flags"] = sorted(set(obs["flags"]) | {"rgb_depth_dimension_mismatch"})
        obs["dimensions_equal"] = rgb.shape[:2] == depth.shape[:2]
        validate_observation(obs)
        directory = self.directory(obs["observation_id"])
        if (directory / "observation.json").exists():
            raise ValueError("Observation already exists; observations are immutable")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "rgb.png").write_bytes(rgb_png)
        (directory / "depth.png").write_bytes(depth_png)
        atomic_write(directory / "observation.json", canonical_json(obs))
        return obs

    def load(self, observation_id, verify=True):
        directory = self.directory(observation_id)
        obs = validate_observation(json.loads((directory / "observation.json").read_text()))
        if obs["mode"] != self.mode:
            raise ValueError("Observation mode does not match store mode")
        if verify:
            for art in obs["artifacts"]:
                path = directory / art["path"]
                if not path.exists():
                    raise FileNotFoundError(f"Missing artifact {art['kind']}")
                if sha256_path(path) != art["sha256"]:
                    raise ValueError(f"Artifact hash mismatch for {art['kind']}")
        return obs

    def local_artifact_status(self, obs):
        directory = self.directory(obs["observation_id"])
        status = {}
        for art in obs["artifacts"]:
            path = directory / art["path"]
            if not path.exists():
                status[art["kind"]] = "missing"
            elif sha256_path(path) != art["sha256"]:
                status[art["kind"]] = "hash_mismatch"
            else:
                status[art["kind"]] = "present"
        return status

    def load_arrays(self, observation_id):
        directory = self.directory(observation_id)
        obs = self.load(observation_id)
        rgb, _ = decode_png((directory / artifact(obs, "rgb")["path"]).read_bytes())
        depth, _ = decode_png((directory / artifact(obs, "depth")["path"]).read_bytes())
        return obs, rgb, depth

    def list_ids(self):
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / "observation.json").exists())


def new_observation(mode, setup_id, episode_id, calibration_version, alignment, camera, intrinsics=None,
                    captured_at=None, timing=None, flags=(), observation_id=None):
    return {
        "schema_version": SCHEMA_VERSION,
        "observation_id": observation_id or f"obs_{uuid.uuid4().hex[:16]}",
        "episode_id": episode_id,
        "mode": mode,
        "setup_id": setup_id,
        "camera": camera,
        "captured_at": captured_at or utc_now(),
        "timing": timing or {"clock_source": "local_wall_clock", "per_image_timestamps": "not_provided",
                             "rgb_depth_skew_ms": None,
                             "uncertainty_note": "Single response timestamp; RGB/depth skew not measured"},
        "intrinsics": intrinsics or {"rgb": None, "depth": None},
        "depth": {"unit": "mm", "scale_m_per_unit": 0.001, "invalid_value": 0, "dtype": "uint16"},
        "alignment": alignment,
        "calibration_version": calibration_version,
        "artifacts": [],
        "flags": list(flags),
    }

