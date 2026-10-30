"""Deterministic synthetic RGB-D scenes for offline geometry fixtures (mock mode only).

A top-down pinhole camera renders a slightly tilted table plane and flat block tops with known poses in the
robot_base frame. Masks are ground truth per block (provider ``synthetic-ground-truth``), not SAM output, so
these scenes exercise geometry, clearance and proposal plumbing; they say nothing about real mask quality.
Everything is labelled synthetic and never mixes with live records.
"""
import json
import math

from . import rgbd
from .grasp_geometry import GeometryCalibrationStore
from .local_store import atomic_write

SETUP_ID = "synthetic-topdown"
CALIBRATION_VERSION = "synthetic-cal-v1"
GEOMETRY_VERSION = "synthetic-geom-v1"
PROVIDER = "synthetic-ground-truth"
BLOCK = (75.0, 25.0, 15.0)
TABLE_Z0_MM = 10.0             # support surface at the base origin: the "about 10 mm" estimate, here exact by construction
TABLE_TILT = (0.004, -0.0025)  # dz/dx, dz/dy: a small tilt so the 1/depth fit matters
CAMERA_POSITION = (300.0, 150.0, 450.0)
ROTATION = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))  # camera x = base x, y = -base y, z = -base z
INTRINSICS = {"width_px": 320, "height_px": 240, "focal_x_px": 228.0, "focal_y_px": 228.0, "center_x_px": 160.0, "center_y_px": 120.0}

# name -> [(x_mm, y_mm, long_side_deg, level)] in robot_base; level 1 rests on a level-0 block at the same pose
PRESETS = {
    "separated": [("A", 230.0, 90.0, 0.0, 0), ("B", 330.0, 200.0, 60.0, 0), ("C", 400.0, 80.0, -35.0, 0)],
    # B's finger strips (47-65 mm from its centre across, and along for the end grip) are occupied by neighbours
    "blocked": [("A", 210.0, 80.0, 10.0, 0), ("B", 330.0, 170.0, 0.0, 0), ("N1", 330.0, 230.0, 0.0, 0), ("N2", 390.0, 170.0, 90.0, 0)],
    "stacked": [("A", 230.0, 90.0, 20.0, 0), ("S0", 350.0, 190.0, 0.0, 0), ("S1", 350.0, 190.0, 0.0, 1)],
}


def table_z(x, y):
    return TABLE_Z0_MM + TABLE_TILT[0] * x + TABLE_TILT[1] * y


def render(preset, seed=7, invalid_fraction=0.01):
    """Return (rgb uint8 HxWx3, depth uint16 HxW mm, masks [bool HxW], truth list)."""
    import numpy as np
    if preset not in PRESETS:
        raise ValueError(f"Unknown synthetic preset {preset!r}; choose from {sorted(PRESETS)}")
    rng = np.random.default_rng(seed)
    w, h = INTRINSICS["width_px"], INTRINSICS["height_px"]
    fx, fy, cx, cy = (INTRINSICS[k] for k in ("focal_x_px", "focal_y_px", "center_x_px", "center_y_px"))
    v, u = np.mgrid[0:h, 0:w]
    dcam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones((h, w))], axis=-1)
    r = np.array(ROTATION)
    d = dcam @ r.T                       # ray directions in robot_base, camera z component = 1 -> depth = t
    c = np.array(CAMERA_POSITION)
    ax, ay = TABLE_TILT

    def intersect(z0):
        t = (z0 + ax * c[0] + ay * c[1] - c[2]) / (d[..., 2] - ax * d[..., 0] - ay * d[..., 1])
        return t, c + t[..., None] * d

    depth, _ = intersect(TABLE_Z0_MM)
    owner = np.full((h, w), -1, dtype=int)
    length, width, height = BLOCK
    truth = []
    for index, (name, bx, by, deg, level) in enumerate(PRESETS[preset]):
        top = TABLE_Z0_MM + (level + 1) * height
        t, p = intersect(top)
        a = math.radians(deg)
        dx, dy = p[..., 0] - bx, p[..., 1] - by
        along = dx * math.cos(a) + dy * math.sin(a)
        across = -dx * math.sin(a) + dy * math.cos(a)
        inside = (np.abs(along) <= length / 2) & (np.abs(across) <= width / 2) & (t < depth)
        depth = np.where(inside, t, depth)
        owner = np.where(inside, index, owner)
        truth.append({"name": name, "center_xy_mm": [bx, by], "long_side_deg": deg, "level": level,
                      "top_z_mm": table_z(bx, by) + (level + 1) * height, "height_above_table_mm": (level + 1) * height,
                      "dimensions_mm": list(BLOCK)})
    depth = depth + rng.normal(0.0, 0.8, depth.shape)
    invalid = rng.random(depth.shape) < invalid_fraction
    invalid[h - 30:h - 10, 10:50] = True
    depth16 = np.where(invalid, 0, np.clip(np.rint(depth), 1, 65535)).astype(np.uint16)
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[:] = (118, 108, 98)
    rgb = (rgb.astype(float) + rng.normal(0, 3, rgb.shape)).clip(0, 255).astype(np.uint8)
    masks = []
    for index in range(len(PRESETS[preset])):
        mask = owner == index
        shade = rng.integers(-15, 15)
        rgb[mask] = np.clip(np.array((205, 172, 122)) + shade, 0, 255).astype(np.uint8)
        masks.append(mask)
    return rgb, depth16, masks, truth


def geometry_document():
    return {"schema_version": 1, "setup_id": SETUP_ID, "version": GEOMETRY_VERSION, "source": "synthetic_ground_truth",
            "support_surface": {"z_in_robot_base_mm": TABLE_Z0_MM, "sign_convention": "positive_above_robot_base_origin",
                                "verified": True, "source": "synthetic_ground_truth", "version": GEOMETRY_VERSION,
                                "uncertainty_mm": 0.0, "note": "exact by construction; real setups must be measured"},
            "extrinsics": {"rotation": [list(r) for r in ROTATION], "translation_mm": list(CAMERA_POSITION), "frame_from": "camera",
                           "frame_to": "robot_base", "verified": True, "source": "synthetic_ground_truth", "version": GEOMETRY_VERSION,
                           "pose_time": "synthetic"},
            "finger": {"open_gap_mm": 104.0, "claw_thickness_mm": 8.0, "finger_width_mm": 45.0, "margin_mm": 5.0, "verified": True,
                       "source": "synthetic_assumed_equal_to_sam_defaults"},
            "block": {"length_mm": BLOCK[0], "width_mm": BLOCK[1], "height_mm": BLOCK[2], "verified": True, "source": "synthetic_ground_truth"},
            "workspace": {"x_mm": [100.0, 550.0], "y_mm": [-100.0, 400.0], "z_mm": [0.0, 200.0]}}


def observation_record(preset, observation_id, episode_id="ep_synthetic"):
    intr = dict(INTRINSICS)
    return rgbd.new_observation("mock", SETUP_ID, episode_id, CALIBRATION_VERSION,
                                alignment={"status": "verified", "method": "synthetic_ground_truth", "checked_by": "synthetic_scene.py",
                                           "checked_at": rgbd.utc_now(), "calibration_version": CALIBRATION_VERSION},
                                camera={"resource_name": "synthetic", "serial": None, "model": "synthetic-pinhole", "preset": preset},
                                intrinsics={"rgb": intr, "depth": intr, "note": "synthetic: depth rendered in the colour frame"},
                                timing={"clock_source": "synthetic", "per_image_timestamps": "identical", "rgb_depth_skew_ms": 0,
                                        "uncertainty_note": "synthetic render; no sensor timing"},
                                flags=["synthetic"], observation_id=observation_id)


def save_scene(data_dir, preset, observation_id=None, seed=7):
    """Persist a synthetic observation (mock store), its ground-truth masks, registry entry and geometry calibration.
    Returns the saved observation. Refuses to overwrite an existing observation (immutability)."""
    observation_id = observation_id or f"obs_synthetic_{preset}"
    rgb, depth, masks, truth = render(preset, seed)
    registry = rgbd.CalibrationRegistry(data_dir, "mock")
    registry.record(SETUP_ID, CALIBRATION_VERSION, "verified", "synthetic_ground_truth", "synthetic_scene.py",
                    viewing_pose="top_down", notes="synthetic scene; not a measurement", block_dimensions_mm=list(BLOCK))
    GeometryCalibrationStore(data_dir, "mock").record(SETUP_ID, GEOMETRY_VERSION, geometry_document())
    store = rgbd.ObservationStore(data_dir, "mock")
    obs = store.save(observation_record(preset, observation_id), rgb, depth)
    seg_dir = store.directory(observation_id) / "segmentation" / PROVIDER
    seg_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (mask, block) in enumerate(zip(masks, truth)):
        png = rgbd.encode_mask_png(mask)
        path = seg_dir / f"mask-{index:03d}.png"
        path.write_bytes(png)
        import numpy as np
        ys, xs = np.nonzero(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None
        records.append({"mask_index": index, "path": path.name, "mask_sha256": rgbd.sha256_bytes(png), "bbox": bbox,
                        "area": int(mask.sum()), "score": None, "truth": block})
    atomic_write(seg_dir / "masks.json", rgbd.canonical_json({"provider": PROVIDER, "model_version": "render-v1", "preset": preset,
                                                               "seed": seed, "masks": records}))
    return obs


def load_masks(data_dir, observation_id):
    store = rgbd.ObservationStore(data_dir, "mock")
    seg_dir = store.directory(observation_id) / "segmentation" / PROVIDER
    doc = json.loads((seg_dir / "masks.json").read_text())
    out = []
    for rec in doc["masks"]:
        data = (seg_dir / rec["path"]).read_bytes()
        if rgbd.sha256_bytes(data) != rec["mask_sha256"]:
            raise ValueError("Synthetic mask hash mismatch")
        array, kind = rgbd.decode_png(data)
        out.append({**rec, "mask": array if kind == "mask" else array > 0})
    return doc, out
