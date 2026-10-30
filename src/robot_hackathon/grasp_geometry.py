"""Pure grasp geometry (AGENT_HEURISTICS_PLAN steps 3 and 6), adapted from Sam's jenga.py.

No hardware imports and no motion. Lengths are millimetres, angles degrees. Frames are explicit:
``rgb_pixels`` (u right, v down), ``camera`` (optical: x right, y down, z forward) and ``robot_base``.
The support-surface offset, camera extrinsics, RGB/depth registration and claw-tip geometry are separate
calibration quantities with their own source/version/verified fields; missing, unverified or inconsistent
calibration prevents a grasp proposal with recorded reasons instead of fabricating a height.

Sam's constants (block 75x25x15, 104 mm claw opening, 45 mm fingers, 8 mm claws) are defaults for the
*data model*, flagged unverified until the hardware owner confirms them for this setup.
"""
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path

from . import rgbd
from .local_store import atomic_write

GEOMETRY_CODE_VERSION = "grasp-geometry-v1"
ACROSS = 90   # claws close across the block's narrow sides (middle and side grips)
ALONG = 0     # claws close on the block's two ends


def _np():
    import numpy
    return numpy


# ---------------------------------------------------------------- calibration data model

@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width_px: int | None = None
    height_px: int | None = None
    source: str = "observation.intrinsics.depth"

    @classmethod
    def from_observation(cls, obs, which="depth"):
        intr = (obs.get("intrinsics") or {}).get(which)
        if not intr or any(intr.get(k) is None for k in ("focal_x_px", "focal_y_px", "center_x_px", "center_y_px")):
            return None
        return cls(float(intr["focal_x_px"]), float(intr["focal_y_px"]), float(intr["center_x_px"]), float(intr["center_y_px"]),
                   intr.get("width_px"), intr.get("height_px"), f"observation.intrinsics.{which}")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SupportSurface:
    """Signed height of the support surface (table top) in the robot-base frame. Positive = above base origin.
    ~10 mm is an estimate from the planning discussion, not a measurement; sign and value stay unverified
    until the hardware owner records them."""
    z_in_robot_base_mm: float | None = None
    verified: bool = False
    source: str = "unspecified"
    version: str | None = None
    uncertainty_mm: float | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CameraExtrinsics:
    """Rigid camera -> robot_base transform applicable at capture time (wrist camera: pose at that instant)."""
    rotation: tuple           # 3x3, rows; columns are camera axes expressed in robot_base
    translation_mm: tuple     # camera origin in robot_base
    frame_from: str = "camera"
    frame_to: str = "robot_base"
    verified: bool = False
    source: str = "unspecified"
    version: str | None = None
    pose_time: str | None = None   # when the arm pose used for this transform was read (provenance)

    def apply(self, point):
        r, t = self.rotation, self.translation_mm
        return tuple(r[i][0] * point[0] + r[i][1] * point[1] + r[i][2] * point[2] + t[i] for i in range(3))

    def to_dict(self):
        d = asdict(self)
        d["rotation"] = [list(row) for row in self.rotation]
        d["translation_mm"] = list(self.translation_mm)
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(tuple(tuple(float(x) for x in row) for row in d["rotation"]), tuple(float(x) for x in d["translation_mm"]),
                   d.get("frame_from", "camera"), d.get("frame_to", "robot_base"), bool(d.get("verified", False)),
                   d.get("source", "unspecified"), d.get("version"), d.get("pose_time"))


@dataclass(frozen=True)
class FingerGeometry:
    """Claw geometry from Sam's setup; measure again for another gripper."""
    open_gap_mm: float = 104.0        # gap between the open claws' inner faces
    claw_thickness_mm: float = 8.0    # one claw, measured across the block
    finger_width_mm: float = 45.0     # claw size along the block's long side
    margin_mm: float = 5.0            # gap beside the block so its own edges are not an obstruction
    verified: bool = False
    source: str = "sam_jenga_py_defaults"

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class BlockDimensions:
    length_mm: float = 75.0
    width_mm: float = 25.0
    height_mm: float = 15.0
    verified: bool = False
    source: str = "config_default"

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class GraspParams:
    grasp_depth_mm: float = 8.0        # claw tips below the block's top surface before closing
    ends_extra_depth_mm: float = 10.0  # the end grip goes this much deeper
    tips_min_mm: float = 1.0           # never below this above the table
    side_shift_mm: float = 15.0        # side grip offset from the centre along the long side
    obstruction_min_mm: float = 6.0    # anything lower in the claws' path is table noise
    pick_lift_mm: float = 100.0
    min_blocked_samples: int = 3
    max_unknown_fraction: float = 0.3

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Workspace:
    """Reachable box in robot_base for block top-centres. None means unchecked (flagged, not rejected)."""
    x_mm: tuple
    y_mm: tuple
    z_mm: tuple

    def contains(self, p):
        return all(lo <= v <= hi for v, (lo, hi) in zip(p, (self.x_mm, self.y_mm, self.z_mm)))

    def to_dict(self):
        return {"x_mm": list(self.x_mm), "y_mm": list(self.y_mm), "z_mm": list(self.z_mm)}


@dataclass(frozen=True)
class CalibrationBundle:
    intrinsics: Intrinsics | None
    extrinsics: CameraExtrinsics | None
    support_surface: SupportSurface
    finger: FingerGeometry
    block: BlockDimensions
    registration_verified: bool
    calibration_version: str | None
    setup_id: str | None = None
    workspace: Workspace | None = None
    grasp: GraspParams = field(default_factory=GraspParams)
    source: str = "unspecified"

    def missing(self):
        """Reasons that prevent a grasp proposal. Empty means geometry may be proposed."""
        reasons = []
        if self.intrinsics is None:
            reasons.append("intrinsics_missing")
        if not self.registration_verified:
            reasons.append("registration_not_verified")
        if self.extrinsics is None:
            reasons.append("extrinsics_missing")
        elif not self.extrinsics.verified:
            reasons.append("extrinsics_not_verified")
        if self.support_surface.z_in_robot_base_mm is None:
            reasons.append("support_surface_offset_missing")
        elif not self.support_surface.verified:
            reasons.append("support_surface_offset_not_verified")
        if not self.finger.verified:
            reasons.append("finger_geometry_not_verified")
        if not self.block.verified:
            reasons.append("block_dimensions_not_verified")
        return reasons

    def to_dict(self):
        return {"geometry_code_version": GEOMETRY_CODE_VERSION, "setup_id": self.setup_id,
                "calibration_version": self.calibration_version, "source": self.source,
                "registration_verified": self.registration_verified,
                "intrinsics": None if self.intrinsics is None else self.intrinsics.to_dict(),
                "extrinsics": None if self.extrinsics is None else self.extrinsics.to_dict(),
                "support_surface": self.support_surface.to_dict(), "finger": self.finger.to_dict(),
                "block": self.block.to_dict(), "grasp": self.grasp.to_dict(),
                "workspace": None if self.workspace is None else self.workspace.to_dict(), "missing": self.missing()}


class GeometryCalibrationStore:
    """Operator-recorded geometry calibration per setup/version: extrinsics, support-surface offset, claw and
    block dimensions. Separate from the alignment registry (rgbd.CalibrationRegistry) so SKY-66 identities are
    untouched. Files are immutable per version; conflicting rewrites are refused."""

    def __init__(self, data_dir, mode):
        self.root = Path(data_dir) / mode / "calibration"

    def path(self, setup_id, version):
        for value in (setup_id, version):
            if not value or "/" in value or value.startswith("."):
                raise ValueError("Invalid setup_id or version")
        return self.root / setup_id / f"geometry-{version}.json"

    def record(self, setup_id, version, doc, make_current=True):
        doc = validate_geometry_doc(dict(doc, setup_id=setup_id, version=version))
        path = self.path(setup_id, version)
        if path.exists():
            existing = json.loads(path.read_text())
            if {k: v for k, v in existing.items() if k != "recorded_at"} != {k: v for k, v in doc.items() if k != "recorded_at"}:
                raise ValueError("Geometry calibration version already recorded with different content; use a new version")
            return existing
        doc["recorded_at"] = rgbd.utc_now()
        atomic_write(path, rgbd.canonical_json(doc))
        if make_current:
            atomic_write(path.parent / "geometry-current", version)
        return doc

    def current(self, setup_id):
        path = self.path(setup_id, "x").parent / "geometry-current"
        return path.read_text().strip() if path.exists() else None

    def load(self, setup_id, version=None):
        version = version or self.current(setup_id)
        if version is None:
            return None
        path = self.path(setup_id, version)
        return json.loads(path.read_text()) if path.exists() else None

    def content_sha256(self, setup_id, version):
        path = self.path(setup_id, version)
        return rgbd.sha256_path(path) if path.exists() else None


def validate_geometry_doc(doc):
    required = {"schema_version", "setup_id", "version", "support_surface", "finger", "block"}
    if not isinstance(doc, dict) or not required.issubset(doc) or doc["schema_version"] != 1:
        raise ValueError("Invalid geometry calibration document")
    ss = doc["support_surface"]
    if ss.get("z_in_robot_base_mm") is not None and not isinstance(ss["z_in_robot_base_mm"], (int, float)):
        raise ValueError("support_surface.z_in_robot_base_mm must be a number or null")
    if "sign_convention" not in ss or ss["sign_convention"] != "positive_above_robot_base_origin":
        raise ValueError("support_surface.sign_convention must be 'positive_above_robot_base_origin'")
    ext = doc.get("extrinsics")
    if ext is not None and ext.get("from_observation"):
        # Use the camera pose recorded by rgbd-capture at capture time; the operator still owns the `verified` flag.
        if "verified" not in ext:
            raise ValueError("extrinsics.from_observation needs an explicit verified flag")
    elif ext is not None:
        rot = ext.get("rotation")
        if not (isinstance(rot, list) and len(rot) == 3 and all(isinstance(r, list) and len(r) == 3 for r in rot)):
            raise ValueError("extrinsics.rotation must be 3x3")
        if not (isinstance(ext.get("translation_mm"), list) and len(ext["translation_mm"]) == 3):
            raise ValueError("extrinsics.translation_mm must have three values")
        if not rotation_is_orthonormal(rot):
            raise ValueError("extrinsics.rotation is not orthonormal")
    for key in ("finger", "block"):
        if not isinstance(doc[key], dict) or "verified" not in doc[key]:
            raise ValueError(f"{key} needs a verified flag")
    return doc


def rotation_is_orthonormal(rot, tol=1e-3):
    for i in range(3):
        for j in range(3):
            dot = sum(rot[k][i] * rot[k][j] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > tol:
                return False
    det = (rot[0][0] * (rot[1][1] * rot[2][2] - rot[1][2] * rot[2][1]) - rot[0][1] * (rot[1][0] * rot[2][2] - rot[1][2] * rot[2][0])
           + rot[0][2] * (rot[1][0] * rot[2][1] - rot[1][1] * rot[2][0]))
    return abs(det - 1.0) < tol


def extrinsics_from_observation(obs, verified, source="observation.camera.pose_in_world"):
    """Camera->world transform recorded by rgbd-capture at capture time, or None if it was not read/is not rigid."""
    pose = (obs.get("camera") or {}).get("pose_in_world") or {}
    if pose.get("status") != "read" or not rotation_is_orthonormal(pose["rotation"], tol=1e-2):
        return None
    return CameraExtrinsics(tuple(tuple(float(x) for x in row) for row in pose["rotation"]), tuple(float(x) for x in pose["translation_mm"]),
                            pose.get("frame_from", "camera"), pose.get("frame_to", "world"), verified, source, obs.get("observation_id"),
                            pose.get("read_at"))


def bundle_from_records(obs, registration_verified, geometry_doc, block_dims=None):
    """Combine an observation's intrinsics, the SKY-66 registration verdict and a geometry calibration document."""
    intr = Intrinsics.from_observation(obs, "depth")
    if geometry_doc is None:
        return CalibrationBundle(intr, None, SupportSurface(), FingerGeometry(), block_dims or BlockDimensions(),
                                 registration_verified, obs.get("calibration_version"), obs.get("setup_id"), source="none")
    ss = geometry_doc["support_surface"]
    fg = geometry_doc["finger"]
    bd = geometry_doc["block"]
    ext = geometry_doc.get("extrinsics")
    ws = geometry_doc.get("workspace")
    grasp = GraspParams(**geometry_doc["grasp"]) if geometry_doc.get("grasp") else GraspParams()
    extrinsics = None
    if ext is not None and ext.get("from_observation"):
        extrinsics = extrinsics_from_observation(obs, bool(ext["verified"]), ext.get("source", "observation.camera.pose_in_world"))
    elif ext is not None:
        extrinsics = CameraExtrinsics.from_dict(ext)
    return CalibrationBundle(
        intr, extrinsics,
        SupportSurface(ss.get("z_in_robot_base_mm"), bool(ss.get("verified", False)), ss.get("source", "unspecified"),
                       ss.get("version"), ss.get("uncertainty_mm")),
        FingerGeometry(fg.get("open_gap_mm", 104.0), fg.get("claw_thickness_mm", 8.0), fg.get("finger_width_mm", 45.0),
                       fg.get("margin_mm", 5.0), bool(fg["verified"]), fg.get("source", "unspecified")),
        BlockDimensions(bd.get("length_mm", 75.0), bd.get("width_mm", 25.0), bd.get("height_mm", 15.0), bool(bd["verified"]),
                        bd.get("source", "unspecified")),
        registration_verified, obs.get("calibration_version"), obs.get("setup_id"),
        None if ws is None else Workspace(tuple(ws["x_mm"]), tuple(ws["y_mm"]), tuple(ws["z_mm"])), grasp,
        source=f"geometry-{geometry_doc['version']}")


# ---------------------------------------------------------------- pinhole and frames

def deproject(u, v, depth_mm, intr):
    """Pixel + depth -> point in the camera frame (mm)."""
    x = (u - intr.cx) * depth_mm / intr.fx
    y = (v - intr.cy) * depth_mm / intr.fy
    return (x, y, float(depth_mm))


def project(point, intr):
    x, y, z = point
    if z <= 0:
        raise ValueError("Point behind the camera")
    return (intr.fx * x / z + intr.cx, intr.fy * y / z + intr.cy)


def pixel_to_base(u, v, depth_mm, intr, extrinsics):
    return extrinsics.apply(deproject(u, v, depth_mm, intr))


def world_angle(a, b):
    """Direction from a to b in the base XY plane, degrees (Sam's world_angle)."""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def wrap_90(angle):
    """The gripper is symmetric: an angle and angle + 180 are the same grasp."""
    return (angle + 90) % 180 - 90


# ---------------------------------------------------------------- table plane

RELATIVE_TOLERANCES = (0.03, 0.02, 0.012, 0.008)   # of the median sampled depth; Sam's 15/8/5/4 mm at ~450 mm
MIN_PLANE_INLIER_FRACTION = 0.5
# The first fit uses only samples within this fraction of the median depth, so a region far off the table (floor past
# the table edge, a laptop) cannot skew it; later iterations re-admit any sample within tolerance of the fitted plane.
INITIAL_DEPTH_BAND = 0.1


@dataclass(frozen=True)
class TablePlane:
    """1/depth is linear in (u, v) for a plane seen through a pinhole camera: 1000/d = c0*u/1000 + c1*v/1000 + c2."""
    c0: float
    c1: float
    c2: float
    inlier_fraction: float
    rms_mm: float
    samples: int
    tolerances_mm: tuple
    median_depth_mm: float = 0.0
    excluded_fraction: float = 0.0

    @property
    def reliable(self):
        """Enough of the (non-object) samples sit within the final tolerance. Below this, heights are not trusted."""
        return self.inlier_fraction >= MIN_PLANE_INLIER_FRACTION

    def depth_at(self, u, v):
        return 1000.0 / (self.c0 * u / 1000.0 + self.c1 * v / 1000.0 + self.c2)

    def height_map(self, depth, invalid_value=0):
        """Height above the table per pixel (mm); NaN where depth is invalid."""
        np = _np()
        h, w = depth.shape
        v, u = np.mgrid[0:h, 0:w]
        table = self.depth_at(u.astype(float), v.astype(float))
        out = table - depth.astype(float)
        out[depth == invalid_value] = np.nan
        return out

    def to_dict(self):
        d = asdict(self)
        d["tolerances_mm"] = list(self.tolerances_mm)
        d["model"] = "inverse_depth_linear_in_uv"
        d["reliable"] = self.reliable
        d["min_inlier_fraction"] = MIN_PLANE_INLIER_FRACTION
        return d


def dilate(mask, radius, step=1):
    np = _np()
    out = mask.copy()
    for dy in range(-radius, radius + 1, step):
        for dx in range(-radius, radius + 1, step):
            if dy or dx:
                out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return out


def fit_table_plane(depth, invalid_value=0, step=4, tolerances_mm=None, min_samples=200, exclude=None):
    """Iteratively refit on inliers so blocks do not skew the plane (Sam's fit_table). Returns None if too sparse.

    tolerances_mm defaults to RELATIVE_TOLERANCES times the median sampled depth, because depth noise grows with
    distance; `exclude` (bool HxW, e.g. dilated detection masks) removes known objects from the fit up front.
    """
    np = _np()
    h, w = depth.shape
    v, u = np.mgrid[0:h:step, 0:w:step]
    d = depth[::step, ::step].astype(float)
    valid = d != invalid_value
    excluded_fraction = 0.0
    if exclude is not None:
        ex = exclude[::step, ::step]
        excluded_fraction = float((valid & ex).sum() / max(1, valid.sum()))
        valid &= ~ex
    u, v, d = u[valid].astype(float), v[valid].astype(float), d[valid]
    if d.size < min_samples:
        return None
    median_depth = float(np.median(d))
    if tolerances_mm is None:
        tolerances_mm = tuple(r * median_depth for r in RELATIVE_TOLERANCES)
    a = np.column_stack([u / 1000.0, v / 1000.0, np.ones_like(u)])
    b = 1000.0 / d
    inliers = np.abs(d - median_depth) <= INITIAL_DEPTH_BAND * median_depth
    if inliers.sum() < max(3, min_samples // 4):
        inliers = np.ones(d.size, dtype=bool)
    coef = None
    for tol in tolerances_mm:
        if inliers.sum() < max(3, min_samples // 4):
            break
        coef, *_ = np.linalg.lstsq(a[inliers], b[inliers], rcond=None)
        pred = 1000.0 / (a @ coef)
        inliers = np.abs(pred - d) < tol
    if coef is None or inliers.sum() < 3:
        return None
    residual = (1000.0 / (a @ coef) - d)[inliers]
    return TablePlane(float(coef[0]), float(coef[1]), float(coef[2]), float(inliers.mean()),
                      float(np.sqrt(np.mean(residual ** 2))), int(d.size), tuple(float(t) for t in tolerances_mm),
                      median_depth, excluded_fraction)


def table_plane_in_base(plane, intr, extrinsics, width, height, grid=5):
    """Transform sampled table points to robot_base: mean z, spread and normal tilt from vertical.
    For a level setup the mean should agree with the configured support-surface offset; a scalar offset
    cannot correct tilt or camera rotation errors, so the spread is reported too."""
    np = _np()
    us = np.linspace(width * 0.1, width * 0.9, grid)
    vs = np.linspace(height * 0.1, height * 0.9, grid)
    pts = np.array([pixel_to_base(u, v, plane.depth_at(u, v), intr, extrinsics) for u in us for v in vs])
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(normal[2])))))
    return {"frame": "robot_base", "mean_z_mm": float(pts[:, 2].mean()), "z_spread_mm": float(pts[:, 2].max() - pts[:, 2].min()),
            "normal_tilt_deg": tilt, "samples": int(len(pts))}


# ---------------------------------------------------------------- block estimate

@dataclass
class BlockEstimate:
    mask_index: int
    candidate_id: str | None
    pixel_center: tuple | None = None
    image_angle_deg: float | None = None
    length_px: float | None = None
    width_px: float | None = None
    depth_mm: float | None = None
    height_above_table_mm: float | None = None
    valid_fraction: float = 0.0
    pixels: int = 0
    mm_per_px: float | None = None
    length_mm: float | None = None
    width_mm: float | None = None
    camera_point_mm: tuple | None = None
    base_point_mm: tuple | None = None
    base_long_side_deg: float | None = None
    top_z_in_robot_base_mm: float | None = None
    flags: list = field(default_factory=list)
    rejection_reasons: list = field(default_factory=list)

    @property
    def rejected(self):
        return bool(self.rejection_reasons)

    def to_dict(self):
        d = asdict(self)
        d["units"] = {"length": "mm", "angle": "deg", "pixel_frame": "rgb_pixels", "pose_frame": "robot_base"}
        d["rejected"] = self.rejected
        return d


def erode(mask, radius):
    np = _np()
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy or dx:
                out &= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    # roll wraps around; pixels on the border are not trusted after erosion
    if radius:
        out[:radius, :] = out[-radius:, :] = False
        out[:, :radius] = out[:, -radius:] = False
    return out


def estimate_block(mask, depth, height_map, calib, mask_index=0, candidate_id=None, invalid_value=0,
                   erode_px=1, min_valid_fraction=0.5, min_pixels=30, top_band_mm=6.0):
    """Fit block pose and dimensions from one mask and registered depth. Never invents geometry: anything the
    calibration cannot support stays None with a reason."""
    np = _np()
    est = BlockEstimate(mask_index, candidate_id)
    dims = calib.block
    region = erode(mask, erode_px) if erode_px and mask.sum() > 4 * min_pixels else mask
    valid = region & (depth != invalid_value) & ~np.isnan(height_map)
    est.pixels = int(region.sum())
    if est.pixels < min_pixels:
        est.rejection_reasons.append("mask_too_small")
        return est
    est.valid_fraction = float(valid.sum() / est.pixels)
    if est.valid_fraction < min_valid_fraction:
        est.rejection_reasons.append("insufficient_valid_depth")
        return est
    heights = height_map[valid]
    # The mask covers the whole visible block including sides; use pixels near the top face for the fit.
    band = heights >= (np.percentile(heights, 75) - top_band_mm)
    if band.sum() < min_pixels:
        band = np.ones_like(heights, dtype=bool)
    vs, us = np.nonzero(valid)
    us, vs = us[band].astype(float), vs[band].astype(float)
    u0, v0 = float(us.mean()), float(vs.mean())
    cuu, cvv = float(((us - u0) ** 2).mean()), float(((vs - v0) ** 2).mean())
    cuv = float(((us - u0) * (vs - v0)).mean())
    mid, half_gap = (cuu + cvv) / 2, math.hypot((cuu - cvv) / 2, cuv)
    est.pixel_center = (u0, v0)
    est.image_angle_deg = math.degrees(0.5 * math.atan2(2 * cuv, cuu - cvv))
    est.length_px = math.sqrt(12 * (mid + half_gap))
    est.width_px = math.sqrt(12 * max(mid - half_gap, 0.0))
    est.depth_mm = float(np.median(depth[valid][band]))
    est.height_above_table_mm = float(np.median(heights[band]))
    if calib.intrinsics is None:
        est.rejection_reasons.append("intrinsics_missing")
        return est
    intr = calib.intrinsics
    est.mm_per_px = est.depth_mm / intr.fx
    est.length_mm = est.length_px * est.mm_per_px
    est.width_mm = est.width_px * est.mm_per_px
    # Plausibility against the configured block, with Sam's allowances for trimmed edges and partial views.
    if not (0.5 * dims.length_mm <= est.length_mm <= 1.3 * dims.length_mm):
        est.rejection_reasons.append(f"implausible_length:{est.length_mm:.1f}mm")
    if not (0.3 * dims.width_mm <= est.width_mm <= 1.15 * dims.width_mm):
        est.rejection_reasons.append(f"implausible_width:{est.width_mm:.1f}mm")
    if est.height_above_table_mm < 0.5 * dims.height_mm:
        est.rejection_reasons.append(f"height_too_low:{est.height_above_table_mm:.1f}mm")
    elif est.height_above_table_mm > 3 * dims.height_mm + 5:
        est.rejection_reasons.append(f"implausible_height:{est.height_above_table_mm:.1f}mm")
    elif est.height_above_table_mm > 1.5 * dims.height_mm:
        est.flags.append("possibly_stacked_or_resting_on_another")
    if est.width_mm < 0.8 * dims.width_mm and est.height_above_table_mm > 1.3 * dims.height_mm:
        est.flags.append("possibly_on_edge")
    if est.length_mm < 0.8 * dims.length_mm:
        est.flags.append("short_visible_top:possibly_partially_covered")
    est.camera_point_mm = deproject(u0, v0, est.depth_mm, intr)
    if calib.extrinsics is not None:
        est.base_point_mm = calib.extrinsics.apply(est.camera_point_mm)
        a = math.radians(est.image_angle_deg)
        along = pixel_to_base(u0 + 50 * math.cos(a), v0 + 50 * math.sin(a), est.depth_mm, intr, calib.extrinsics)
        est.base_long_side_deg = wrap_90(world_angle(est.base_point_mm, along))
    else:
        est.flags.append("pose_in_robot_base_unavailable:extrinsics_missing")
    ss = calib.support_surface
    if ss.z_in_robot_base_mm is not None:
        est.top_z_in_robot_base_mm = ss.z_in_robot_base_mm + est.height_above_table_mm
        if est.base_point_mm is not None and ss.verified and calib.extrinsics.verified:
            if abs(est.base_point_mm[2] - est.top_z_in_robot_base_mm) > 15.0:
                est.rejection_reasons.append("calibration_inconsistent:extrinsics_vs_support_surface")
    if calib.workspace is not None and est.base_point_mm is not None and not calib.workspace.contains(est.base_point_mm):
        est.rejection_reasons.append("out_of_workspace")
    elif calib.workspace is None:
        est.flags.append("workspace_unchecked")
    for reason in calib.missing():
        est.rejection_reasons.append(f"calibration:{reason}")
    return est


def geometry_quality(est, dims):
    """0..1 deterministic quality for ranking: depth coverage times dimensional agreement. Not a success probability."""
    if est.length_mm is None or est.width_mm is None:
        return 0.0
    err = (abs(est.length_mm - dims.length_mm) / dims.length_mm + abs(est.width_mm - dims.width_mm) / dims.width_mm) / 2
    return round(max(0.0, est.valid_fraction * (1.0 - min(1.0, err))), 4)


# ---------------------------------------------------------------- grasps and finger clearance

MIN_GRIP_DEPTH_MM = 8.0           # the claws must reach at least this far below a block's top to hold it
OBSTRUCTION_CLEARANCE_MM = 2.0    # tips stop this far above a low neighbour they would otherwise land on


ON_EDGE_TOP_MARGIN_MM = 6.0       # a top up to one block width + this is taken as resting on the table (on its edge) ...
ON_EDGE_MAX_TOP_WIDTH = 0.9       # ... when its visible top is narrow: at most this fraction of the block width, or flagged on
                                  # edge (on edge 11-20 mm on 30 Oct; flat tops 22.9-23.8). A full-width top at ~30 mm is a flat
                                  # block on another and keeps that level, so an aligned pair is not gripped together.


def narrow_top(flags, width_mm, dims):
    """The visible top is a block's narrow side (it lies on its edge): flagged on edge, or measured no wider than
    ON_EDGE_MAX_TOP_WIDTH of the block width."""
    return "possibly_on_edge" in (flags or []) or (width_mm is not None and width_mm <= ON_EDGE_MAX_TOP_WIDTH * dims.width_mm)


def resting_level_mm(block_top_mm, block_thickness_mm, block_width_mm=None):
    """Height of the surface a block rests on: 0 for the table, one thickness for a block lying on another, ... The
    level is the rounded count of thicknesses under the top, so a block on a tilted or edge-resting one is treated as
    resting on the nearest whole level below it. block_width_mm is passed only for a block whose top is narrow
    (narrow_top: it lies on its edge); then a top no higher than one block width + ON_EDGE_TOP_MARGIN_MM rests on the
    table: a block on its edge (25 mm tall, Jenga logo up) read 22-29 mm, which rounds to "on a block at 15 mm", and
    the claws stopped 15 mm too high and missed it (30 Oct 20:15-20:21, six picks). A lower block that is really there
    shows in the claw strips, and claw_obstruction raises the tips over it."""
    if not block_thickness_mm or block_top_mm is None:
        return 0.0
    if block_width_mm and block_top_mm <= block_width_mm + ON_EDGE_TOP_MARGIN_MM:
        return 0.0
    return max(0, round(block_top_mm / block_thickness_mm) - 1) * block_thickness_mm


# ---------------------------------------------------------------- lying flat (gravity reorientation)

REORIENT_TILT_DEG = 8.0   # a top face tilted more than this is re-laid by a drop (flat blocks measured <= 6.8 deg, 30 Oct)


def top_face_tilt_deg(mask, height_map, mm_per_px, band_mm=6.0, trim_mm=3.0, min_pixels=30):
    """Tilt of a block's visible top face against the table (deg), from the height map. SAM masks also cover side
    faces, so the plane is seeded with the highest pixels only and refitted on every mask pixel within trim_mm of it.
    None when there is too little valid depth to fit a face."""
    np = _np()
    if not mm_per_px:
        return None
    vs, us = np.nonzero(mask & ~np.isnan(height_map))
    if len(us) < min_pixels:
        return None
    h = height_map[vs, us]
    a = np.stack([us * mm_per_px, vs * mm_per_px, np.ones(len(us))], axis=1)
    seed = h >= np.percentile(h, 75) - band_mm
    if seed.sum() < min_pixels:
        seed = np.ones(len(h), dtype=bool)
    coef = np.linalg.lstsq(a[seed], h[seed], rcond=None)[0]
    face = np.abs(a @ coef - h) <= trim_mm
    if face.sum() >= min_pixels:
        coef = np.linalg.lstsq(a[face], h[face], rcond=None)[0]
    return math.degrees(math.atan(math.hypot(coef[0], coef[1])))


def reorient_reasons(geometry, tilt_deg, max_tilt_deg=REORIENT_TILT_DEG):
    """Why a pick candidate should be laid flat by a drop before it goes to the pallet: its top face is tilted, or it
    stands on its long edge (narrow top, more than one block high). Empty when it already lies flat."""
    reasons = []
    if tilt_deg is not None and tilt_deg > max_tilt_deg:
        reasons.append(f"tilted:{tilt_deg:.1f}deg")
    if "possibly_on_edge" in (geometry.get("flags") or []):
        reasons.append("on_edge")
    return reasons


def lying_flat_reasons(geometry, tilt_deg, dims, max_tilt_deg=REORIENT_TILT_DEG):
    """Check a block after the drop: no thicker than one block (not on its edge, 25 mm, or end, 75 mm), the full
    footprint visible, top face level. Empty when it lies flat."""
    reasons = []
    h = geometry.get("height_above_table_mm")
    # Only too high counts: on its edge reads ~25 mm, on end ~75. Low readings are depth bias, not an orientation
    # (30 Oct: a block that had landed flat read 10.4 mm; flat pallet blocks read 11.9-14.7).
    if h is None or h > 1.3 * dims.height_mm:
        reasons.append("height:unknown" if h is None else f"height:{h:.1f}mm_not_one_block")
    if "possibly_on_edge" in (geometry.get("flags") or []):
        reasons.append("on_edge")
    length, width = geometry.get("length_mm"), geometry.get("width_mm")
    if length is None or length < 0.85 * dims.length_mm:
        reasons.append("length:unknown" if length is None else f"short:{length:.1f}mm")
    if width is None or not (0.7 * dims.width_mm <= width <= 1.2 * dims.width_mm):
        reasons.append("width:unknown" if width is None else f"width:{width:.1f}mm")
    if tilt_deg is None or tilt_deg > max_tilt_deg:
        reasons.append("tilt:unknown" if tilt_deg is None else f"tilt:{tilt_deg:.1f}deg")
    return reasons


def standing_on_end(geometry, dims):
    """A block standing upright on its end: about one block length tall, with only its small end face (25 x 15 mm)
    visible from above. Such a block is rejected for picking (implausible height and length); it is knocked over."""
    h, length, width = geometry.get("height_above_table_mm"), geometry.get("length_mm"), geometry.get("width_mm")
    return (h is not None and 0.7 * dims.length_mm <= h <= 1.2 * dims.length_mm and length is not None
            and length <= 0.6 * dims.length_mm and (width is None or width <= 1.3 * dims.width_mm))


def on_edge_on_table(geometry, dims):
    """A block lying on its long narrow side directly on the table: about one block width tall (25 mm) with its 15 mm
    side on top (estimate_block flags it possibly_on_edge). resting_level_mm reads its 25 mm as resting on a block;
    it is not (30 Oct, kazy-palletize-30oct: two drops left such a block on its edge both times)."""
    h = geometry.get("height_above_table_mm")
    return "possibly_on_edge" in (geometry.get("flags") or []) and h is not None and 1.3 * dims.height_mm < h <= dims.width_mm + 6.0


def upright_kind(geometry, dims):
    """'standing' (on its end, ~75 mm), 'on_edge' (on its narrow side on the table, ~25 mm) or None. Both are swept
    over across their thin side so they fall onto their wide face."""
    if standing_on_end(geometry, dims):
        return "standing"
    if on_edge_on_table(geometry, dims):
        return "on_edge"
    return None


PALLET_YAW_TOLERANCE_DEG = 10.0   # a pallet block turned further than this from the row's long side is taken off (pushes cannot turn it)
PALLET_TILT_DEG = 15.0            # tilt alone takes a pallet block off only above this: touching blocks read 8.2-8.3 deg at a normal
                                  # 13 mm height (30 Oct 20:11/20:13, a good block taken off twice); resting on a neighbour shows in height


def misplaced_on_pallet_reasons(geometry, tilt_deg, dims, max_tilt_deg=PALLET_TILT_DEG, row_long_deg=None,
                                max_yaw_deg=PALLET_YAW_TOLERANCE_DEG):
    """A pallet block that must be taken off and laid flat again: not one block thick (on a neighbour, on its edge),
    standing on its edge, tilted, or (given row_long_deg) turned off the row's long side, e.g. rotated 90 deg or lying
    diagonally. Only positive evidence counts here: pallet blocks touch, so their masks can be trimmed, and a short
    footprint or an unmeasurable face is not a reason to disturb a good row."""
    reasons = [r for r in lying_flat_reasons(geometry, tilt_deg, dims, max_tilt_deg)
               if r == "on_edge" or (r.startswith("height:") and r != "height:unknown") or (r.startswith("tilt:") and r != "tilt:unknown")]
    long_deg = geometry.get("base_long_side_deg")
    if row_long_deg is not None and long_deg is not None and abs(wrap_90(long_deg - row_long_deg)) > max_yaw_deg:
        reasons.append(f"rotated:{wrap_90(long_deg - row_long_deg):+.0f}deg")
    return reasons


def grip_tips_mm(block_height, claws_deg, params, block_thickness_mm=None, block_width_mm=None):
    """How far above the table the claw tips go to grip a block whose top is block_height above the table (Sam's
    grip_tips_mm). The floor tips_min_mm is relative to the surface the block rests on, so a block on top of another
    in a heap is gripped with the same depth as one on the table, never with claws driven into the block below."""
    depth = params.grasp_depth_mm + (params.ends_extra_depth_mm if claws_deg == ALONG else 0.0)
    return max(block_height - depth, resting_level_mm(block_height, block_thickness_mm, block_width_mm) + params.tips_min_mm)


def grip_points(params):
    return {"middle": (ACROSS, [0.0]), "side": (ACROSS, [params.side_shift_mm, -params.side_shift_mm]), "ends": (ALONG, [0.0])}


def claw_obstruction(height_map, est, shift_mm, claws_deg, calib, step_px=2):
    """Sample the two strips where the open claws come down beside the block (Sam's claw_obstruction).

    Returns {status: clear|blocked|unobservable, ...}. Unknown (invalid or off-image) samples are counted and make
    the strip unobservable above a fraction, rather than being silently treated as clear.
    """
    np = _np()
    finger, dims, params = calib.finger, calib.block, calib.grasp
    h, w = height_map.shape
    scale = 1.0 / est.mm_per_px
    a = math.radians(est.image_angle_deg)
    c = math.radians(est.image_angle_deg + claws_deg)
    close_u, close_v = math.cos(c), math.sin(c)   # the way the claws close
    face_u, face_v = -close_v, close_u            # across each claw's face
    # Anything in the claw strips above where the tips will stop is an obstruction: for a block resting on another,
    # that includes the lower block if it lies under the strips.
    edge_width = dims.width_mm if narrow_top(est.flags, est.width_mm, dims) else None
    limit = max(grip_tips_mm(est.height_above_table_mm, claws_deg, params, dims.height_mm, edge_width), params.obstruction_min_mm)
    # Hard limit: the claws may stop on top of a lower neighbour as long as MIN_GRIP_DEPTH_MM of the block remains
    # below the tips (heap: a block lying across two others). Anything higher than that is a real obstruction.
    hard_limit = max(est.height_above_table_mm - MIN_GRIP_DEPTH_MM, limit)
    half_block = (dims.length_mm if claws_deg == ALONG else dims.width_mm) / 2
    inner = max(half_block + finger.margin_mm, finger.open_gap_mm / 2 - finger.margin_mm)
    outer = finger.open_gap_mm / 2 + finger.claw_thickness_mm + finger.margin_mm
    cu = est.pixel_center[0] + shift_mm * scale * math.cos(a)
    cv = est.pixel_center[1] + shift_mm * scale * math.sin(a)
    step_mm = step_px / scale
    blocked, unknown, total = [], 0, 0
    for side in (1, -1):
        t = -finger.finger_width_mm / 2
        while t <= finger.finger_width_mm / 2:
            r = inner
            while r <= outer:
                closing, facing = side * r * scale, t * scale
                u = int(round(cu + closing * close_u + facing * face_u))
                v = int(round(cv + closing * close_v + facing * face_v))
                total += 1
                if not (0 <= u < w and 0 <= v < h) or np.isnan(height_map[v, u]):
                    unknown += 1
                elif height_map[v, u] > limit:
                    blocked.append(float(height_map[v, u]))
                r += step_mm
            t += step_mm
    hard = [b for b in blocked if b > hard_limit]
    result = {"samples": total, "blocked_samples": len(blocked), "hard_blocked_samples": len(hard), "unknown_samples": unknown,
              "limit_mm_above_table": limit, "hard_limit_mm_above_table": hard_limit, "strip_inner_mm": inner, "strip_outer_mm": outer,
              "obstruction_height_mm": float(np.median(blocked)) if blocked else None, "raised_tips_mm": None}
    if len(hard) >= params.min_blocked_samples:
        result["status"] = "blocked"
    elif total and unknown / total > params.max_unknown_fraction:
        result["status"] = "unobservable"
    else:
        result["status"] = "clear"
        if len(blocked) >= params.min_blocked_samples:
            # Low obstruction under the claws: stop the tips just above its top instead of beside the block's bottom.
            result["raised_tips_mm"] = float(np.percentile(blocked, 90)) + OBSTRUCTION_CLEARANCE_MM
    return result


def evaluate_grasps(est, height_map, calib, styles=("middle", "side", "ends")):
    """Pure geometric eligibility per grasp style/shift. Does not establish motion-path safety."""
    options = []
    params, finger, dims = calib.grasp, calib.finger, calib.block
    for style in styles:
        claws_deg, shifts = grip_points(params)[style]
        for shift in shifts:
            opt = {"style": style, "claws_deg_to_long_side": claws_deg, "shift_mm_along_long_side": shift,
                   "approach": "vertical_from_above", "retreat": "vertical_lift", "lift_mm": params.pick_lift_mm,
                   "tips_above_table_mm": None, "clearance": None, "eligible": False, "reasons": []}
            if est.rejected:
                opt["reasons"].append("block_rejected")
            if style == "ends" and finger.open_gap_mm / 2 <= dims.length_mm / 2 + finger.margin_mm:
                opt["reasons"].append("gripper_too_narrow_for_ends")
            if est.pixel_center is None or est.mm_per_px is None or est.height_above_table_mm is None:
                opt["reasons"].append("geometry_unavailable")
            if not opt["reasons"]:
                edge_width = dims.width_mm if narrow_top(est.flags, est.width_mm, dims) else None
                opt["tips_above_table_mm"] = grip_tips_mm(est.height_above_table_mm, claws_deg, params, dims.height_mm, edge_width)
                opt["resting_level_mm"] = resting_level_mm(est.height_above_table_mm, dims.height_mm, edge_width)
                clearance = claw_obstruction(height_map, est, shift, claws_deg, calib)
                opt["clearance"] = clearance
                if clearance["status"] != "clear":
                    opt["reasons"].append(f"fingers_{clearance['status']}")
                elif clearance.get("raised_tips_mm") is not None and clearance["raised_tips_mm"] > opt["tips_above_table_mm"]:
                    opt["tips_above_table_mm"] = clearance["raised_tips_mm"]
                    opt["tips_raised_over_neighbour"] = True
            opt["eligible"] = not opt["reasons"]
            if est.base_long_side_deg is not None and opt["eligible"]:
                opt["claws_close_direction_deg_in_base"] = wrap_90(est.base_long_side_deg + claws_deg)
            options.append(opt)
    return options
