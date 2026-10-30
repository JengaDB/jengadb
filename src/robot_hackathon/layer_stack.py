"""Supervised layer stacking: a human builds a three-block layer, the robot grips it as one bundle and stacks it on the
tower, rotated 90 degrees to the layer below. One run per command; the operator presses Enter before any motion.

Precision comes from measurement, not learning: both layers are measured from a close hover capture (not the overview),
the claws are aligned to the measured angle, the bundle's offset in the gripper is read at grip time and compensated
over the tower, and every run ends with a verification capture whose residual is appended to a per-setup log. A
constant offset fitted from those residuals can be applied (`--apply-offset`) once at least MIN_RUNS_FOR_OFFSET runs agree.

Geometry follows live_executor (Sam's jenga.py conventions). No retries: a refused move, a failed grab or a failed check
stops the run, and the next decision is the operator's. Sessions stay enabled.
"""
import asyncio
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .grasp_geometry import ACROSS, GraspParams, wrap_90

BLOCKS_PER_LAYER = 3
PARALLEL_TOLERANCE_DEG = 4.0      # every block within this of the layer's long-side angle
SPACING_RANGE_MM = (22.0, 28.0)   # neighbour pitch for a packed layer of 25 mm blocks (the bundle to be gripped)
BASE_SPACING_RANGE_MM = (20.0, 33.0)  # the tower's top layer only has to give a centre and an angle; its blocks may
                                      # have settled apart a little and the overview measures it from far above
                                      # (robot-live-8 18:50: pitch 28.3 measured, run refused)
BASE_MIN_SHARED_CONTACT_MM = 40.0
MIN_SHARED_CONTACT_MM = 55.0      # 45 mm finger width plus margin must overlap all three blocks' long sides
HEIGHT_BAND = 0.3                 # of a block height (4.5 mm): beyond this a top surface is between layers (tilted block)
MIN_BLOCK_HEIGHT_FRACTION = 0.5   # detections lower than half a block are flat objects (paper, tape), not blocks
MAX_TARGET_OFFSET_MM = 15.0       # refuse to stack if the refined base moved this far from the overview estimate
CENTER_TOLERANCE_MM = 3.0         # success: new layer centre within this of the target
YAW_TOLERANCE_DEG = 3.0           # success: new layer within this of exactly 90 deg to the base
MIN_RUNS_FOR_OFFSET = 3
MAX_OFFSET_SPREAD_MM = 2.0        # residual standard deviation above which a constant offset is not trusted
MIN_TIPS_FLOOR_MM = -6.0          # --tips-floor-mm may not send the claw tips further below the calibrated table than this
EDGE_MARGIN_PX = 4                # a mask this close to the image border is cut off: its centre and angle are wrong
# Default flange pose for the overview capture. live_executor.HOME (z 400) did not fit two layers 210 mm apart in one
# image (Kazy, 29 Oct); z 460 did and saw x 199-546, y -128..491 mm of table (Kris, 30 Oct). Raise z or shift x/y with
# --view to cover more table; coverage grows linearly with camera height, block masks shrink the same way (51 px wide
# at z 460, 42 px at 560, 36 px at 650).
OVERVIEW_POSE = (300.0, 150.0, 460.0)


class LayerCheckError(ValueError):
    """A measured layer that must not be gripped or stacked on. The message is the operator-facing reason."""


class NoBaseLayer(LayerCheckError):
    """The tower zone held no blocks in this capture."""


@dataclass
class LayerPose:
    center_xy: tuple            # robot_base mm, mean of the three top centres
    long_side_deg: float        # world direction of the blocks' long side, wrapped to [-90, 90)
    height_above_table_mm: float  # median top-surface height of the layer
    spacing_mm: list            # neighbour pitch across the layer
    shared_contact_mm: float    # length along the long side that all three blocks cover
    level: int                  # 0 = on the table
    block_centers: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["center_xy"] = [round(v, 2) for v in self.center_xy]
        return d


def _axial_mean_deg(angles):
    """Mean of undirected angles (a block's long side is the same at a and a+180)."""
    s = sum(math.sin(math.radians(2 * a)) for a in angles)
    c = sum(math.cos(math.radians(2 * a)) for a in angles)
    return wrap_90(math.degrees(math.atan2(s, c)) / 2)


def _usable(block):
    return block.get("base_point_mm") is not None and block.get("base_long_side_deg") is not None \
        and block.get("height_above_table_mm") is not None


def layer_level(height_above_table_mm, block_height_mm):
    """0 for a block on the table, 1 for one on a single layer, ...; None when between levels."""
    n = height_above_table_mm / block_height_mm
    level = round(n) - 1
    return level if level >= 0 and abs(n - round(n)) <= HEIGHT_BAND else None


def summarize_layer(blocks, block_height_mm=15.0, spacing_range=SPACING_RANGE_MM, min_shared_contact_mm=MIN_SHARED_CONTACT_MM):
    """Validate three measured blocks as one packed, flat, parallel layer and return its pose. The defaults are the
    bundle requirements; use base_layer() for the tower's top layer."""
    blocks = [b for b in blocks if _usable(b)]
    if any(b.get("clipped") for b in blocks):
        raise LayerCheckError("a block is cut off by the image edge; move the view so the whole layer is visible")
    if len(blocks) != BLOCKS_PER_LAYER:
        raise LayerCheckError(f"expected {BLOCKS_PER_LAYER} measured blocks in the layer, saw {len(blocks)}")
    levels = {layer_level(b["height_above_table_mm"], block_height_mm) for b in blocks}
    if None in levels or len(levels) != 1:
        raise LayerCheckError("blocks are not all at one layer height: " +
                              ", ".join(f"{b['height_above_table_mm']:.1f}" for b in blocks) + " mm")
    theta = _axial_mean_deg([b["base_long_side_deg"] for b in blocks])
    worst = max(abs(wrap_90(b["base_long_side_deg"] - theta)) for b in blocks)
    if worst > PARALLEL_TOLERANCE_DEG:
        raise LayerCheckError(f"blocks not parallel: {worst:.1f} deg off the layer angle (limit {PARALLEL_TOLERANCE_DEG})")
    ax, ay = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    xy = [(b["base_point_mm"][0], b["base_point_mm"][1]) for b in blocks]
    across = sorted(x * -ay + y * ax for x, y in xy)
    along = [x * ax + y * ay for x, y in xy]
    spacing = [b - a for a, b in zip(across, across[1:])]
    lo, hi = spacing_range
    if not all(lo <= s <= hi for s in spacing):
        raise LayerCheckError("layer has gaps or overlaps: pitch " + ", ".join(f"{s:.1f}" for s in spacing) +
                              f" mm (want {lo:.0f}-{hi:.0f}); push it together by hand")
    contact = 75.0 - (max(along) - min(along))
    if contact < min_shared_contact_mm:
        raise LayerCheckError(f"block ends are staggered by {75.0 - contact:.1f} mm; line the ends up")
    cx = sum(p[0] for p in xy) / len(xy)
    cy = sum(p[1] for p in xy) / len(xy)
    height = statistics.median(b["height_above_table_mm"] for b in blocks)
    return LayerPose((cx, cy), theta, height, spacing, contact, levels.pop(), [list(p) for p in xy])


def base_layer(blocks, block_height_mm=15.0):
    """The tower's top layer: same flatness/parallel checks as a bundle, looser spacing and end-stagger limits."""
    return summarize_layer(blocks, block_height_mm, spacing_range=BASE_SPACING_RANGE_MM, min_shared_contact_mm=BASE_MIN_SHARED_CONTACT_MM)


def split_scene(blocks, tower_origin_xy, zone_radius_mm, block_height_mm=15.0, ignored=None):
    """Source layer = usable blocks outside the tower zone; base = the highest layer inside it.

    Detections in the zone whose height falls between layer levels are tolerated when they lie *below* the top layer:
    with alternating layers the lower layer's ends stick out beside the top one and their masks mix two depths
    (robot-live-8 verify: 18.4 and 20.4 mm next to a 29 mm top layer). One above the top layer is a tilted or
    edge-resting block and stops the run. `ignored`, if given, receives the tolerated detections."""
    # Flat detections (paper notes, tape) are not blocks: Kris's estimate_block flags them height_too_low as well.
    usable = [b for b in blocks if _usable(b) and b["height_above_table_mm"] >= MIN_BLOCK_HEIGHT_FRACTION * block_height_mm]
    def in_zone(b):
        return math.hypot(b["base_point_mm"][0] - tower_origin_xy[0], b["base_point_mm"][1] - tower_origin_xy[1]) < zone_radius_mm
    outside = [b for b in usable if not in_zone(b)]
    inside = [b for b in usable if in_zone(b)]
    by_level = {}
    for b in inside:
        lvl = layer_level(b["height_above_table_mm"], block_height_mm)
        by_level.setdefault(lvl, []).append(b)
    between = by_level.pop(None, [])
    if not by_level:
        if between:
            raise LayerCheckError("a block in the tower zone is between layer heights (tilted or resting on an edge?)")
        raise NoBaseLayer("no base layer in the tower zone")
    top = max(by_level)
    top_height = min(b["height_above_table_mm"] for b in by_level[top])
    above = [b for b in between if b["height_above_table_mm"] > top_height - HEIGHT_BAND * block_height_mm]
    if above:
        raise LayerCheckError("a block on top of the tower is between layer heights (tilted or resting on an edge?): " +
                              ", ".join(f"{b['height_above_table_mm']:.1f}" for b in above) + " mm")
    if ignored is not None:
        ignored.extend(between)
    return outside, by_level[top]


def tower_layers(top_layer, block_height_mm=15.0):
    """How many layers the tower has, from the measured top layer: its height divided by one block, rounded."""
    return int(round(top_layer.height_above_table_mm / block_height_mm))


@dataclass
class StackPlan:
    source: LayerPose
    base: LayerPose
    claws_close_at_pick_deg: float    # world direction the claws close along at the pick (across the three blocks)
    claws_close_at_place_deg: float   # same, over the tower: the bundle's long side ends up base + 90
    target_center_xy: tuple           # where the bundle's centre should land (base centre + any fitted offset)
    target_long_side_deg: float
    pick_tips_above_table_mm: float   # claw tips at the pick, relative to the table
    release_tips_above_table_mm: float
    applied_offset_xy: tuple = (0.0, 0.0)
    grip_below_top_mm: float = 0.0    # achieved grip depth (requested depth, unless the tips floor clamped it)
    requested_grip_below_top_mm: float = 0.0

    def to_dict(self):
        d = asdict(self)
        d["source"], d["base"] = self.source.to_dict(), self.base.to_dict()
        return d


def plan_stack(source, base, block_height_mm=15.0, params=GraspParams(), release_drop_mm=1.0, offset_xy=(0.0, 0.0)):
    """Grip across the bundle's 75 mm width (claws close perpendicular to the long sides, pressing the three together),
    and release with the bundle's long side at 90 deg to the base layer, centred on it."""
    if source.level != 0:
        raise LayerCheckError("the hand-built layer must lie on the table")
    target_long = wrap_90(base.long_side_deg + 90.0)
    tips_pick = max(source.height_above_table_mm - params.grasp_depth_mm, params.tips_min_mm)
    # The grip depth actually reached once the floor (tips_min_mm) applies; the bundle hangs this far below the block
    # tops in the claws, so the release height has to use it or a clamped pick would push the bundle into the base.
    grip_below_top = source.height_above_table_mm - tips_pick
    tips_release = base.height_above_table_mm + block_height_mm - grip_below_top + release_drop_mm
    target = (base.center_xy[0] - offset_xy[0], base.center_xy[1] - offset_xy[1])
    return StackPlan(source, base, wrap_90(source.long_side_deg + ACROSS), wrap_90(target_long + ACROSS),
                     target, target_long, tips_pick, tips_release, tuple(offset_xy), grip_below_top, params.grasp_depth_mm)


def placement_residual(plan, measured):
    """How far the stacked layer landed from where it should be, in world and in the target layer's own axes."""
    dx = measured.center_xy[0] - plan.target_center_xy[0]
    dy = measured.center_xy[1] - plan.target_center_xy[1]
    a = math.radians(plan.target_long_side_deg)
    along = dx * math.cos(a) + dy * math.sin(a)
    across = -dx * math.sin(a) + dy * math.cos(a)
    yaw = wrap_90(measured.long_side_deg - plan.target_long_side_deg)
    # The physical error is relative to the base's actual centre, not to the offset-shifted command.
    true_dx = measured.center_xy[0] - plan.base.center_xy[0]
    true_dy = measured.center_xy[1] - plan.base.center_xy[1]
    level_ok = measured.level == plan.base.level + 1
    error = math.hypot(true_dx, true_dy)
    return {"command_residual_xy_mm": [round(dx, 2), round(dy, 2)], "along_mm": round(along, 2), "across_mm": round(across, 2),
            "center_error_vs_base_mm": round(error, 2), "yaw_error_deg": round(yaw, 2), "level": measured.level,
            "success": bool(level_ok and error <= CENTER_TOLERANCE_MM and abs(yaw) <= YAW_TOLERANCE_DEG),
            "tolerances": {"center_mm": CENTER_TOLERANCE_MM, "yaw_deg": YAW_TOLERANCE_DEG}}


def blocks_left_behind(source, blocks, block_height_mm=15.0, radius_mm=15.0):
    """How many table-level blocks are still within radius_mm of the source layer's block centres (after a lift)."""
    left = 0
    for c in source.block_centers:
        if any(_usable(b) and layer_level(b["height_above_table_mm"], block_height_mm) == 0
               and math.dist(c, b["base_point_mm"][:2]) <= radius_mm for b in blocks):
            left += 1
    return left


def visible_table_extent(observation, table_z_mm=0.0, margin_px=None):
    """World x/y bounds (mm) of the table area the capture could see: the image corners (inset by margin_px) projected
    through the recorded camera pose onto the plane z = table_z_mm. None when the pose or intrinsics are missing or the
    camera does not look down at the table. Tells the operator where a layer has to lie to be measurable."""
    cam = (observation.get("camera") or {}).get("pose_in_world") or {}
    intr = (observation.get("intrinsics") or {}).get("rgb") or {}
    if cam.get("status") != "read" or not intr:
        return None
    margin_px = EDGE_MARGIN_PX if margin_px is None else margin_px
    t, r = cam["translation_mm"], cam["rotation"]
    fx, fy, cx, cy = intr["focal_x_px"], intr["focal_y_px"], intr["center_x_px"], intr["center_y_px"]
    w, h = intr["width_px"], intr["height_px"]
    pts = []
    for u, v in ((margin_px, margin_px), (w - 1 - margin_px, margin_px), (margin_px, h - 1 - margin_px), (w - 1 - margin_px, h - 1 - margin_px)):
        d = ((u - cx) / fx, (v - cy) / fy, 1.0)
        dw = [sum(r[i][j] * d[j] for j in range(3)) for i in range(3)]
        if dw[2] >= -1e-6:
            return None  # ray does not reach the table
        s = (table_z_mm - t[2]) / dw[2]
        pts.append((t[0] + s * dw[0], t[1] + s * dw[1]))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return {"x_mm": [round(min(xs), 1), round(max(xs), 1)], "y_mm": [round(min(ys), 1), round(max(ys), 1)]}


class ResidualLog:
    """Append-only per-setup record of verified placements. The fitted offset is a calibration, applied only on request."""

    def __init__(self, data_dir, setup_id):
        self.path = Path(data_dir) / "live" / "layer_stack" / f"residuals-{setup_id}.jsonl"

    def append(self, row):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(row, default=str) + "\n")

    def rows(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def fitted_offset(self):
        """Mean command residual over runs placed *without* an offset. None, with a reason, until it is trustworthy."""
        rows = [r for r in self.rows() if r.get("applied_offset_xy") in (None, [0.0, 0.0]) and r.get("residual")]
        if len(rows) < MIN_RUNS_FOR_OFFSET:
            return None, f"{len(rows)} of {MIN_RUNS_FOR_OFFSET} offset-free verified runs"
        xs = [r["residual"]["command_residual_xy_mm"][0] for r in rows]
        ys = [r["residual"]["command_residual_xy_mm"][1] for r in rows]
        spread = max(statistics.pstdev(xs), statistics.pstdev(ys))
        if spread > MAX_OFFSET_SPREAD_MM:
            return None, f"residual spread {spread:.1f} mm exceeds {MAX_OFFSET_SPREAD_MM} mm: error is not a constant offset"
        return (statistics.mean(xs), statistics.mean(ys)), f"mean of {len(rows)} runs, spread {spread:.1f} mm"


# ---------------------------------------------------------------- live run

async def _open_client(config, direct_local, disable_sessions=False):
    """direct_local dials viam-server on the LAN over gRPC instead of the cloud WebRTC relay (measured 29 Oct: median
    0.37 s vs 1.13 s per RGB-D read, max 0.87 s vs 5.39 s). Sessions stay enabled unless the operator disables them
    (--no-sessions): with sessions on, a missed heartbeat expires the session and ends the run mid-flow
    (SESSION_EXPIRED, robot-live-8 18:43, as Sam saw before); with them off, a dropped connection no longer stops a
    move that is already executing, so the e-stop is the only stop for that move."""
    from .capture import open_client
    if not direct_local:
        return await open_client(config, disable_sessions=disable_sessions)
    from viam.robot.client import RobotClient
    config.require_viam()
    host = config.address.split(":")[0]
    if not host.endswith(".viam.cloud") or ".local." in host:
        raise LayerCheckError("--direct-local needs the machine's <name>.viam.cloud address in MACHINE_ADDRESS")
    options = RobotClient.Options.with_api_key(api_key=config.key, api_key_id=config.key_id)
    options.dial_options.disable_webrtc = True
    if disable_sessions:
        # Also stop the SDK's connectivity probe (ResourceNames with a 1 s deadline every 10 s): a server busy with
        # motion planning misses it and the SDK tears down in-flight calls ("Lost connection", robot-live-8 18:36).
        options.disable_sessions = True
        options.check_connection_interval = 0
        options.attempt_reconnect_interval = 0
    return await asyncio.wait_for(RobotClient.at_address(host.replace(".viam.cloud", ".local.viam.cloud") + ":8080", options), timeout=15)


def _blocks(context, width, height):
    """Measured block geometries, each marked `clipped` when its mask touches the image border."""
    out = []
    for c in context["candidates"]:
        if c["geometry"].get("base_point_mm") is None:
            continue
        x0, y0, x1, y1 = c["bbox"]
        clipped = x0 < EDGE_MARGIN_PX or y0 < EDGE_MARGIN_PX or x1 > width - 1 - EDGE_MARGIN_PX or y1 > height - 1 - EDGE_MARGIN_PX
        out.append(dict(c["geometry"], clipped=clipped))
    return out


async def run_layer_stack(config, data_dir, operator, episode_id, tower_origin, seg_provider, direct_local=False,
                          apply_offset=False, confirm=input, plan_only=False, view_pose=None, grip_lower_mm=0.0,
                          source_hint_xy=None, tips_floor_mm=None, expected_layers=None, grip_retries=1, disable_sessions=False,
                          arm_speed=None, approach_speed=None, align_speed=None):
    """One supervised stack. Returns 0 on a verified success, 1 otherwise. See module docstring.
    plan_only: capture from the current pose, measure both layers and print the plan; no motion command is sent.
    source_hint_xy: operator-supplied world x,y (mm) of the hand-built layer when it lies outside the overview image;
    the run hovers there and measures it close up instead of requiring it in the overview.
    tips_floor_mm: lowest claw-tip height above the calibrated table the pick may use (default GraspParams.tips_min_mm);
    negative values are the operator's statement that the calibrated table z sits above the real surface.
    expected_layers: how many layers the tower has before this run (--layers); the measured count (top height divided by
    one block height) must agree or the run stops. Without it the measurement is trusted. The episode state keeps
    `tower_layers` and `layers_stacked`, updated on every verified placement."""
    if expected_layers is not None and expected_layers < 1:
        raise LayerCheckError("--layers counts the layers already on the tower; at least 1")
    if not plan_only:
        config.require_motion(operator)  # before any SDK import or connection
    if tips_floor_mm is not None and not (MIN_TIPS_FLOOR_MM <= tips_floor_mm <= GraspParams().tips_min_mm):
        raise LayerCheckError(f"--tips-floor-mm must be between {MIN_TIPS_FLOOR_MM} and {GraspParams().tips_min_mm}")
    grasp = GraspParams(grasp_depth_mm=GraspParams().grasp_depth_mm + grip_lower_mm,
                        tips_min_mm=GraspParams().tips_min_mm if tips_floor_mm is None else tips_floor_mm)
    from . import proposal, rgbd, segmentation
    from . import grasp_geometry as geom
    from .capture import capture_with_camera
    from .live_executor import PICK_LIFT_MM, TIP_ABOVE_GRIPPER_FRAME_MM, DEFAULT_ARM_SPEED_DEG_S, Episode, Robot, grip_failed
    # Speeds as in run_flow: slow final approach and slow claw alignment, travel speed restored afterwards.
    travel_speed = arm_speed if arm_speed is not None else (DEFAULT_ARM_SPEED_DEG_S if (approach_speed or align_speed) else None)
    from .pipeline import candidate_id
    from .placement import TowerTarget, footprint_clearance_mm
    # The SDK builds resource clients only for APIs registered (imported) before connecting; see live_executor.run_flow.
    from viam.components.arm import Arm  # noqa: F401
    from viam.components.camera import Camera
    from viam.components.gripper import Gripper  # noqa: F401
    from viam.services.motion import MotionClient  # noqa: F401

    ep = Episode(data_dir, episode_id, operator)
    client = await _open_client(config, direct_local, disable_sessions=disable_sessions)
    bot = None
    holding = False
    try:
        await asyncio.wait_for(client.refresh(), timeout=10)  # resource list can lag a viam-server reconfigure
        bot = Robot(client, config)
        camera = Camera.from_robot(client, config.camera_resource)

        size = [0, 0]  # width, height of the latest capture
        last_obs = [None]  # observation_id of the latest capture
        last_doc = [None]  # observation record of the latest capture (camera pose, intrinsics)

        async def measure(label):
            obs = await capture_with_camera(config, data_dir, camera, episode_id, client=client)
            obs, rgb, depth = rgbd.ObservationStore(data_dir, "live").load_arrays(obs["observation_id"])
            raw = await asyncio.to_thread(seg_provider.segment, rgb)
            masks = segmentation.finalize_masks(raw, rgb.shape[1], rgb.shape[0], 200)
            for m in masks:
                m.update(candidate_id=candidate_id(obs["observation_id"], "layer-stack", m["mask_index"]), embedding=None, candidate=None)
            doc = geom.GeometryCalibrationStore(data_dir, "live").load(obs["setup_id"])
            calib = geom.bundle_from_records(obs, False, doc)
            scene = {"source": "observation", "observation": obs, "rgb": rgb, "depth": depth, "masks": masks, "calibration": calib,
                     "geometry_calibration_sha256": None, "segmentation": {"provider": seg_provider.provider_id, "model_version": "layer-stack"},
                     "processing_version": "layer-stack"}
            ctx = proposal.build_context(scene, tower_origin_xy_mm=tower_origin, dry_run=True)
            if ctx["table_plane"].get("status") != "fitted":
                raise LayerCheckError(f"{label}: table plane {ctx['table_plane'].get('status')}; heights are not trustworthy")
            size[:] = [rgb.shape[1], rgb.shape[0]]
            last_obs[0] = obs["observation_id"]
            last_doc[0] = obs
            seen = visible_table_extent(obs, calib.support_surface.z_in_robot_base_mm)
            ep.event("stack", phase="measured", label=label, observation_id=obs["observation_id"], masks=len(masks), table_seen_mm=seen,
                     blocks=[{"xy": [round(b["base_point_mm"][0], 1), round(b["base_point_mm"][1], 1)], "long_deg": round(b["base_long_side_deg"], 1),
                              "height": round(b["height_above_table_mm"], 1)} for b in _blocks(ctx, *size) if _usable(b)])
            if label == "overview" and seen:
                print(f"Overview covers table x {seen['x_mm'][0]:.0f}..{seen['x_mm'][1]:.0f}, y {seen['y_mm'][0]:.0f}..{seen['y_mm'][1]:.0f} mm "
                      f"(raise or shift --view to widen)")
            return ctx, calib

        def source_from_overview(outside, table_z):
            """The hand-built layer as seen in the overview, or the operator's hint when it is out of frame."""
            try:
                return summarize_layer(outside, dims.height_mm), None
            except LayerCheckError as exc:
                if source_hint_xy is not None and not plan_only:
                    return None, f"overview did not show a usable source layer ({exc}); using --source {source_hint_xy}"
                seen = visible_table_extent(last_doc[0], table_z) if last_doc[0] else None
                where = (f"camera saw the table over x {seen['x_mm'][0]:.0f}..{seen['x_mm'][1]:.0f}, y {seen['y_mm'][0]:.0f}..{seen['y_mm'][1]:.0f} mm"
                         if seen else "visible area unknown (camera pose missing)")
                raise LayerCheckError(f"{exc}. Source layer: {where}; {len(outside)} block(s) outside the tower zone. Put the "
                                      f"layer inside that area at least {zone:.0f} mm from the tower, raise --view, or pass "
                                      f"--source x,y with its approximate position") from None

        async def hover_over(xy):
            g = await bot.gripper_origin()
            ok, _ = await bot.move_delta(dx=xy[0] - g[0], dy=xy[1] - g[1])
            if not ok:
                raise LayerCheckError(f"planner refused the hover move to {xy[0]:.1f}, {xy[1]:.1f}")

        async def move_z_to(tips_above_table, table_z):
            g = await bot.gripper_origin()
            target_z = table_z + tips_above_table - TIP_ABOVE_GRIPPER_FRAME_MM
            if target_z < g[2]:
                ok = await bot.descend_to(target_z, approach_speed, travel_speed)   # slow for the last stretch
                await bot.set_speed(travel_speed)
            else:
                ok, _ = await bot.move_delta(dz=target_z - g[2], near_table=True)
            if not ok:
                raise LayerCheckError("planner refused the vertical move")

        if await bot.arm.is_moving():
            raise LayerCheckError("arm is moving; wait for it to stop")
        view = tuple(view_pose) if view_pose else OVERVIEW_POSE   # flange pose for the overview capture
        f = await bot.flange_pose()
        if plan_only and math.dist((f.x, f.y, f.z), view) > 5:
            raise LayerCheckError(f"plan-only measures from the view pose {view}; arm is at {f.x:.0f}, {f.y:.0f}, {f.z:.0f} and plan-only never moves it")
        if math.dist((f.x, f.y, f.z), view) > 5:
            await asyncio.to_thread(confirm, f"Arm is at {f.x:.0f}, {f.y:.0f}, {f.z:.0f}, not the view pose {view}. Press Enter to move there (Ctrl-C to stop) ")
            ok, _ = await bot.move_to(*view)
            if not ok:
                raise LayerCheckError("planner refused the move to the view pose")
        if not plan_only:
            await bot.gripper.open()

        # 1. Overview: find both layers.
        ctx, calib = await measure("overview")
        dims = calib.block
        table_z = calib.support_surface.z_in_robot_base_mm
        zone = footprint_clearance_mm(TowerTarget(tuple(tower_origin), calib.support_surface, dims, calib.finger))
        overview_blocks = _blocks(ctx, *size)
        seen = visible_table_extent(last_doc[0], table_z) if last_doc[0] else None
        in_view = bool(seen and seen["x_mm"][0] <= tower_origin[0] <= seen["x_mm"][1] and seen["y_mm"][0] <= tower_origin[1] <= seen["y_mm"][1])
        where = (f"camera saw x {seen['x_mm'][0]:.0f}..{seen['x_mm'][1]:.0f}, y {seen['y_mm'][0]:.0f}..{seen['y_mm'][1]:.0f} mm"
                 if seen else "visible area unknown")
        look_at_tower = None   # reason to hover over the tower origin and measure the base from there
        try:
            outside, base_blocks = split_scene(overview_blocks, tower_origin, zone, dims.height_mm)
            if any(b.get("clipped") for b in base_blocks):
                look_at_tower = "the base layer is cut off by the overview image edge"
        except NoBaseLayer as exc:
            if in_view or plan_only:
                raise LayerCheckError(f"{exc} at {tower_origin[0]:.0f}, {tower_origin[1]:.0f} ({where}); is --tower-origin the tower?") from None
            look_at_tower = f"tower origin {tower_origin[0]:.0f}, {tower_origin[1]:.0f} is outside the overview image"
            outside = [b for b in overview_blocks if _usable(b) and b["height_above_table_mm"] >= MIN_BLOCK_HEIGHT_FRACTION * dims.height_mm]
        if look_at_tower and not plan_only:
            # Keep the overview's source blocks; go and look at the tower from directly above.
            note = f"{look_at_tower}; hovering over the tower origin to measure the base"
            print(note)
            ep.event("stack", phase="base_out_of_view", tower_origin=list(tower_origin), seen=seen, note=note)
            await hover_over(tower_origin)
            ctx_t, _ = await measure("base_find")
            base_blocks = split_scene(_blocks(ctx_t, *size), tower_origin, zone, dims.height_mm)[1]
        base0 = base_layer(base_blocks, dims.height_mm)

        def at_source(blocks):
            """With a known source spot (the pallet), only blocks within the footprint zone of it count as the source
            layer; loose heap blocks elsewhere outside the tower zone are not part of it (robot-live-8 18:56: the
            heap made the overview see 6 and the close-up refuse on a clipped heap block)."""
            if source_hint_xy is None:
                return blocks
            return [b for b in blocks if math.hypot(b["base_point_mm"][0] - source_hint_xy[0], b["base_point_mm"][1] - source_hint_xy[1]) < zone]

        source0, hint_note = source_from_overview(at_source(outside), table_z)
        source0_xy = source0.center_xy if source0 else tuple(source_hint_xy)
        if hint_note:
            print(hint_note)
            ep.event("stack", phase="source_hint", source_xy=list(source0_xy), note=hint_note)

        def check_layers(top, label):
            """Layer count from the measured top layer; --layers, when given, must agree."""
            n = tower_layers(top, dims.height_mm)
            blocks_high = top.height_above_table_mm / dims.height_mm
            stacked = ep.state.get("layers_stacked", 0)
            print(f"Tower ({label}): top layer {top.height_above_table_mm:.1f} mm = {blocks_high:.2f} blocks -> {n} layer(s); "
                  f"this run places layer {n + 1}; stacked by the robot in this episode so far: {stacked}")
            ep.event("stack", phase="layers", label=label, tower_layers=n, top_height_mm=round(top.height_above_table_mm, 1),
                     blocks_high=round(blocks_high, 2), expected_layers=expected_layers, layers_stacked=stacked)
            if expected_layers is not None and n != expected_layers:
                # The measurement wins (operator decision, 30 Oct): --layers is a cross-check that is reported, not enforced.
                print(f"NOTE: --layers {expected_layers} but the tower measures {n} layer(s) ({label}: top {top.height_above_table_mm:.1f} mm "
                      f"= {blocks_high:.2f} blocks); using the measurement", flush=True)
                ep.event("stack", phase="layers_mismatch", label=label, expected_layers=expected_layers, measured_layers=n)
            return n

        layers_before = check_layers(base0, "overview")
        if plan_only:
            log = ResidualLog(data_dir, config.setup_id)
            offset, offset_note = log.fitted_offset()
            plan = plan_stack(source0, base0, dims.height_mm, params=grasp)
            ep.event("stack", phase="plan_only", plan=plan.to_dict(), fitted_offset=offset, fitted_offset_note=offset_note,
                     table_z=table_z, table_z_verified=calib.support_surface.verified, zone_radius_mm=round(zone, 1))
            print(f"\nPLAN ONLY (overview capture, no motion). Source ({source0.center_xy[0]:.1f}, {source0.center_xy[1]:.1f}) "
                  f"long {source0.long_side_deg:.1f} deg, pitch {[round(v, 1) for v in source0.spacing_mm]} mm; base ({base0.center_xy[0]:.1f}, "
                  f"{base0.center_xy[1]:.1f}) long {base0.long_side_deg:.1f} deg level {base0.level}; place long {plan.target_long_side_deg:.1f} deg")
            return 0

        # 2. Refine each layer from a hover directly above it (closer to the image centre, less extrinsic leverage).
        await hover_over(base0.center_xy)
        ctx_b, _ = await measure("base_close")
        tolerated = []
        base = base_layer(split_scene(_blocks(ctx_b, *size), tower_origin, zone, dims.height_mm, ignored=tolerated)[1], dims.height_mm)
        if tolerated:
            ep.event("stack", phase="between_level_ignored", label="base_close",
                     heights_mm=[round(b["height_above_table_mm"], 1) for b in tolerated])
        if math.dist(base.center_xy, base0.center_xy) > MAX_TARGET_OFFSET_MM:
            raise LayerCheckError(f"base measured {math.dist(base.center_xy, base0.center_xy):.1f} mm apart in two captures")
        layers_before = check_layers(base, "close-up")
        base_view = await bot.gripper_origin()
        base_view_claws_deg = await bot.close_direction_deg()  # wrist camera: viewpoint includes the claw angle
        await hover_over(source0_xy)
        ctx_s, _ = await measure("source_close")
        source_blocks = at_source(split_scene(_blocks(ctx_s, *size), tower_origin, zone, dims.height_mm)[0])
        try:
            source = summarize_layer(source_blocks, dims.height_mm)
        except LayerCheckError as exc:
            seen = visible_table_extent(last_doc[0], table_z) if last_doc[0] else None
            where = f"; close-up saw x {seen['x_mm'][0]:.0f}..{seen['x_mm'][1]:.0f}, y {seen['y_mm'][0]:.0f}..{seen['y_mm'][1]:.0f} mm" if seen else ""
            raise LayerCheckError(f"source close-up: {exc}{where}") from None

        log = ResidualLog(data_dir, config.setup_id)
        offset, offset_note = log.fitted_offset()
        use = offset if (apply_offset and offset) else (0.0, 0.0)
        # Deeper grip changes pick and release tips together, so the bundle still lands release_drop above the base.
        plan = plan_stack(source, base, dims.height_mm, params=grasp, offset_xy=use)
        ep.event("stack", phase="plan", plan=plan.to_dict(), fitted_offset=offset, fitted_offset_note=offset_note,
                 table_z=table_z, table_z_verified=calib.support_surface.verified, tips_floor_mm=grasp.tips_min_mm,
                 tower_layers=layers_before)
        print(f"\nSource layer at ({source.center_xy[0]:.1f}, {source.center_xy[1]:.1f}) long side {source.long_side_deg:.1f} deg")
        print(f"Base layer   at ({base.center_xy[0]:.1f}, {base.center_xy[1]:.1f}) long side {base.long_side_deg:.1f} deg, level {base.level}")
        print(f"Place centre ({plan.target_center_xy[0]:.1f}, {plan.target_center_xy[1]:.1f}) long side {plan.target_long_side_deg:.1f} deg;"
              f" offset applied {use} ({offset_note})")
        print(f"Pick: claw tips {plan.pick_tips_above_table_mm:.1f} mm above the calibrated table (z {table_z:.1f}, "
              f"{'verified' if calib.support_surface.verified else 'UNVERIFIED'}), {plan.grip_below_top_mm:.1f} mm below the block tops"
              + (f" -- requested {plan.requested_grip_below_top_mm:.1f} mm, limited by the tips floor {grasp.tips_min_mm:.1f} mm; "
                 f"lower it with --tips-floor-mm" if plan.grip_below_top_mm < plan.requested_grip_below_top_mm - 1e-6 else "")
              + f". Release: tips {plan.release_tips_above_table_mm:.1f} mm above the table")
        # In a thread, so the SDK heartbeat and connection checks keep running while the operator reads the plan.
        await asyncio.to_thread(confirm, "Press Enter to pick and stack (Ctrl-C to stop) ")

        # 3. Pick: centre, align claws across the bundle, recentre (rotation moves the gripper origin), descend, grip.
        # The gripper's own report (grab() result, is_holding_something, opening width) is the grip check; an empty
        # grip repeats the same movement up to grip_retries times (operator request, 30 Oct 2026).
        sx, sy = source.center_xy
        travel_z = (await bot.gripper_origin())[2]
        attempt = 0
        while True:
            attempt += 1
            await bot.gripper.open()   # always approach with open claws
            await hover_over(source.center_xy)
            aligned, align_log = await bot.align_claws(plan.claws_close_at_pick_deg, tolerance_deg=1.5, align_speed=align_speed, travel_speed=travel_speed)
            if not aligned:
                raise LayerCheckError(f"claws did not align for the pick: errors {align_log}")
            await hover_over(source.center_xy)
            await move_z_to(plan.pick_tips_above_table_mm, table_z)
            held = await bot.world_to_frame("gripper", sx, sy, table_z + source.height_above_table_mm)
            got = await bot.gripper.grab()
            state = await bot.grip_state()
            failed = grip_failed(state, got)
            holding = not failed
            ep.event("stack", phase="grab", attempt=attempt, gripper_reports_holding=bool(got), gripper=state, failed=failed,
                     bundle_in_gripper_frame=[round(v, 2) for v in held])
            print(f"Grab {'ok' if got else 'FALSE'}; gripper reports holding={state['holding']} width={state['width']} ({state['source']}) "
                  f"-> {'FAILED' if failed else 'held'}", flush=True)
            if not failed:
                break
            await bot.gripper.open()
            g = await bot.gripper_origin()
            await bot.move_delta(dz=travel_z - g[2], near_table=True)
            if attempt > grip_retries:
                raise LayerCheckError(f"gripper reports nothing held after {attempt} attempt(s); stopped")
            print(f"Retrying the pick ({attempt}/{grip_retries} retries used)", flush=True)
        ok, _ = await bot.move_delta(dz=PICK_LIFT_MM, near_table=True)
        if not ok:
            raise LayerCheckError("planner refused the lift; bundle still held")
        # The gripper's "holding" only says the claws stopped on something. Look at the pick spot before travelling,
        # from travel height: 100 mm up the camera is ~260 mm from the table, inside the D435's minimum range at
        # 1280x720 (robot-live-7: 0.6% valid depth, "table plane unavailable"), and the bundle fills the frame.
        g = await bot.gripper_origin()
        ok, _ = await bot.move_delta(dz=travel_z - g[2], near_table=True)
        if not ok:
            raise LayerCheckError("planner refused the rise to travel height; bundle still held")
        ctx_l, _ = await measure("after_lift")
        left = blocks_left_behind(source, _blocks(ctx_l, *size), dims.height_mm)
        ep.event("stack", phase="lift_check", blocks_left_at_pick=left)
        if left == BLOCKS_PER_LAYER:
            await bot.gripper.open()
            holding = False
            raise LayerCheckError("grip failed: all three blocks are still at the pick spot after the lift (claws slipped or closed too high)")
        if left:
            raise LayerCheckError(f"partial grip: {left} block(s) left at the pick spot; the rest may still be in the gripper")

        # 4. Place: over the base, turn the bundle to base + 90, put the *bundle* centre (not the gripper) on target.
        await hover_over(plan.target_center_xy)
        aligned, align_log = await bot.align_claws(plan.claws_close_at_place_deg, tolerance_deg=1.0, align_speed=align_speed, travel_speed=travel_speed)
        if not aligned:
            raise LayerCheckError(f"claws did not align over the tower: errors {align_log}; bundle still held")
        hx, hy, _ = await bot.point_in_world("gripper", x=held[0], y=held[1])
        ok, _ = await bot.move_delta(dx=plan.target_center_xy[0] - hx, dy=plan.target_center_xy[1] - hy)
        if not ok:
            raise LayerCheckError("planner refused the final alignment over the tower; bundle still held")
        await move_z_to(plan.release_tips_above_table_mm, table_z)
        rx, ry, _ = await bot.point_in_world("gripper", x=held[0], y=held[1])
        await bot.gripper.open()
        holding = False
        g = await bot.gripper_origin()
        ok, _ = await bot.move_delta(dz=travel_z - g[2], near_table=True)
        ep.event("stack", phase="released", commanded_bundle_xy=[round(rx, 2), round(ry, 2)], retreat_accepted=ok)
        if not ok:
            raise LayerCheckError("planner refused the retreat after release")

        # 5. Verify from the same viewpoint as the base measurement, then record the residual.
        await hover_over(base_view[:2])
        await bot.align_claws(base_view_claws_deg, tolerance_deg=1.5, align_speed=align_speed, travel_speed=travel_speed)
        await hover_over(base_view[:2])
        ctx_v, _ = await measure("verify")
        try:
            tolerated = []
            top = split_scene(_blocks(ctx_v, *size), tower_origin, zone, dims.height_mm, ignored=tolerated)[1]
            measured = base_layer(top, dims.height_mm)   # the stacked layer is measured like a tower layer
        except LayerCheckError as exc:
            ep.event("stack", phase="verify_failed", reason=str(exc), observed_outcome="unknown", observation_id=last_obs[0])
            print(f"Verification could not measure the new layer: {exc}. Outcome unknown; inspect by eye.")
            return 1
        if tolerated:
            ep.event("stack", phase="between_level_ignored", label="verify",
                     heights_mm=[round(b["height_above_table_mm"], 1) for b in tolerated])
        layers_after = tower_layers(measured, dims.height_mm)
        if measured.level != base.level + 1:
            ep.event("stack", phase="verify_failed", reason="no new layer on the base", observed_outcome="failure",
                     top_level=measured.level, tower_layers=layers_after, observation_id=last_obs[0])
            print(f"\nFAILURE: the tower still measures {layers_after} layer(s) (top {measured.height_above_table_mm:.1f} mm); nothing was stacked.")
            return 1
        res = placement_residual(plan, measured)
        log.append({"episode_id": episode_id, "observation_id": last_obs[0], "operator": operator,
                    "applied_offset_xy": list(use), "plan": plan.to_dict(), "measured": measured.to_dict(), "residual": res,
                    "direct_local": direct_local, "tower_layers": layers_after})
        ep.state["tower_layers"] = layers_after
        ep.state["layers_stacked"] = ep.state.get("layers_stacked", 0) + 1
        ep.save()
        ep.event("stack", phase="verified", residual=res, measured=measured.to_dict(), tower_layers=layers_after,
                 layers_stacked=ep.state["layers_stacked"])
        print(f"\nCentre error {res['center_error_vs_base_mm']} mm (along {res['along_mm']}, across {res['across_mm']}), "
              f"yaw error {res['yaw_error_deg']} deg -> {'SUCCESS' if res['success'] else 'OUT OF TOLERANCE'}")
        print(f"Tower now has {layers_after} layers (top {measured.height_above_table_mm:.1f} mm = "
              f"{measured.height_above_table_mm / dims.height_mm:.2f} blocks); stacked by the robot in this episode: {ep.state['layers_stacked']}. "
              f"Next run: --layers {layers_after}")
        await bot.move_to(*view)
        return 0 if res["success"] else 1
    except (LayerCheckError, KeyboardInterrupt) as exc:
        reason = str(exc) or type(exc).__name__
        ep.event("stack", phase="stop", reason=reason, bundle_still_held=holding)
        print(f"\nSTOPPED: {reason}" + (" -- the bundle is still in the gripper; lower it by hand-guided live-step" if holding else ""))
        return 1
    except BaseException as exc:
        ep.event("stack", phase="stop", reason=f"{type(exc).__name__}: {str(exc)[:200]}", bundle_still_held=holding)
        if bot is not None and not plan_only:
            for part in (bot.arm, bot.gripper):
                try:
                    await asyncio.wait_for(part.stop(), timeout=5)
                    print(f"{part.name}: stop sent")
                except Exception as stop_exc:  # noqa: BLE001
                    print(f"{part.name}: stop NOT confirmed ({type(stop_exc).__name__}); use the e-stop")
        raise
    finally:
        await asyncio.wait_for(client.close(), timeout=5)


async def run_palletize(config, data_dir, operator, episode_id, tower_origin, pallet_xy, seg_provider, cycles=1, max_blocks=3,
                        direct_local=False, view_pose=None, grip_lower_mm=0.0, tips_floor_mm=None, expected_layers=None,
                        confirm=input, apply_offset=False, arm_speed=None, arm_accel=None, place_spacing_mm=None, grip_retries=1,
                        references="none", approach_speed=None, push_speed=None, disable_sessions=False, align_speed=None):
    """Full flow, `cycles` times: build a packed three-block layer on the pallet spot from the loose blocks in view
    (live_executor.run_flow: pick one at a time, place back to back, push the row tight and centre it), then grip that
    layer as a bundle and stack it on the tower (run_layer_stack). The tower is a keep-out for the picking stage. The
    stacking stage keeps its Enter hold point unless `confirm` is replaced. Stops at the first stage that does not
    complete: fewer than three blocks on the pallet, or a stack that did not add a verified layer."""
    from .live_executor import Episode, run_flow
    config.require_motion(operator)
    if cycles < 1:
        raise LayerCheckError("--cycles must be at least 1")
    pallet_xy, tower_origin = tuple(pallet_xy), tuple(tower_origin)
    for cycle in range(1, cycles + 1):
        ep = Episode(data_dir, episode_id, operator)
        # A row left incomplete by the previous palletize run is continued, not built over; run_flow verifies that the
        # declared slots are actually occupied and stops otherwise, after which the next run starts from an empty pallet.
        carry = [s for s in ep.state.get("filled", []) if s.startswith("L0")] if ep.state.get("pallet_incomplete") else []
        ep.state.pop("filled", None)
        ep.state["pallet_incomplete"] = False
        ep.save()
        ep.event("palletize", phase="assemble_start", cycle=cycle, cycles=cycles, pallet_xy=list(pallet_xy), tower_origin=list(tower_origin),
                 continuing_row=carry)
        print(f"\n=== Cycle {cycle}/{cycles}: assemble a layer on the pallet at ({pallet_xy[0]:.0f}, {pallet_xy[1]:.0f}) from the loose blocks"
              + (f", continuing the row {carry}" if carry else "") + " ===", flush=True)
        rc = await run_flow(config, data_dir, operator, episode_id, pallet_xy, carry, 0.0, max(max_blocks, BLOCKS_PER_LAYER + 2), seg_provider,
                            arm_speed=arm_speed, arm_accel=arm_accel, grip_lower_mm=grip_lower_mm, tips_floor_mm=tips_floor_mm,
                            keep_out_xy=tower_origin, view_pose=view_pose, direct_local=direct_local, drop_unseen_filled=True,
                            grip_retries=grip_retries, references=references, skip_if_complete=True, layers=1, disable_sessions=disable_sessions,
                            **({"place_spacing_mm": place_spacing_mm} if place_spacing_mm is not None else {}),
                            **({"approach_speed": approach_speed} if approach_speed is not None else {}),
                            **({"push_speed": push_speed} if push_speed is not None else {}),
                            **({"align_speed": align_speed} if align_speed is not None else {}))
        ep = Episode(data_dir, episode_id, operator)
        filled = [s for s in ep.state.get("filled", []) if s.startswith("L0")]
        # The pallet count is the measured one from the final capture (falls back to the commanded slots).
        count = ep.state.get("pallet_count", len(filled))
        if rc != 0 or count < BLOCKS_PER_LAYER:
            ep.state["pallet_incomplete"] = count > 0
            ep.state["filled"] = [f"L0S{i}" for i in range(min(count, BLOCKS_PER_LAYER))]
            ep.save()
            ep.event("palletize", phase="stop", cycle=cycle, reason="pallet_incomplete", measured=count, filled=filled, flow_rc=rc)
            print(f"\nSTOPPED: the pallet shows {count} of {BLOCKS_PER_LAYER} blocks after the assembly stage; nothing stacked. "
                  + ("Add loose blocks in view and rerun; the row is continued." if count else
                     "Check the loose blocks (lying flat, clear of each other, in view) and rerun."))
            return 1
        before = Episode(data_dir, episode_id, operator).state.get("layers_stacked", 0)
        print(f"\n=== Cycle {cycle}/{cycles}: stack the pallet onto the tower at ({tower_origin[0]:.0f}, {tower_origin[1]:.0f}) ===", flush=True)
        rc = await run_layer_stack(config, data_dir, operator, episode_id, tower_origin, seg_provider, direct_local=direct_local,
                                   apply_offset=apply_offset, confirm=confirm, view_pose=view_pose, grip_lower_mm=grip_lower_mm,
                                   source_hint_xy=pallet_xy, tips_floor_mm=tips_floor_mm, expected_layers=expected_layers,
                                   grip_retries=grip_retries, disable_sessions=disable_sessions,
                                   arm_speed=arm_speed, approach_speed=approach_speed, align_speed=align_speed)
        state = Episode(data_dir, episode_id, operator).state
        if state.get("layers_stacked", 0) <= before:
            ep.event("palletize", phase="stop", cycle=cycle, reason="stack_not_verified", stack_rc=rc)
            print("\nSTOPPED: the stacking stage did not add a verified layer; see the reason above.")
            return 1
        if expected_layers is not None:
            expected_layers = state.get("tower_layers", expected_layers + 1)
        ep.event("palletize", phase="cycle_done", cycle=cycle, tower_layers=state.get("tower_layers"), stack_rc=rc,
                 within_tolerance=(rc == 0))
        print(f"=== Cycle {cycle}/{cycles} done: tower has {state.get('tower_layers')} layers"
              + ("" if rc == 0 else " (last layer outside the 3 mm / 3 deg tolerance; stacked nonetheless)") + " ===", flush=True)
    return 0
