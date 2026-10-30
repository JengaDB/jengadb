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
SPACING_RANGE_MM = (22.0, 28.0)   # neighbour pitch for a packed layer of 25 mm blocks
MIN_SHARED_CONTACT_MM = 55.0      # 45 mm finger width plus margin must overlap all three blocks' long sides
HEIGHT_BAND = 0.3                 # of a block height (4.5 mm): beyond this a top surface is between layers (tilted block)
MIN_BLOCK_HEIGHT_FRACTION = 0.5   # detections lower than half a block are flat objects (paper, tape), not blocks
MAX_TARGET_OFFSET_MM = 15.0       # refuse to stack if the refined base moved this far from the overview estimate
CENTER_TOLERANCE_MM = 3.0         # success: new layer centre within this of the target
YAW_TOLERANCE_DEG = 3.0           # success: new layer within this of exactly 90 deg to the base
MIN_RUNS_FOR_OFFSET = 3
MAX_OFFSET_SPREAD_MM = 2.0        # residual standard deviation above which a constant offset is not trusted


class LayerCheckError(ValueError):
    """A measured layer that must not be gripped or stacked on. The message is the operator-facing reason."""


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


def summarize_layer(blocks, block_height_mm=15.0):
    """Validate three measured blocks as one packed, flat, parallel layer and return its pose."""
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
    lo, hi = SPACING_RANGE_MM
    if not all(lo <= s <= hi for s in spacing):
        raise LayerCheckError("layer has gaps or overlaps: pitch " + ", ".join(f"{s:.1f}" for s in spacing) +
                              f" mm (want {lo:.0f}-{hi:.0f}); push it together by hand")
    contact = 75.0 - (max(along) - min(along))
    if contact < MIN_SHARED_CONTACT_MM:
        raise LayerCheckError(f"block ends are staggered by {75.0 - contact:.1f} mm; line the ends up")
    cx = sum(p[0] for p in xy) / len(xy)
    cy = sum(p[1] for p in xy) / len(xy)
    height = statistics.median(b["height_above_table_mm"] for b in blocks)
    return LayerPose((cx, cy), theta, height, spacing, contact, levels.pop(), [list(p) for p in xy])


def split_scene(blocks, tower_origin_xy, zone_radius_mm, block_height_mm=15.0):
    """Source layer = usable blocks outside the tower zone; base = the highest complete layer inside it."""
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
    if None in by_level:
        raise LayerCheckError("a block in the tower zone is between layer heights (tilted or resting on an edge?)")
    if not by_level:
        raise LayerCheckError("no base layer in the tower zone")
    top = max(by_level)
    return outside, by_level[top]


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

    def to_dict(self):
        d = asdict(self)
        d["source"], d["base"] = self.source.to_dict(), self.base.to_dict()
        return d


def plan_stack(source, base, block_height_mm=15.0, params=GraspParams(), release_drop_mm=1.0, offset_xy=(0.0, 0.0)):
    """Grip across the bundle's 75 mm width (claws close perpendicular to the long sides, pressing the three together),
    and release with the bundle's long side at 90 deg to the base layer, centred on it."""
    if source.level != 0:
        raise LayerCheckError("the hand-built layer must lie on the table")
    grip_below_top = params.grasp_depth_mm
    target_long = wrap_90(base.long_side_deg + 90.0)
    tips_pick = max(source.height_above_table_mm - grip_below_top, params.tips_min_mm)
    tips_release = base.height_above_table_mm + block_height_mm - grip_below_top + release_drop_mm
    target = (base.center_xy[0] - offset_xy[0], base.center_xy[1] - offset_xy[1])
    return StackPlan(source, base, wrap_90(source.long_side_deg + ACROSS), wrap_90(target_long + ACROSS),
                     target, target_long, tips_pick, tips_release, tuple(offset_xy))


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

async def _open_client(config, direct_local):
    """Sessions stay enabled. direct_local dials viam-server on the LAN over gRPC instead of the cloud WebRTC relay
    (measured 29 Oct: median 0.37 s vs 1.13 s per RGB-D read, max 0.87 s vs 5.39 s)."""
    from .capture import open_client
    if not direct_local:
        return await open_client(config)
    from viam.robot.client import RobotClient
    config.require_viam()
    host = config.address.split(":")[0]
    if not host.endswith(".viam.cloud") or ".local." in host:
        raise LayerCheckError("--direct-local needs the machine's <name>.viam.cloud address in MACHINE_ADDRESS")
    options = RobotClient.Options.with_api_key(api_key=config.key, api_key_id=config.key_id)
    options.dial_options.disable_webrtc = True
    return await asyncio.wait_for(RobotClient.at_address(host.replace(".viam.cloud", ".local.viam.cloud") + ":8080", options), timeout=15)


EDGE_MARGIN_PX = 4  # a mask this close to the image border is cut off: its centre and angle are wrong


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
                          apply_offset=False, confirm=input, plan_only=False, view_pose=None, grip_lower_mm=0.0):
    """One supervised stack. Returns 0 on a verified success, 1 otherwise. See module docstring.
    plan_only: capture from the current pose, measure both layers and print the plan; no motion command is sent."""
    if not plan_only:
        config.require_motion(operator)  # before any SDK import or connection
    from . import proposal, rgbd, segmentation
    from . import grasp_geometry as geom
    from .capture import capture_with_camera
    from .live_executor import HOME, PICK_LIFT_MM, TIP_ABOVE_GRIPPER_FRAME_MM, Episode, Robot
    from .pipeline import candidate_id
    from .placement import TowerTarget, footprint_clearance_mm
    # The SDK builds resource clients only for APIs registered (imported) before connecting; see live_executor.run_flow.
    from viam.components.arm import Arm  # noqa: F401
    from viam.components.camera import Camera
    from viam.components.gripper import Gripper  # noqa: F401
    from viam.services.motion import MotionClient  # noqa: F401

    ep = Episode(data_dir, episode_id, operator)
    client = await _open_client(config, direct_local)
    bot = None
    holding = False
    try:
        await asyncio.wait_for(client.refresh(), timeout=10)  # resource list can lag a viam-server reconfigure
        bot = Robot(client, config)
        camera = Camera.from_robot(client, config.camera_resource)

        size = [0, 0]  # width, height of the latest capture
        last_obs = [None]  # observation_id of the latest capture

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
            ep.event("stack", phase="measured", label=label, observation_id=obs["observation_id"], masks=len(masks),
                     blocks=[{"xy": [round(b["base_point_mm"][0], 1), round(b["base_point_mm"][1], 1)], "long_deg": round(b["base_long_side_deg"], 1),
                              "height": round(b["height_above_table_mm"], 1)} for b in _blocks(ctx, *size) if _usable(b)])
            size[:] = [rgb.shape[1], rgb.shape[0]]
            last_obs[0] = obs["observation_id"]
            return ctx, calib

        async def hover_over(xy):
            g = await bot.gripper_origin()
            ok, _ = await bot.move_delta(dx=xy[0] - g[0], dy=xy[1] - g[1])
            if not ok:
                raise LayerCheckError(f"planner refused the hover move to {xy[0]:.1f}, {xy[1]:.1f}")

        async def move_z_to(tips_above_table, table_z):
            g = await bot.gripper_origin()
            ok, _ = await bot.move_delta(dz=(table_z + tips_above_table - TIP_ABOVE_GRIPPER_FRAME_MM) - g[2], near_table=True)
            if not ok:
                raise LayerCheckError("planner refused the vertical move")

        if await bot.arm.is_moving():
            raise LayerCheckError("arm is moving; wait for it to stop")
        view = tuple(view_pose) if view_pose else HOME   # flange pose for the overview capture
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
        outside, base_blocks = split_scene(_blocks(ctx, *size), tower_origin, zone, dims.height_mm)
        source0 = summarize_layer(outside, dims.height_mm)
        base0 = summarize_layer(base_blocks, dims.height_mm)
        if plan_only:
            log = ResidualLog(data_dir, config.setup_id)
            offset, offset_note = log.fitted_offset()
            plan = plan_stack(source0, base0, dims.height_mm, params=GraspParams(grasp_depth_mm=GraspParams().grasp_depth_mm + grip_lower_mm))
            ep.event("stack", phase="plan_only", plan=plan.to_dict(), fitted_offset=offset, fitted_offset_note=offset_note,
                     table_z=table_z, table_z_verified=calib.support_surface.verified, zone_radius_mm=round(zone, 1))
            print(f"\nPLAN ONLY (overview capture, no motion). Source ({source0.center_xy[0]:.1f}, {source0.center_xy[1]:.1f}) "
                  f"long {source0.long_side_deg:.1f} deg, pitch {[round(v, 1) for v in source0.spacing_mm]} mm; base ({base0.center_xy[0]:.1f}, "
                  f"{base0.center_xy[1]:.1f}) long {base0.long_side_deg:.1f} deg level {base0.level}; place long {plan.target_long_side_deg:.1f} deg")
            return 0

        # 2. Refine each layer from a hover directly above it (closer to the image centre, less extrinsic leverage).
        await hover_over(base0.center_xy)
        ctx_b, _ = await measure("base_close")
        base = summarize_layer(split_scene(_blocks(ctx_b, *size), tower_origin, zone, dims.height_mm)[1], dims.height_mm)
        if math.dist(base.center_xy, base0.center_xy) > MAX_TARGET_OFFSET_MM:
            raise LayerCheckError(f"base measured {math.dist(base.center_xy, base0.center_xy):.1f} mm apart in two captures")
        base_view = await bot.gripper_origin()
        base_view_claws_deg = await bot.close_direction_deg()  # wrist camera: viewpoint includes the claw angle
        await hover_over(source0.center_xy)
        ctx_s, _ = await measure("source_close")
        source = summarize_layer(split_scene(_blocks(ctx_s, *size), tower_origin, zone, dims.height_mm)[0], dims.height_mm)

        log = ResidualLog(data_dir, config.setup_id)
        offset, offset_note = log.fitted_offset()
        use = offset if (apply_offset and offset) else (0.0, 0.0)
        # Deeper grip changes pick and release tips together, so the bundle still lands release_drop above the base.
        params = GraspParams(grasp_depth_mm=GraspParams().grasp_depth_mm + grip_lower_mm)
        plan = plan_stack(source, base, dims.height_mm, params=params, offset_xy=use)
        ep.event("stack", phase="plan", plan=plan.to_dict(), fitted_offset=offset, fitted_offset_note=offset_note,
                 table_z=table_z, table_z_verified=calib.support_surface.verified)
        print(f"\nSource layer at ({source.center_xy[0]:.1f}, {source.center_xy[1]:.1f}) long side {source.long_side_deg:.1f} deg")
        print(f"Base layer   at ({base.center_xy[0]:.1f}, {base.center_xy[1]:.1f}) long side {base.long_side_deg:.1f} deg, level {base.level}")
        print(f"Place centre ({plan.target_center_xy[0]:.1f}, {plan.target_center_xy[1]:.1f}) long side {plan.target_long_side_deg:.1f} deg;"
              f" offset applied {use} ({offset_note})")
        # In a thread, so the SDK heartbeat and connection checks keep running while the operator reads the plan.
        await asyncio.to_thread(confirm, "Press Enter to pick and stack (Ctrl-C to stop) ")

        # 3. Pick: centre, align claws across the bundle, recentre (rotation moves the gripper origin), descend, grip.
        await hover_over(source.center_xy)
        aligned, align_log = await bot.align_claws(plan.claws_close_at_pick_deg, tolerance_deg=1.5)
        if not aligned:
            raise LayerCheckError(f"claws did not align for the pick: errors {align_log}")
        await hover_over(source.center_xy)
        travel_z = (await bot.gripper_origin())[2]
        await move_z_to(plan.pick_tips_above_table_mm, table_z)
        sx, sy = source.center_xy
        held = await bot.world_to_frame("gripper", sx, sy, table_z + source.height_above_table_mm)
        holding = await bot.gripper.grab()
        ep.event("stack", phase="grab", gripper_reports_holding=bool(holding), bundle_in_gripper_frame=[round(v, 2) for v in held])
        if not holding:
            await bot.gripper.open()
            await move_z_to(plan.pick_tips_above_table_mm + 30.0, table_z)
            raise LayerCheckError("gripper reports nothing held; stopped without retry")
        ok, _ = await bot.move_delta(dz=PICK_LIFT_MM, near_table=True)
        if not ok:
            raise LayerCheckError("planner refused the lift; bundle still held")
        # The gripper's "holding" only says the claws stopped on something. Look at the pick spot before travelling.
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
        aligned, align_log = await bot.align_claws(plan.claws_close_at_place_deg, tolerance_deg=1.0)
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
        await bot.align_claws(base_view_claws_deg, tolerance_deg=1.5)
        await hover_over(base_view[:2])
        ctx_v, _ = await measure("verify")
        top = split_scene(_blocks(ctx_v, *size), tower_origin, zone, dims.height_mm)[1]
        try:
            measured = summarize_layer(top, dims.height_mm)
        except LayerCheckError as exc:
            ep.event("stack", phase="verify_failed", reason=str(exc), observed_outcome="unknown")
            print(f"Verification could not measure the new layer: {exc}. Outcome unknown; inspect by eye.")
            return 1
        if measured.level != base.level + 1:
            ep.event("stack", phase="verify_failed", reason="no new layer on the base", observed_outcome="failure",
                     top_level=measured.level, observation_id=last_obs[0])
            print("\nFAILURE: the top of the tower is still the base layer; nothing was stacked.")
            return 1
        res = placement_residual(plan, measured)
        log.append({"episode_id": episode_id, "observation_id": last_obs[0], "operator": operator,
                    "applied_offset_xy": list(use), "plan": plan.to_dict(), "measured": measured.to_dict(), "residual": res,
                    "direct_local": direct_local})
        ep.event("stack", phase="verified", residual=res, measured=measured.to_dict())
        print(f"\nCentre error {res['center_error_vs_base_mm']} mm (along {res['along_mm']}, across {res['across_mm']}), "
              f"yaw error {res['yaw_error_deg']} deg -> {'SUCCESS' if res['success'] else 'OUT OF TOLERANCE'}")
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
