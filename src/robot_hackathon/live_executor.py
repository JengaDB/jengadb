"""Operator-run, one-segment-at-a-time executor for a single validated proposal (docs/LIVE_SINGLE_PLACEMENT_PLAN.md s.5).

Authorized by the project owner on 29 Oct 2026 with the operator at the physical e-stop. Every invocation performs
exactly one motion segment, prints the planned target before moving and the measured pose after, appends an event
to the episode log and exits. The hold point between segments is the human. There is no retry path: a failed grab,
a planner error or any exception ends the step, and the next decision is the operator's.

Geometry follows Sam's jenga.py: the gripper frame origin is what gets positioned over block centres, claws close
along the gripper's +y axis, descents use a linear constraint with the gripper-table collision allowance, and the
claw-tip height uses his touch-off relation (flange z 171.5 at contact with a gripper frame 172 below the flange ->
tips sit 0.5 mm above the gripper frame origin). The table height comes from the proposal's support-surface offset,
which is unverified until a touch-off replaces it; the margin of error is printed before every descent.
"""
import asyncio
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .config import ConfigError
from .local_store import atomic_write

HOME = (300.0, 150.0, 400.0)      # flange pose, gripper pointing straight down (Sam's HOME_POSE on this machine)
SAFE_Z_MM = 150.0                 # gripper origin height that is clear of flat blocks
TIP_ABOVE_GRIPPER_FRAME_MM = 0.5  # from Sam's touch-off: 171.5 (flange at contact) - 172 (gripper frame below flange)
PLACE_DROP_MM = 1.0
PICK_LIFT_MM = 100.0
CLAW_THICKNESS_MM = 8.0           # Sam's; closed claws are two claws thick
PUSH_START_GAP_MM = 5.0           # gap between the closed claws and the last block before the push starts
PUSH_TIPS_MM = 5.0                # claw tips above the table while pushing
PUSH_EXTRA_MM = 8.0               # shove past the point where the gaps have closed
ADJACENT_CLEARANCE_MM = 1.5       # placing exactly one pitch away nudged the neighbour; leave this much and push later
PUSH_IF_GAPS_OVER_MM = 2.0
MOVE_TIMEOUT_S = 120.0
# Blue tape square on the table, measured from obs_bd83a42d69914db0 (marks at x 397..481, y 165..241 world):
# centre (439, 203); x pre-compensated by -4 mm for the placement bias measured on the first two rows.
BLUE_SQUARE_ORIGIN = (435.0, 203.0)
STEPS = ("status", "home", "hover", "descend", "grab", "lift", "to-slot", "release", "retreat", "abort-up", "open", "push")


def _now():
    return datetime.now(timezone.utc).isoformat()


def wrap_90(angle):
    return (angle + 90) % 180 - 90


class Episode:
    def __init__(self, data_dir, episode_id, operator):
        if not episode_id or "/" in episode_id:
            raise ConfigError("--episode is required")
        self.dir = Path(data_dir) / "live" / "episodes" / episode_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"episode_id": episode_id, "operator": operator}
        self.operator = operator

    def save(self):
        atomic_write(self.state_path, json.dumps(self.state, indent=1, sort_keys=True, default=str))

    def event(self, step, **payload):
        row = {"at": _now(), "episode_id": self.state["episode_id"], "operator": self.operator, "step": step, **payload}
        with (self.dir / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
        print(json.dumps(row, default=str))
        return row


class Robot:
    def __init__(self, client, config):
        from viam.components.arm import Arm
        from viam.components.gripper import Gripper
        from viam.services.motion import MotionClient
        from viam.proto.service.motion import CollisionSpecification, Constraints, LinearConstraint
        self.client = client
        self.arm = Arm.from_robot(client, "arm")
        self.gripper = Gripper.from_robot(client, "gripper")
        self.motion = MotionClient.from_robot(client, "builtin")
        self.near_table = Constraints(linear_constraint=[LinearConstraint()],
                                      collision_specification=[CollisionSpecification(
                                          allows=[CollisionSpecification.AllowedFrameCollisions(frame1="gripper", frame2="table")])])

    async def point_in_world(self, frame, x=0.0, y=0.0, z=0.0):
        from viam.proto.common import Pose, PoseInFrame
        p = (await self.client.transform_pose(PoseInFrame(reference_frame=frame, pose=Pose(x=x, y=y, z=z, o_z=1)), "world")).pose
        return (p.x, p.y, p.z)

    async def world_to_frame(self, frame, x, y, z):
        from viam.proto.common import Pose, PoseInFrame
        p = (await self.client.transform_pose(PoseInFrame(reference_frame="world", pose=Pose(x=x, y=y, z=z, o_z=1)), frame)).pose
        return (p.x, p.y, p.z)

    async def flange_pose(self):
        return (await self.motion.get_pose("arm", "world")).pose

    async def gripper_origin(self):
        return await self.point_in_world("gripper")

    async def close_direction_deg(self):
        o = await self.gripper_origin()
        y = await self.point_in_world("gripper", y=100.0)
        return math.degrees(math.atan2(y[1] - o[1], y[0] - o[0]))

    async def snapshot(self):
        f = await self.flange_pose()
        g = await self.gripper_origin()
        return {"flange": [round(f.x, 1), round(f.y, 1), round(f.z, 1)], "orientation": [round(f.o_x, 3), round(f.o_y, 3), round(f.o_z, 3), round(f.theta, 1)],
                "gripper_origin": [round(v, 1) for v in g], "claws_close_deg": round(await self.close_direction_deg(), 1),
                "arm_moving": await self.arm.is_moving(), "gripper_moving": await self.gripper.is_moving()}

    async def move_delta(self, dx=0.0, dy=0.0, dz=0.0, rotate=0.0, near_table=False):
        """Shift the flange by (dx, dy, dz) mm in world and spin theta by `rotate` degrees, planner-checked (Sam's move)."""
        from viam.proto.common import Pose, PoseInFrame
        cur = await self.flange_pose()
        target = Pose(x=cur.x + dx, y=cur.y + dy, z=cur.z + dz, o_x=cur.o_x, o_y=cur.o_y, o_z=cur.o_z, theta=cur.theta + rotate)
        ok = await self.motion.move("arm", PoseInFrame(reference_frame="world", pose=target),
                                    constraints=self.near_table if near_table else None, timeout=MOVE_TIMEOUT_S)
        return ok, [round(target.x, 1), round(target.y, 1), round(target.z, 1), round(target.theta, 1)]

    async def move_to(self, x, y, z):
        from viam.proto.common import Pose, PoseInFrame
        target = Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=-1, theta=0)
        ok = await self.motion.move("arm", PoseInFrame(reference_frame="world", pose=target), timeout=MOVE_TIMEOUT_S)
        return ok, [x, y, z]

    async def align_claws(self, target_deg, passes=3, tolerance_deg=3.0):
        """Rotate until the claws close along target_deg (either way round). Sam's sign-flip trick for theta."""
        sign, last = 1, None
        log = []
        for _ in range(passes):
            error = wrap_90(target_deg - await self.close_direction_deg())
            log.append(round(error, 1))
            if abs(error) < tolerance_deg:
                return True, log
            if last is not None and abs(error) > abs(last):
                sign = -sign
            last = error
            await self.move_delta(rotate=-sign * error)
        error = wrap_90(target_deg - await self.close_direction_deg())
        log.append(round(error, 1))
        return abs(error) < tolerance_deg, log


async def run_flow(config, data_dir, operator, episode_id, tower_origin, filled, gap_mm, max_blocks=3, seg_provider=None,
                   final_capture=True, arm_speed=None, arm_accel=None):
    """One connection, one capture: segment, rank, then place blocks back to back. The next slot is positioned from the
    previous block's *actual* release pose (plus one pitch along the row) rather than the nominal slot, so release
    nudges do not accumulate. Grasp per slot: ends when the slot touches a placed neighbour, else middle/side/ends.
    No intermediate checks; a failed grab or planner refusal stops the flow. Only the final capture is measured."""
    from . import proposal, rgbd, segmentation
    from . import grasp_geometry as geom
    from .capture import open_client, capture_with_camera
    from .pipeline import candidate_id
    from .placement import TowerTarget, next_open_slot
    from viam.components.arm import Arm  # noqa: F401
    from viam.components.gripper import Gripper  # noqa: F401
    from viam.components.camera import Camera
    from viam.services.motion import MotionClient  # noqa: F401
    config.require_motion(operator)
    ep = Episode(data_dir, episode_id, operator)
    client = await open_client(config)
    placed = []
    try:
        await asyncio.wait_for(client.refresh(), timeout=10)
        bot = Robot(client, config)
        camera = Camera.from_robot(client, config.camera_resource)
        if arm_speed is not None or arm_accel is not None:
            # UFactory xArm module do_command keys; operator-requested, applied for this session only.
            result = {}
            if arm_speed is not None:
                result["set_speed"] = await bot.arm.do_command({"set_speed": float(arm_speed)})
            if arm_accel is not None:
                result["set_acceleration"] = await bot.arm.do_command({"set_acceleration": float(arm_accel)})
            ep.event("flow", phase="arm_speed", speed_deg_s=arm_speed, accel_deg_s2=arm_accel, result=result)

        async def scene_from_capture():
            obs = await capture_with_camera(config, data_dir, camera, episode_id, client=client)
            obs, rgb, depth = rgbd.ObservationStore(data_dir, "live").load_arrays(obs["observation_id"])
            raw = await asyncio.to_thread(seg_provider.segment, rgb)  # the provider runs its own event loop internally
            masks = segmentation.finalize_masks(raw, rgb.shape[1], rgb.shape[0], 200)
            for m in masks:
                m.update(candidate_id=candidate_id(obs["observation_id"], "live-flow", m["mask_index"]), embedding=None, candidate=None)
            store = geom.GeometryCalibrationStore(data_dir, "live")
            doc = store.load(obs["setup_id"])
            scene = {"source": "observation", "observation": obs, "rgb": rgb, "depth": depth, "masks": masks,
                     "calibration": geom.bundle_from_records(obs, False, doc), "geometry_calibration_sha256": None,
                     "segmentation": {"provider": seg_provider.provider_id, "model_version": "live-flow"}, "processing_version": "live-flow"}
            return obs, scene

        obs, scene = await scene_from_capture()
        ctx = proposal.build_context(scene, tower_origin_xy_mm=tower_origin, filled_slots=filled, dry_run=True, tower_gap_mm=gap_mm)
        proposal.save_context(ctx, {"kind": "flow_start"}, Path(data_dir) / "live" / "proposals" / obs["observation_id"])
        calib = scene["calibration"]
        table_z = calib.support_surface.z_in_robot_base_mm
        target = TowerTarget(tuple(tower_origin), calib.support_surface, calib.block, calib.finger, 0.0, gap_mm=gap_mm)
        queue = [proposal.candidate_entry(ctx, r["candidate_id"]) for r in ctx["ranking"]]
        ep.event("flow", phase="scene", observation_id=obs["observation_id"], masks=len(scene["masks"]), eligible=[c["candidate_id"] for c in queue],
                 rejections={c["candidate_id"]: c["rejection_reasons"] for c in ctx["candidates"] if not c["eligible"]},
                 table_plane=ctx["table_plane"].get("status"), tower_origin=list(tower_origin), gap_mm=gap_mm, filled=list(filled))
        filled = list(filled)
        # Declared filled slots must be visible: a block in the tower zone at that layer's height for each one.
        if filled:
            in_zone = [c["geometry"] for c in ctx["candidates"] if "in_tower_zone" in c["rejection_reasons"] and c["geometry"]["height_above_table_mm"] is not None]
            layers_declared = {int(s[1]) for s in filled}
            missing = []
            for layer in layers_declared:
                expect = (layer + 1) * calib.block.height_mm
                seen = [g for g in in_zone if abs(g["height_above_table_mm"] - expect) < 0.6 * calib.block.height_mm]
                count = sum(1 for s in filled if int(s[1]) == layer)
                if len(seen) < min(count, 1):
                    missing.append({"layer": layer, "expected_height_mm": expect, "seen_heights": [round(g["height_above_table_mm"], 1) for g in in_zone]})
            if missing:
                ep.event("flow", phase="stop", reason="declared_filled_slots_not_visible", missing=missing,
                         note="nothing placed; fix --filled-slots or put the base layer back")
                return 1
        last_pose = None   # (x, y, long_side_deg) of the previous block as released

        def row_blocks(context, layer):
            """Measured blocks in the tower zone at this layer's height, ordered along the row's across direction."""
            expect = (layer + 1) * calib.block.height_mm
            found = [c["geometry"] for c in context["candidates"] if "in_tower_zone" in c["rejection_reasons"]
                     and c["geometry"]["height_above_table_mm"] is not None and abs(c["geometry"]["height_above_table_mm"] - expect) < 0.6 * calib.block.height_mm]
            across = math.radians(target.layer0_long_side_deg + 90.0 * (layer + 1))
            return sorted(found, key=lambda g: g["base_point_mm"][0] * math.cos(across) + g["base_point_mm"][1] * math.sin(across))

        for _ in range(max_blocks):
            slot = next_open_slot(target, filled)
            if slot is None:
                ep.event("flow", phase="layer_complete", filled=filled)
                break
            if last_pose is None and slot["index"] > 0:
                # Resuming a partly built layer: continue from the measured end of the row, not the nominal slot.
                row = row_blocks(ctx, slot["layer"])
                if row:
                    g = row[-1]
                    last_pose = (g["base_point_mm"][0], g["base_point_mm"][1], slot["long_side_deg"])
                    ep.event("flow", phase="resume_from_measured_row", row_end_xy=[round(last_pose[0], 1), round(last_pose[1], 1)], blocks_seen=len(row))
            order = proposal.grasp_order(slot)
            pick = None
            for c in queue:
                g = next((g for style in order for g in c["grasps"] if g["style"] == style and g["eligible"]), None)
                if g is not None:
                    pick, grasp = c, g
                    break
            if pick is None:
                ep.event("flow", phase="stop", reason="no_candidate_with_required_grasp", required=list(order), slot=slot["slot_id"])
                break
            queue.remove(pick)
            geo = pick["geometry"]
            bx, by, bz = geo["base_point_mm"]
            # Destination: nominal for the first block of a layer; otherwise previous release pose plus one pitch.
            if last_pose is None or slot["index"] == 0:
                sx, sy = slot["block_center_mm"][0], slot["block_center_mm"][1]
            else:
                across = math.radians(slot["long_side_deg"] + 90.0)
                pitch = slot["block_dimensions_mm"][1] + gap_mm + ADJACENT_CLEARANCE_MM
                sx, sy = last_pose[0] + pitch * math.cos(across), last_pose[1] + pitch * math.sin(across)
            tips = grasp["tips_above_table_mm"]
            ep.event("flow", phase="plan_block", slot=slot["slot_id"], candidate=pick["candidate_id"], grasp=grasp["style"], pick_xy=[round(bx, 1), round(by, 1)],
                     pick_long_deg=round(geo["base_long_side_deg"], 1), dest_xy=[round(sx, 1), round(sy, 1)], dest_long_deg=slot["long_side_deg"],
                     support_z=slot["support_z_mm"], tips_above_table_mm=round(tips, 1))
            # --- pick
            g0 = await bot.gripper_origin()
            travel_z = g0[2]
            ok, _ = await bot.move_delta(dx=bx - g0[0], dy=by - g0[1])
            if not ok:
                ep.event("flow", phase="stop", reason="hover_refused"); break
            close_deg = grasp.get("claws_close_direction_deg_in_base")
            aligned, log = await bot.align_claws(close_deg)
            g = await bot.gripper_origin()
            ok, _ = await bot.move_delta(dz=(table_z + tips - TIP_ABOVE_GRIPPER_FRAME_MM) - g[2], near_table=True)
            if not ok:
                ep.event("flow", phase="stop", reason="descend_refused"); break
            held = await bot.world_to_frame("gripper", bx, by, bz)
            got = await bot.gripper.grab()
            if not got:
                await bot.gripper.open()
                g = await bot.gripper_origin()
                await bot.move_delta(dz=travel_z - g[2], near_table=True)
                ep.event("flow", phase="stop", reason="grab_failed", candidate=pick["candidate_id"]); break
            ok, _ = await bot.move_delta(dz=PICK_LIFT_MM, near_table=True)
            # --- place
            g = await bot.gripper_origin()
            ok, _ = await bot.move_delta(dx=sx - g[0], dy=sy - g[1])
            await bot.align_claws(wrap_90(slot["long_side_deg"] + grasp["claws_deg_to_long_side"]), tolerance_deg=1.5)
            hx, hy, _ = await bot.point_in_world("gripper", x=held[0], y=held[1])
            ok, _ = await bot.move_delta(dx=sx - hx, dy=sy - hy)
            g = await bot.gripper_origin()
            release_z = slot["support_z_mm"] + tips + PLACE_DROP_MM - TIP_ABOVE_GRIPPER_FRAME_MM
            ok, _ = await bot.move_delta(dz=release_z - g[2], near_table=True)
            if not ok:
                ep.event("flow", phase="stop", reason="place_descend_refused", note="block still held; use live-step abort-up/open"); break
            rx, ry, _ = await bot.point_in_world("gripper", x=held[0], y=held[1])
            await bot.gripper.open()
            g = await bot.gripper_origin()
            await bot.move_delta(dz=travel_z - g[2], near_table=True)
            last_pose = (rx, ry, slot["long_side_deg"])
            filled.append(slot["slot_id"])
            placed.append({"slot": slot["slot_id"], "candidate": pick["candidate_id"], "released_at": [round(rx, 1), round(ry, 1)], "grasp": grasp["style"]})
            ep.event("flow", phase="placed", slot=slot["slot_id"], released_block_xy=[round(rx, 1), round(ry, 1)], aligned_at_pick=aligned)
        # --- home, measure, push the table row together if the measured gaps say so (layer 0 only), measure again
        await bot.gripper.open()
        ok, _ = await bot.move_to(*HOME)
        ep.event("flow", phase="home", accepted=ok, placed=placed)
        async def push_row(edge_block_xy, ux, uy, distance, label):
            """Closed claws come down just outside `edge_block_xy` on the side opposite to (ux, uy), then slide `distance`."""
            w = calib.block.width_mm
            back = w / 2 + CLAW_THICKNESS_MM + PUSH_START_GAP_MM
            start = (edge_block_xy[0] - ux * back, edge_block_xy[1] - uy * back)
            await bot.gripper.grab()
            g = await bot.gripper_origin()
            travel_z = g[2]
            ep.event("flow", phase=f"push_{label}", start_xy=[round(v, 1) for v in start], direction=[round(ux, 2), round(uy, 2)], distance_mm=round(distance, 1))
            await bot.move_delta(dx=start[0] - g[0], dy=start[1] - g[1])
            await bot.align_claws(wrap_90(target.layer0_long_side_deg + 90.0), tolerance_deg=1.5)
            g = await bot.gripper_origin()
            await bot.move_delta(dx=start[0] - g[0], dy=start[1] - g[1])
            g = await bot.gripper_origin()
            ok, _ = await bot.move_delta(dz=(table_z + PUSH_TIPS_MM - TIP_ABOVE_GRIPPER_FRAME_MM) - g[2], near_table=True)
            if ok:
                ok, _ = await bot.move_delta(dx=ux * distance, dy=uy * distance, near_table=True)
            g = await bot.gripper_origin()
            await bot.move_delta(dz=travel_z - g[2], near_table=True)
            await bot.gripper.open()
            await bot.move_to(*HOME)
            return ok

        layer0_only = not any(p["slot"].startswith("L1") for p in placed) and not any(s.startswith("L1") for s in filled)
        if layer0_only and (placed or any(s.startswith("L0") for s in filled)):
            # Close the gaps, then centre the tight row on the target origin along the row direction. Two rounds at most.
            across = math.radians(target.layer0_long_side_deg + 90.0)
            ax, ay = math.cos(across), math.sin(across)
            w = calib.block.width_mm
            for _round in range(3):
                obs_m, scene_m = await scene_from_capture()
                ctx_m = proposal.build_context(scene_m, tower_origin_xy_mm=tower_origin, filled_slots=filled, dry_run=True, tower_gap_mm=gap_mm)
                row = row_blocks(ctx_m, 0)
                pts = [g["base_point_mm"] for g in row]
                gaps = sum(max(0.0, math.hypot(b[0] - a[0], b[1] - a[1]) - w) for a, b in zip(pts, pts[1:]))
                centre = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)) if pts else None
                offset = ((tower_origin[0] - centre[0]) * ax + (tower_origin[1] - centre[1]) * ay) if centre else 0.0
                ep.event("flow", phase="row_measured", blocks=len(row), row_gaps_mm=round(gaps, 1), centre_offset_along_row_mm=round(offset, 1),
                         centres=[[round(p[0], 1), round(p[1], 1)] for p in pts])
                if len(row) < 2:
                    break
                if gaps > PUSH_IF_GAPS_OVER_MM:
                    # Push from the far end toward the first block (the first block was placed nominally, so it anchors the row).
                    await push_row(pts[-1], -ax, -ay, gaps + PUSH_START_GAP_MM + PUSH_EXTRA_MM, "gaps")
                    continue
                if abs(offset) > 3.0:
                    # Whole row off-centre: push from the side opposite to the needed shift by exactly that amount (no extra).
                    sign = 1.0 if offset > 0 else -1.0
                    edge = pts[0] if sign > 0 else pts[-1]
                    await push_row(edge, sign * ax, sign * ay, abs(offset) + PUSH_START_GAP_MM, "centre")
                    continue
                break
        if final_capture:
            obs2, scene2 = await scene_from_capture()
            ctx2 = proposal.build_context(scene2, tower_origin_xy_mm=tower_origin, filled_slots=filled, dry_run=True, tower_gap_mm=gap_mm)
            measured = [{"xy": [round(c["geometry"]["base_point_mm"][0], 1), round(c["geometry"]["base_point_mm"][1], 1)],
                         "z": round(c["geometry"]["base_point_mm"][2], 1), "long_deg": round(c["geometry"]["base_long_side_deg"], 1),
                         "height": round(c["geometry"]["height_above_table_mm"], 1)}
                        for c in ctx2["candidates"] if c["geometry"]["base_point_mm"] and "in_tower_zone" in c["rejection_reasons"]]
            ep.event("flow", phase="final_validation", observation_id=obs2["observation_id"], blocks_in_tower_zone=sorted(measured, key=lambda m: (m["z"], m["xy"][1])))
        ep.state["filled"] = filled
        ep.save()
        return 0 if (placed or filled) else 1
    finally:
        await asyncio.wait_for(client.close(), timeout=5)


def load_decision(path):
    decision = json.loads(Path(path).read_text())
    if decision.get("kind") not in ("pick_and_place_proposal", "geometry_preview"):
        raise ConfigError(f"Decision file is a {decision.get('kind')}, nothing to execute")
    if not decision["source_pose"].get("block_top_center_mm"):
        raise ConfigError("Decision has no block pose in the robot frame")
    return decision


async def run_step(config, data_dir, step, operator, episode_id, decision_path=None, table_z=None, push_distance=None):
    config.require_motion(operator)
    if step not in STEPS:
        raise ConfigError(f"Unknown step {step}; choose from {STEPS}")
    from .capture import open_client
    # The SDK registers resource clients per subtype at import time; these imports must precede the connection or
    # the manager never creates the arm/gripper/motion clients (ResourceNotFoundError despite the names being listed).
    from viam.components.arm import Arm  # noqa: F401
    from viam.components.gripper import Gripper  # noqa: F401
    from viam.services.motion import MotionClient  # noqa: F401
    ep = Episode(data_dir, episode_id, operator)
    client = await open_client(config)
    try:
        await asyncio.wait_for(client.refresh(), timeout=10)
        names = sorted(r.name for r in client.resource_names)
        for needed in ("arm", "gripper", "builtin"):
            if needed not in names:
                raise ConfigError(f"Machine does not list {needed!r} right now (resources: {names}); not moving")
        bot = Robot(client, config)
        before = await bot.snapshot()
        if step == "status":
            ep.event(step, measured=before, state={k: v for k, v in ep.state.items() if k != "decision"})
            return 0

        if step == "home":
            await bot.arm.stop()
            await bot.gripper.stop()
            await bot.gripper.open()
            g = await bot.gripper_origin()
            if g[2] < SAFE_Z_MM:
                ep.event(step, phase="lift_clear_first", planned_dz=SAFE_Z_MM - g[2])
                await bot.move_delta(dz=SAFE_Z_MM - g[2], near_table=True)
            ep.event(step, phase="plan", target_flange=list(HOME), before=before)
            ok, target = await bot.move_to(*HOME)
            ep.event(step, accepted=ok, target_flange=target, measured=await bot.snapshot())
            ep.state.pop("held", None)
            ep.save()
            return 0 if ok else 1

        if step == "hover":
            if not decision_path:
                raise ConfigError("hover needs --decision FILE (the agent-propose decision json)")
            decision = load_decision(decision_path)
            ep.state["decision"] = decision
            ep.state["decision_file"] = str(decision_path)
            ctx_path = Path(str(decision_path).replace("-decision.json", ".json"))
            if ctx_path.exists():
                placement = json.loads(ctx_path.read_text()).get("placement", {})
                if placement.get("target"):
                    ep.state["slots"] = placement["target"]["slots"]
            ep.state["table_z_mm"] = table_z if table_z is not None else decision["source_pose"].get("top_z_from_support_offset_mm", 0) - decision["source_pose"]["height_above_table_mm"]
            ep.save()
            bx, by, bz = decision["source_pose"]["block_top_center_mm"]
            g = await bot.gripper_origin()
            ep.event(step, phase="plan", decision_kind=decision["kind"], candidate_id=decision["candidate_id"], block_top_center=[bx, by, bz],
                     grasp=decision["grasp"]["style"], planned_dxdy=[round(bx - g[0], 1), round(by - g[1], 1)], travel_gripper_z=round(g[2], 1), before=before)
            ok, target = await bot.move_delta(dx=bx - g[0], dy=by - g[1])
            after = await bot.snapshot()
            ep.event(step, phase="translated", accepted=ok, target_flange=target, measured=after,
                     residual_xy=[round(bx - after["gripper_origin"][0], 1), round(by - after["gripper_origin"][1], 1)])
            if not ok:
                return 1
            close_deg = decision["grasp"].get("claws_close_direction_deg_in_base")
            if close_deg is None:
                close_deg = wrap_90(decision["source_pose"]["long_side_deg"] + decision["grasp"]["claws_deg_to_long_side"])
            aligned, log = await bot.align_claws(close_deg)
            after = await bot.snapshot()
            ep.event(step, phase="aligned", target_close_deg=round(close_deg, 1), aligned=aligned, error_log_deg=log, measured=after)
            ep.state["travel_gripper_z"] = after["gripper_origin"][2]
            ep.save()
            return 0 if aligned else 1

        decision = ep.state.get("decision")
        if decision is None:
            raise ConfigError("Run hover first in this episode")
        table_z = ep.state["table_z_mm"]
        tips_mm = decision["grasp"]["tips_above_table_mm"]

        if step == "descend":
            g = await bot.gripper_origin()
            target_z = table_z + tips_mm - TIP_ABOVE_GRIPPER_FRAME_MM
            ep.event(step, phase="plan", table_z_mm=table_z, tips_above_table_mm=tips_mm, target_gripper_z=round(target_z, 1), planned_dz=round(target_z - g[2], 1),
                     note="table_z is the unverified support offset; if the table is really at 0 the tips stop this much higher: " + str(round(table_z, 1)), before=before)
            ok, target = await bot.move_delta(dz=target_z - g[2], near_table=True)
            ep.event(step, accepted=ok, target_flange=target, measured=await bot.snapshot())
            return 0 if ok else 1

        if step == "grab":
            bx, by, bz = decision["source_pose"]["block_top_center_mm"]
            held = await bot.world_to_frame("gripper", bx, by, bz)
            got = await bot.gripper.grab()
            ep.event(step, grab_returned=got, held_offset_in_gripper_frame=[round(v, 1) for v in held], measured=await bot.snapshot())
            if not got:
                await bot.gripper.open()
                ep.event(step, phase="opened_after_failed_grab", note="no retry; operator decides (abort-up or home)")
                return 1
            ep.state["held"] = {"offset": list(held), "tips_mm": tips_mm, "claws_deg": decision["grasp"]["claws_deg_to_long_side"]}
            ep.save()
            return 0

        if step == "lift":
            ep.event(step, phase="plan", planned_dz=PICK_LIFT_MM, before=before)
            ok, target = await bot.move_delta(dz=PICK_LIFT_MM, near_table=True)
            ep.event(step, accepted=ok, target_flange=target, measured=await bot.snapshot(), still_holding=not await bot.gripper.is_moving())
            return 0 if ok else 1

        if step == "to-slot":
            held = ep.state.get("held")
            if held is None:
                raise ConfigError("Nothing recorded as held; run grab first")
            slot = decision["destination"]
            sx, sy, _ = slot["block_center_mm"]
            g = await bot.gripper_origin()
            ep.event(step, phase="plan_translate", slot=slot["slot_id"], slot_center=[sx, sy], planned_dxdy=[round(sx - g[0], 1), round(sy - g[1], 1)], before=before)
            ok, target = await bot.move_delta(dx=sx - g[0], dy=sy - g[1])
            if not ok:
                ep.event(step, accepted=False, target_flange=target)
                return 1
            close_deg = wrap_90(slot["long_side_deg"] + held["claws_deg"])
            aligned, log = await bot.align_claws(close_deg, tolerance_deg=1.5)
            # The block centre sits at the recorded offset in the gripper frame; put the block, not the gripper, on the slot.
            bx, by, _ = await bot.point_in_world("gripper", x=held["offset"][0], y=held["offset"][1])
            ok2, target2 = await bot.move_delta(dx=sx - bx, dy=sy - by)
            after = await bot.snapshot()
            bx2, by2, _ = await bot.point_in_world("gripper", x=held["offset"][0], y=held["offset"][1])
            ep.event(step, phase="positioned", aligned=aligned, error_log_deg=log, accepted=ok2, target_flange=target2, measured=after,
                     block_residual_xy=[round(sx - bx2, 1), round(sy - by2, 1)])
            return 0 if (ok2 and aligned) else 1

        if step == "release":
            held = ep.state.get("held")
            if held is None:
                raise ConfigError("Nothing recorded as held; run grab first")
            g = await bot.gripper_origin()
            target_z = table_z + held["tips_mm"] + PLACE_DROP_MM - TIP_ABOVE_GRIPPER_FRAME_MM
            ep.event(step, phase="plan_descend", target_gripper_z=round(target_z, 1), planned_dz=round(target_z - g[2], 1), before=before)
            ok, target = await bot.move_delta(dz=target_z - g[2], near_table=True)
            ep.event(step, phase="descended", accepted=ok, target_flange=target, measured=await bot.snapshot())
            if not ok:
                return 1
            await bot.gripper.open()
            ep.event(step, phase="opened", measured=await bot.snapshot())
            ep.state.pop("held", None)
            ep.save()
            return 0

        if step == "retreat":
            g = await bot.gripper_origin()
            travel = ep.state.get("travel_gripper_z", SAFE_Z_MM + 50)
            ep.event(step, phase="plan", planned_dz=round(travel - g[2], 1), before=before)
            ok, target = await bot.move_delta(dz=travel - g[2], near_table=True)
            ep.event(step, accepted=ok, target_flange=target, measured=await bot.snapshot())
            return 0 if ok else 1

        if step == "abort-up":
            g = await bot.gripper_origin()
            dz = max(SAFE_Z_MM - g[2], 0.0)
            ep.event(step, phase="plan", planned_dz=round(dz, 1), note="gripper left as is; use `open` separately", before=before)
            ok, target = await bot.move_delta(dz=dz, near_table=True) if dz else (True, None)
            ep.event(step, accepted=ok, target_flange=target, measured=await bot.snapshot())
            return 0 if ok else 1

        if step == "open":
            await bot.gripper.open()
            ep.state.pop("held", None)
            ep.save()
            ep.event(step, measured=await bot.snapshot())
            return 0

        if step == "push":
            # Sam's push_together: closed claws come down just outside the last block of the row, on the side away from
            # the first, and slide toward the first block to close the gaps left for the claws.
            slots = [s for s in ep.state.get("slots", []) if s["layer"] == 0]
            if len(slots) < 2:
                raise ConfigError("No layer-0 slot plan in this episode; run hover with a decision first")
            if ep.state.get("held"):
                raise ConfigError("Still recorded as holding a block; release first")
            first, last = slots[0]["block_center_mm"], slots[-1]["block_center_mm"]
            d = math.hypot(first[0] - last[0], first[1] - last[1])
            ux, uy = (first[0] - last[0]) / d, (first[1] - last[1]) / d
            block_w = slots[0]["block_dimensions_mm"][1]
            back = block_w / 2 + CLAW_THICKNESS_MM + PUSH_START_GAP_MM
            start = (last[0] - ux * back, last[1] - uy * back)
            gaps = sum(max(0.0, math.hypot(b["block_center_mm"][0] - a["block_center_mm"][0], b["block_center_mm"][1] - a["block_center_mm"][1]) - block_w)
                       for a, b in zip(slots, slots[1:]))
            distance = push_distance if push_distance is not None else gaps + PUSH_START_GAP_MM + PUSH_EXTRA_MM
            await bot.gripper.grab()  # closed claws make a solid pusher
            g = await bot.gripper_origin()
            travel_z = g[2]
            ep.event(step, phase="plan", start_xy=[round(v, 1) for v in start], direction=[round(ux, 3), round(uy, 3)], distance_mm=round(distance, 1),
                     row_gaps_mm=round(gaps, 1), tips_above_table_mm=PUSH_TIPS_MM, before=before)
            ok, _ = await bot.move_delta(dx=start[0] - g[0], dy=start[1] - g[1])
            if not ok:
                ep.event(step, phase="translate_failed")
                return 1
            # Push with the claws' flat outer face, which runs along the blocks' long sides when they close across them.
            aligned, log = await bot.align_claws(wrap_90(slots[0]["long_side_deg"] + 90.0), tolerance_deg=1.5)
            g = await bot.gripper_origin()
            ok, _ = await bot.move_delta(dx=start[0] - g[0], dy=start[1] - g[1])  # turning may have shifted it
            g = await bot.gripper_origin()
            target_z = table_z + PUSH_TIPS_MM - TIP_ABOVE_GRIPPER_FRAME_MM
            ep.event(step, phase="at_start", aligned=aligned, error_log_deg=log, measured=await bot.snapshot(), target_gripper_z=round(target_z, 1))
            ok, _ = await bot.move_delta(dz=target_z - g[2], near_table=True)
            if not ok:
                ep.event(step, phase="descend_failed")
                return 1
            ok, target = await bot.move_delta(dx=ux * distance, dy=uy * distance, near_table=True)
            ep.event(step, phase="pushed", accepted=ok, target_flange=target, measured=await bot.snapshot())
            g = await bot.gripper_origin()
            ok2, _ = await bot.move_delta(dz=travel_z - g[2], near_table=True)
            await bot.gripper.open()
            ep.event(step, phase="retreated_and_opened", accepted=ok2, measured=await bot.snapshot())
            return 0 if (ok and ok2) else 1
    finally:
        await asyncio.wait_for(client.close(), timeout=5)
    raise ConfigError(f"Unhandled step {step}")
