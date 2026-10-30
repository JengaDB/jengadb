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
MOVE_TIMEOUT_S = 120.0
STEPS = ("status", "home", "hover", "descend", "grab", "lift", "to-slot", "release", "retreat", "abort-up", "open")


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


def load_decision(path):
    decision = json.loads(Path(path).read_text())
    if decision.get("kind") not in ("pick_and_place_proposal", "geometry_preview"):
        raise ConfigError(f"Decision file is a {decision.get('kind')}, nothing to execute")
    if not decision["source_pose"].get("block_top_center_mm"):
        raise ConfigError("Decision has no block pose in the robot frame")
    return decision


async def run_step(config, data_dir, step, operator, episode_id, decision_path=None, table_z=None):
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
    finally:
        await asyncio.wait_for(client.close(), timeout=5)
    raise ConfigError(f"Unhandled step {step}")
