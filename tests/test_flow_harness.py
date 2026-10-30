"""Offline harness for live_executor.run_flow (Kris's heap -> pallet assembly, used by live-run and palletize).
Copied from codex/layer-stack (tests/test_flow_harness.py) with tests for this branch: the released-row lane, the
pallet pick-and-drop repair and the drop-spot checks.

A simulated arm, gripper and camera stand in for the robot: captures are synthetic renders of the current world, the
gripper picks up the block under it and releases it where it opens, and closed claws moving low push blocks ahead of
them. Kris's own Robot.align_claws and Robot.descend_to run on top of the simulated moves. No network, no robot; the
viam SDK is imported (run_flow imports resource APIs before connecting) but nothing connects.
Needs numpy, Pillow and viam (RGBD_TESTS_REQUIRED=1 fails instead of skipping).
"""
import asyncio
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy  # noqa: F401
    from PIL import Image  # noqa: F401
    import viam  # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False
    if os.environ.get("RGBD_TESTS_REQUIRED") == "1":
        raise

FLANGE_ABOVE_GRIPPER_MM = 172.0
PUSHER_HALF_THICKNESS_MM = 8.0     # closed claws: two claws, measured along the push
PUSHER_HALF_WIDTH_MM = 22.5        # finger width 45 across the push


class World:
    """Blocks as [name, x, y, long_deg, level, tilt_deg, lift_mm] (synthetic_scene preset entries) plus the arm."""

    def __init__(self, blocks):
        self.blocks = [list(b) + [0.0] * (7 - len(b)) for b in blocks]
        self.g = [300.0, 150.0, 400.0 - FLANGE_ABOVE_GRIPPER_MM]   # gripper origin
        self.claws = 90.0                                            # world direction the claws close along
        self.closed = False
        self.held = None                                             # (block, offset in gripper frame, angle offset)
        self.captures = 0
        self.last_masks = []
        self.refuse = lambda kind, **kw: False                       # test hook: return True to refuse a move
        self.moves = []

    def table_z(self, x, y):
        from robot_hackathon import synthetic_scene as synth
        return synth.table_z(x, y)

    def frame(self):
        c = math.radians(self.claws)
        return (math.sin(c), -math.cos(c)), (math.cos(c), math.sin(c))   # gripper x axis, y axis (claws close along y)

    def held_follow(self):
        if self.held is None:
            return
        b, (fx, fy), dang = self.held
        ex, ey = self.frame()
        b[1], b[2] = self.g[0] + fx * ex[0] + fy * ey[0], self.g[1] + fx * ex[1] + fy * ey[1]
        b[3] = self.claws + dang

    def push(self, old_xy, new_xy):
        """Closed claws low over the table push every block their face covers that lay ahead of them; a pushed block
        pushes the next one only end to end (it must have been ahead of that block's face before the move, so blocks
        lying side by side are not shoved along). Contact needs more than 1 mm of sideways overlap."""
        dx, dy = new_xy[0] - old_xy[0], new_xy[1] - old_xy[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return
        ux, uy = dx / dist, dy / dist
        face_before = old_xy[0] * ux + old_xy[1] * uy + PUSHER_HALF_THICKNESS_MM
        face = new_xy[0] * ux + new_xy[1] * uy + PUSHER_HALF_THICKNESS_MM

        def extents(b):
            a = math.radians(b[3])
            along = abs(37.5 * (math.cos(a) * ux + math.sin(a) * uy)) + abs(12.5 * (-math.sin(a) * ux + math.cos(a) * uy))
            across = abs(37.5 * (-math.cos(a) * uy + math.sin(a) * ux)) + abs(12.5 * (math.sin(a) * uy + math.cos(a) * ux))
            return along, across

        # Every block the claw face covers is pushed by it; a pushed block in turn pushes a block ahead of it, end to end.
        pushers = [(-new_xy[0] * uy + new_xy[1] * ux, PUSHER_HALF_WIDTH_MM, face_before, face)]
        for b in sorted((b for b in self.blocks if b[4] == 0), key=lambda b: b[1] * ux + b[2] * uy):
            along, across = extents(b)
            p = b[1] * ux + b[2] * uy
            lat = -b[1] * uy + b[2] * ux
            shift = 0.0
            for plat, phalf, pbefore, pface in pushers:
                if abs(lat - plat) <= phalf + across - 1.0 and p - along >= pbefore - 1.0 and p - along < pface:
                    shift = max(shift, pface - (p - along))
            if shift > 0:
                b[1] += shift * ux
                b[2] += shift * uy
                pushers.append((lat, across, p + along, p + shift + along))

    def preset(self):
        return [tuple(b[:5]) for b in self.blocks]   # synthetic_scene renders flat blocks: name, x, y, long_deg, level


class FakeGripper:
    def __init__(self, world):
        self.w = world

    async def open(self):
        w = self.w
        w.closed = False
        if w.held is not None:
            b = w.held[0]
            w.held = None
            others = [o for o in w.blocks if o is not b and math.hypot(o[1] - b[1], o[2] - b[2]) < 20]
            b[4] = (max(o[4] for o in others) + 1) if others else 0
            b[5] = b[6] = 0.0
        return None

    async def grab(self):
        w = self.w
        w.closed = True
        tips = w.g[2] + 0.5 - w.table_z(w.g[0], w.g[1])
        under = [b for b in w.blocks if math.hypot(b[1] - w.g[0], b[2] - w.g[1]) < 15
                 and (b[4] + 1) * 15 + b[6] > tips]
        if not under:
            return False
        b = max(under, key=lambda b: b[4])
        ex, ey = w.frame()
        rx, ry = b[1] - w.g[0], b[2] - w.g[1]
        w.held = (b, (rx * ex[0] + ry * ex[1], rx * ey[0] + ry * ey[1]), b[3] - w.claws)
        return True

    async def is_moving(self):
        return False


class FakeArm:
    async def do_command(self, cmd):
        return {}

    async def stop(self):
        return None

    async def is_moving(self):
        return False


def make_robot_class(world):
    from robot_hackathon.live_executor import Robot

    class FakeRobot(Robot):
        def __init__(self, client, config):   # no viam resources
            self.client, self.arm, self.gripper, self.motion, self.near_table = client, FakeArm(), FakeGripper(world), None, None

        async def flange_pose(self):
            return types.SimpleNamespace(x=world.g[0], y=world.g[1], z=world.g[2] + FLANGE_ABOVE_GRIPPER_MM,
                                         o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)

        async def point_in_world(self, frame, x=0.0, y=0.0, z=0.0):
            ex, ey = world.frame()
            return (world.g[0] + x * ex[0] + y * ey[0], world.g[1] + x * ex[1] + y * ey[1], world.g[2] + z)

        async def world_to_frame(self, frame, x, y, z):
            ex, ey = world.frame()
            dx, dy = x - world.g[0], y - world.g[1]
            return (dx * ex[0] + dy * ex[1], dx * ey[0] + dy * ey[1], z - world.g[2])

        async def move_delta(self, dx=0.0, dy=0.0, dz=0.0, rotate=0.0, near_table=False):
            kind = "lift" if dz > 0 and not (dx or dy) else ("rotate" if rotate else ("down" if dz < 0 else "xy"))
            target = [world.g[0] + dx, world.g[1] + dy, world.g[2] + dz]
            if world.refuse(kind, held=world.held is not None, closed=world.closed, target=target):
                world.moves.append(("refused", kind))
                return False, target
            old = (world.g[0], world.g[1])
            world.g = target
            world.claws -= rotate
            world.moves.append((kind, world.held is not None))
            if world.closed and world.held is None and world.g[2] + 0.5 - world.table_z(*old) < 12:
                world.push(old, (world.g[0], world.g[1]))
            world.held_follow()
            return True, [round(v, 1) for v in target]

        async def move_to(self, x, y, z):
            if world.refuse("move_to", held=world.held is not None, closed=world.closed, target=[x, y, z]):
                world.moves.append(("refused", "move_to"))
                return False, [x, y, z]
            world.g = [x, y, z - FLANGE_ABOVE_GRIPPER_MM]
            world.claws = 90.0
            world.moves.append(("move_to", world.held is not None))
            world.held_follow()
            return True, [x, y, z]

        async def set_speed(self, deg_s):
            return None

        async def grip_state(self):
            return {"holding": world.held is not None, "width": 25.0 if world.held is not None else 0.0, "source": "fake", "meta": {}}

    return FakeRobot


@unittest.skipUnless(HAVE_DEPS, "numpy, Pillow and viam required for the run_flow harness")
class RunFlowHarness(unittest.TestCase):
    PALLET = (200.0, 150.0)

    def setUp(self):
        from robot_hackathon import synthetic_scene as synth, rgbd
        from robot_hackathon.grasp_geometry import GeometryCalibrationStore
        self.synth, self.rgbd = synth, rgbd
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        GeometryCalibrationStore(self.data, "live").record(synth.SETUP_ID, synth.GEOMETRY_VERSION, synth.geometry_document())
        self.addCleanup(synth.PRESETS.pop, "_flow", None)

    def run_flow(self, world, **kw):
        from robot_hackathon import live_executor, layer_stack, capture
        from robot_hackathon.config import Config
        from viam.components.camera import Camera
        synth, rgbd = self.synth, self.rgbd

        async def fake_capture(config, data_dir, camera, episode_id=None, client=None):
            world.captures += 1
            synth.PRESETS["_flow"] = world.preset()
            rgb, depth, masks, _ = synth.render("_flow", seed=world.captures)
            world.last_masks = masks
            rec = synth.observation_record("_flow", f"obs_flow_{world.captures:03d}", episode_id or "ep")
            rec["mode"] = "live"
            return rgbd.ObservationStore(data_dir, "live").save(rec, rgb, depth)

        class Seg:
            provider_id = "harness-ground-truth"

            def segment(self, rgb):
                return [(m, 0.9) for m in world.last_masks]

        class Client:
            resource_names = [types.SimpleNamespace(name=n) for n in ("arm", "gripper", "builtin", "cam")]

            async def refresh(self):
                return None

            async def close(self):
                return None

        async def fake_open(config, direct_local, disable_sessions=False):
            return Client()

        config = Config.from_env({"MODE": "live", "ENABLE_MOTION": "true"})
        with mock.patch.object(layer_stack, "_open_client", fake_open), \
                mock.patch.object(capture, "capture_with_camera", fake_capture), \
                mock.patch.object(live_executor, "Robot", make_robot_class(world)), \
                mock.patch.object(Camera, "from_robot", staticmethod(lambda client, name: None)):
            rc = asyncio.run(live_executor.run_flow(config, self.data, "Tester", "ep-harness", self.PALLET, [], 0.0, 5, Seg(),
                                                    layers=1, **kw))
        events = [json.loads(line) for line in (self.data / "live" / "episodes" / "ep-harness" / "events.jsonl").read_text().splitlines()]
        return rc, events

    def flat_heap(self):
        return World([("F", 360.0, 40.0, 30.0, 0), ("G", 330.0, 170.0, 10.0, 0), ("H", 470.0, 260.0, 90.0, 0),
                      ("K", 420.0, 330.0, -20.0, 0)])

    def on_pallet(self, world, radius=102.5):
        return sorted(b[0] for b in world.blocks if b[4] == 0 and math.hypot(b[1] - self.PALLET[0], b[2] - self.PALLET[1]) < radius)

    def test_third_across_grasp_block_is_counted_not_repicked(self):
        world = self.flat_heap()
        rc, events = self.run_flow(world)
        picks = [e for e in events if e.get("phase") == "plan_block"]
        self.assertEqual(rc, 0)
        self.assertEqual(len(picks), 3)
        self.assertTrue(all(p["grasp"] == "middle" for p in picks))       # all across: released 77.5 mm apart
        self.assertFalse(any(e.get("phase") == "top_up" for e in events))
        rows = [e for e in events if e.get("phase") == "row_measured"]
        self.assertEqual(rows[0]["blocks"], 3)                            # the third, ~138 mm out, is counted
        self.assertEqual([e for e in events if e.get("phase") == "final_validation"][-1]["pallet_count"], 3)
        self.assertEqual(len(self.on_pallet(world)), 3)

    def test_block_put_down_on_its_neighbour_is_taken_off_and_replaced(self):
        world = self.flat_heap()
        orig_open = FakeGripper.open
        pallet = []

        async def open_(gripper):
            b = gripper.w.held[0] if gripper.w.held is not None else None
            await orig_open(gripper)
            if b is not None and math.hypot(b[1] - self.PALLET[0], b[2] - self.PALLET[1]) < 102.5:
                pallet.append(b)
                if len(pallet) == 2:                                      # lands on top of the first pallet block
                    b[1], b[2], b[4] = pallet[0][1] + 3.0, pallet[0][2] + 8.0, 1
        with mock.patch.object(FakeGripper, "open", open_):
            rc, events = self.run_flow(world, drop_spot_xy=(500.0, 120.0))
        repair = [e for e in events if e.get("phase") == "pallet_repair"]
        self.assertEqual(len(repair), 1)
        self.assertTrue(repair[0]["misplaced"][0].startswith("height:"))
        check = [e for e in events if e.get("phase") == "reorient_check" and e.get("slot") == "repair"]
        self.assertTrue(check and check[0]["flat"])
        self.assertEqual([e for e in events if e.get("phase") == "final_validation"][-1]["pallet_count"], 3)
        self.assertEqual(len(self.on_pallet(world)), 3)
        self.assertEqual([b[0] for b in world.blocks if b[4] == 1], [])  # nothing left lying on the row

    def run_with_upright(self, world, prefix, dims, tips_range, fall_mm, hi_res=False):
        """Blocks named prefix* are upright with footprint/height `dims` (rendered separately and composited); the
        simulated claws topple one when they sweep through it with their tips inside tips_range, leaving it flat
        `fall_mm` further along the sweep. hi_res doubles the camera resolution so a small end face clears run_flow's
        200 px mask minimum (on the D435 a lying block covers ~9000 px, an end face ~1800)."""
        import numpy as np
        from robot_hackathon import synthetic_scene as synth
        orig_render = synth.render

        def render(preset, seed=7, invalid_fraction=0.01):
            entries = synth.PRESETS[preset]
            synth.PRESETS["_flat"] = [e for e in entries if not e[0].startswith(prefix)]
            synth.PRESETS["_up"] = [e for e in entries if e[0].startswith(prefix)]
            rgb, depth, masks, truth = orig_render("_flat", seed, invalid_fraction)
            if synth.PRESETS["_up"]:
                with mock.patch.object(synth, "BLOCK", dims):
                    rgb2, depth2, masks2, _ = orig_render("_up", seed, invalid_fraction)
                nearer = (depth2 > 0) & ((depth == 0) | (depth2 < depth - 2))
                depth = np.where(nearer, depth2, depth)
                rgb[nearer] = rgb2[nearer]
                masks = [m & ~nearer for m in masks] + [m & nearer for m in masks2]
            return rgb, depth, masks, truth

        robot_cls = make_robot_class(world)
        base_move = robot_cls.move_delta

        async def move_delta(robot, dx=0.0, dy=0.0, dz=0.0, rotate=0.0, near_table=False):
            old = (world.g[0], world.g[1])
            res = await base_move(robot, dx, dy, dz, rotate, near_table)
            tips = world.g[2] + 0.5 - world.table_z(*old)
            if res[0] and world.closed and world.held is None and tips_range[0] < tips < tips_range[1] and (dx or dy):
                d = math.hypot(dx, dy)
                ux, uy = dx / d, dy / d
                for b in world.blocks:
                    ahead_before = (b[1] - old[0]) * ux + (b[2] - old[1]) * uy
                    ahead_after = (b[1] - world.g[0]) * ux + (b[2] - world.g[1]) * uy
                    side = abs(-(b[1] - world.g[0]) * uy + (b[2] - world.g[1]) * ux)
                    if b[0].startswith(prefix) and side < 30.0 and ahead_before > 0 and ahead_after < 15.0:
                        b[0] = "FELL" + b[0][len(prefix):]
                        b[1], b[2] = b[1] + ux * fall_mm, b[2] + uy * fall_mm
                        if prefix == "UP":
                            b[3] = math.degrees(math.atan2(uy, ux))   # a standing block falls with its length along the sweep
            return res
        robot_cls.move_delta = move_delta
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(synth, "render", render))
            stack.enter_context(mock.patch(__name__ + ".make_robot_class", lambda w: robot_cls))
            if hi_res:
                stack.enter_context(mock.patch.object(synth, "INTRINSICS", dict(synth.INTRINSICS, width_px=640, height_px=480, focal_x_px=456.0,
                                                                                focal_y_px=456.0, center_x_px=320.0, center_y_px=240.0)))
            return self.run_flow(world, knock_over=True)

    def test_block_standing_on_end_is_knocked_over_then_picked(self):
        world = World([("A", 430.0, 60.0, 30.0, 0), ("C", 470.0, 250.0, -40.0, 0), ("UP1", 380.0, 120.0, 0.0, 0)])
        rc, events = self.run_with_upright(world, "UP", (25.0, 15.0, 75.0), (25.0, 70.0), 45.0, hi_res=True)
        knock = [e for e in events if e.get("phase") == "knock_over"]
        self.assertEqual(len(knock), 1)
        self.assertEqual((knock[0]["onto"], knock[0]["upright"], knock[0]["tips_mm"]), ("wide_face", "standing", 45.0))
        self.assertGreater(knock[0]["height_mm"], 60.0)
        check = [e for e in events if e.get("phase") == "knock_check"][0]
        self.assertFalse(check["still_standing"])
        self.assertLess(check["landed_height_mm"], 20.0)
        self.assertEqual([e for e in events if e.get("phase") == "final_validation"][-1]["pallet_count"], 3)
        self.assertEqual(sorted(b[0] for b in world.blocks), ["A", "C", "FELL1"])

    def test_block_on_its_edge_is_swept_over_not_dropped(self):
        # 30 Oct 19:59: dropping a block on its edge left it on its edge twice. Now it is swept over where it lies.
        world = World([("A", 430.0, 60.0, 30.0, 0), ("C", 470.0, 250.0, -40.0, 0), ("EDGE1", 360.0, 180.0, 10.0, 0)])
        rc, events = self.run_with_upright(world, "EDGE", (75.0, 15.0, 25.0), (12.0, 30.0), 20.0)
        sweep = [e for e in events if e.get("phase") == "knock_over"]
        self.assertEqual(len(sweep), 1)
        self.assertEqual((sweep[0]["upright"], sweep[0]["onto"], sweep[0]["tips_mm"]), ("on_edge", "wide_face", 16.0))
        self.assertFalse(any(e.get("phase") == "dropped" for e in events))
        self.assertFalse([e for e in events if e.get("phase") == "knock_check"][0]["still_standing"])
        self.assertEqual([e for e in events if e.get("phase") == "final_validation"][-1]["pallet_count"], 3)
        self.assertEqual(sorted(b[0] for b in world.blocks), ["A", "C", "FELL1"])

    def test_low_travel_goes_over_a_tall_block_near_its_path(self):
        # Spec: the closed claws travel 30 mm above the table during the push stage. A 60 mm-tall object (here a block
        # rendered at level 3, rejected for picking) right on the way from home to the push start is gone over instead.
        world = self.flat_heap()
        world.blocks.append(["TALL", 250.0, 232.0, 60.0, 3, 0.0, 0.0])
        rc, events = self.run_flow(world)
        detours = [e for e in events if e.get("phase") == "low_travel_detour"]
        self.assertTrue(detours)
        self.assertTrue(any(abs(o[0] - 250.0) < 5 and abs(o[1] - 232.0) < 5 for d in detours for o in d["over"]))
        self.assertEqual([e for e in events if e.get("phase") == "final_validation"][-1]["pallet_count"], 3)
        self.assertEqual([b[1:3] for b in world.blocks if b[0] == "TALL"], [[250.0, 232.0]])   # never touched

    def refused_before_picking(self, spot, reason):
        rc, events = self.run_flow(self.flat_heap(), drop_spot_xy=spot)
        self.assertEqual(rc, 1)
        self.assertEqual([e for e in events if e.get("phase") == "stop"][-1]["reason"], reason)
        self.assertFalse(any(e.get("phase") == "plan_block" for e in events))

    def test_drop_spot_too_close_to_the_pallet_is_refused(self):
        self.refused_before_picking((260.0, 150.0), "drop_spot_too_close")

    def test_drop_spot_in_line_with_the_row_is_refused(self):
        self.refused_before_picking((200.0, 330.0), "drop_spot_in_pallet_lane")

if __name__ == "__main__":
    unittest.main()
