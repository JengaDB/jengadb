"""Offline tests for supervised layer stacking (src/robot_hackathon/layer_stack.py). No SDK, network or robot."""
import asyncio
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_hackathon import layer_stack as ls  # noqa: E402
from robot_hackathon.config import Config, ConfigError  # noqa: E402


def layer(center, long_deg, level=0, pitch=25.0, stagger=0.0, jitter_deg=0.0):
    """Three measured block geometries forming one layer, in the shape build_context produces."""
    a = math.radians(long_deg)
    ax, ay = math.cos(a), math.sin(a)
    bx, by = -ay, ax
    blocks = []
    for i, k in enumerate((-1, 0, 1)):
        along = stagger if i == 2 else 0.0
        x = center[0] + k * pitch * bx + along * ax
        y = center[1] + k * pitch * by + along * ay
        blocks.append({"base_point_mm": (x, y, 0.0), "base_long_side_deg": long_deg + (jitter_deg if i == 1 else 0.0),
                       "height_above_table_mm": 15.0 * (level + 1)})
    return blocks


class SummarizeLayer(unittest.TestCase):
    def test_packed_layer_pose(self):
        pose = ls.summarize_layer(layer((400, 170), -6.5))
        self.assertAlmostEqual(pose.center_xy[0], 400, places=6)
        self.assertAlmostEqual(pose.center_xy[1], 170, places=6)
        self.assertAlmostEqual(pose.long_side_deg, -6.5, places=6)
        self.assertEqual(pose.level, 0)
        self.assertTrue(all(abs(s - 25) < 1e-6 for s in pose.spacing_mm))

    def test_angle_wraps_across_plus_minus_90(self):
        blocks = layer((0, 0), 89.0)
        blocks[0]["base_long_side_deg"] = -89.0  # same physical direction
        self.assertLess(abs(ls.wrap_90(ls.summarize_layer(blocks).long_side_deg - 89.0)), 1.0)

    def test_rejections(self):
        with self.assertRaisesRegex(ls.LayerCheckError, "gaps or overlaps"):
            ls.summarize_layer(layer((0, 0), 0, pitch=32))
        with self.assertRaisesRegex(ls.LayerCheckError, "not parallel"):
            ls.summarize_layer(layer((0, 0), 0, jitter_deg=8))
        with self.assertRaisesRegex(ls.LayerCheckError, "staggered"):
            ls.summarize_layer(layer((0, 0), 0, stagger=25))
        with self.assertRaisesRegex(ls.LayerCheckError, "expected 3"):
            ls.summarize_layer(layer((0, 0), 0)[:2])
        mixed = layer((0, 0), 0)
        mixed[0]["height_above_table_mm"] = 30.0
        with self.assertRaisesRegex(ls.LayerCheckError, "one layer height"):
            ls.summarize_layer(mixed)
        clipped = layer((0, 0), 0)
        clipped[2]["clipped"] = True
        with self.assertRaisesRegex(ls.LayerCheckError, "cut off by the image edge"):
            ls.summarize_layer(clipped)
        tilted = layer((0, 0), 0)
        tilted[0]["height_above_table_mm"] = 22.0
        with self.assertRaisesRegex(ls.LayerCheckError, "one layer height"):
            ls.summarize_layer(tilted)


class SplitAndPlan(unittest.TestCase):
    def test_split_takes_top_complete_layer_in_zone(self):
        blocks = layer((300, 100), 0) + layer((435, 203), 0, level=0) + layer((435, 203), 90, level=1)
        outside, top = ls.split_scene(blocks, (435, 203), 102.5)
        self.assertEqual(len(outside), 3)
        self.assertEqual(ls.summarize_layer(top).level, 1)

    def test_split_ignores_flat_clipped_detections(self):
        paper = {"base_point_mm": (474.8, -8.0, 0.0), "base_long_side_deg": 70.4, "height_above_table_mm": -0.8, "clipped": True}
        outside, top = ls.split_scene(layer((570, 216), 2.4) + layer((447, 202), 6.8) + [paper], (435, 203), 102.5)
        self.assertEqual(len(outside), 3)
        ls.summarize_layer(outside)

    def test_split_without_base(self):
        with self.assertRaisesRegex(ls.LayerCheckError, "no base layer"):
            ls.split_scene(layer((300, 100), 0), (435, 203), 102.5)

    def test_plan_rotates_ninety_and_centres(self):
        src = ls.summarize_layer(layer((300, 100), -6.5))
        base = ls.summarize_layer(layer((435, 203), 2.0))
        plan = ls.plan_stack(src, base)
        self.assertAlmostEqual(plan.target_long_side_deg, ls.wrap_90(92.0))
        self.assertAlmostEqual(plan.claws_close_at_pick_deg, ls.wrap_90(-6.5 + 90))   # claws squeeze the three together
        self.assertAlmostEqual(plan.claws_close_at_place_deg, ls.wrap_90(92.0 + 90))
        self.assertAlmostEqual(plan.target_center_xy[0], 435)
        self.assertAlmostEqual(plan.pick_tips_above_table_mm, 15 - 8)
        self.assertAlmostEqual(plan.release_tips_above_table_mm, 15 + 15 - 8 + 1)  # one layer up plus the release drop

    def test_deeper_grip_lowers_pick_and_release_together(self):
        src = ls.summarize_layer(layer((300, 100), 0))
        base = ls.summarize_layer(layer((435, 203), 0))
        a = ls.plan_stack(src, base)
        b = ls.plan_stack(src, base, params=ls.GraspParams(grasp_depth_mm=11.0))
        self.assertAlmostEqual(a.pick_tips_above_table_mm - b.pick_tips_above_table_mm, 3.0)
        self.assertAlmostEqual(a.release_tips_above_table_mm - b.release_tips_above_table_mm, 3.0)

    def test_plan_refuses_source_not_on_table(self):
        with self.assertRaisesRegex(ls.LayerCheckError, "on the table"):
            ls.plan_stack(ls.summarize_layer(layer((300, 100), 0, level=1)), ls.summarize_layer(layer((435, 203), 0)))

    def test_offset_shifts_command_not_success_reference(self):
        src = ls.summarize_layer(layer((300, 100), 0))
        base = ls.summarize_layer(layer((435, 203), 0))
        plan = ls.plan_stack(src, base, offset_xy=(2.0, -1.0))
        self.assertEqual(plan.target_center_xy, (433.0, 204.0))
        landed = ls.summarize_layer(layer((435.5, 203), 90, level=1))
        res = ls.placement_residual(plan, landed)
        self.assertAlmostEqual(res["center_error_vs_base_mm"], 0.5)
        self.assertTrue(res["success"])


class Residuals(unittest.TestCase):
    def plan(self):
        return ls.plan_stack(ls.summarize_layer(layer((300, 100), 0)), ls.summarize_layer(layer((435, 203), 0)))

    def test_residual_axes_and_failure(self):
        plan = self.plan()   # target long side 90 deg: world +y is "along"
        res = ls.placement_residual(plan, ls.summarize_layer(layer((435, 207), 90, level=1)))
        self.assertAlmostEqual(abs(res["along_mm"]), 4.0)   # sign follows the wrapped target angle (-90)
        self.assertAlmostEqual(res["across_mm"], 0.0)
        self.assertFalse(res["success"])
        res = ls.placement_residual(plan, ls.summarize_layer(layer((435, 203), 85, level=1)))
        self.assertAlmostEqual(res["yaw_error_deg"], -5.0)
        self.assertFalse(res["success"])
        res = ls.placement_residual(plan, ls.summarize_layer(layer((435, 203), 90, level=0)))
        self.assertFalse(res["success"])   # wrong level: it did not end up on top

    def test_fitted_offset_needs_consistent_runs(self):
        with tempfile.TemporaryDirectory() as d:
            log = ls.ResidualLog(d, "setup-a")
            self.assertIsNone(log.fitted_offset()[0])
            for dx in (2.0, 2.5, 1.5):
                log.append({"applied_offset_xy": [0.0, 0.0], "residual": {"command_residual_xy_mm": [dx, -1.0]}})
            log.append({"applied_offset_xy": [2.0, -1.0], "residual": {"command_residual_xy_mm": [9.0, 9.0]}})  # ignored
            offset, _ = log.fitted_offset()
            self.assertAlmostEqual(offset[0], 2.0)
            self.assertAlmostEqual(offset[1], -1.0)
            log.append({"applied_offset_xy": [0.0, 0.0], "residual": {"command_residual_xy_mm": [12.0, -1.0]}})
            offset, note = log.fitted_offset()
            self.assertIsNone(offset)
            self.assertIn("not a constant offset", note)


class LiftCheck(unittest.TestCase):
    def test_counts_blocks_still_at_pick_spot(self):
        src = ls.summarize_layer(layer((570, 213), 2.5))
        self.assertEqual(ls.blocks_left_behind(src, layer((570.5, 213.3), 3.0)), 3)   # run 3: nothing lifted
        self.assertEqual(ls.blocks_left_behind(src, layer((570, 213), 2.5)[:1]), 1)
        self.assertEqual(ls.blocks_left_behind(src, layer((448, 201), 6.8)), 0)
        held = [dict(b, height_above_table_mm=110.0) for b in layer((570, 213), 2.5)]   # bundle in the air is not on the table
        self.assertEqual(ls.blocks_left_behind(src, held), 0)


class MotionGate(unittest.TestCase):
    def test_requires_motion_flag_before_connecting(self):
        config = Config.from_env({"MODE": "live"})
        with self.assertRaises(ConfigError):
            asyncio.run(ls.run_layer_stack(config, Path(tempfile.gettempdir()), "Kazy", "ep1", (435, 203), seg_provider=None))


if __name__ == "__main__":
    unittest.main()


class ClippedMasks(unittest.TestCase):
    def test_bbox_near_border_marks_clipped(self):
        ctx = {"candidates": [{"bbox": [10, 0, 50, 40], "geometry": {"base_point_mm": (1, 2, 3)}},
                              {"bbox": [10, 10, 50, 40], "geometry": {"base_point_mm": (1, 2, 3)}},
                              {"bbox": [10, 10, 50, 40], "geometry": {"base_point_mm": None}}]}
        out = ls._blocks(ctx, 1280, 720)
        self.assertEqual([b["clipped"] for b in out], [True, False])
