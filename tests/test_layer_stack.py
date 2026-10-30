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

    def test_between_level_detections_below_the_top_are_tolerated_above_it_not(self):
        # robot-live-8 verify: two layers crossed; the lower layer's protruding ends read 18.4 and 20.4 mm (mixed depth)
        # next to the 29 mm top layer. Those are not a tilted block on top.
        tower = layer((523, -15), -85, level=0) + layer((523, -18), 5, level=1)
        ends = [{"base_point_mm": (531, 22, 0), "base_long_side_deg": 4, "height_above_table_mm": 18.4},
                {"base_point_mm": (499, 18, 0), "base_long_side_deg": 2, "height_above_table_mm": 20.4}]
        ignored = []
        outside, top = ls.split_scene(layer((439, 193), 4) + tower + ends, (550, 0), 115.0, ignored=ignored)
        measured = ls.summarize_layer(top)
        self.assertEqual(measured.level, 1)
        self.assertEqual(ls.tower_layers(measured), 2)
        self.assertEqual([b["height_above_table_mm"] for b in ignored], [20.4])   # 18.4 still rounds to level 0
        tilted_on_top = [{"base_point_mm": (523, -18, 0), "base_long_side_deg": 5, "height_above_table_mm": 37.0}]
        with self.assertRaisesRegex(ls.LayerCheckError, "on top of the tower is between layer heights"):
            ls.split_scene(tower + tilted_on_top, (550, 0), 115.0)
        with self.assertRaisesRegex(ls.LayerCheckError, "between layer heights"):
            ls.split_scene(ends, (550, 0), 115.0)   # nothing but between-level detections: still a stop

    def test_tower_layers_is_height_over_block_height(self):
        self.assertEqual(ls.tower_layers(ls.summarize_layer(layer((0, 0), 0, level=0))), 1)
        three = ls.summarize_layer([dict(b, height_above_table_mm=44.0) for b in layer((0, 0), 0)])   # measured 44 for 45
        self.assertEqual(ls.tower_layers(three), 3)
        self.assertEqual(ls.tower_layers(three, block_height_mm=14.5), 3)

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

    def test_tips_floor_clamps_pick_and_release_follows_the_achieved_grip(self):
        # robot-live-7, --grip-lower-mm 15 on 14.2 mm blocks: tips were clamped at the 1 mm floor (13.2 mm grip, not 23),
        # but the release was computed from the requested 23 mm and would have pushed the bundle 8 mm into the base.
        src = ls.summarize_layer(layer((300, 100), 0))
        base = ls.summarize_layer(layer((435, 203), 0))
        plan = ls.plan_stack(src, base, params=ls.GraspParams(grasp_depth_mm=23.0))
        self.assertAlmostEqual(plan.pick_tips_above_table_mm, 1.0)
        self.assertAlmostEqual(plan.grip_below_top_mm, 15 - 1.0)
        self.assertAlmostEqual(plan.requested_grip_below_top_mm, 23.0)
        self.assertAlmostEqual(plan.release_tips_above_table_mm, 15 + 15 - 14.0 + 1)   # bundle bottom 1 mm above the base
        lower = ls.plan_stack(src, base, params=ls.GraspParams(grasp_depth_mm=23.0, tips_min_mm=-3.0))
        self.assertAlmostEqual(lower.pick_tips_above_table_mm, -3.0)   # operator lowered the floor
        self.assertAlmostEqual(lower.release_tips_above_table_mm, 15 + 15 - 18.0 + 1)

    def test_tips_floor_is_bounded(self):
        config = Config.from_env({"MODE": "live", "ENABLE_MOTION": "true"})
        with self.assertRaises(ls.LayerCheckError):
            asyncio.run(ls.run_layer_stack(config, Path(tempfile.gettempdir()), "Kris", "ep1", (550, 0), seg_provider=None, tips_floor_mm=-20.0))
        with self.assertRaises(ls.LayerCheckError):
            asyncio.run(ls.run_layer_stack(config, Path(tempfile.gettempdir()), "Kris", "ep1", (550, 0), seg_provider=None, tips_floor_mm=5.0))

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


class VisibleExtent(unittest.TestCase):
    # Camera pose and intrinsics as recorded on obs_cb99bc7f5df4440d (30 Oct, view flange 300,150,460): a wrist D435
    # looking straight down from world z 446, image columns along -y, rows along -x.
    OBS = {"camera": {"pose_in_world": {"status": "read", "translation_mm": [366.2, 182.1, 445.6],
                                        "rotation": [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]}},
           "intrinsics": {"rgb": {"focal_x_px": 907.8, "focal_y_px": 906.6, "center_x_px": 638.0, "center_y_px": 372.8,
                                  "width_px": 1280, "height_px": 720}}}

    def test_projects_image_corners_onto_table(self):
        seen = ls.visible_table_extent(self.OBS, table_z_mm=3.0, margin_px=0)
        # height above the table 442.6 mm; the principal point is off-centre, so the extent is asymmetric:
        # rows run along -x: top edge +372.8 px -> +182 mm, bottom edge -346.2 px -> -169 mm
        # columns run along -y: left edge +638 px -> +311 mm, right edge -641 px -> -313 mm
        self.assertAlmostEqual(seen["x_mm"][0], 366.2 - 169, delta=2)
        self.assertAlmostEqual(seen["x_mm"][1], 366.2 + 182, delta=2)
        self.assertAlmostEqual(seen["y_mm"][0], 182.1 - 313, delta=2)
        self.assertAlmostEqual(seen["y_mm"][1], 182.1 + 311, delta=2)
        # the tower on the blue square was in frame; Kazy's pick spot at x 656 was not
        self.assertTrue(seen["x_mm"][0] < 435 < seen["x_mm"][1] and seen["y_mm"][0] < 203 < seen["y_mm"][1])
        self.assertGreater(656, seen["x_mm"][1])

    def test_empty_tower_zone_is_a_distinct_stop(self):
        # run_layer_stack goes and looks at the tower origin only for this error, and only when it is out of frame
        with self.assertRaises(ls.NoBaseLayer):
            ls.split_scene(layer((440, 201), 2.0), (700, 200), 115.0)
        self.assertTrue(issubclass(ls.NoBaseLayer, ls.LayerCheckError))

    def test_missing_pose_or_upward_camera_gives_none(self):
        self.assertIsNone(ls.visible_table_extent({"camera": {"pose_in_world": {"status": "unavailable"}}, "intrinsics": self.OBS["intrinsics"]}))
        up = {"camera": {"pose_in_world": dict(self.OBS["camera"]["pose_in_world"], rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])},
              "intrinsics": self.OBS["intrinsics"]}
        self.assertIsNone(ls.visible_table_extent(up))
