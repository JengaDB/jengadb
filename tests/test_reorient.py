"""Offline tests for gravity reorientation checks (grasp_geometry.top_face_tilt_deg / reorient_reasons /
lying_flat_reasons). Height maps are built directly with known tilts; no robot, network or SAM.
Needs numpy (RGBD_TESTS_REQUIRED=1 fails instead of skipping)."""
import math
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
    if os.environ.get("RGBD_TESTS_REQUIRED") == "1":
        raise

from robot_hackathon import grasp_geometry as geom  # noqa: E402

MM_PER_PX = 0.8


def block(tilt_deg=0.0, base_mm=15.0, side_faces=True):
    """A 75 x 25 mm block top seen from above at MM_PER_PX, tilted along its length, plus a 2 px ring of side-face
    pixels at mid height (what a SAM mask also covers)."""
    h, w = 140, 160
    hm = np.full((h, w), 0.0)
    mask = np.zeros((h, w), dtype=bool)
    length_px, width_px = int(75 * math.cos(math.radians(tilt_deg)) / MM_PER_PX), int(25 / MM_PER_PX)
    u0, v0 = (w - length_px) // 2, (h - width_px) // 2
    for u in range(u0, u0 + length_px):
        along_mm = (u - (u0 + length_px / 2)) * MM_PER_PX
        hm[v0:v0 + width_px, u] = base_mm + along_mm * math.tan(math.radians(tilt_deg))
        mask[v0:v0 + width_px, u] = True
    if side_faces:
        ring = np.zeros_like(mask)
        ring[v0 - 2:v0 + width_px + 2, u0 - 2:u0 + length_px + 2] = True
        ring &= ~mask
        hm[ring] = base_mm / 2
        mask |= ring
    return mask, hm


def flat_geometry(**kw):
    g = {"height_above_table_mm": 15.0, "length_mm": 74.0, "width_mm": 24.0, "flags": []}
    g.update(kw)
    return g


@unittest.skipUnless(HAVE_NUMPY, "numpy required")
class TopFaceTilt(unittest.TestCase):
    def test_recovers_known_tilt_despite_side_faces(self):
        for tilt in (0.0, 5.0, 12.0, 25.0):
            mask, hm = block(tilt)
            self.assertAlmostEqual(geom.top_face_tilt_deg(mask, hm, MM_PER_PX), tilt, delta=1.5, msg=tilt)

    def test_unknown_without_depth(self):
        mask, hm = block(0.0)
        hm[:] = np.nan
        self.assertIsNone(geom.top_face_tilt_deg(mask, hm, MM_PER_PX))
        self.assertIsNone(geom.top_face_tilt_deg(mask, block(0.0)[1], None))


class Reasons(unittest.TestCase):
    dims = geom.BlockDimensions()

    def test_reorient_only_tilted_or_on_edge(self):
        self.assertEqual(geom.reorient_reasons(flat_geometry(), 1.0), [])
        self.assertEqual(geom.reorient_reasons(flat_geometry(height_above_table_mm=30.0), 2.0), [])   # flat on another block
        self.assertEqual(geom.reorient_reasons(flat_geometry(), 14.0), ["tilted:14.0deg"])
        self.assertEqual(geom.reorient_reasons(flat_geometry(flags=["possibly_on_edge"]), None), ["on_edge"])

    def test_lying_flat_after_drop(self):
        self.assertEqual(geom.lying_flat_reasons(flat_geometry(), 1.2, self.dims), [])
        on_edge = geom.lying_flat_reasons(flat_geometry(height_above_table_mm=25.0, width_mm=15.0, flags=["possibly_on_edge"]), 1.0, self.dims)
        self.assertTrue(any(r.startswith("height:25.0") for r in on_edge) and "on_edge" in on_edge, on_edge)
        on_end = geom.lying_flat_reasons(flat_geometry(height_above_table_mm=75.0, length_mm=25.0, width_mm=15.0), 0.5, self.dims)
        self.assertTrue(any(r.startswith("height:75.0") for r in on_end) and any(r.startswith("short:") for r in on_end), on_end)
        leaning = geom.lying_flat_reasons(flat_geometry(height_above_table_mm=19.0, length_mm=60.0), 15.0, self.dims)
        self.assertIn("tilt:15.0deg", leaning)
        self.assertTrue(any(r.startswith("short:60.0") for r in leaning))
        self.assertIn("tilt:unknown", geom.lying_flat_reasons(flat_geometry(), None, self.dims))
        # 30 Oct, kazy-palletize-30oct: a block that had landed flat read 10.4 mm high; low readings are not an orientation
        self.assertEqual(geom.lying_flat_reasons(flat_geometry(height_above_table_mm=10.4), 4.6, self.dims), [])

    def test_pallet_check_needs_positive_evidence(self):
        # A good pallet block whose mask is trimmed by its neighbours, or whose face cannot be fitted, is left alone.
        self.assertEqual(geom.misplaced_on_pallet_reasons(flat_geometry(length_mm=55.0, width_mm=16.0), None, self.dims), [])
        self.assertEqual(geom.misplaced_on_pallet_reasons(flat_geometry(height_above_table_mm=11.9), 2.0, self.dims), [])   # 30 Oct pallet
        on_neighbour = geom.misplaced_on_pallet_reasons(flat_geometry(height_above_table_mm=30.0), 1.0, self.dims)
        self.assertEqual(on_neighbour, ["height:30.0mm_not_one_block"])
        # 30 Oct 20:11/20:13: a good pallet block read 8.2 / 8.3 deg at 13.2 / 13.1 mm and was taken off twice; tilt alone
        # now needs more than 15 deg (a block resting on a neighbour shows in its height)
        self.assertEqual(geom.misplaced_on_pallet_reasons(flat_geometry(height_above_table_mm=13.2), 8.3, self.dims), [])
        self.assertEqual(geom.misplaced_on_pallet_reasons(flat_geometry(height_above_table_mm=18.0), 18.0, self.dims), ["tilt:18.0deg"])
        self.assertIn("on_edge", geom.misplaced_on_pallet_reasons(flat_geometry(height_above_table_mm=25.0, flags=["possibly_on_edge"]), 0.5, self.dims))
        # turned off the row (edge-case list: rotated 90, diagonal): the stacking stage needs the three parallel within 4 deg
        self.assertEqual(geom.misplaced_on_pallet_reasons(flat_geometry(base_long_side_deg=2.0), 1.0, self.dims, row_long_deg=0.0), [])
        self.assertEqual(geom.misplaced_on_pallet_reasons(flat_geometry(base_long_side_deg=-88.0), 1.0, self.dims, row_long_deg=0.0), ["rotated:-88deg"])
        self.assertEqual(geom.misplaced_on_pallet_reasons(flat_geometry(base_long_side_deg=35.0), 1.0, self.dims, row_long_deg=0.0), ["rotated:+35deg"])
        self.assertEqual(geom.misplaced_on_pallet_reasons(flat_geometry(base_long_side_deg=89.0), 1.0, self.dims, row_long_deg=-89.0), [])   # same direction



class StandingOnEnd(unittest.TestCase):
    dims = geom.BlockDimensions()

    def test_recognises_only_an_upright_block(self):
        self.assertTrue(geom.standing_on_end({"height_above_table_mm": 74.0, "length_mm": 24.0, "width_mm": 14.0}, self.dims))
        self.assertTrue(geom.standing_on_end({"height_above_table_mm": 60.3, "length_mm": 27.0, "width_mm": 16.0}, self.dims))
        # a tower top (layers of long blocks) and flat or on-edge blocks are not standing blocks
        self.assertFalse(geom.standing_on_end({"height_above_table_mm": 59.0, "length_mm": 73.0, "width_mm": 24.0}, self.dims))
        self.assertFalse(geom.standing_on_end({"height_above_table_mm": 15.0, "length_mm": 74.0, "width_mm": 24.0}, self.dims))
        self.assertFalse(geom.standing_on_end({"height_above_table_mm": 25.0, "length_mm": 74.0, "width_mm": 14.0}, self.dims))
        self.assertFalse(geom.standing_on_end({"height_above_table_mm": None, "length_mm": 24.0, "width_mm": 14.0}, self.dims))


class OnEdge(unittest.TestCase):
    dims = geom.BlockDimensions()

    def test_on_its_edge_on_the_table(self):
        # kazy-palletize-30oct 19:59: after both drops the block read 23.3 / 22.1 mm high, 11-12 mm wide, flagged on edge
        for h, w in ((23.3, 11.1), (22.1, 12.1)):
            g = {"height_above_table_mm": h, "length_mm": 72.0, "width_mm": w, "flags": ["possibly_on_edge"]}
            self.assertTrue(geom.on_edge_on_table(g, self.dims))
            self.assertEqual(geom.upright_kind(g, self.dims), "on_edge")
        # the first pick: on its edge on top of a two-block stack (top 38.6 mm) is not on the table; it is picked and dropped
        self.assertFalse(geom.on_edge_on_table({"height_above_table_mm": 38.6, "width_mm": 12.0, "flags": ["possibly_on_edge"]}, self.dims))
        self.assertIsNone(geom.upright_kind({"height_above_table_mm": 15.0, "length_mm": 74.0, "width_mm": 24.0, "flags": []}, self.dims))
        self.assertEqual(geom.upright_kind({"height_above_table_mm": 74.0, "length_mm": 24.0, "width_mm": 14.0, "flags": []}, self.dims), "standing")


class OnEdgeGripHeight(unittest.TestCase):
    def test_a_block_on_its_edge_is_gripped_from_the_table(self):
        # 30 Oct 20:21:21: on its edge (Jenga logo up) at (270.8, 92.4), top 23.7 mm; read as "resting on a block at 15 mm"
        # the claws stopped at 15.0 mm and missed. Tops 22-29 mm (six picks that run) now rest on the table.
        dims = geom.BlockDimensions()
        params = geom.GraspParams(grasp_depth_mm=8.0 + 15.0, tips_min_mm=0.0)          # --grip-lower-mm 15 --tips-floor-mm 0
        self.assertEqual(geom.resting_level_mm(23.7, dims.height_mm), 15.0)               # without the width: the old reading
        for top in (22.1, 23.7, 27.6, 28.9):
            self.assertEqual(geom.resting_level_mm(top, dims.height_mm, dims.width_mm), 0.0, top)
        self.assertEqual(geom.grip_tips_mm(23.7, geom.ALONG, params, dims.height_mm, dims.width_mm), 0.0)
        self.assertEqual(geom.grip_tips_mm(23.7, geom.ALONG, params, dims.height_mm), 15.0)
        # a block clearly on top of others keeps its level
        self.assertEqual(geom.resting_level_mm(38.6, dims.height_mm, dims.width_mm), 30.0)
        self.assertEqual(geom.resting_level_mm(44.0, dims.height_mm, dims.width_mm), 30.0)

    def test_only_a_narrow_top_counts_as_on_its_edge(self):
        # A full-width top at ~30 mm is a flat block on another: it keeps "on a block at 15 mm", so an aligned pair is not
        # gripped together. On 30 Oct on-edge tops read 11.1-19.6 mm wide, flat tops 22.9-23.8.
        dims = geom.BlockDimensions()
        self.assertTrue(geom.narrow_top(["possibly_on_edge"], 19.6, dims))
        self.assertTrue(geom.narrow_top([], 12.1, dims))
        self.assertFalse(geom.narrow_top([], 23.8, dims))
        self.assertFalse(geom.narrow_top([], None, dims))


class KnockPlan(unittest.TestCase):
    dims = geom.BlockDimensions()

    @staticmethod
    def flat(x, y, deg=0.0, top=15.0):
        return {"xy": (x, y), "long_deg": deg, "size": (75.0, 25.0), "top": top}

    def plan(self, obstacles=(), avoid=(), forbidden=None, upright="standing", xy=(350.0, 0.0)):
        from robot_hackathon.live_executor import knock_plan
        # end face's long side (25 mm) along x: the thin 15 mm side faces +y/-y
        return knock_plan(xy, 0.0, list(obstacles), list(avoid), self.dims, forbidden=forbidden, upright=upright)

    def test_prefers_tipping_onto_the_wide_face_with_room(self):
        from robot_hackathon.live_executor import KNOCK_START_GAP_MM, KNOCK_PAST_MM, CLAW_THICKNESS_MM
        plan = self.plan(obstacles=[self.flat(350.0, 75.0)])            # a block lies where a +y fall would land
        self.assertEqual(plan["onto"], "wide_face")
        self.assertAlmostEqual(plan["u_deg"], 270.0)                     # falls toward -y, away from it
        self.assertAlmostEqual(plan["landing_xy"][1], -(7.5 + 37.5))
        # operator, 30 Oct: come down further from the block (20 mm from its face, was 5)
        self.assertAlmostEqual(plan["start_xy"][1], 7.5 + KNOCK_START_GAP_MM + CLAW_THICKNESS_MM)
        self.assertAlmostEqual(plan["distance_mm"], KNOCK_START_GAP_MM + 7.5 + KNOCK_PAST_MM)

    def test_never_comes_down_on_another_block(self):
        # A flat block whose centre is 36 mm from where the claws would come down on the +y side, but whose body lies
        # under them: the old centre-distance check (30 mm) let the claws come down on it (operator: "don't hit other
        # blocks"). Now that side is out and the claws come down on the -y side.
        import math as m
        from robot_hackathon.live_executor import KNOCK_START_GAP_MM, CLAW_THICKNESS_MM
        start_plus_y = (350.0, 7.5 + KNOCK_START_GAP_MM + CLAW_THICKNESS_MM)
        self.assertGreater(m.dist(start_plus_y, (380.0, 55.0)), 30.0)
        plan = self.plan(obstacles=[self.flat(380.0, 55.0)], upright="on_edge")
        self.assertAlmostEqual(plan["u_deg"], 90.0)                        # claws on the -y side, sweeping toward +y
        self.assertLess(plan["start_xy"][1], 0.0)
        self.assertIsNone(self.plan(obstacles=[self.flat(380.0, 55.0), self.flat(380.0, -55.0)], upright="on_edge"))

    def test_falls_back_to_the_edge_then_gives_up(self):
        blocked_y = [self.flat(350.0, 60.0), self.flat(350.0, -60.0)]
        plan = self.plan(obstacles=blocked_y)
        self.assertEqual(plan["onto"], "edge")                            # along x instead
        self.assertIn(round(plan["u_deg"]), (0, 180))
        self.assertIsNone(self.plan(obstacles=blocked_y + [self.flat(440.0, 0.0, deg=90.0), self.flat(260.0, 0.0, deg=90.0)]))
        # the tower's keep-out and the pallet lane count as occupied too
        self.assertIsNone(self.plan(avoid=[((350.0, 0.0), 120.0)]))
        self.assertIsNone(self.plan(forbidden=lambda xy: True))

    def test_on_edge_tips_across_its_thin_side_only(self):
        from robot_hackathon.live_executor import KNOCK_START_GAP_MM, KNOCK_PAST_MM
        plan = self.plan(upright="on_edge", xy=(300.0, 330.0))
        self.assertEqual(plan["onto"], "wide_face")
        self.assertIn(round(plan["u_deg"]), (90, 270))
        self.assertAlmostEqual(math.dist(plan["landing_xy"], (300.0, 330.0)), 7.5 + 12.5)
        self.assertAlmostEqual(plan["tips_mm"], 16.0)                      # above flat 15 mm neighbours
        self.assertAlmostEqual(plan["distance_mm"], KNOCK_START_GAP_MM + 7.5 + KNOCK_PAST_MM)
        self.assertAlmostEqual(self.plan()["tips_mm"], 45.0)
        both = [self.flat(300.0, 375.0), self.flat(300.0, 285.0)]
        self.assertIsNone(self.plan(obstacles=both, upright="on_edge", xy=(300.0, 330.0)))


class FlattenSweepPlan(unittest.TestCase):
    """Random heap-flattening sweeps (operator, 30 Oct: after going home, sweep a few times at random)."""
    dims = geom.BlockDimensions()
    # a small heap: one block on its edge among flat ones, the pallet 160 mm away
    EDGE = {"xy": (330.0, 60.0), "long_deg": 20.0, "size": (75.0, 15.0), "top": 25.0}
    FLAT = [{"xy": (300.0, 0.0), "long_deg": 60.0, "size": (75.0, 25.0), "top": 14.0},
            {"xy": (390.0, 80.0), "long_deg": -30.0, "size": (75.0, 25.0), "top": 13.0},
            {"xy": (320.0, 110.0), "long_deg": 0.0, "size": (75.0, 25.0), "top": 15.0}]
    PALLET = ((440.0, 195.0), 112.5)

    def plan(self, seed, avoid=(PALLET,), forbidden=None):
        import random
        from robot_hackathon.live_executor import flatten_sweep_plan
        return flatten_sweep_plan(random.Random(seed), [self.EDGE["xy"]], [self.EDGE] + self.FLAT, list(avoid), self.dims, forbidden=forbidden)

    def test_claws_come_down_on_bare_table_and_sweep_through_the_block(self):
        from robot_hackathon.live_executor import (CLAW_FACE_WIDTH_MM, CLAW_THICKNESS_MM, FLATTEN_PAST_MM, KNOCK_LOWER_MARGIN_MM,
                                                   _rects_overlap, block_footprint)
        for seed in range(30):
            p = self.plan(seed)
            self.assertIsNotNone(p)
            self.assertEqual(p["tips_mm"], 18.0)                                     # over flat tops (15), into a 25 mm edge
            lower = (*p["start_xy"], p["u_deg"], CLAW_THICKNESS_MM + KNOCK_LOWER_MARGIN_MM, CLAW_FACE_WIDTH_MM / 2 + KNOCK_LOWER_MARGIN_MM)
            for o in [self.EDGE] + self.FLAT:
                self.assertFalse(_rects_overlap(lower, block_footprint(o["xy"], o["long_deg"], o["size"])))
            u = (math.cos(math.radians(p["u_deg"])), math.sin(math.radians(p["u_deg"])))
            back = (self.EDGE["xy"][0] - p["start_xy"][0]) * u[0] + (self.EDGE["xy"][1] - p["start_xy"][1]) * u[1]
            self.assertAlmostEqual(p["distance_mm"], back + FLATTEN_PAST_MM)          # through the block, 60 mm past it

    def test_stroke_and_carried_blocks_stay_off_the_pallet(self):
        from robot_hackathon.live_executor import CLAW_FACE_WIDTH_MM, FLATTEN_CARRY_MM, point_segment_distance
        directions = set()
        for seed in range(60):
            p = self.plan(seed)
            u = (math.cos(math.radians(p["u_deg"])), math.sin(math.radians(p["u_deg"])))
            reach = (p["end_xy"][0] + u[0] * FLATTEN_CARRY_MM, p["end_xy"][1] + u[1] * FLATTEN_CARRY_MM)
            self.assertGreaterEqual(point_segment_distance(self.PALLET[0], p["start_xy"], reach), self.PALLET[1] + CLAW_FACE_WIDTH_MM / 2)
            directions.add(int(p["u_deg"] // 90))
        self.assertGreater(len(directions), 1)                                       # random, not one fixed direction
        self.assertEqual(self.plan(7), self.plan(7))                                 # a seed repeats its draws

    def test_x_sweep_crosses_a_packed_heap_from_bare_table(self):
        # run 10 start (20:52): flat blocks lying parallel at ~35 deg, 33-54 mm apart, every claw strip blocked
        from robot_hackathon.live_executor import (CLAW_FACE_WIDTH_MM, CLAW_THICKNESS_MM, FLATTEN_CARRY_MM, KNOCK_LOWER_MARGIN_MM,
                                                   PUSH_TIPS_MM, XSWEEP_BOUNDS_MARGIN_MM, _rects_overlap, block_footprint,
                                                   cross_sweep_plan, point_segment_distance, x_sweep_line)
        heap = [{"xy": xy, "long_deg": a, "size": (75.0, 25.0), "top": 14.0} for xy, a in
                (((369, -48), 25.9), ((370, -84), 32.2), ((352, 11), 39.0), ((344, -25), 35.8), ((315, 25), 35.6),
                 ((307, 78), 56.8), ((296, -90), 27.9), ((274, -41), -37.9))]
        pallet, tower, drop = ((440.0, 195.0), 112.5), ((550.0, 0.0), 112.5), ((340.0, 340.0), 60.0)
        bounds = ((209.4, 638.6), (-201.3, 563.7))                                       # what the home photo saw
        # strokes through the whole heap had no clear way here (wedged by the pallet, the tower and the photo's edge);
        # 70 mm arms and turning the X find two
        line, clear = x_sweep_line(heap, heap, 35.0 + 45.0, [pallet, tower, drop], bounds=bounds)
        self.assertEqual(clear, 2)
        for k in (0, 1):
            p = cross_sweep_plan(heap, heap, line + 90.0 * k, [pallet, tower, drop], bounds=bounds)
            self.assertIsNotNone(p)
            self.assertEqual(p["tips_mm"], PUSH_TIPS_MM)                                 # pushes the blocks on the table
            u = (math.cos(math.radians(p["u_deg"])), math.sin(math.radians(p["u_deg"])))
            self.assertLess(point_segment_distance(p["centre_xy"], p["start_xy"], p["end_xy"]), 1e-6)   # through the centre
            lower = (*p["start_xy"], p["u_deg"], CLAW_THICKNESS_MM + KNOCK_LOWER_MARGIN_MM, CLAW_FACE_WIDTH_MM / 2 + KNOCK_LOWER_MARGIN_MM)
            self.assertFalse(any(_rects_overlap(lower, block_footprint(o["xy"], o["long_deg"], o["size"])) for o in heap))
            reach = (p["end_xy"][0] + u[0] * FLATTEN_CARRY_MM, p["end_xy"][1] + u[1] * FLATTEN_CARRY_MM)
            for c, r in (pallet, tower, drop):
                self.assertGreaterEqual(point_segment_distance(c, p["start_xy"], reach), r + CLAW_FACE_WIDTH_MM / 2)
            m = XSWEEP_BOUNDS_MARGIN_MM
            for q in (p["start_xy"], reach):
                self.assertTrue(209.4 + m <= q[0] <= 638.6 - m and -201.3 + m <= q[1] <= 563.7 - m)
        a = cross_sweep_plan(heap, heap, line, [pallet, tower, drop], bounds=bounds)
        b = cross_sweep_plan(heap, heap, line + 90.0, [pallet, tower, drop], bounds=bounds)
        self.assertAlmostEqual((a["u_deg"] - b["u_deg"]) % 180.0, 90.0)                  # the two strokes cross: an X
        self.assertIsNone(cross_sweep_plan(heap, heap, 80.0, [pallet, tower, drop], bounds=bounds,
                                           forbidden=lambda xy: True))

    @staticmethod
    def row(n, centre=(330.0, 0.0), long_deg=30.0, pitch=25.0, along=None, top=14.0):
        a = math.radians(long_deg)
        ax, ay, bx, by = math.cos(a), math.sin(a), -math.sin(a), math.cos(a)
        along = along or [0.0] * n
        return [{"xy": (centre[0] + k * pitch * bx + along[k] * ax, centre[1] + k * pitch * by + along[k] * ay), "long_deg": long_deg,
                 "size": (75.0, 25.0), "top": top} for k in range(n)]

    def test_rows_of_three_or_more_flat_blocks_side_by_side(self):
        # operator, 30 Oct: "3 or more blocks perfectly adjacent to each other on the bottom layer"
        from robot_hackathon.live_executor import side_by_side_rows
        three = self.row(3)
        pair = self.row(2, centre=(330.0, 200.0))
        staggered = self.row(3, centre=(500.0, -150.0), along=[0.0, 30.0, 0.0])        # middle end 30 mm out: not a row
        on_edge = self.row(3, centre=(250.0, -150.0), top=25.0)                       # on their edges: not the bottom layer
        blocks = three + pair + staggered + on_edge
        rows = side_by_side_rows(blocks)
        self.assertEqual(rows, [[2, 1, 0]] if blocks[2]["xy"][1] < blocks[0]["xy"][1] else [[0, 1, 2]])
        four = self.row(4, pitch=26.0, along=[0.0, 8.0, -6.0, 3.0])
        self.assertEqual(len(side_by_side_rows(four)[0]), 4)

    def test_middle_blocks_are_pushed_out_along_their_length_edge_on(self):
        from robot_hackathon.live_executor import (CLAWS_ACROSS_MM, PUSH_OUT_GAP_MM, PUSH_OUT_MM, PUSH_TIPS_MM, push_out_plan)
        three = self.row(3)
        (p,) = push_out_plan(three, three, [], self.dims)
        self.assertEqual(p["index"], 1)
        self.assertAlmostEqual(p["u_deg"], 30.0)                                       # along the long side
        self.assertAlmostEqual(abs(geom.wrap_90(p["claws_close_deg"] - p["u_deg"])), 90.0)   # turned edge-on
        self.assertLess(CLAWS_ACROSS_MM, 25.0)                                        # narrower than the block's end
        back = math.dist(p["start_xy"], three[1]["xy"])
        self.assertAlmostEqual(back, 37.5 + PUSH_OUT_GAP_MM + 22.5)                    # beyond its end, claws 45 deep
        self.assertEqual((p["distance_mm"], p["tips_mm"]), (PUSH_OUT_GAP_MM + PUSH_OUT_MM, PUSH_TIPS_MM))
        # a keep-out (pallet, tower, drop spot) where it would slide to on that side: pushed out the other way
        a = math.radians(30.0)
        ahead = ((three[1]["xy"][0] + 120 * math.cos(a), three[1]["xy"][1] + 120 * math.sin(a)), 40.0)
        behind = ((three[1]["xy"][0] - 120 * math.cos(a), three[1]["xy"][1] - 120 * math.sin(a)), 40.0)
        (p,) = push_out_plan(three, three, [ahead], self.dims)
        self.assertAlmostEqual(p["u_deg"], 210.0)
        self.assertEqual(push_out_plan(three, three, [ahead, behind], self.dims), [])          # both sides: no push
        # the claws come down on bare table: nothing under them in the plan; a block lying there (which is also in the
        # way of the push the other way) leaves no push
        from robot_hackathon.live_executor import CLAW_FACE_WIDTH_MM, PUSH_OUT_LOWER_MARGIN_MM, _rects_overlap, block_footprint
        (p,) = push_out_plan(three, three, [], self.dims)
        lower = (*p["start_xy"], p["u_deg"], CLAW_FACE_WIDTH_MM / 2 + PUSH_OUT_LOWER_MARGIN_MM, CLAWS_ACROSS_MM / 2 + PUSH_OUT_LOWER_MARGIN_MM)
        self.assertFalse(any(_rects_overlap(lower, block_footprint(o["xy"], o["long_deg"], o["size"])) for o in three))
        blocker = {"xy": (three[1]["xy"][0] - 75 * math.cos(a), three[1]["xy"][1] - 75 * math.sin(a)), "long_deg": -60.0,
                   "size": (75.0, 25.0), "top": 14.0}
        self.assertEqual(push_out_plan(three, three + [blocker], [], self.dims), [])
        # four side by side: both middle ones, out on opposite sides
        four = self.row(4)
        pushes = push_out_plan(four, four, [], self.dims)
        self.assertEqual([q["index"] for q in pushes], [1, 2])
        self.assertAlmostEqual(abs((pushes[0]["u_deg"] - pushes[1]["u_deg"]) % 360.0), 180.0)
        # never into the pallet's clearance
        self.assertEqual(push_out_plan(three, three, [(three[1]["xy"], 150.0)], self.dims), [])

    def test_x_sweep_refuses_when_no_way_is_clear(self):
        from robot_hackathon.live_executor import cross_sweep_plan
        heap = [{"xy": (330.0, 0.0), "long_deg": 0.0, "size": (75.0, 25.0), "top": 14.0}]
        self.assertIsNone(cross_sweep_plan(heap, heap, 0.0, [((330.0, 0.0), 200.0)]))      # the heap is inside a keep-out
        self.assertIsNone(cross_sweep_plan(heap, heap, 0.0, [], bounds=((300.0, 360.0), (-30.0, 30.0))))   # off the seen table
        self.assertIsNone(cross_sweep_plan([], [], 0.0, []))

    def test_no_sweep_when_every_draw_fails(self):
        self.assertIsNone(self.plan(1, avoid=[((330.0, 60.0), 50.0)]))               # the block itself is in a keep-out
        self.assertIsNone(self.plan(1, forbidden=lambda xy: True))

if __name__ == "__main__":
    unittest.main()
