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
        self.assertEqual(geom.misplaced_on_pallet_reasons(flat_geometry(height_above_table_mm=18.0), 12.0, self.dims), ["tilt:12.0deg"])
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

if __name__ == "__main__":
    unittest.main()
