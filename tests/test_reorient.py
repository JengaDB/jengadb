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

    def test_pallet_check_needs_positive_evidence(self):
        # A good pallet block whose mask is trimmed by its neighbours, or whose face cannot be fitted, is left alone.
        self.assertEqual(geom.misplaced_on_pallet_reasons(flat_geometry(length_mm=55.0, width_mm=16.0), None, self.dims), [])
        self.assertEqual(geom.misplaced_on_pallet_reasons(flat_geometry(height_above_table_mm=11.9), 2.0, self.dims), [])   # 30 Oct pallet
        on_neighbour = geom.misplaced_on_pallet_reasons(flat_geometry(height_above_table_mm=30.0), 1.0, self.dims)
        self.assertEqual(on_neighbour, ["height:30.0mm_not_one_block"])
        self.assertEqual(geom.misplaced_on_pallet_reasons(flat_geometry(height_above_table_mm=18.0), 12.0, self.dims), ["tilt:12.0deg"])
        self.assertIn("on_edge", geom.misplaced_on_pallet_reasons(flat_geometry(height_above_table_mm=25.0, flags=["possibly_on_edge"]), 0.5, self.dims))


if __name__ == "__main__":
    unittest.main()
