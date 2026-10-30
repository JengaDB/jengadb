"""Offline tests for the retrieval-and-stacking proposal slice (docs/AGENT_HEURISTICS_PLAN.md, task AGENT-01).

Covers: threshold boundaries and Atlas score conversion, curated-only references, self/same-observation exclusion,
no-match behaviour, frame conversions and table-relative heights on synthetic fixtures, missing/inconsistent
calibration blocking proposals, a similar-looking block rejected for blocked fingers, six-slot placement, the
selection validator, deterministic abstention and the OpenAI tool loop with a fake transport.
Geometry tests need numpy/Pillow (RGBD_TESTS_REQUIRED=1 fails instead of skipping).
"""
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
    from PIL import Image  # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False
    if os.environ.get("RGBD_TESTS_REQUIRED") == "1":
        raise

from robot_hackathon import retrieval, placement, proposal, coordinator, cli  # noqa: E402
from robot_hackathon import grasp_geometry as geom  # noqa: E402
from robot_hackathon.config import Config, ConfigError  # noqa: E402


def unit(seed, dim=8):
    vals = [math.sin(seed * 7.1 + i * 1.3) for i in range(dim)]
    n = math.sqrt(sum(v * v for v in vals))
    return [v / n for v in vals]


def ref(cid, obs="obs_ref", cosine=0.9, label_source="human", status="pickable", **extra):
    return {"candidate_id": cid, "observation_id": obs, "mode": "mock", "phase": "pick", "setup_id": "s", "encoder_version": "enc-v1",
            "embedding_dimension": 8, "processing_version": "pv", "pickability_label": {"status": status, "label_source": label_source},
            "pickable": status == "pickable", "score": retrieval.atlas_score_from_cosine(cosine), **extra}


QUERY = {"candidate_id": "cand_query", "observation_id": "obs_q", "encoder_version": "enc-v1", "embedding_dimension": 8, "mode": "mock"}
INDEX = {"source": "test", "name": "candidate_context_v1", "similarity": "cosine"}


class RetrievalTests(unittest.TestCase):
    def test_score_conversion_and_threshold_boundaries(self):
        self.assertAlmostEqual(retrieval.cosine_from_atlas_score(0.75), 0.5)
        self.assertAlmostEqual(retrieval.atlas_score_from_cosine(0.5), 0.75)
        self.assertAlmostEqual(retrieval.cosine_from_atlas_score(1.0), 1.0)
        policy = retrieval.ReferencePolicy()
        self.assertEqual(policy.effective_settings()["atlas_score_threshold"], 0.75)
        hits = [ref("at", cosine=0.5), ref("below", cosine=0.4999), ref("above", cosine=0.5001)]
        ev = retrieval.shortlist(QUERY, hits, policy, INDEX)
        self.assertEqual([m["reference_candidate_id"] for m in ev["matches"]], ["above", "at"])
        excluded = {e["reference_candidate_id"]: e["reasons"] for e in ev["excluded"]}
        self.assertTrue(excluded["below"][0].startswith("below_threshold:0.4999"))
        self.assertEqual(ev["status"], "matched")
        self.assertEqual(ev["score_conversion"], retrieval.SCORE_CONVERSION)
        for m in ev["matches"]:
            self.assertAlmostEqual(m["cosine_similarity"], retrieval.cosine_from_atlas_score(m["atlas_vector_search_score"]))

    def test_curation_requires_explicit_human_label_and_reference_set(self):
        policy = retrieval.ReferencePolicy()
        hits = [ref("human_ok"), ref("default_label", label_source="human_default"), ref("not_pickable", status="not_pickable"),
                ref("no_label", label_source=None, status=None), ref("excluded_review", review={"status": "excluded"})]
        ev = retrieval.shortlist(QUERY, hits, policy, INDEX)
        self.assertEqual([m["reference_candidate_id"] for m in ev["matches"]], ["human_ok"])
        reasons = {e["reference_candidate_id"]: e["reasons"] for e in ev["excluded"]}
        self.assertEqual(reasons["default_label"], ["label_source:human_default"])
        self.assertIn("label_status:not_pickable", reasons["not_pickable"])
        self.assertIn("label_status:missing", reasons["no_label"])
        self.assertIn("review_status:excluded", reasons["excluded_review"])
        scoped = retrieval.ReferencePolicy(reference_set_id="golden-v1")
        ev = retrieval.shortlist(QUERY, [ref("in", reference_set="golden-v1"), ref("out")], scoped, INDEX)
        self.assertEqual([m["reference_candidate_id"] for m in ev["matches"]], ["in"])
        self.assertIn("not_in_reference_set", ev["excluded"][0]["reasons"])

    def test_self_same_observation_held_out_and_compatibility_exclusions(self):
        policy = retrieval.ReferencePolicy(exclude_observation_ids=("obs_heldout",))
        hits = [ref("cand_query", obs="obs_q"), ref("sibling", obs="obs_q"), ref("held", obs="obs_heldout"),
                ref("other_encoder", encoder_version="enc-v2"), ref("other_dim", embedding_dimension=512),
                ref("live_record", mode="live"), ref("place_phase", phase="place"), ref("good")]
        ev = retrieval.shortlist(QUERY, hits, policy, INDEX)
        self.assertEqual([m["reference_candidate_id"] for m in ev["matches"]], ["good"])
        reasons = {e["reference_candidate_id"]: e["reasons"] for e in ev["excluded"]}
        self.assertIn("self_match", reasons["cand_query"])
        self.assertIn("same_observation", reasons["sibling"])
        self.assertIn("held_out_observation", reasons["held"])
        self.assertIn("encoder_version_mismatch", reasons["other_encoder"])
        self.assertIn("embedding_dimension_mismatch", reasons["other_dim"])
        self.assertIn("mode_mismatch", reasons["live_record"])
        self.assertIn("phase_not_pick", reasons["place_phase"])
        # Same-observation copies are allowed only when the policy says so, and still never the query itself.
        ev = retrieval.shortlist(QUERY, hits[:2], retrieval.ReferencePolicy(exclude_same_observation=False), INDEX)
        self.assertEqual([m["reference_candidate_id"] for m in ev["matches"]], ["sibling"])

    def test_no_match_is_empty_and_threshold_is_not_lowered(self):
        ev = retrieval.shortlist(QUERY, [ref("weak", cosine=0.3), ref("weaker", cosine=0.1)], retrieval.ReferencePolicy(), INDEX)
        self.assertEqual((ev["status"], ev["matches"], ev["best_cosine_similarity"]), ("no_match", [], None))
        self.assertEqual(ev["policy"]["cosine_threshold"], 0.5)
        ev = retrieval.unavailable_evidence(QUERY, "no_query_embedding", retrieval.ReferencePolicy())
        self.assertEqual((ev["status"], ev["unavailable_reason"], ev["matches"]), ("unavailable", "no_query_embedding", []))

    def test_local_bruteforce_matches_atlas_shape_and_top_k(self):
        docs = [dict(ref(f"r{i}", obs=f"o{i}"), embedding=unit(i)) for i in range(6)]
        docs.append(dict(ref("wrong_dim", obs="o9"), embedding=unit(1, dim=4)))
        query_vec = unit(2)
        search = retrieval.LocalReferenceSearch(docs)
        self.assertEqual(search.identity()["source"], "local_bruteforce")
        hits = search.search(query_vec, retrieval.index_filters(QUERY, retrieval.ReferencePolicy()), 10)
        self.assertEqual(hits[0]["candidate_id"], "r2")
        self.assertAlmostEqual(hits[0]["score"], 1.0)
        self.assertTrue(all("embedding" not in h for h in hits))
        self.assertNotIn("wrong_dim", [h["candidate_id"] for h in hits])
        ev = retrieval.retrieve(QUERY, query_vec, search, retrieval.ReferencePolicy(top_k=2))
        self.assertEqual(len(ev["matches"]), 2)
        self.assertEqual(ev["matches"][0]["reference_candidate_id"], "r2")
        self.assertEqual(ev["index"]["documents"], 7)
        with self.assertRaises(ValueError):
            retrieval.LocalReferenceSearch([ref("none")])
        self.assertEqual(retrieval.retrieve(QUERY, None, search, retrieval.ReferencePolicy())["status"], "unavailable")
        self.assertEqual(retrieval.retrieve(QUERY, query_vec, None, retrieval.ReferencePolicy())["unavailable_reason"], "no_reference_index")

    def test_index_filters_use_only_exposed_paths(self):
        from robot_hackathon.rgbd_mongo import VECTOR_FILTERS
        filters = retrieval.index_filters(QUERY, retrieval.ReferencePolicy())
        self.assertTrue(set(filters) <= set(VECTOR_FILTERS))
        self.assertEqual(filters["pickability_label.status"], "pickable")


class PlacementTests(unittest.TestCase):
    def target(self, gap=0.0):
        return placement.TowerTarget((100.0, 200.0), geom.SupportSurface(10.0, True, "test"), geom.BlockDimensions(75, 25, 15, True),
                                     geom.FingerGeometry(verified=True), layer0_long_side_deg=0.0, gap_mm=gap)

    def test_six_slots_alternate_and_stack(self):
        slots = self.target().slots()
        self.assertEqual([s["slot_id"] for s in slots], ["L0S0", "L0S1", "L0S2", "L1S0", "L1S1", "L1S2"])
        self.assertEqual({s["long_side_deg"] for s in slots if s["layer"] == 0}, {0.0})
        self.assertEqual({s["long_side_deg"] for s in slots if s["layer"] == 1}, {-90.0})
        self.assertEqual([round(s["block_center_mm"][2], 1) for s in slots], [17.5] * 3 + [32.5] * 3)
        self.assertEqual([s["support_z_mm"] for s in slots], [10.0] * 3 + [25.0] * 3)
        # layer 0 blocks are offset across the long side (y) by one block width; layer 1 across x
        self.assertEqual([round(s["block_center_mm"][1], 1) for s in slots[:3]], [175.0, 200.0, 225.0])
        self.assertEqual([round(s["block_center_mm"][0], 1) for s in slots[3:]], [75.0, 100.0, 125.0])
        self.assertEqual([round(s["block_center_mm"][1], 1) for s in slots[3:]], [200.0] * 3)
        self.assertEqual(slots[3]["requires_filled"], ["L0S0", "L0S1", "L0S2"])
        self.assertEqual(slots[3]["validation_status"], "unvalidated_second_layer")
        self.assertTrue(slots[0]["neighbour_release_ok"])       # nothing placed yet: no claw-room problem
        self.assertEqual(slots[1]["neighbours"], ["L0S0", "L0S2"])
        beside = self.target().slots(filled=["L0S0"])
        self.assertFalse(beside[1]["neighbour_release_ok"])     # touching a placed block leaves no room to open a claw
        self.assertEqual(beside[1]["filled_neighbours"], ["L0S0"])
        self.assertTrue(beside[2]["neighbour_release_ok"])      # far end is still free
        self.assertTrue(self.target(gap=11.0).slots(filled=["L0S0"])[1]["neighbour_release_ok"])

    def test_next_open_slot_respects_layer_dependencies(self):
        t = self.target()
        self.assertEqual(placement.next_open_slot(t, [])["slot_id"], "L0S0")
        self.assertEqual(placement.next_open_slot(t, ["L0S0", "L0S1"])["slot_id"], "L0S2")
        self.assertEqual(placement.next_open_slot(t, ["L0S0", "L0S1", "L0S2"])["slot_id"], "L1S0")
        self.assertIsNone(placement.next_open_slot(t, [s["slot_id"] for s in t.slots()]))
        with self.assertRaises(ValueError):
            placement.TowerTarget((0, 0), geom.SupportSurface(None), geom.BlockDimensions(), geom.FingerGeometry()).slots()


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow required for geometry tests")
class GeometryTests(unittest.TestCase):
    def setUp(self):
        from robot_hackathon import synthetic_scene as synth
        self.synth = synth

    def scene(self, preset):
        rgb, depth, masks, truth = self.synth.render(preset)
        obs = self.synth.observation_record(preset, "obs_x")
        calib = geom.bundle_from_records(obs, True, self.synth.geometry_document())
        plane = geom.fit_table_plane(depth)
        return depth, masks, truth, calib, plane, plane.height_map(depth)

    def test_pinhole_round_trip_and_extrinsics(self):
        intr = geom.Intrinsics(228.0, 228.0, 160.0, 120.0)
        point = geom.deproject(200, 90, 450.0, intr)
        self.assertAlmostEqual(point[2], 450.0)
        u, v = geom.project(point, intr)
        self.assertAlmostEqual(u, 200.0)
        self.assertAlmostEqual(v, 90.0)
        ext = geom.CameraExtrinsics(((1, 0, 0), (0, -1, 0), (0, 0, -1)), (300.0, 150.0, 450.0), verified=True)
        base = ext.apply((0.0, 0.0, 440.0))
        self.assertEqual([round(x, 6) for x in base], [300.0, 150.0, 10.0])  # optical axis hits the table under the camera
        self.assertFalse(geom.rotation_is_orthonormal([[1, 0, 0], [0, 1, 0], [0, 0, 2]]))
        with self.assertRaises(ValueError):
            geom.validate_geometry_doc(dict(self.synth.geometry_document(), support_surface={"z_in_robot_base_mm": 10}))

    def test_table_plane_and_heights_match_ground_truth(self):
        depth, masks, truth, calib, plane, hm = self.scene("separated")
        self.assertGreater(plane.inlier_fraction, 0.9)
        self.assertLess(plane.rms_mm, 1.5)
        in_base = geom.table_plane_in_base(plane, calib.intrinsics, calib.extrinsics, 320, 240)
        # the synthetic table is 10 mm at the base origin with a small tilt; the fitted surface under the camera agrees
        self.assertAlmostEqual(in_base["mean_z_mm"], self.synth.table_z(300, 150), delta=1.0)
        self.assertLess(in_base["normal_tilt_deg"], 0.5)
        for i, (mask, t) in enumerate(zip(masks, truth)):
            est = geom.estimate_block(mask, depth, hm, calib, i)
            self.assertEqual(est.rejection_reasons, [])
            self.assertAlmostEqual(est.height_above_table_mm, t["height_above_table_mm"], delta=1.0)
            self.assertAlmostEqual(est.base_point_mm[0], t["center_xy_mm"][0], delta=1.5)
            self.assertAlmostEqual(est.base_point_mm[1], t["center_xy_mm"][1], delta=1.5)
            self.assertAlmostEqual(est.base_point_mm[2], t["top_z_mm"], delta=1.0)
            self.assertAlmostEqual(abs(geom.wrap_90(est.base_long_side_deg - t["long_side_deg"])), 0.0, delta=1.0)
            self.assertAlmostEqual(est.length_mm, 75.0, delta=8.0)
            self.assertAlmostEqual(est.width_mm, 25.0, delta=6.0)   # eroded edges, as in Sam's notes
            self.assertAlmostEqual(est.top_z_in_robot_base_mm, 10.0 + 15.0, delta=1.0)

    def test_missing_or_unverified_calibration_blocks_proposals(self):
        depth, masks, truth, calib, plane, hm = self.scene("separated")
        obs = self.synth.observation_record("separated", "obs_x")
        no_geometry = geom.bundle_from_records(obs, False, None)
        self.assertEqual(no_geometry.missing(), ["registration_not_verified", "extrinsics_missing", "support_surface_offset_missing",
                                                 "finger_geometry_not_verified", "block_dimensions_not_verified"])
        est = geom.estimate_block(masks[0], depth, hm, no_geometry, 0)
        self.assertIsNone(est.base_point_mm)
        self.assertIsNone(est.top_z_in_robot_base_mm)
        self.assertIn("pose_in_robot_base_unavailable:extrinsics_missing", est.flags)
        self.assertTrue(all(r.startswith("calibration:") for r in est.rejection_reasons))
        self.assertTrue(all(not g["eligible"] and "block_rejected" in g["reasons"] for g in geom.evaluate_grasps(est, hm, no_geometry)))
        # unverified offset: height is still estimated relative to the table, but no proposal
        doc = json.loads(json.dumps(self.synth.geometry_document()))
        doc["support_surface"]["verified"] = False
        unverified = geom.bundle_from_records(obs, True, doc)
        self.assertEqual(unverified.missing(), ["support_surface_offset_not_verified"])
        est = geom.estimate_block(masks[0], depth, hm, unverified, 0)
        self.assertAlmostEqual(est.height_above_table_mm, 15.0, delta=1.0)
        self.assertEqual(est.rejection_reasons, ["calibration:support_surface_offset_not_verified"])
        # inconsistent offset versus verified extrinsics is a rejection, not a silent average
        doc = json.loads(json.dumps(self.synth.geometry_document()))
        doc["support_surface"]["z_in_robot_base_mm"] = 60.0
        est = geom.estimate_block(masks[0], depth, hm, geom.bundle_from_records(obs, True, doc), 0)
        self.assertIn("calibration_inconsistent:extrinsics_vs_support_surface", est.rejection_reasons)
        # no intrinsics -> no mm, no pose
        est = geom.estimate_block(masks[0], depth, hm, geom.bundle_from_records(dict(obs, intrinsics={"rgb": None, "depth": None}), True,
                                                                                 self.synth.geometry_document()), 0)
        self.assertIn("intrinsics_missing", est.rejection_reasons)
        self.assertIsNone(est.length_mm)

    def test_invalid_depth_and_implausible_masks_are_rejected(self):
        depth, masks, truth, calib, plane, hm = self.scene("separated")
        holed = depth.copy()
        holed[masks[0]] = 0
        est = geom.estimate_block(masks[0], holed, plane.height_map(holed), calib, 0)
        self.assertEqual(est.rejection_reasons, ["insufficient_valid_depth"])
        tiny = np.zeros_like(masks[0])
        tiny[100:103, 100:103] = True
        self.assertEqual(geom.estimate_block(tiny, depth, hm, calib, 0).rejection_reasons, ["mask_too_small"])
        table_patch = np.zeros_like(masks[0])
        table_patch[20:60, 20:100] = True  # a "block" that is really the table surface
        est = geom.estimate_block(table_patch, depth, hm, calib, 0)
        self.assertTrue(any(r.startswith("height_too_low") for r in est.rejection_reasons))
        merged = masks[0] | masks[1]  # two blocks in one mask
        self.assertTrue(any(r.startswith("implausible_") for r in geom.estimate_block(merged, depth, hm, calib, 0).rejection_reasons))
        far = geom.CalibrationBundle(calib.intrinsics, calib.extrinsics, calib.support_surface, calib.finger, calib.block, True,
                                     "v", "s", geom.Workspace((0, 50), (0, 50), (0, 50)))
        self.assertIn("out_of_workspace", geom.estimate_block(masks[0], depth, hm, far, 0).rejection_reasons)

    def test_similar_block_rejected_for_blocked_fingers(self):
        depth, masks, truth, calib, plane, hm = self.scene("blocked")
        by_name = {t["name"]: i for i, t in enumerate(truth)}
        free = geom.estimate_block(masks[by_name["A"]], depth, hm, calib, by_name["A"])
        boxed = geom.estimate_block(masks[by_name["B"]], depth, hm, calib, by_name["B"])
        self.assertEqual((free.rejection_reasons, boxed.rejection_reasons), ([], []))  # both look like blocks
        self.assertAlmostEqual(free.height_above_table_mm, boxed.height_above_table_mm, delta=1.5)
        free_grasps = geom.evaluate_grasps(free, hm, calib)
        boxed_grasps = geom.evaluate_grasps(boxed, hm, calib)
        self.assertTrue(all(g["eligible"] for g in free_grasps))
        self.assertTrue(all(not g["eligible"] and g["reasons"] == ["fingers_blocked"] for g in boxed_grasps))
        self.assertTrue(all(g["clearance"]["obstruction_height_mm"] > g["clearance"]["limit_mm_above_table"] for g in boxed_grasps))
        self.assertAlmostEqual(free_grasps[0]["tips_above_table_mm"], free.height_above_table_mm - 8.0)  # measured height minus grasp depth
        self.assertEqual(free_grasps[3]["tips_above_table_mm"], 1.0)   # ends grip goes 10 mm deeper, floored at 1 mm
        self.assertEqual(geom.grip_tips_mm(15.0, geom.ACROSS, geom.GraspParams()), 7.0)
        # the neighbour beside B is only blocked across; its ends are free
        n1 = geom.evaluate_grasps(geom.estimate_block(masks[by_name["N1"]], depth, hm, calib, by_name["N1"]), hm, calib)
        self.assertEqual([g["eligible"] for g in n1], [False, False, False, True])
        # unobservable strips (off-image) are not treated as clear
        shifted = geom.BlockEstimate(0, None, pixel_center=(2.0, 2.0), image_angle_deg=0.0, depth_mm=435.0, height_above_table_mm=15.0,
                                     valid_fraction=1.0, pixels=400, mm_per_px=435.0 / 228.0, length_mm=75.0, width_mm=25.0)
        self.assertEqual(geom.claw_obstruction(hm, shifted, 0.0, geom.ACROSS, calib)["status"], "unobservable")
        narrow = geom.CalibrationBundle(calib.intrinsics, calib.extrinsics, calib.support_surface, geom.FingerGeometry(open_gap_mm=80, verified=True),
                                        calib.block, True, "v", "s", calib.workspace)
        self.assertIn("gripper_too_narrow_for_ends", geom.evaluate_grasps(free, hm, narrow)[3]["reasons"])

    def test_floor_past_the_table_edge_does_not_skew_the_plane(self):
        # 30 Oct, overview from flange z 650 (obs_b174ec2236ec429a): the left 15% of the image was the floor at ~1400 mm
        # while the table sat at ~633 mm; the fit seeded from all samples converged on 5% inliers (fitted_unreliable).
        rgb, depth, masks, truth = self.synth.render("separated")
        clean = geom.fit_table_plane(depth)
        with_floor = depth.copy()
        with_floor[:, : depth.shape[1] * 15 // 100] = 1400
        plane = geom.fit_table_plane(with_floor)
        self.assertTrue(plane.reliable)
        self.assertGreater(plane.inlier_fraction, 0.75)
        self.assertLess(plane.inlier_fraction, 0.9)       # the floor itself is correctly left out
        h, w = depth.shape
        for u, v in ((w // 2, h // 2), (w - 10, 10), (w // 4, h - 10)):
            self.assertAlmostEqual(plane.depth_at(u, v), clean.depth_at(u, v), delta=1.0)   # same table, within 1 mm
        far = np.full_like(depth, 1400)
        self.assertIsNotNone(geom.fit_table_plane(far))   # a band that empties out falls back to every sample

    def test_noisy_table_makes_plane_unreliable_and_blocks_proposals(self):
        rgb, depth, masks, truth = self.synth.render("separated")
        rng = np.random.default_rng(1)
        noisy = np.where(depth > 0, np.clip(depth.astype(float) + rng.normal(0, 12.0, depth.shape), 1, 65535), 0).astype(np.uint16)
        plane = geom.fit_table_plane(noisy)
        self.assertFalse(plane.reliable)
        self.assertLess(plane.tolerances_mm[-1], 4.0)        # relative schedule: 0.8% of ~440 mm
        clean = geom.fit_table_plane(depth, exclude=geom.dilate(masks[0] | masks[1] | masks[2], 4))
        self.assertTrue(clean.reliable)
        self.assertGreater(clean.excluded_fraction, 0.02)
        with tempfile.TemporaryDirectory() as tmp:
            self.synth.save_scene(Path(tmp), "separated")
            scene = proposal.synthetic_scene(Path(tmp), "obs_synthetic_separated")
            scene["depth"] = noisy
            ctx = proposal.build_context(scene, tower_origin_xy_mm=(160.0, 300.0))
            self.assertEqual(ctx["table_plane"]["status"], "fitted_unreliable")
            self.assertEqual(ctx["eligible_candidate_ids"], [])
            self.assertTrue(all(any(r.startswith("table_plane_unreliable") for r in c["rejection_reasons"]) for c in ctx["candidates"]))
            self.assertEqual(proposal.DeterministicCoordinator().decide(ctx)["reason"], "no_eligible_candidate")

    def test_extrinsics_from_recorded_camera_pose(self):
        from robot_hackathon import capture
        obs = self.synth.observation_record("separated", "obs_x")
        doc = json.loads(json.dumps(self.synth.geometry_document()))
        doc["extrinsics"] = {"from_observation": True, "verified": False, "source": "viam frame system at capture"}
        geom.validate_geometry_doc(doc)
        with self.assertRaises(ValueError):
            geom.validate_geometry_doc(dict(doc, extrinsics={"from_observation": True}))
        # no pose recorded -> extrinsics missing
        self.assertIn("extrinsics_missing", geom.bundle_from_records(obs, True, doc).missing())
        # recorded pose (the synthetic camera, as rgbd-capture would store it) -> usable but unverified
        pose = {"status": "read", "frame_from": "cam", "frame_to": "world", "rotation": [list(r) for r in self.synth.ROTATION],
                "translation_mm": list(self.synth.CAMERA_POSITION), "read_at": "2026-10-29T00:00:00+00:00"}
        obs["camera"]["pose_in_world"] = pose
        bundle = geom.bundle_from_records(obs, True, doc)
        self.assertEqual(bundle.missing(), ["extrinsics_not_verified"])
        self.assertEqual(bundle.extrinsics.frame_to, "world")
        self.assertEqual([round(x, 3) for x in bundle.extrinsics.apply((0.0, 0.0, 440.0))], [300.0, 150.0, 10.0])
        # the capture-side record flags motion between the two reads
        before = dict(pose)
        after = dict(pose, translation_mm=[300.0, 150.0, 455.0])
        rec = capture._camera_pose_record(before, after)
        self.assertFalse(rec["stable"])
        self.assertEqual(rec["max_change_mm_between_reads"], 5.0)
        self.assertTrue(capture._camera_pose_record(before, dict(pose))["stable"])
        self.assertEqual(capture._camera_pose_record(None, None)["status"], "not_read")
        self.assertEqual(capture._camera_pose_record(before, {"status": "unavailable"})["status"], "unavailable")

    def test_stacked_block_is_flagged_not_rejected(self):
        depth, masks, truth, calib, plane, hm = self.scene("stacked")
        top = geom.estimate_block(masks[2], depth, hm, calib, 2)
        self.assertEqual(top.rejection_reasons, [])
        self.assertAlmostEqual(top.height_above_table_mm, 30.0, delta=1.0)
        self.assertIn("possibly_stacked_or_resting_on_another", top.flags)
        hidden = geom.estimate_block(masks[1], depth, hm, calib, 1)  # the covered block shows only a sliver
        self.assertTrue(hidden.rejected)


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow required for proposal tests")
class ProposalTests(unittest.TestCase):
    def setUp(self):
        from robot_hackathon import synthetic_scene as synth
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.synth = synth
        synth.save_scene(self.dir, "blocked")
        self.scene = proposal.synthetic_scene(self.dir, "obs_synthetic_blocked")
        self.names = {m["truth"]["name"]: m["candidate_id"] for m in self.scene["masks"]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_saved_scene_is_immutable_and_replayable(self):
        with self.assertRaises(ValueError):
            self.synth.save_scene(self.dir, "blocked")
        self.assertEqual(len(self.scene["masks"]), 4)
        self.assertEqual(self.scene["calibration"].missing(), [])
        self.assertEqual(self.scene["segmentation"]["provider"], "synthetic-ground-truth")

    def test_context_traces_observation_to_proposal(self):
        ctx = proposal.build_context(self.scene, tower_origin_xy_mm=(160.0, 300.0))
        self.assertEqual(ctx["observation_id"], "obs_synthetic_blocked")
        self.assertEqual(ctx["mode"], "mock")
        self.assertEqual(ctx["table_plane"]["status"], "fitted")
        self.assertLess(abs(ctx["table_plane"]["offset_minus_fitted_mean_mm"]), 2.0)
        by_id = {c["candidate_id"]: c for c in ctx["candidates"]}
        self.assertFalse(by_id[self.names["B"]]["eligible"])
        self.assertEqual(by_id[self.names["B"]]["rejection_reasons"], ["no_eligible_grasp"])
        self.assertTrue(by_id[self.names["A"]]["eligible"])
        self.assertEqual(ctx["ranking"][0]["candidate_id"], self.names["A"])   # all four grasps clear beats ends-only
        self.assertTrue(all(c["retrieval"]["status"] == "unavailable" for c in ctx["candidates"]))
        self.assertEqual(ctx["placement"]["next_slot"]["slot_id"], "L0S0")
        decision = proposal.DeterministicCoordinator().decide(ctx)
        self.assertEqual(decision["kind"], "pick_and_place_proposal")
        self.assertEqual(decision["candidate_id"], self.names["A"])
        self.assertEqual(decision["grasp"]["style"], "middle")
        self.assertFalse(decision["executable"])
        self.assertEqual(decision["expiry"]["policy"], "undefined")
        self.assertEqual(decision["source_pose"]["frame"], "robot_base")
        self.assertAlmostEqual(decision["source_pose"]["block_top_center_mm"][0], 210.0, delta=1.5)
        self.assertEqual(decision["destination"]["slot_id"], "L0S0")
        self.assertEqual(decision["evidence"]["retrieval_status"], "unavailable")
        self.assertEqual(decision["evidence"]["reference_candidate_ids"], [])
        # replay: same inputs -> same context identity (timestamps aside)
        again = proposal.build_context(self.scene, tower_origin_xy_mm=(160.0, 300.0))
        self.assertEqual(again["context_id"], ctx["context_id"])
        paths = proposal.save_context(ctx, decision, self.dir / "out")
        self.assertEqual(json.loads(paths[1].read_text())["proposal_id"], decision["proposal_id"])

    def test_retrieval_evidence_enters_ranking_without_overriding_geometry(self):
        # Give the ends-only neighbour a vector that matches a curated reference; the fully blocked block too.
        refs = [dict(ref("ref_good", obs="obs_ref"), embedding=unit(1))]
        for m in self.scene["masks"]:
            m["candidate"] = {"encoder_version": "enc-v1", "embedding_dimension": 8}
            m["embedding"] = unit(1) if m["truth"]["name"] in ("N1", "B") else unit(5)
        ctx = proposal.build_context(self.scene, retrieval.ReferencePolicy(), retrieval.LocalReferenceSearch(refs), (160.0, 300.0))
        by_id = {c["candidate_id"]: c for c in ctx["candidates"]}
        self.assertEqual(by_id[self.names["N1"]]["retrieval"]["status"], "matched")
        self.assertEqual(by_id[self.names["B"]]["retrieval"]["status"], "matched")
        self.assertFalse(by_id[self.names["B"]]["eligible"])                      # resemblance never overrides blocked fingers
        self.assertEqual(ctx["ranking"][0]["candidate_id"], self.names["N1"])     # retrieval-assisted prefers the matched eligible block
        self.assertEqual(ctx["geometry_only_ranking"][0], self.names["A"])        # baseline kept separately
        decision = proposal.DeterministicCoordinator().decide(ctx)
        self.assertEqual((decision["candidate_id"], decision["grasp"]["style"]), (self.names["N1"], "ends"))
        self.assertEqual(decision["evidence"]["reference_candidate_ids"], ["ref_good"])
        self.assertAlmostEqual(decision["evidence"]["best_cosine_similarity"], 1.0, places=6)

    def test_validator_rejects_bad_selections_and_coordinator_abstains(self):
        ctx = proposal.build_context(self.scene, tower_origin_xy_mm=(160.0, 300.0))
        with self.assertRaisesRegex(proposal.SelectionError, "unknown_candidate"):
            proposal.validate_selection(ctx, "cand_nope", "middle", 0.0, "L0S0")
        with self.assertRaisesRegex(proposal.SelectionError, "candidate_not_eligible"):
            proposal.validate_selection(ctx, self.names["B"], "middle", 0.0, "L0S0")
        with self.assertRaisesRegex(proposal.SelectionError, "grasp_not_eligible"):
            proposal.validate_selection(ctx, self.names["N1"], "middle", 0.0, "L0S0")
        with self.assertRaisesRegex(proposal.SelectionError, "unknown_grasp"):
            proposal.validate_selection(ctx, self.names["A"], "side", 3.0, "L0S0")
        with self.assertRaisesRegex(proposal.SelectionError, "slot_not_open"):
            proposal.validate_selection(ctx, self.names["A"], "middle", 0.0, "L1S0")
        proposal.validate_selection(ctx, self.names["A"], "side", -15.0, "L0S0")
        # no placement configured -> abstention; tower complete -> abstention; nothing eligible -> abstention with rejections
        self.assertEqual(proposal.DeterministicCoordinator().decide(proposal.build_context(self.scene))["reason"], "tower_origin_not_configured")
        full = proposal.build_context(self.scene, tower_origin_xy_mm=(160.0, 300.0), filled_slots=[f"L{l}S{i}" for l in range(2) for i in range(3)])
        self.assertEqual(proposal.DeterministicCoordinator().decide(full)["reason"], "all_slots_filled")
        # second block of a touching row: open slot exists but has no claw room -> blocked, validator refuses it too
        beside = proposal.build_context(self.scene, tower_origin_xy_mm=(160.0, 300.0), filled_slots=["L0S0"])
        self.assertEqual(beside["placement"]["status"], "available")
        self.assertEqual(beside["placement"]["next_slot"]["slot_id"], "L0S1")
        self.assertTrue(beside["placement"]["next_slot"]["requires_ends_grasp"])
        with self.assertRaisesRegex(proposal.SelectionError, "slot_requires_ends_grasp"):
            proposal.validate_selection(beside, self.names["A"], "middle", 0.0, "L0S1")
        decision = proposal.DeterministicCoordinator().decide(beside)
        self.assertEqual((decision["kind"], decision["grasp"]["style"]), ("pick_and_place_proposal", "ends"))
        doc = json.loads(json.dumps(self.synth.geometry_document()))
        doc["support_surface"]["verified"] = False
        unverified = dict(self.scene, calibration=geom.bundle_from_records(self.scene["observation"], True, doc))
        decision = proposal.DeterministicCoordinator().decide(proposal.build_context(unverified, tower_origin_xy_mm=(160.0, 300.0)))
        self.assertEqual(decision["reason"], "no_eligible_candidate")
        self.assertTrue(all("calibration:support_surface_offset_not_verified" in r for r in decision["details"]["rejections"].values()))

    def test_dry_run_waives_only_unverified_calibration_and_yields_a_preview(self):
        doc = json.loads(json.dumps(self.synth.geometry_document()))
        doc["support_surface"]["verified"] = False
        doc["finger"]["verified"] = False
        unverified = dict(self.scene, calibration=geom.bundle_from_records(self.scene["observation"], True, doc))
        strict = proposal.build_context(unverified, tower_origin_xy_mm=(160.0, 300.0))
        self.assertEqual(strict["eligible_candidate_ids"], [])
        preview = proposal.build_context(unverified, tower_origin_xy_mm=(160.0, 300.0), dry_run=True)
        self.assertTrue(preview["dry_run"])
        self.assertIn(self.names["A"], preview["eligible_candidate_ids"])
        a = next(c for c in preview["candidates"] if c["candidate_id"] == self.names["A"])
        self.assertIn("preview_waived:calibration:support_surface_offset_not_verified", a["geometry"]["flags"])
        self.assertNotIn(self.names["B"], preview["eligible_candidate_ids"])     # blocked fingers are never waived
        decision = proposal.DeterministicCoordinator().decide(preview)
        self.assertEqual((decision["kind"], decision["validation_status"]), ("geometry_preview", "unverified_calibration_preview"))
        self.assertTrue(decision["proposal_id"].startswith("preview_"))
        self.assertFalse(decision["executable"])
        self.assertEqual(sorted(decision["calibration_missing"]), ["finger_geometry_not_verified", "support_surface_offset_not_verified"])
        # missing (not merely unverified) data is still a rejection in dry run
        doc["support_surface"]["z_in_robot_base_mm"] = None
        missing = dict(self.scene, calibration=geom.bundle_from_records(self.scene["observation"], True, doc))
        ctx = proposal.build_context(missing, tower_origin_xy_mm=(160.0, 300.0), dry_run=True)
        self.assertEqual(ctx["eligible_candidate_ids"], [])
        self.assertTrue(all("calibration:support_surface_offset_missing" in c["rejection_reasons"] for c in ctx["candidates"]))

    def test_openai_tool_loop_validates_and_abstains(self):
        ctx = proposal.build_context(self.scene, tower_origin_xy_mm=(160.0, 300.0))
        tools = coordinator.tool_definitions()
        self.assertEqual([t["name"] for t in tools], ["inspect_scene", "get_retrieval_evidence", "get_grasp_options", "list_placement_slots", "submit_decision"])
        self.assertTrue(all(t["strict"] and set(t["parameters"]["required"]) == set(t["parameters"]["properties"]) for t in tools))
        handlers = coordinator.ContextTools(ctx)
        summary = handlers.dispatch("inspect_scene", {})
        self.assertEqual(len(summary["candidates"]), 4)
        self.assertNotIn("rgb", json.dumps(summary))
        self.assertEqual(handlers.dispatch("get_grasp_options", {"candidate_id": "cand_nope"}), {"error": "unknown_candidate:cand_nope"})
        with self.assertRaises(coordinator.ToolError):
            handlers.dispatch("move_arm", {})
        # Model picks a rejected candidate -> abstention, never a proposal
        bad = coordinator.apply_decision(ctx, {"action": "propose", "candidate_id": self.names["B"], "grasp_style": "middle", "shift_mm": 0,
                                               "slot_id": "L0S0", "rationale": "looks similar"}, "test")
        self.assertEqual((bad["kind"], bad["reason"]), ("abstention", "invalid_selection"))
        self.assertEqual(coordinator.apply_decision(ctx, {"action": "fly"}, "test")["reason"], "malformed_decision")
        good = coordinator.apply_decision(ctx, {"action": "propose", "candidate_id": self.names["A"], "grasp_style": "middle", "shift_mm": 0,
                                                "slot_id": "L0S0", "rationale": "clear"}, "test")
        self.assertEqual(good["kind"], "pick_and_place_proposal")

        # Fake Responses API transport: first turn asks for the scene, second submits.
        class Resp:
            def __init__(self, body): self.body = body
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return json.dumps(self.body).encode()
        sent = []

        def opener(req, timeout=0):
            body = json.loads(req.data.decode())
            sent.append(body)
            if len(sent) == 1:
                self.assertEqual(body["tool_choice"], "required")
                self.assertFalse(body["store"])
                return Resp({"id": "r1", "output": [{"type": "function_call", "call_id": "c1", "name": "inspect_scene", "arguments": "{}"}]})
            self.assertNotIn("previous_response_id", body)  # store=False: the whole exchange is resent
            self.assertEqual([i.get("type") or i.get("role") for i in body["input"]], ["user", "function_call", "function_call_output"])
            self.assertEqual(body["input"][2]["call_id"], "c1")
            top = json.loads(body["input"][2]["output"])["ranking"][0]["candidate_id"]
            args = {"action": "propose", "candidate_id": top, "grasp_style": "middle", "shift_mm": 0, "slot_id": "L0S0", "rationale": "ranked first"}
            return Resp({"id": "r2", "output": [{"type": "function_call", "call_id": "c2", "name": "submit_decision", "arguments": json.dumps(args)}]})
        agent = coordinator.OpenAICoordinator(Config(openai_key="k"), allow_cloud=True, model="test-model", opener=opener)
        decision = agent.decide(ctx)
        self.assertEqual((decision["kind"], decision["decided_by"], decision["candidate_id"]), ("pick_and_place_proposal", "openai:test-model", self.names["A"]))
        self.assertEqual(agent.calls, 2)
        # Budget exhaustion and malformed payloads become abstentions
        loop = coordinator.OpenAICoordinator(Config(openai_key="k"), allow_cloud=True, max_turns=2, opener=lambda req, timeout=0: Resp(
            {"id": "x", "output": [{"type": "function_call", "call_id": "c", "name": "list_placement_slots", "arguments": "{}"}]}))
        self.assertEqual(loop.decide(ctx)["reason"], "turn_budget_exhausted")
        self.assertEqual(coordinator.OpenAICoordinator(Config(openai_key="k"), allow_cloud=True, opener=lambda req, timeout=0: Resp({"id": "x", "output": "?"})).decide(ctx)["reason"],
                         "openai_malformed_response")
        self.assertEqual(coordinator.OpenAICoordinator(Config(openai_key="k"), allow_cloud=True, opener=lambda req, timeout=0: Resp(
            {"id": "x", "output": [{"type": "message"}]})).decide(ctx)["reason"], "no_tool_call")
        with self.assertRaises(ConfigError):
            coordinator.OpenAICoordinator(Config(openai_key="k"), allow_cloud=False).decide(ctx)
        with self.assertRaises(ConfigError):
            coordinator.OpenAICoordinator(Config(), allow_cloud=True).decide(ctx)

    def test_cli_runs_synthetic_offline_and_gates_cloud(self):
        env = dict(os.environ)
        os.environ.pop("ALLOW_CLOUD", None)
        os.environ.pop("MODE", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                report = Path(tmp) / "r.html"
                self.assertEqual(cli.main(["agent-propose", "--synthetic", "separated", "--data-dir", tmp, "--report", str(report)]), 0)
                self.assertIn("Pick-and-place proposal review", report.read_text())
                self.assertIn("SYNTHETIC SCENE", report.read_text())
                self.assertTrue(list((Path(tmp) / "mock" / "proposals" / "obs_synthetic_separated").glob("ctx_*-decision.json")))
                self.assertEqual(cli.main(["agent-propose", "--synthetic", "nope", "--data-dir", tmp]), 1)  # unknown preset fails safely
                self.assertEqual(cli.main(["agent-propose", "--data-dir", tmp]), 2)
                self.assertEqual(cli.main(["agent-propose", "--synthetic", "separated", "--coordinator", "openai", "--data-dir", tmp]), 2)
                self.assertEqual(cli.main(["agent-propose", "--synthetic", "separated", "--references", "atlas", "--data-dir", tmp]), 2)
                self.assertEqual(cli.main(["agent-calibration", "--data-dir", tmp]), 0)
                self.assertEqual(cli.main(["agent-propose", "--observation", "obs_missing", "--data-dir", tmp]), 1)
        finally:
            os.environ.clear()
            os.environ.update(env)


if __name__ == "__main__":
    unittest.main()
