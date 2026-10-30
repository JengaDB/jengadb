import copy
import json
import unittest
from robot_hackathon.pick_policy import propose


class PickPolicyTests(unittest.TestCase):
    def setUp(self):
        self.obs = {"observation_id": "now", "episode_id": "ep", "setup_id": "robot",
                    "mode": "mock", "calibration_version": "cal1", "alignment": {"status": "verified"}}
        self.candidates = [{**{k: v for k, v in self.obs.items() if k != "alignment"},
            "candidate_id": cid, "registration_verified": True, "exposure": {"status": "exposed"},
            "depth_stats": {"median_mm": 400}, "encoder_version": "encoder1",
            "segmentation_score": confidence, "bbox": [1, 2, 3, 4]} for cid, confidence in [("a", .99), ("b", .8)]]
        self.geometry = {c["candidate_id"]: {"observation_id": "now", "setup_id": "robot", "calibration_version": "cal1",
            "verified": True, "height_above_table_mm": h} for c, h in zip(self.candidates, [15, 30])}

    def run_policy(self, comparisons=None):
        return propose(self.obs, self.candidates, comparisons or {}, self.geometry)

    def test_top_first_proposal_is_never_executable(self):
        result = self.run_policy()
        self.assertEqual(result["selected_candidate_id"], "b")
        self.assertEqual(result["decision"], "propose_pick")
        self.assertFalse(result["executable"])
        self.assertIsNone(result["robot_pose"])
        self.assertIn("fresh_observation", result["required_before_execution"])

    def test_occluded_higher_block_is_excluded(self):
        self.candidates[1]["exposure"]["status"] = "occluded"
        self.assertEqual(self.run_policy()["selected_candidate_id"], "a")

    def test_unverified_scene_abstains_despite_default_pickable_labels(self):
        self.obs["alignment"]["status"] = "unverified"
        for c in self.candidates:
            c.update(pickable=True, pickability_label={"status": "pickable", "label_source": "human_default"})
        result = self.run_policy()
        self.assertEqual(result["decision"], "request_verification")
        self.assertIsNone(result["selected_candidate_id"])

    def test_mismatched_observation_and_stale_geometry_rejected(self):
        self.candidates[0]["observation_id"] = "old"
        self.geometry["b"]["calibration_version"] = "old"
        self.assertIsNone(self.run_policy()["selected_candidate_id"])

    def test_camera_depth_is_not_used_as_table_height(self):
        self.geometry = {}
        self.assertIsNone(self.run_policy()["selected_candidate_id"])

    def test_nonfinite_and_boolean_heights_rejected(self):
        for invalid in (True, float('nan'), float('inf'), -1, "15"):
            with self.subTest(value=invalid):
                for g in self.geometry.values():g["height_above_table_mm"] = invalid
                self.assertIsNone(self.run_policy()["selected_candidate_id"])

    def test_only_compatible_named_reviews_break_height_ties(self):
        self.geometry["b"]["height_above_table_mm"] = 15
        match = {"candidate_id": "historic", "observation_id": "previous", "encoder_version": "encoder1",
                 "mode": "mock", "score": .99,
                 "pickability_label": {"status": "pickable", "label_source": "human", "reviewed_by": "sam"}}
        self.assertEqual(self.run_policy({"b": [match]})["selected_candidate_id"], "b")
        for key, value in [("observation_id", "now"), ("encoder_version", "old"), ("mode", "live")]:
            self.assertEqual(self.run_policy({"b": [{**match, key: value}]})["selected_candidate_id"], "a")
        for label in ({"status": "pickable", "label_source": "human_default", "reviewed_by": "sam"},
                      {"status": "pickable", "label_source": "human", "reviewed_by": None}):
            self.assertEqual(self.run_policy({"b": [{**match, "pickability_label": label}]})["selected_candidate_id"], "a")

    def test_negative_review_blocks_selection_and_duplicates_fail(self):
        self.candidates[1]["pickability_label"] = {"status": "not_pickable"}
        self.assertEqual(self.run_policy()["selected_candidate_id"], "a")
        self.candidates.append(copy.deepcopy(self.candidates[0]))
        with self.assertRaises(ValueError):self.run_policy()

    def test_deterministic_and_does_not_mutate_inputs(self):
        before = json.dumps([self.obs, self.candidates, self.geometry], sort_keys=True)
        first = self.run_policy()
        second = propose(self.obs, list(reversed(self.candidates)), {}, self.geometry)
        self.assertEqual(first, second)
        self.assertEqual(before, json.dumps([self.obs, self.candidates, self.geometry], sort_keys=True))
