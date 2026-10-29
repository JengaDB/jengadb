"""Offline tests for the RGB-D collection pipeline (SKY-66).

Synthetic scene, stub providers and in-memory remote live here only; the shipped pipeline has no fake paths.
Set RGBD_TESTS_REQUIRED=1 to fail (instead of skip) when numpy/Pillow are unavailable.
"""
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
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

from robot_hackathon import rgbd, exposure, cutouts, embeddings, segmentation, pipeline, media, candidate_store, cli  # noqa: E402
from robot_hackathon.config import Config, ConfigError  # noqa: E402

UTC = timezone.utc
BLOCKS = (("A", (20, 30, 110, 70), (220, 40, 40), 600), ("C", (150, 90, 300, 140), (40, 60, 220), 650),
          ("B", (200, 60, 260, 170), (40, 200, 60), 500))  # B is closer and overlaps C; A is clear


def scene(invalid_patch=True):
    rgb = np.full((240, 320, 3), (90, 90, 90), dtype=np.uint8)
    depth = np.full((240, 320), 900, dtype=np.uint16)
    for _, (x0, y0, x1, y1), color, mm in BLOCKS:
        rgb[y0:y1, x0:x1] = color
        depth[y0:y1, x0:x1] = mm
    if invalid_patch:
        depth[200:230, 10:60] = 0
    return rgb, depth


class StubSegmenter(segmentation.SegmentationProvider):
    provider_id = "stub-colour"

    def __init__(self, dominance=60, fail=False):
        self.dominance, self.fail, self.calls = dominance, fail, 0

    def effective_settings(self):
        return {"provider": self.provider_id, "model_version": "stub-1", "dominance": self.dominance, "min_area": 50, "parser_version": "n/a"}

    def segment(self, rgb):
        self.calls += 1
        if self.fail:
            raise TimeoutError("segmentation timeout")
        rgb = rgb.astype(np.int16)
        return [((rgb[..., c] - np.maximum(rgb[..., [o for o in range(3) if o != c][0]], rgb[..., [o for o in range(3) if o != c][1]])) > self.dominance, 1.0)
                for c in range(3)]


class StubEmbedder(embeddings.EmbeddingProvider):
    encoder_version = "stub/ctx-v1"

    def __init__(self, template_id="geom-v1", fail=False, dimension=1024):
        self.template_id, self.fail, self.dimension, self.calls = template_id, fail, dimension, 0

    def effective_settings(self):
        return {"provider": "stub", "model": "sha256", "dimension": self.dimension, "input_type": "document",
                "input_parts": ["text", "image"], "image_preprocessing": {"resize_max_px": None, "format": "png"}}

    def embed(self, text, image_png):
        self.calls += 1
        if self.fail:
            raise TimeoutError("embedding timeout")
        seed = hashlib.sha256(text.encode() + image_png).digest()
        values = []
        i = 0
        while len(values) < self.dimension:
            values.extend(b / 127.5 - 1 for b in hashlib.sha256(seed + i.to_bytes(4, "big")).digest())
            i += 1
        values = values[:self.dimension]
        n = math.sqrt(sum(v * v for v in values))
        return [v / n for v in values]


def make_observation(data_dir, alignment="verified", calibration="cal-v1", register=True, intrinsics=True, obs_id="obs_test", **kw):
    rgb, depth = kw.pop("arrays", scene())
    registry = rgbd.CalibrationRegistry(data_dir, "mock")
    if register:
        registry.record("test-setup", calibration, alignment, "test-overlay", "tester", viewing_pose="top_down")
    obs = rgbd.new_observation("mock", "test-setup", "ep1", calibration,
                               alignment={"status": alignment, "method": "test-overlay", "calibration_version": calibration,
                                          "checked_by": "tester", "checked_at": rgbd.utc_now()},
                               camera={"resource_name": "cam", "serial": None, "model": "test"},
                               intrinsics={"rgb": {"focal_x_px": 300}, "depth": {"focal_x_px": 300}} if intrinsics else None,
                               observation_id=obs_id)
    return rgbd.ObservationStore(data_dir, "mock").save(obs, rgb, depth)


def make_pipeline(data_dir, seg=None, emb=None, params=None):
    return pipeline.Pipeline(data_dir, "mock", seg or StubSegmenter(), emb or StubEmbedder(), params or exposure.ExposureParams())


# ---------------------------------------------------------------- in-memory remote

class MemoryBackend:
    def __init__(self):
        self.attempts, self.files, self.manifest = {}, {}, {}
        self.clock = datetime(2026, 1, 1, tzinfo=UTC)
        self.fail_once = set()
        self.n = 0

    def _fail(self, op):
        if op in self.fail_once:
            self.fail_once.remove(op)
            raise TimeoutError(op)

    def now(self):
        return self.clock

    def insert_attempt(self, sha256, source_alias, started_at):
        self.n += 1
        aid = f"att{self.n}"
        self.attempts[aid] = {"_id": aid, "sha256": sha256, "source_alias": source_alias, "started_at": started_at, "state": "writing"}
        return aid

    def set_attempt_state(self, aid, state):
        self.attempts[aid]["state"] = state

    def get_attempt(self, aid):
        return self.attempts.get(aid)

    def write_file(self, aid, data, metadata):
        self._fail("write_file_before")
        if "write_file_partial" in self.fail_once:
            self.fail_once.remove("write_file_partial")
            self.files[aid] = (data[: len(data) // 2], metadata)
            raise TimeoutError("partial")
        self.files[aid] = (data, metadata)

    def file_length(self, fid):
        return None if fid not in self.files else len(self.files[fid][0])

    def download(self, fid):
        return self.files[fid][0]

    def delete_file(self, fid):
        self.files.pop(fid, None)

    def find_manifest(self, sha256):
        return self.manifest.get(sha256)

    def insert_manifest(self, doc):
        if doc["_id"] in self.manifest:
            return False
        self.manifest[doc["_id"]] = doc
        self._fail("insert_manifest_after")
        return True

    def list_attempts(self, states, older_than):
        return [a for a in self.attempts.values() if a["state"] in states and a["started_at"] < older_than]


class MemoryRemote:
    target = "memtarget"

    def __init__(self):
        self.backend = MemoryBackend()
        self.media = media.MediaUploader(self.backend, timedelta(hours=24))
        self.observations, self.candidates, self.artifact_sync = {}, {}, {}
        self.fail_once = set()

    def _fail(self, op):
        if op in self.fail_once:
            self.fail_once.remove(op)
            raise TimeoutError(op)

    def upsert_observation(self, obs):
        self.observations.setdefault(obs["observation_id"], obs)
        self._fail("observation_after")
        assert self.observations[obs["observation_id"]] == obs

    def upsert_candidate(self, doc):
        self.candidates.setdefault(doc["candidate_id"], doc)
        self._fail("candidate_after")
        assert self.candidates[doc["candidate_id"]] == doc

    def set_artifact_sync(self, sha, obs_id, status, gid, size, local_check):
        row = self.artifact_sync.setdefault(sha, {"remote_status": "pending", "observation_ids": set()})
        row["observation_ids"].add(obs_id)
        row["last_local_check"] = local_check
        if status == "synced":
            row["remote_status"], row["gridfs_id"] = "synced", gid


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow required for RGB-D tests")
class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_hash_validation_and_immutability(self):
        obs = make_observation(self.dir)
        store = rgbd.ObservationStore(self.dir, "mock")
        loaded, rgb, depth = store.load_arrays("obs_test")
        self.assertEqual(loaded["artifacts"], obs["artifacts"])
        self.assertEqual(depth.dtype, np.uint16)
        self.assertEqual(int(depth[40, 50]), 600)
        self.assertTrue(loaded["dimensions_equal"])
        (store.directory("obs_test") / "depth.png").write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            store.load("obs_test")
        self.assertEqual(store.local_artifact_status(loaded)["depth"], "hash_mismatch")
        with self.assertRaises(ValueError):
            rgbd.ObservationStore(self.dir, "mock").save(obs, *scene())
        with self.assertRaises(ValueError):
            rgbd.ObservationStore(self.dir, "live").save(dict(obs, observation_id="x"), *scene())

    def test_dimension_mismatch_is_a_flag_not_alignment(self):
        rgb, depth = scene()
        obs = make_observation(self.dir, arrays=(rgb, depth[:-10]), obs_id="obs_mismatch")
        self.assertIn("rgb_depth_dimension_mismatch", obs["flags"])
        self.assertEqual(obs["alignment"]["status"], "verified")  # recorded verdict is preserved, gate rejects it
        self.assertFalse(exposure.gate(obs, "cal-v1", rgbd.CalibrationRegistry(self.dir, "mock").load("test-setup", "cal-v1"))[0])

    def test_viam_depth_decode(self):
        _, depth = scene()
        data = rgbd.encode_viam_depth(depth)
        self.assertTrue(np.array_equal(rgbd.decode_viam_depth(data), depth))
        with self.assertRaises(ValueError):
            rgbd.decode_viam_depth(data[:-2])

    def test_calibration_registry_rejects_conflicting_rewrite(self):
        reg = rgbd.CalibrationRegistry(self.dir, "mock")
        reg.record("s", "v1", "verified", "m", "me")
        reg.record("s", "v1", "verified", "m", "me")
        with self.assertRaises(ValueError):
            reg.record("s", "v1", "unverified", "m", "me")
        self.assertEqual(reg.current("s"), "v1")


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow required for RGB-D tests")
class ExposureAndCutoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def by_name(self, result):
        colours = {(220, 40, 40): "A", (40, 60, 220): "C", (40, 200, 60): "B"}
        rgb, _ = scene()
        out = {}
        for c in result["candidates"]:
            x0, y0 = c["bbox"][0], c["bbox"][1]
            out[colours[tuple(int(v) for v in rgb[y0, x0])]] = c
        return out

    def test_gated_in_scene_classification(self):
        make_observation(self.dir)
        result = make_pipeline(self.dir).process("obs_test")
        self.assertTrue(result["gated_in"])
        named = self.by_name(result)
        self.assertEqual({k: v["exposure"]["status"] for k, v in named.items()}, {"A": "exposed", "B": "exposed", "C": "occluded"})
        for c in named.values():
            self.assertEqual(c["pickability"]["status"], "unknown")
            self.assertIsNone(c["depth_crop_reason"])
            self.assertIsNotNone(c["files"]["depth_crop_png"])
        self.assertAlmostEqual(named["C"]["depth_stats"]["median_mm"], 650)
        self.assertIn("mm", named["C"]["description_text"])

    def test_unverified_alignment_yields_unknown_and_no_depth_crop(self):
        make_observation(self.dir, alignment="unverified")
        result = make_pipeline(self.dir).process("obs_test")
        self.assertFalse(result["gated_in"])
        self.assertIn("alignment_not_verified", result["gate_reasons"])
        for c in result["candidates"]:
            self.assertEqual(c["exposure"]["status"], "unknown")
            self.assertEqual(c["depth_crop_reason"], "registration_not_verified")
            self.assertIsNone(c["files"]["depth_crop_png"])
            self.assertIn("registration not verified", c["description_text"])

    def test_calibration_mismatch_and_missing_intrinsics_gate_out(self):
        make_observation(self.dir, calibration="cal-v1")
        rgbd.CalibrationRegistry(self.dir, "mock").record("test-setup", "cal-v2", "verified", "m", "me")
        result = make_pipeline(self.dir).process("obs_test")
        self.assertIn("calibration_mismatch", result["gate_reasons"])
        make_observation(self.dir, intrinsics=False, obs_id="obs_noint", calibration="cal-v2", register=False)
        self.assertIn("intrinsics_missing", make_pipeline(self.dir).process("obs_noint")["gate_reasons"])

    def test_invalid_depth_is_unknown(self):
        rgb, depth = scene()
        depth[:] = 0
        make_observation(self.dir, arrays=(rgb, depth))
        result = make_pipeline(self.dir).process("obs_test")
        self.assertTrue(result["gated_in"])
        self.assertTrue(all(c["exposure"]["status"] == "unknown" and "insufficient_valid_depth" in c["exposure"]["reasons"] for c in result["candidates"]))

    def test_crop_transform_maps_back(self):
        make_observation(self.dir)
        result = make_pipeline(self.dir).process("obs_test")
        named = self.by_name(result)
        a = named["A"]
        self.assertEqual((a["crop_transform"]["x0"], a["crop_transform"]["y0"], a["crop_transform"]["width"], a["crop_transform"]["height"]), (20, 30, 90, 40))
        cut, kind = rgbd.decode_png(pipeline.load_candidate_file(self.dir, a, "cutout_png"))
        self.assertEqual(cut.shape, (40, 90, 3))
        mask, kind = rgbd.decode_png(pipeline.load_candidate_file(self.dir, a, "mask_png"))
        self.assertEqual(kind, "mask")
        self.assertEqual(mask.shape, (240, 320))
        self.assertTrue(mask[30:70, 20:110].all())
        ctx = a["context_transform"]
        self.assertEqual((ctx["x0"], ctx["y0"]), (0, 6))

    def test_empty_candidate_set_is_valid(self):
        rgb, depth = scene()
        rgb[:] = 90
        make_observation(self.dir, arrays=(rgb, depth))
        result = make_pipeline(self.dir).process("obs_test")
        self.assertEqual(result["candidates"], [])


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow required for RGB-D tests")
class IdentityAndCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        make_observation(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rerun_is_noop_and_identity_changes_make_new_ids(self):
        seg, emb = StubSegmenter(), StubEmbedder()
        first = make_pipeline(self.dir, seg, emb).process("obs_test")
        second = make_pipeline(self.dir, seg, emb).process("obs_test")
        self.assertEqual(first["processing_version"], second["processing_version"])
        self.assertEqual([c["candidate_id"] for c in first["candidates"]], [c["candidate_id"] for c in second["candidates"]])
        self.assertEqual((seg.calls, emb.calls), (1, 3))
        base_ids = {c["candidate_id"] for c in first["candidates"]}
        variants = [
            make_pipeline(self.dir, params=exposure.ExposureParams(block_dimensions_mm=(76.0, 25.0, 15.0))),
            make_pipeline(self.dir, params=exposure.ExposureParams(block_dimensions_verified=True)),
            make_pipeline(self.dir, emb=StubEmbedder(dimension=512)),
            make_pipeline(self.dir, seg=StubSegmenter(dominance=61)),
        ]
        versions = {first["processing_version"]}
        for p in variants:
            r = p.process("obs_test")
            self.assertNotIn(r["processing_version"], versions)
            versions.add(r["processing_version"])
            self.assertTrue(base_ids.isdisjoint({c["candidate_id"] for c in r["candidates"]}))
        # Registry state change (new current calibration) affects the gate, so it is part of the identity too.
        rgbd.CalibrationRegistry(self.dir, "mock").record("test-setup", "cal-v2", "verified", "m", "me")
        r = make_pipeline(self.dir, seg, emb).process("obs_test")
        self.assertNotIn(r["processing_version"], versions)
        self.assertIn("calibration_mismatch", r["gate_reasons"])
        self.assertTrue(base_ids.isdisjoint({c["candidate_id"] for c in r["candidates"]}))
        self.assertEqual(seg.calls, 1)  # segmentation cache shared across identities
        # Old candidates untouched on disk
        proc = rgbd.ObservationStore(self.dir, "mock").directory("obs_test") / "processing" / first["processing_version"]
        self.assertEqual(json.loads((proc / "candidates.json").read_text()), first["candidates"])

    def test_restart_after_embedding_timeout_skips_segmentation(self):
        seg, emb = StubSegmenter(), StubEmbedder(fail=True)
        with self.assertRaises(TimeoutError):
            make_pipeline(self.dir, seg, emb).process("obs_test")
        self.assertEqual(seg.calls, 1)
        emb.fail = False
        result = make_pipeline(self.dir, seg, emb).process("obs_test")
        self.assertEqual(seg.calls, 1)
        self.assertEqual(len(result["candidates"]), 3)

    def test_segmentation_cache_shared_across_exposure_params(self):
        seg = StubSegmenter()
        make_pipeline(self.dir, seg).process("obs_test")
        make_pipeline(self.dir, seg, params=exposure.ExposureParams(closer_margin_mm=5.0)).process("obs_test")
        self.assertEqual(seg.calls, 1)
        seg2 = StubSegmenter(dominance=61)  # same label semantics, different threshold -> cache miss
        make_pipeline(self.dir, seg2).process("obs_test")
        self.assertEqual(seg2.calls, 1)

    def test_tampered_checkpoint_is_recomputed(self):
        seg, emb = StubSegmenter(), StubEmbedder()
        first = make_pipeline(self.dir, seg, emb).process("obs_test")
        proc = rgbd.ObservationStore(self.dir, "mock").directory("obs_test") / "processing" / first["processing_version"]
        # 1. delete a cutout output -> cutouts stage recomputed, segmentation not re-called
        (proc / "cutouts" / "000-cutout.png").unlink()
        make_pipeline(self.dir, seg, emb).process("obs_test")
        self.assertTrue((proc / "cutouts" / "000-cutout.png").exists())
        self.assertTrue(list((proc / "stale").glob("cutouts-*.json")))
        self.assertEqual(seg.calls, 1)
        # 2. alter a mask output -> segmentation recomputed (provider called again)
        seg_dir = next((rgbd.ObservationStore(self.dir, "mock").directory("obs_test") / "segmentation").iterdir())
        mask_path = next(seg_dir.glob("mask-*.png"))
        mask_path.write_bytes(rgbd.encode_mask_png(np.zeros((240, 320), dtype=bool)))
        make_pipeline(self.dir, seg, emb).process("obs_test")
        self.assertEqual(seg.calls, 2)
        # 3. edit a checkpoint body -> self-hash mismatch -> recomputed
        path = proc / "exposure.json"
        doc = json.loads(path.read_text())
        doc["payload"]["results"][0]["exposure"]["status"] = "exposed"
        path.write_text(json.dumps(doc))
        make_pipeline(self.dir, seg, emb).process("obs_test")
        self.assertNotEqual(json.loads(path.read_text())["payload"]["results"], doc["payload"]["results"]) if doc["payload"]["results"][0]["exposure"]["status"] != "exposed" else None
        self.assertTrue(list((proc / "stale").glob("exposure-*.json")))


class ProviderParserTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_DEPS, "numpy and Pillow required")
    def test_meta_sam3_fixture_stream_parses_and_places_masks(self):
        try:
            import meta_sam_parser  # noqa: F401
        except ImportError:
            if os.environ.get("RGBD_TESTS_REQUIRED") == "1":
                raise
            self.skipTest("meta_sam_parser required")
        fixture = json.loads((Path(__file__).resolve().parents[1] / "fixtures/providers/meta-sam3/responses-stream.json").read_text())
        w, h = fixture["image"]["width"], fixture["image"]["height"]
        masks, info = segmentation.parse_events(fixture["events"], w, h)
        self.assertEqual(len(masks), fixture["expected"]["masks"])
        self.assertEqual([int(m.sum()) for m, _ in masks], fixture["expected"]["areas_px"])
        self.assertEqual([c for _, c in masks], fixture["expected"]["confidences"])
        self.assertEqual(info["outcome"], "completed")
        self.assertEqual(masks[0][0].shape, (h, w))
        # Mask bounds that do not fit a smaller frame are rejected rather than silently cropped.
        with self.assertRaises(ValueError):
            segmentation.parse_events(fixture["events"], 10, 10)
        # Zero matches: a completed stream whose output_text lane is empty is a valid empty result.
        empty = []
        for e in fixture["events"]:
            if e["type"] == "response.output_text.delta":
                continue
            e = json.loads(json.dumps(e))
            if e["type"] == "response.output_text.done":
                e["text"] = ""
            if e["type"] == "response.content_part.done":
                e["part"]["text"] = ""
            empty.append(e)
        no_masks, info = segmentation.parse_events(empty, w, h)
        self.assertEqual((no_masks, info["outcome"]), ([], "completed"))
        self.assertEqual(segmentation.finalize_masks(no_masks, w, h, 0), [])
        # SSE framing: blank-line separated data blocks, [DONE] ignored
        raw = b"event: x\ndata: {\"type\": \"a\"}\n\ndata: [DONE]\n\n"
        self.assertEqual(list(segmentation.iter_sse(io.BytesIO(raw))), [{"type": "a"}])

    def test_meta_request_body_shape(self):
        seg = segmentation.MetaSam3Provider(Config(meta_api_key="k"), allow_cloud=True, prompt="wooden block")
        body = seg.request_body(b"png")
        self.assertEqual(body["model"], "sam-3.1")
        self.assertTrue(body["stream"])
        parts = body["input"][0]["content"]
        self.assertEqual(parts[0], {"type": "input_text", "text": "wooden block"})
        self.assertTrue(parts[1]["image_url"].startswith("data:image/png;base64,"))
        self.assertEqual(body["metadata"]["mask_encoding"], "one_bit")
        self.assertEqual(seg.url, "https://api.meta.ai/v1/responses")
        self.assertIn("prompt", seg.effective_settings())

    def test_voyage_parser_fixture(self):
        fixture = json.loads((Path(__file__).resolve().parents[1] / "fixtures/providers/voyage/response.json").read_text())
        vector = embeddings.parse_voyage_response(fixture, len(fixture["data"][0]["embedding"]))
        self.assertEqual(len(vector), len(fixture["data"][0]["embedding"]))
        with self.assertRaises(ValueError):
            embeddings.parse_voyage_response(fixture, 7)
        with self.assertRaises(ValueError):
            embeddings.parse_voyage_response({"data": []}, 1024)

    def test_voyage_default_is_image_only_with_description_as_metadata(self):
        cfg = Config(voyage_key="k")
        img_only = embeddings.VoyageMultimodalProvider(cfg, allow_cloud=True)
        fused = embeddings.VoyageMultimodalProvider(cfg, allow_cloud=True, input_parts=("text", "image"))
        self.assertFalse(img_only.uses_text)
        self.assertTrue(fused.uses_text)
        self.assertNotEqual(img_only.encoder_version, fused.encoder_version)
        self.assertNotEqual(img_only.effective_settings(), fused.effective_settings())
        sent = {}
        class Resp:
            def __init__(self, body): self.body = body
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return json.dumps({"data": [{"embedding": [0.0] * 1024}]}).encode()
        def opener(req, timeout=0):
            sent["body"] = json.loads(req.data.decode()); return Resp(None)
        img_only.opener = opener
        img_only.embed("ignored text", rgbd.encode_rgb_png(np.zeros((4, 4, 3), dtype=np.uint8)) if HAVE_DEPS else b"")
        parts = [p["type"] for p in sent["body"]["inputs"][0]["content"]]
        self.assertEqual(parts, ["image_base64"])
        self.assertTrue(sent["body"]["inputs"][0]["content"][0]["image_base64"].startswith("data:image/png;base64,"))

    def test_cloud_providers_refuse_without_allow_cloud(self):
        config = Config(meta_api_key="x", voyage_key="k")
        with self.assertRaises(ConfigError):
            segmentation.MetaSam3Provider(config, allow_cloud=False).segment_png(b"", 1, 1)
        with self.assertRaises(ConfigError):
            embeddings.VoyageMultimodalProvider(config, allow_cloud=False).embed("t", b"")
        with self.assertRaises(ConfigError):
            segmentation.MetaSam3Provider(Config(), allow_cloud=True).segment_png(b"", 1, 1)
        with self.assertRaises(ConfigError):
            embeddings.VoyageMultimodalProvider(Config(), allow_cloud=True).embed("t", b"")

    def test_segmentation_key_ignores_labels_but_not_parameters(self):
        a = segmentation.segmentation_key("sha", {"provider": "p", "pred_iou_thresh": 0.88})
        b = segmentation.segmentation_key("sha", {"provider": "p", "pred_iou_thresh": 0.90})
        self.assertNotEqual(a, b)
        self.assertEqual(a, segmentation.segmentation_key("sha", {"pred_iou_thresh": 0.88, "provider": "p"}))


class ConfigAndCliTests(unittest.TestCase):
    def test_machine_auth_none_loopback_only(self):
        Config(machine_auth="none", address="localhost:8080").require_viam()
        Config(machine_auth="none", address="127.0.0.1:8080").require_viam()
        with self.assertRaises(ConfigError):
            Config(machine_auth="none", address="robot.example.viam.cloud").require_viam()
        with self.assertRaises(ConfigError):
            Config.from_env({"MACHINE_AUTH": "token"})
        with self.assertRaises(ConfigError):
            Config.from_env({"BLOCK_DIMENSIONS_MM": "75,25"})
        cfg = Config.from_env({"BLOCK_DIMENSIONS_MM": "74.5,24.9,14.8", "SETUP_ID": "mac-desk-d435"})
        self.assertEqual(cfg.block_dimensions_mm, (74.5, 24.9, 14.8))
        self.assertFalse(cfg.block_dimensions_verified)

    def test_cli_gates_cloud_and_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ)
            os.environ.pop("ALLOW_CLOUD", None)
            try:
                self.assertEqual(cli.main(["rgbd-process", "--all", "--data-dir", tmp]), 2)
                self.assertEqual(cli.main(["rgbd-capture", "--data-dir", tmp]), 2)
                os.environ["ALLOW_CLOUD"] = "true"
                self.assertEqual(cli.main(["rgbd-flush", "--allow-cloud", "--data-dir", tmp]), 2)
                self.assertEqual(cli.main(["rgbd-index", "--allow-cloud", "--data-dir", tmp]), 2)
                self.assertEqual(cli.main(["rgbd-calibration", "--data-dir", tmp]), 0)
                self.assertEqual(cli.main(["rgbd-calibration", "--data-dir", tmp, "--alignment", "verified"]), 2)
            finally:
                os.environ.clear()
                os.environ.update(env)


class MediaProtocolTests(unittest.TestCase):
    def setUp(self):
        self.backend = MemoryBackend()
        self.up = media.MediaUploader(self.backend, timedelta(hours=24))
        self.data = b"depth-bytes" * 100
        self.sha = hashlib.sha256(self.data).hexdigest()

    def test_timeout_before_manifest_then_reupload_single_manifest(self):
        self.backend.fail_once.add("write_file_partial")
        with self.assertRaises(TimeoutError):
            self.up.ensure_present(self.sha, self.data, "image/png", "depth", "lap")
        self.assertEqual(self.backend.attempts["att1"]["state"], "writing")
        gid = self.up.ensure_present(self.sha, self.data, "image/png", "depth", "lap")
        self.assertEqual(gid, "att2")
        self.assertEqual(list(self.backend.manifest), [self.sha])
        # young partial attempt is not cleaned; after the bound it is
        self.assertEqual(self.up.clean_orphans(), 0)
        self.backend.clock += timedelta(hours=25)
        self.assertEqual(self.up.clean_orphans(), 1)
        self.assertNotIn("att1", self.backend.files)
        self.assertIn("att2", self.backend.files)

    def test_timeout_after_manifest_verifies_without_second_upload(self):
        self.backend.fail_once.add("insert_manifest_after")
        with self.assertRaises(TimeoutError):
            self.up.ensure_present(self.sha, self.data, "image/png", "depth", "lap")
        gid = self.up.ensure_present(self.sha, self.data, "image/png", "depth", "lap")
        self.assertEqual(gid, "att1")
        self.assertEqual(len(self.backend.files), 1)
        self.assertEqual(self.backend.attempts["att1"]["state"], "writing")  # state update was lost; manifest is authoritative
        self.backend.clock += timedelta(hours=25)
        self.assertEqual(self.up.clean_orphans(), 0)  # referenced by manifest, never cleaned

    def test_two_writers_interleave_cleanup_and_upload(self):
        a_id = self.up.start_attempt(self.sha, self.data, "image/png", "rgb", "A")
        other = media.MediaUploader(self.backend, timedelta(hours=24))
        self.backend.clock += timedelta(hours=25)
        self.backend.attempts[a_id]["started_at"] = self.backend.clock  # A is young relative to now
        self.assertEqual(other.clean_orphans(), 0)
        b_gid = other.ensure_present(self.sha, self.data, "image/png", "rgb", "B")
        self.assertIn(a_id, self.backend.files)  # B never touched A's in-progress file
        winner = self.up.commit_attempt(a_id, self.sha, self.data, "A")
        self.assertEqual(winner, b_gid)
        self.assertEqual(self.backend.attempts[a_id]["state"], "superseded")
        self.assertNotIn(a_id, self.backend.files)
        self.assertEqual(set(self.backend.files), {b_gid})
        self.assertEqual(len(self.backend.manifest), 1)

    def test_old_attempt_is_abandoned_not_committed(self):
        a_id = self.up.start_attempt(self.sha, self.data, "image/png", "rgb", "A")
        self.backend.clock += timedelta(hours=25)
        with self.assertRaises(RuntimeError):
            self.up.commit_attempt(a_id, self.sha, self.data, "A")
        self.assertEqual(self.backend.attempts[a_id]["state"], "abandoned")
        self.assertEqual(self.backend.manifest, {})

    def test_corrupt_manifest_file_triggers_reupload(self):
        gid = self.up.ensure_present(self.sha, self.data, "image/png", "rgb", "A")
        self.backend.files[gid] = (b"corrupted", {})
        self.assertIsNone(self.up.verified_manifest(self.sha))
        with self.assertRaises(ValueError):
            self.up.ensure_present("wrong-sha", self.data, "image/png", "rgb", "A")


@unittest.skipUnless(HAVE_DEPS, "numpy and Pillow required for RGB-D tests")
class SyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        make_observation(self.dir)
        result = make_pipeline(self.dir).process("obs_test")
        self.store = candidate_store.CandidateLocalStore(self.dir, "mock")
        self.store.put_many(result["candidates"])
        self.remote = MemoryRemote()

    def tearDown(self):
        self.tmp.cleanup()

    def test_local_store_rejects_conflicts_and_mixed_modes(self):
        rows = self.store.read()
        self.store.put_many(rows)
        with self.assertRaises(ValueError):
            self.store.put_many([dict(rows[0], bbox=[0, 0, 1, 1])])
        with self.assertRaises(ValueError):
            self.store.put_many([dict(rows[0], mode="live", candidate_id="other")])
        with self.assertRaises(ValueError):
            candidate_store.validate_candidate(dict(rows[0], pickability={"status": "pickable"}))

    def test_flush_acks_are_separate_and_idempotent(self):
        first = candidate_store.sync(self.dir, "mock", self.remote, observation_ids=["obs_test"])
        self.assertEqual((first["media_uploaded"], first["observations_sent"], first["candidates_sent"], first["error"]), (2, 1, 3, None))
        self.assertTrue(all(a["remote_status"] == "synced" for a in first["artifacts"]))
        second = candidate_store.sync(self.dir, "mock", self.remote, observation_ids=["obs_test"])
        self.assertEqual((second["media_uploaded"], second["observations_sent"], second["candidates_sent"]), (0, 0, 0))
        self.assertEqual(len(self.remote.candidates), 3)
        doc = next(iter(self.remote.candidates.values()))
        self.assertEqual(len(doc["embedding"]), 1024)
        self.assertIsInstance(doc["cutout_png"], bytes)
        self.assertNotIn("embedding", self.store.read()[0])

    def test_sync_requires_explicit_ids_and_delete_refuses_after_upload(self):
        with self.assertRaises(ValueError):
            candidate_store.sync(self.dir, "mock", self.remote)
        make_observation(self.dir, obs_id="obs_other", register=False)
        self.store.put_many(make_pipeline(self.dir).process("obs_other")["candidates"])
        r = candidate_store.sync(self.dir, "mock", self.remote, observation_ids=["obs_test"])
        self.assertEqual((r["observations_sent"], r["error"]), (1, None))
        self.assertNotIn("obs_other", self.remote.observations)  # unselected capture never leaves the laptop
        self.assertFalse(candidate_store.upload_state(self.dir, "mock", "obs_other")["any"])
        self.assertTrue(candidate_store.upload_state(self.dir, "mock", "obs_test")["complete"])
        with self.assertRaises(PermissionError):
            candidate_store.delete_observation(self.dir, "mock", "obs_test")
        out = candidate_store.delete_observation(self.dir, "mock", "obs_other")
        self.assertEqual((out["removed_candidates"], out["removed_directory"]), (3, True))
        self.assertNotIn("obs_other", rgbd.ObservationStore(self.dir, "mock").list_ids())
        self.assertTrue(all(c["observation_id"] == "obs_test" for c in self.store.read()))

    def test_prune_stale_keeps_uploaded_rows(self):
        old_ids = {c["candidate_id"] for c in self.store.read()}
        candidate_store.sync(self.dir, "mock", self.remote, observation_ids=["obs_test"])  # old version uploaded
        new = make_pipeline(self.dir, emb=StubEmbedder(dimension=512)).process("obs_test")
        self.store.put_many(new["candidates"])
        self.assertEqual(self.store.prune_stale("obs_test", new["processing_version"]), 0)  # uploaded rows survive
        self.assertEqual(len(self.store.read()), 6)
        newer = make_pipeline(self.dir, emb=StubEmbedder(dimension=256)).process("obs_test")
        self.store.put_many(newer["candidates"])
        self.assertEqual(self.store.prune_stale("obs_test", newer["processing_version"]), 3)  # the un-uploaded 512 rows go
        self.assertEqual({c["processing_version"] for c in self.store.read()}, {newer["processing_version"]} | {c["processing_version"] for c in self.store.read() if c["candidate_id"] in old_ids})

    def test_pickability_label_defaults_and_overrides(self):
        rows = self.store.read()
        review = candidate_store.Review(self.dir, "mock")
        review.set_labels("obs_test", {rows[0]["candidate_id"]: "not_pickable"}, reviewed_by="tester")
        with self.assertRaises(ValueError):
            review.set_labels("obs_test", {rows[1]["candidate_id"]: "maybe"})
        review.set_ignored("obs_test", [rows[2]["candidate_id"]])  # labels survive an ignore update
        candidate_store.sync(self.dir, "mock", self.remote, observation_ids=["obs_test"])
        d0 = self.remote.candidates[rows[0]["candidate_id"]]
        d1 = self.remote.candidates[rows[1]["candidate_id"]]
        self.assertEqual((d0["pickability_label"]["status"], d0["pickability_label"]["label_source"], d0["pickable"]), ("not_pickable", "human", False))
        self.assertEqual((d1["pickability_label"]["status"], d1["pickability_label"]["label_source"], d1["pickable"]), ("pickable", "human_default", True))
        self.assertEqual(d0["pickability"]["status"], "unknown")  # heuristic field untouched
        self.assertNotIn(rows[2]["candidate_id"], self.remote.candidates)

    def test_ignored_candidates_are_never_uploaded(self):
        rows = self.store.read()
        review = candidate_store.Review(self.dir, "mock")
        review.set_ignored("obs_test", [rows[0]["candidate_id"], rows[1]["candidate_id"]], reviewed_by="tester")
        r = candidate_store.sync(self.dir, "mock", self.remote, observation_ids=["obs_test"])
        self.assertEqual((r["candidates_sent"], r["candidates_ignored"], r["error"]), (1, 2, None))
        self.assertEqual(set(self.remote.candidates), {rows[2]["candidate_id"]})
        self.assertEqual(next(iter(self.remote.candidates.values()))["review"]["reviewed_by"], "tester")
        state = candidate_store.upload_state(self.dir, "mock", "obs_test")
        self.assertTrue(state["complete"])
        self.assertEqual((state["candidates_acked"], state["candidates_total"]), (1, 1))
        # un-ignoring later makes the capture incomplete again and a resume uploads only the newly included one
        review.set_ignored("obs_test", [rows[0]["candidate_id"]])
        self.assertFalse(candidate_store.upload_state(self.dir, "mock", "obs_test")["complete"])
        r = candidate_store.sync(self.dir, "mock", self.remote, observation_ids=["obs_test"])
        self.assertEqual((r["candidates_sent"], len(self.remote.candidates)), (1, 2))
        self.assertNotIn(rows[0]["candidate_id"], self.remote.candidates)

    def test_timeout_after_commit_acks_only_after_readback(self):
        self.remote.fail_once.update({"observation_after", "candidate_after"})
        r1 = candidate_store.sync(self.dir, "mock", self.remote, observation_ids=["obs_test"])
        self.assertEqual(r1["error"], "TimeoutError")
        self.assertEqual(r1["observations_sent"], 0)
        r2 = candidate_store.sync(self.dir, "mock", self.remote, observation_ids=["obs_test"])
        self.assertEqual(r2["error"], "TimeoutError")
        self.assertEqual(r2["observations_sent"], 1)
        r3 = candidate_store.sync(self.dir, "mock", self.remote, observation_ids=["obs_test"])
        self.assertEqual((r3["error"], r3["candidates_sent"], r3["pending_candidates"]), (None, 3, 0))
        self.assertEqual(len(self.remote.observations), 1)

    def test_missing_media_then_restored(self):
        obs_dir = rgbd.ObservationStore(self.dir, "mock").directory("obs_test")
        depth = (obs_dir / "depth.png").read_bytes()
        (obs_dir / "depth.png").unlink()
        r1 = candidate_store.sync(self.dir, "mock", self.remote, observation_ids=["obs_test"])
        depth_line = next(a for a in r1["artifacts"] if a["kind"] == "depth")
        self.assertEqual((depth_line["local_status"], depth_line["remote_status"]), ("missing", "pending"))
        self.assertEqual(r1["observations_sent"], 1)  # observation metadata still uploaded; gap visible remotely
        sha = depth_line["sha256"]
        self.assertEqual(self.remote.artifact_sync[sha]["remote_status"], "pending")
        obs_before = dict(self.remote.observations["obs_test"])
        (obs_dir / "depth.png").write_bytes(depth)
        r2 = candidate_store.sync(self.dir, "mock", self.remote, observation_ids=["obs_test"])
        self.assertEqual(next(a for a in r2["artifacts"] if a["kind"] == "depth")["remote_status"], "synced")
        self.assertEqual(self.remote.artifact_sync[sha]["remote_status"], "synced")
        self.assertEqual(self.remote.observations["obs_test"], obs_before)
        # synced artifact whose local copy disappears stays synced
        (obs_dir / "depth.png").unlink()
        r3 = candidate_store.sync(self.dir, "mock", self.remote, observation_ids=["obs_test"])
        line = next(a for a in r3["artifacts"] if a["kind"] == "depth")
        self.assertEqual((line["local_status"], line["remote_status"]), ("missing", "synced"))
        self.assertEqual(self.remote.artifact_sync[sha]["remote_status"], "synced")


if __name__ == "__main__":
    unittest.main()
