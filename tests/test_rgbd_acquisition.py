"""Offline tests for RGB-D acquisition, media/Atlas connection and cached embeddings.

Extracted from the SKY-66 test module; the processing tests (segmentation, exposure, cutouts, candidates) stay there.
Synthetic scene, stub provider and in-memory remote live here only. Set RGBD_TESTS_REQUIRED=1 to fail (instead of
skip) when numpy/Pillow are unavailable.
"""
import hashlib
import json
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

from robot_hackathon import rgbd, embeddings, media, observation_sync, cli  # noqa: E402
from robot_hackathon.cached_embedding import CachedEmbeddingProvider  # noqa: E402
from robot_hackathon.result_cache import ResultCache  # noqa: E402
from robot_hackathon.config import Config, ConfigError  # noqa: E402

UTC = timezone.utc
BLOCKS = (("A", (20, 30, 110, 70), (220, 40, 40), 600), ("C", (150, 90, 300, 140), (40, 60, 220), 650),
          ("B", (200, 60, 260, 170), (40, 200, 60), 500))


def scene(invalid_patch=True):
    rgb = np.full((240, 320, 3), (90, 90, 90), dtype=np.uint8)
    depth = np.full((240, 320), 900, dtype=np.uint16)
    for _, (x0, y0, x1, y1), color, mm in BLOCKS:
        rgb[y0:y1, x0:x1] = color
        depth[y0:y1, x0:x1] = mm
    if invalid_patch:
        depth[200:230, 10:60] = 0
    return rgb, depth


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
        self.observations, self.artifact_sync = {}, {}
        self.fail_once = set()

    def _fail(self, op):
        if op in self.fail_once:
            self.fail_once.remove(op)
            raise TimeoutError(op)

    def upsert_observation(self, obs):
        self.observations.setdefault(obs["observation_id"], obs)
        self._fail("observation_after")
        assert self.observations[obs["observation_id"]] == obs

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
        self.assertEqual(obs["alignment"]["status"], "verified")  # recorded verdict is preserved

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


class ProviderTests(unittest.TestCase):
    def test_voyage_parser_fixture(self):
        fixture = json.loads((Path(__file__).resolve().parents[1] / "fixtures/providers/voyage/response.json").read_text())
        vector = embeddings.parse_voyage_response(fixture, len(fixture["data"][0]["embedding"]))
        self.assertEqual(len(vector), len(fixture["data"][0]["embedding"]))
        with self.assertRaises(ValueError):
            embeddings.parse_voyage_response(fixture, 7)
        with self.assertRaises(ValueError):
            embeddings.parse_voyage_response({"data": []}, 1024)

    @unittest.skipUnless(HAVE_DEPS, "numpy and Pillow required")
    def test_voyage_default_is_image_only_and_cache_ignores_text(self):
        cfg = Config(voyage_key="k")
        img_only = embeddings.VoyageMultimodalProvider(cfg, allow_cloud=True)
        fused = embeddings.VoyageMultimodalProvider(cfg, allow_cloud=True, input_parts=("text", "image"))
        self.assertFalse(img_only.uses_text)
        self.assertTrue(fused.uses_text)
        self.assertNotEqual(img_only.encoder_version, fused.encoder_version)
        sent = []
        class Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return json.dumps({"data": [{"embedding": [0.0] * 1024}]}).encode()
        def opener(req, timeout=0):
            sent.append(json.loads(req.data.decode())); return Resp()
        img_only.opener = opener
        png = rgbd.encode_rgb_png(np.zeros((4, 4, 3), dtype=np.uint8))
        with tempfile.TemporaryDirectory() as tmp:
            cached = CachedEmbeddingProvider(img_only, ResultCache(Path(tmp) / "cache.sqlite3"))
            self.assertFalse(cached.uses_text)
            cached.embed("ignored text", png)
            cached.embed("other ignored text", png)
        self.assertEqual(len(sent), 1)
        parts = [p["type"] for p in sent[0]["inputs"][0]["content"]]
        self.assertEqual(parts, ["image_base64"])
        self.assertTrue(sent[0]["inputs"][0]["content"][0]["image_base64"].startswith("data:image/png;base64,"))

    def test_cloud_provider_refuses_without_allow_cloud(self):
        with self.assertRaises(ConfigError):
            embeddings.VoyageMultimodalProvider(Config(voyage_key="k"), allow_cloud=False).embed("t", b"")
        with self.assertRaises(ConfigError):
            embeddings.VoyageMultimodalProvider(Config(), allow_cloud=True).embed("t", b"")


class ConfigAndCliTests(unittest.TestCase):
    def test_machine_auth_none_loopback_only(self):
        Config(machine_auth="none", address="localhost:8080").require_viam()
        Config(machine_auth="none", address="127.0.0.1:8080").require_viam()
        with self.assertRaises(ConfigError):
            Config(machine_auth="none", address="robot.example.viam.cloud").require_viam()
        with self.assertRaises(ConfigError):
            Config.from_env({"MACHINE_AUTH": "token"})
        with self.assertRaises(ConfigError):
            Config.from_env({"MEDIA_ATTEMPT_MAX_AGE_HOURS": "0"})
        cfg = Config.from_env({"SETUP_ID": "mac-desk-d435", "MONGO_DB_MODELS_ENDPOINT_URL": "ai.mongodb.com"})
        self.assertEqual((cfg.setup_id, cfg.voyage_base_url), ("mac-desk-d435", "https://ai.mongodb.com"))
        with self.assertRaises(ConfigError):
            Config(uri="mongodb://h/x", observations_collection="bad name").require_mongo()

    def test_cli_gates_cloud_and_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ)
            os.environ.pop("ALLOW_CLOUD", None)
            try:
                self.assertEqual(cli.main(["rgbd-capture", "--data-dir", tmp]), 2)
                self.assertEqual(cli.main(["rgbd-verify-embedding", "--data-dir", tmp]), 2)
                os.environ["ALLOW_CLOUD"] = "true"
                self.assertEqual(cli.main(["rgbd-flush", "--allow-cloud", "--data-dir", tmp]), 2)
                self.assertEqual(cli.main(["rgbd-index", "--allow-cloud", "--data-dir", tmp]), 2)
                self.assertEqual(cli.main(["rgbd-calibration", "--data-dir", tmp]), 0)
                self.assertEqual(cli.main(["rgbd-calibration", "--data-dir", tmp, "--alignment", "verified"]), 2)
                self.assertEqual(cli.main(["rgbd-status", "--data-dir", tmp]), 0)
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
        self.assertEqual(self.backend.attempts["att1"]["state"], "writing")
        self.backend.clock += timedelta(hours=25)
        self.assertEqual(self.up.clean_orphans(), 0)

    def test_two_writers_interleave_cleanup_and_upload(self):
        a_id = self.up.start_attempt(self.sha, self.data, "image/png", "rgb", "A")
        other = media.MediaUploader(self.backend, timedelta(hours=24))
        self.backend.clock += timedelta(hours=25)
        self.backend.attempts[a_id]["started_at"] = self.backend.clock
        self.assertEqual(other.clean_orphans(), 0)
        b_gid = other.ensure_present(self.sha, self.data, "image/png", "rgb", "B")
        self.assertIn(a_id, self.backend.files)
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
class ObservationSyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        make_observation(self.dir)
        make_observation(self.dir, obs_id="obs_other", register=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_requires_explicit_ids(self):
        with self.assertRaises(ValueError):
            observation_sync.sync_observations(self.dir, "mock", MemoryRemote())

    def test_timeout_then_retry_is_idempotent(self):
        remote = MemoryRemote()
        remote.fail_once.add("observation_after")
        failed = observation_sync.sync_observations(self.dir, "mock", remote, observation_ids=["obs_test", "obs_other"])
        self.assertEqual(failed["error"], "TimeoutError")
        self.assertEqual(failed["pending_observations"], 2)  # ack recorded only after read-back succeeds
        retried = observation_sync.sync_observations(self.dir, "mock", remote, observation_ids=["obs_test", "obs_other"])
        self.assertIsNone(retried["error"])
        self.assertEqual(retried["pending_observations"], 0)
        self.assertEqual(len(remote.observations), 2)
        # Identical bytes across observations share one GridFS copy (one manifest entry per distinct artifact).
        self.assertEqual(len(remote.backend.manifest), 2)
        self.assertEqual(failed["media_uploaded"], 2)
        repeated = observation_sync.sync_observations(self.dir, "mock", remote, observation_ids=["obs_test", "obs_other"])
        self.assertEqual((repeated["media_uploaded"], repeated["observations_sent"]), (0, 0))
        acks = json.loads((self.dir / "mock" / "acks-rgbd-memtarget.json").read_text())
        self.assertEqual(sorted(acks["observations"]), ["obs_other", "obs_test"])
        self.assertEqual(acks["candidates"], [])

    def test_missing_local_artifact_is_reported_pending(self):
        store = rgbd.ObservationStore(self.dir, "mock")
        (store.directory("obs_test") / "depth.png").unlink()
        remote = MemoryRemote()
        result = observation_sync.sync_observations(self.dir, "mock", remote, observation_ids=["obs_test"])
        self.assertIsNone(result["error"])
        statuses = {a["kind"]: a["remote_status"] for a in result["artifacts"]}
        self.assertEqual(statuses, {"rgb": "synced", "depth": "pending"})
        self.assertEqual(result["observations_sent"], 1)
