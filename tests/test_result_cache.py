import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event

from robot_hackathon.result_cache import CacheMiss, CacheOutput, ResultCache, fingerprint


class CacheTests(unittest.TestCase):
    def test_reuse_and_changed_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cache.sqlite3"
            calls = []
            def compute():
                calls.append(1)
                return {"synthetic": True}
            args = dict(operation="test", inputs=[b"image"], model_version="v1",
                        settings={"threshold": 0.5}, compute=compute)
            self.assertFalse(ResultCache(path).get_or_compute(**args)[1])
            self.assertTrue(ResultCache(path).get_or_compute(**args)[1])
            for change in ({"inputs": [b"different"]}, {"model_version": "v2"},
                           {"settings": {"threshold": 0.9}}, {"operation": "other"}):
                self.assertFalse(ResultCache(path).get_or_compute(**(args | change))[1])
            self.assertEqual(len(calls), 5)

    def test_failed_processing_is_not_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = ResultCache(Path(folder) / "cache.sqlite3")
            args = dict(operation="test", inputs=[b"a"], model_version="v1", settings={})
            def fail():
                raise RuntimeError("processing failed")
            with self.assertRaises(RuntimeError):
                cache.get_or_compute(**args, compute=fail)
            self.assertFalse(cache.get_or_compute(**args, compute=lambda: {})[1])

    def test_separate_process_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "fixture.ppm"
            image.write_bytes(b"P6\n1 1\n255\n\xff\xff\xff")
            cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/cache_demo.py"),
                   str(image), "--cache", str(Path(folder) / "cache.sqlite3")]
            first = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
            second = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
            self.assertIn("CACHE MISS", first)
            self.assertIn("CACHE HIT", second)
            self.assertNotIn("Running PRETEND", second)
            hit = subprocess.run(cmd + ["--lookup-only"], check=True, capture_output=True, text=True)
            self.assertIn("CACHE HIT", hit.stdout)
            miss = subprocess.run(cmd + ["--lookup-only", "--stage", "tower"], capture_output=True, text=True)
            self.assertEqual(miss.returncode, 2)
            self.assertNotIn("Running PRETEND", miss.stdout)

    def test_missing_and_corrupt_artifacts_invalidate_result(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = ResultCache(Path(folder) / "cache.sqlite3")
            args = dict(operation="segment", inputs=[b"image"], model_version="v1", settings={})
            calls = []
            def compute():
                calls.append(1)
                ref = cache.artifacts.put(b"crop", "image/png")
                return CacheOutput({"crop": ref}, (ref,))
            value, _, _ = cache.get_or_compute(**args, compute=compute)
            restarted = ResultCache(cache.path)
            self.assertTrue(restarted.get_or_compute(**args)[1])
            restarted.artifacts.path_for(value["crop"]).unlink()
            with self.assertRaises(CacheMiss):
                restarted.get_or_compute(**args)
            self.assertFalse(restarted.get_or_compute(**args, compute=compute)[1])
            restarted.artifacts.path_for(value["crop"]).write_bytes(b"bad!")
            self.assertFalse(restarted.get_or_compute(**args, compute=compute)[1])
            self.assertEqual(len(calls), 3)

    def test_invalid_outputs_are_not_published(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = ResultCache(Path(folder) / "cache.sqlite3")
            args = dict(operation="test", inputs=[b"a"], model_version="v1", settings={})
            ref = cache.artifacts.put(b"crop")
            cache.artifacts.path_for(ref).unlink()
            with self.assertRaises(ValueError):
                cache.store(**args, value={}, artifacts=[ref])
            with self.assertRaises(ValueError):
                cache.store(**args, value={"score": float("nan")})
            self.assertIsNone(cache.lookup(**args))
            first = cache.store(**args, value={"score": 1})
            self.assertEqual(cache.store(**args, value={"score": 2}), first)

    def test_legacy_results_and_corrupt_json(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = ResultCache(Path(folder) / "cache.sqlite3")
            args = dict(operation="legacy", inputs=[b"a"], model_version="v1", settings={})
            key, request = fingerprint(**args)
            db = cache.connect()
            try:
                with db:
                    db.execute("INSERT INTO results VALUES (?, ?, ?, ?)", (key, request, '{"old": true}', "old"))
                self.assertEqual(cache.get_or_compute(**args)[0], {"old": True})
                with db:
                    db.execute("UPDATE results SET result = ? WHERE key = ?", ('{"score": NaN}', key))
                self.assertIsNone(cache.lookup(**args))
            finally:
                db.close()

    def test_processing_does_not_hold_writer_lock_and_first_result_wins(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = ResultCache(Path(folder) / "cache.sqlite3")
            args = dict(operation="test", inputs=[b"a"], model_version="v1", settings={})
            processing, finish = Event(), Event()
            def slow():
                processing.set()
                if not finish.wait(10):
                    raise RuntimeError("test timed out")
                return {"winner": "slow"}
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(cache.get_or_compute, **args, compute=slow)
                try:
                    self.assertTrue(processing.wait(5))
                    cache.store(**args, value={"winner": "fast"})
                finally:
                    finish.set()
                self.assertEqual(future.result(timeout=5)[0], {"winner": "fast"})

    def test_key_order_and_input_validation(self):
        args = dict(operation="test", inputs=[b"a", b"b"], model_version="v1", settings={"x": 1, "y": 2})
        self.assertEqual(fingerprint(**args), fingerprint(**(args | {"settings": {"y": 2, "x": 1}})))
        self.assertNotEqual(fingerprint(**args), fingerprint(**(args | {"inputs": [b"b", b"a"]})))
        for change in ({"inputs": []}, {"inputs": ["not bytes"]}, {"operation": ""},
                       {"model_version": ""}, {"settings": {1: "ambiguous"}},
                       {"settings": {"score": float("nan")}}):
            with self.assertRaises(ValueError):
                fingerprint(**(args | change))
