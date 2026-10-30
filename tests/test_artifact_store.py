from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from robot_hackathon.artifact_store import ArtifactStore


class ArtifactTests(unittest.TestCase):
    def test_dedup_restart_and_repair(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ArtifactStore(folder)
            ref = store.put(b"saved crop", "image/png")
            self.assertEqual(ref, store.put(b"saved crop", "image/png"))
            restarted = ArtifactStore(folder)
            self.assertEqual(restarted.read(ref), b"saved crop")
            restarted.path_for(ref).write_bytes(b"corruption")
            self.assertFalse(restarted.valid(ref))
            self.assertEqual(ref, restarted.put(b"saved crop", "image/png"))
            self.assertEqual(restarted.read(ref), b"saved crop")

    def test_failed_publish_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ArtifactStore(folder)
            with patch("robot_hackathon.artifact_store.os.replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    store.put(b"new crop")
            self.assertEqual([p for p in Path(folder).rglob("*") if p.is_file()], [])

    def test_rejects_invalid_and_escaping_references(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ArtifactStore(Path(folder) / "store")
            ref = store.put(b"crop")
            for invalid in (None, {}, ref | {"path": "../../outside"},
                            ref | {"size_bytes": 0}, ref | {"sha256": "wrong"}):
                self.assertFalse(store.valid(invalid))
            outside = Path(folder) / "outside"
            outside.write_bytes(b"crop")
            store.path_for(ref).unlink()
            (store.directory / ref["path"]).symlink_to(outside)
            self.assertFalse(store.valid(ref))
