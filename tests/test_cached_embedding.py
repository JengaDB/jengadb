from pathlib import Path
import tempfile
import unittest

from robot_hackathon.cached_embedding import CachedEmbeddingProvider
from robot_hackathon.result_cache import ResultCache


class Provider:
    encoder_version = "test-v1"
    dimension = 2
    url = "https://example.invalid/embeddings"

    def __init__(self):
        self.calls = 0
        self.settings = {"provider": "test", "preprocessing": "v1"}
        self.result = [0.25, 0.75]

    def effective_settings(self):
        return dict(self.settings)

    def embed(self, text, image):
        self.calls += 1
        return self.result


class CachedEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "cache.sqlite3"
        self.provider = Provider()
        self.cached = CachedEmbeddingProvider(self.provider, ResultCache(self.path))

    def test_restart_reuses_exact_request_and_preserves_settings(self):
        result = self.cached.embed("geometry", b"image")
        self.assertFalse(self.cached.last_cache_hit)
        restarted = CachedEmbeddingProvider(self.provider, ResultCache(self.path))
        self.assertEqual(restarted.embed("geometry", b"image"), result)
        self.assertTrue(restarted.last_cache_hit)
        self.assertEqual(self.provider.calls, 1)
        self.assertEqual(restarted.effective_settings(), self.provider.effective_settings())

    def test_changed_request_model_settings_endpoint_and_dimension_miss(self):
        self.cached.embed("geometry", b"image")
        self.cached.embed("different geometry", b"image")
        self.cached.embed("geometry", b"different image")
        self.provider.encoder_version = "test-v2"
        self.cached.embed("geometry", b"image")
        self.provider.settings["preprocessing"] = "v2"
        self.cached.embed("geometry", b"image")
        self.provider.url = "https://other.invalid/embeddings"
        self.cached.embed("geometry", b"image")
        self.provider.dimension = 3
        self.provider.result = [0.2, 0.3, 0.5]
        self.cached.embed("geometry", b"image")
        self.assertEqual(self.provider.calls, 7)

    def test_invalid_vectors_are_not_cached(self):
        for result in ([1], [float("nan"), 1], [float("inf"), 1], [True, 1], ["1", "2"]):
            self.provider.result = result
            with self.assertRaises(ValueError):
                self.cached.embed("geometry", b"image")
        self.provider.result = [0.25, 0.75]
        self.cached.embed("geometry", b"image")
        self.assertFalse(self.cached.last_cache_hit)
        self.assertEqual(self.provider.calls, 6)

    def test_image_only_provider_forwards_uses_text_and_ignores_text_in_key(self):
        self.provider.uses_text = False
        cached = CachedEmbeddingProvider(self.provider, ResultCache(self.path))
        self.assertFalse(cached.uses_text)
        cached.embed("geometry", b"image")
        cached.embed("different geometry", b"image")
        self.assertTrue(cached.last_cache_hit)
        self.assertEqual(self.provider.calls, 1)
        # Flipping the contract changes the key: a text-aware provider never reads an image-only entry.
        self.provider.uses_text = True
        self.assertTrue(cached.uses_text)
        cached.embed("geometry", b"image")
        self.assertFalse(cached.last_cache_hit)
        self.assertEqual(self.provider.calls, 2)

    def test_modes_do_not_share_cache(self):
        for mode in ("mock", "live"):
            cache = ResultCache(Path(self.tmp.name) / mode / "provider-cache" / "results.sqlite3")
            CachedEmbeddingProvider(self.provider, cache).embed("geometry", b"image")
        self.assertEqual(self.provider.calls, 2)
