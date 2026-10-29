"""Reuse exact embedding requests without changing provider or candidate schemas."""
import math
from .embeddings import EmbeddingProvider


class CachedEmbeddingProvider(EmbeddingProvider):
    def __init__(self, provider, cache):
        self.provider, self.cache = provider, cache
        self.last_cache_hit = None

    @property
    def encoder_version(self):
        return self.provider.encoder_version

    @property
    def dimension(self):
        return self.provider.dimension

    def effective_settings(self):
        # Caching changes execution, not the processing identity or output format.
        return self.provider.effective_settings()

    def _validate(self, vector):
        if not isinstance(vector, (list, tuple)) or len(vector) != self.dimension:
            raise ValueError("Embedding dimension mismatch")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector):
            raise ValueError("Embedding must contain finite numbers")
        return list(vector)

    def embed(self, text, image_png):
        self.last_cache_hit = None
        settings = {"provider_settings": self.effective_settings(), "dimension": self.dimension,
                    "endpoint": getattr(self.provider, "url", None), "adapter_version": 1}
        vector, hit, _ = self.cache.get_or_compute(
            operation="vision.context-embedding", inputs=[text.encode("utf-8"), image_png],
            model_version=self.encoder_version, settings=settings,
            compute=lambda: self._validate(self.provider.embed(text, image_png)),
        )
        vector = self._validate(vector)
        self.last_cache_hit = hit
        return vector
