"""Candidate context embeddings: Voyage multimodal-3.5 via the MongoDB AI endpoint.

Providers expose effective_settings() (model, dimension, input_type, text template id + content hash,
image preprocessing) which feeds the processing identity. encoder_version is a human-readable label.
"""
import base64
import hashlib
import json
import urllib.request
from .config import ConfigError
from . import rgbd

EMBED_PATH = "/v1/multimodalembeddings"
TEXT_TEMPLATES = {
    "geom-v1": ("Jenga block candidate from setup {setup_id}, phase pick. Exposure {exposure}. "
                "Bounding box {w}x{h} px at ({x0},{y0}). {depth_text}"),
}


def geometry_text(candidate_fields, template_id="geom-v1"):
    st = candidate_fields.get("depth_stats")
    if st and st.get("median_mm") is not None and candidate_fields.get("registration_verified"):
        depth_text = (f"Median depth {st['median_mm']:.0f} mm, p10 {st['p10_mm']:.0f} mm, p90 {st['p90_mm']:.0f} mm, "
                      f"valid depth fraction {st['valid_fraction']:.2f}.")
    else:
        depth_text = "Depth statistics unavailable (registration not verified)."
    ct = candidate_fields["crop_transform"]
    return TEXT_TEMPLATES[template_id].format(setup_id=candidate_fields["setup_id"],
                                             exposure=candidate_fields["exposure"]["status"], w=ct["width"],
                                             h=ct["height"], x0=ct["x0"], y0=ct["y0"], depth_text=depth_text)


def template_sha256(template_id):
    return hashlib.sha256(TEXT_TEMPLATES[template_id].encode()).hexdigest()


class EmbeddingProvider:
    """Interface: effective_settings() feeds the processing identity; embed(text, image_png) -> list[float]."""
    encoder_version = "abstract"
    dimension = 1024

    def effective_settings(self):
        raise NotImplementedError

    def embed(self, text, image_png):
        raise NotImplementedError


class VoyageMultimodalProvider(EmbeddingProvider):
    encoder_version = "voyage-multimodal-3.5/ctx-v1"

    def __init__(self, config, allow_cloud=False, model="voyage-multimodal-3.5", dimension=1024, template_id="geom-v1",
                 resize_max_px=1024, opener=None):
        self.config, self.allow_cloud, self.model, self.dimension = config, allow_cloud, model, dimension
        self.template_id, self.resize_max_px = template_id, resize_max_px
        self.opener = opener or urllib.request.urlopen
        self.calls = 0

    def effective_settings(self):
        return {"provider": "voyage", "model": self.model, "dimension": self.dimension, "input_type": "document",
                "text_template_id": self.template_id, "text_template_sha256": template_sha256(self.template_id),
                "image_preprocessing": {"resize_max_px": self.resize_max_px, "format": "png"}}

    def _prepare_image(self, image_png):
        array, _ = rgbd.decode_png(image_png)
        h, w = array.shape[:2]
        if max(h, w) > self.resize_max_px:
            from PIL import Image
            import io
            image = Image.fromarray(array)
            image.thumbnail((self.resize_max_px, self.resize_max_px))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()
        return image_png

    @property
    def url(self):
        return self.config.voyage_base_url.rstrip("/") + EMBED_PATH

    def _gate(self):
        if not self.allow_cloud:
            raise ConfigError("Embedding sends data to the hosted Voyage endpoint; requires ALLOW_CLOUD=true and --allow-cloud")
        self.config.require_voyage()

    def _post(self, content):
        self._gate()
        self.calls += 1
        body = {"inputs": [{"content": content}], "model": self.model, "input_type": "document"}
        if self.dimension != 1024:
            body["output_dimension"] = self.dimension
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self.config.voyage_key}",
                                              "Content-Type": "application/json"})
        with self.opener(req, timeout=60) as resp:
            return json.loads(resp.read().decode())

    def embed(self, text, image_png):
        self._gate()
        content = [{"type": "text", "text": text},
                   {"type": "image_base64", "image_base64": base64.b64encode(self._prepare_image(image_png)).decode()}]
        return parse_voyage_response(self._post(content), self.dimension)

    def verify(self, text="JengaDB endpoint verification"):
        """Credential/endpoint/dimension check with a text-only input; sends no image."""
        payload = self._post([{"type": "text", "text": text}])
        vector = parse_voyage_response(payload, self.dimension)
        return {"model": payload.get("model"), "dimension": len(vector), "usage": payload.get("usage"), "url": self.url}


def parse_voyage_response(payload, dimension):
    data = payload.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0].get("embedding"), list):
        raise ValueError("Voyage response lacks data[0].embedding")
    vector = [float(v) for v in data[0]["embedding"]]
    if len(vector) != dimension:
        raise ValueError(f"Embedding dimension {len(vector)} does not match the configured {dimension}")
    return vector
