"""Snapshot segmentation via Meta Model API SAM 3.1 (Responses API). Masks are returned in source pixel coordinates.

Cache/checkpoint identity is sha256(canonical {image_sha256, effective_settings()}); a human-readable
params label is never part of any key. Segmentation proposes visible regions; it says nothing about
exposure or graspability.
"""
import asyncio
import base64
import json
import time
import urllib.request
from .config import ConfigError
from . import rgbd

PARSER_VERSION = "meta-sam-parser-0.1.0/adapter-v1"
RESPONSES_PATH = "/v1/responses"
MODEL = "sam-3.1"


def segmentation_key(image_sha256, settings):
    return rgbd.sha256_bytes(rgbd.canonical_json({"image_sha256": image_sha256, "settings": settings}).encode())


def _bbox(mask):
    import numpy as np
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def finalize_masks(masks, width, height, min_area):
    """Normalize provider output to a deterministic ordered list of mask records."""
    out = []
    for mask, score in masks:
        if mask.shape != (height, width):
            raise ValueError("Mask dimensions do not match the source image")
        area = int(mask.sum())
        bbox = _bbox(mask)
        if bbox is None or area < min_area:
            continue
        out.append({"mask": mask, "bbox": bbox, "area": area, "score": None if score is None else float(score)})
    out.sort(key=lambda m: (m["bbox"][1], m["bbox"][0], -m["area"]))
    for index, m in enumerate(out):
        m["mask_index"] = index
    return out


class SegmentationProvider:
    """Interface: effective_settings() feeds the processing identity; segment(rgb) -> [(bool mask HxW, score|None)]."""
    provider_id = "abstract"
    last_latency_ms = None

    def effective_settings(self):
        raise NotImplementedError

    def segment(self, rgb):
        raise NotImplementedError


def iter_sse(stream):
    """Yield parsed JSON objects from a server-sent-events byte stream (one per `data:` block)."""
    data_lines = []
    for raw in stream:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line == "":
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                if payload.strip() != "[DONE]":
                    yield json.loads(payload)
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        payload = "\n".join(data_lines)
        if payload.strip() != "[DONE]":
            yield json.loads(payload)


def parse_events(events, width, height):
    """Feed Responses events to meta_sam_parser and place every mask record on the full frame.

    Returns (masks: [(bool HxW, confidence|None)], info: {records, diagnostics, outcome, raw_output_chars}).
    """
    import numpy as np
    import meta_sam_parser as sam

    async def source():
        for event in events:
            yield event

    async def run():
        parsed = sam.parse_responses_stream(source(), sam.image_segmentation_format())
        async for _snapshot in parsed:
            pass
        final = parsed.final_result() if callable(parsed.final_result) else parsed.final_result
        return await final if asyncio.iscoroutine(final) or asyncio.isfuture(final) else final

    result = asyncio.run(run())
    masks = []
    for record in result.records:
        if getattr(record, "kind", None) != "mask":
            continue
        raster = np.frombuffer(sam.decode_mask_to_raster(record.mask), dtype=np.uint8)
        w, h = record.mask.width, record.mask.height
        if raster.size != w * h:
            raise ValueError("Decoded mask raster size mismatch")
        local = raster.reshape(h, w).astype(bool)
        full = np.zeros((height, width), dtype=bool)
        b = record.bounds
        if b is None:
            if (h, w) != (height, width):
                raise ValueError("Full-frame mask dimensions do not match the source image")
            full = local
        else:
            left, top, right, bottom = int(b.left), int(b.top), int(b.right), int(b.bottom)
            if (bottom - top, right - left) != (h, w) or left < 0 or top < 0 or right > width or bottom > height:
                raise ValueError("Mask bounds do not fit the source image")
            full[top:bottom, left:right] = local
        masks.append((full, None if record.confidence is None else float(record.confidence)))
    info = {"records": len(result.records), "masks": len(masks),
            "diagnostics": [getattr(d, "code", str(d)) for d in result.diagnostics],
            "outcome": getattr(result.outcome, "status", None), "raw_output_chars": len(result.raw_output or "")}
    return masks, info


class MetaSam3Provider(SegmentationProvider):
    """SAM 3.1 on Meta Model API: POST {base}/v1/responses, model sam-3.1, text prompt + input_image, streamed SSE.

    Contract from https://dev.meta.ai/docs/sam/overview and the official `meta_sam_parser` package.
    Verified status is tracked in docs/PROVIDERS.md.
    """
    provider_id = f"meta-model-api/{MODEL}"

    def __init__(self, config, allow_cloud=False, prompt="wooden block", mask_encoding="one_bit", include_confidence=True,
                 min_area=50, timeout=120, opener=None):
        self.config = config
        self.allow_cloud = allow_cloud
        self.prompt = prompt
        self.mask_encoding = mask_encoding
        self.include_confidence = bool(include_confidence)
        self.min_area = int(min_area)
        self.timeout = timeout
        self.opener = opener or urllib.request.urlopen
        self.calls = 0
        self.last_latency_ms = None
        self.last_info = None

    def effective_settings(self):
        return {"provider": self.provider_id, "model_version": MODEL, "prompt": self.prompt, "mask_encoding": self.mask_encoding,
                "include_confidence": self.include_confidence, "image_format": "png", "min_area": self.min_area,
                "parser_version": PARSER_VERSION}

    @property
    def url(self):
        return self.config.meta_base_url.rstrip("/") + RESPONSES_PATH

    def request_body(self, image_png):
        metadata = {"mask_encoding": self.mask_encoding}
        if self.include_confidence:
            metadata["include_confidence"] = "true"
        return {"model": MODEL, "stream": True,
                "input": [{"type": "message", "role": "user",
                           "content": [{"type": "input_text", "text": self.prompt},
                                       {"type": "input_image",
                                        "image_url": "data:image/png;base64," + base64.b64encode(image_png).decode()}]}],
                "metadata": metadata}

    def segment_png(self, image_png, width, height):
        if not self.allow_cloud:
            raise ConfigError("Hosted segmentation sends the image to Meta; requires ALLOW_CLOUD=true and --allow-cloud")
        self.config.require_meta()
        self.calls += 1
        started = time.monotonic()
        req = urllib.request.Request(self.url, data=json.dumps(self.request_body(image_png)).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self.config.meta_api_key}",
                                              "Content-Type": "application/json", "Accept": "text/event-stream"})
        with self.opener(req, timeout=self.timeout) as resp:
            events = list(iter_sse(resp))
        masks, info = parse_events(events, width, height)
        self.last_latency_ms = int((time.monotonic() - started) * 1000)
        self.last_info = {**info, "latency_ms": self.last_latency_ms}
        return masks

    def segment(self, rgb):
        return self.segment_png(rgbd.encode_rgb_png(rgb), rgb.shape[1], rgb.shape[0])
