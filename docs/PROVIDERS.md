# Hosted provider contracts (RGB-D pipeline)

Evidence as of 29 October 2026. Adapters are tested against sanitized real responses in `fixtures/providers/`. No credentials appear here or in fixtures.

## Meta Model API, SAM 3.1 — verified

- Endpoint: `POST https://api.meta.ai/v1/responses` (Responses API), bearer key from `META_API_KEY`; host override `META_API_BASE_URL`.
- Model `sam-3.1`. One user message with an `input_text` noun phrase (default prompt `wooden block`, configurable, part of the processing identity) and one `input_image`; a PNG `data:` URI is accepted. `metadata.mask_encoding: one_bit`, `metadata.include_confidence: "true"`.
- Response: streamed SSE (`response.created`, `response.in_progress`, `response.output_item.added`, `response.content_part.added`, `response.output_text.delta`, `response.output_text.done`, `response.content_part.done`, `response.output_item.done`, `response.completed`). The `output_text` lane carries a special-token line with a box and a mask per object; zero matches is a completed response with an empty lane.
- Decoding: official `meta_sam_parser` 0.1.0 (`parse_responses_stream` + `image_segmentation_format`, `decode_mask_to_raster`). Mask rasters are box-local; the adapter places each on the full frame using `record.bounds` and rejects bounds that do not fit the source image.
- Verification run: `run.py rgbd-verify-segmentation --allow-cloud --image <real photo> --prompt mountain` on the 214x130 macOS "Sonoma Horizon" thumbnail returned 1 mask (1260 px, confidence 0.54) in 2.2 s. Fixture: `fixtures/providers/meta-sam3/responses-stream.json` (full event list). Not yet run on an actual block pile; prompt quality for wooden blocks is unmeasured.
- References: https://dev.meta.ai/docs/sam/overview, https://dev.meta.ai/docs/sam/reading-segmentation, https://dev.meta.ai/docs/sam/client-libraries

## Voyage multimodal embeddings (MongoDB hosted endpoint) — verified

- Endpoint: `POST https://ai.mongodb.com/v1/multimodalembeddings`, bearer key from `MONGO_DB_MODELS_ENDPOINT_KEY` (alias `VOYAGE_API_KEY`); host from `MONGO_DB_MODELS_ENDPOINT_URL`.
- Model `voyage-multimodal-3.5`, default 1024 dimensions, `input_type: document` for stored candidates (queries would use `query`).
- Default input is the **context crop image only** (`encoder_version voyage-multimodal-3.5/img-v2`); the geometry description is stored as `description_text` metadata and does not enter the vector. The fused text+image variant remains available (`input_parts=("text","image")`, `ctx-v2`) and gets its own `encoder_version`/index filter value, so the two are never mixed.
- Request: `{"inputs":[{"content":[{"type":"text","text":...},{"type":"image_base64","image_base64":"data:image/png;base64,..."}]}],"model":...,"input_type":"document"}`. The image value must be a data URL; bare base64 returns HTTP 400 (verified on a real crop, 29 Oct 2026).
- Response: `{"object":"list","data":[{"object":"embedding","embedding":[...1024 floats],"index":0}],"model":...,"usage":{text_tokens,image_pixels,video_pixels,total_tokens}}`.
- Verification run: `run.py rgbd-verify-embedding --allow-cloud` (text-only) returned `voyage-multimodal-3.5`, 1024 dims. Fixture `fixtures/providers/voyage/response.json` (embedding truncated). Image path verified on a real 160x122 context crop: 1024 dims, 89 tokens.
- Reference: https://www.mongodb.com/docs/voyageai/models/multimodal-embeddings/

## Replicate — not used

Dropped on 29 October 2026 in favour of the Meta Model API.
