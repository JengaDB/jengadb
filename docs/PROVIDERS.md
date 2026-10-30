# Hosted provider contracts (RGB-D acquisition and embeddings)

Evidence as of 29 October 2026, recorded on the SKY-66 branch. Adapters are tested against sanitized real responses in `fixtures/providers/`. No credentials appear here or in fixtures.

## Meta Model API, SAM 3.1 — not on this branch

The segmentation adapter, its fixture and `rgbd-verify-segmentation` live on the SKY-66 branch (PR #5) and were deliberately not carried here. This branch covers capture, storage, the Atlas connection and cached embeddings only.

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
