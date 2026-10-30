# Cache integration: acquisition, connection and cached embeddings on main

This branch (`cache-integration-on-main`) is cut from `main`. It replaces the stacked layout of PR #4, which was
based on the SKY-66 branch (PR #5) and inherited all of its processing code. PR #5 is not merged; only its
acquisition and connection modules are copied here, unchanged where possible.

## What is reused from SKY-66 (PR #5)

| Module | Role | Taken as-is |
| --- | --- | --- |
| `rgbd.py` | Observation store, calibration registry, PNG/depth codecs, hashing | yes |
| `capture.py` | Read-only Viam `camera.get_images`, overlay rendering | yes |
| `media.py` | `MediaBackend` protocol and `MediaUploader` (attempt/manifest/orphan protocol) | yes |
| `rgbd_mongo.py` | `RgbdMongoStore` (unique + vector indexes, insert-only records with read-back, GridFS backend) | yes |
| `embeddings.py` | `EmbeddingProvider` base, hosted Voyage client, geometry text template | yes |
| `config.py` | `MACHINE_AUTH`, `SETUP_ID`, `CAMERA_*`, Voyage endpoint/key, RGB-D collection names, media attempt age | subset |
| `cli.py` | Additive `rgbd-*` dispatch hook | yes |
| `configs/viam-local-mac.json`, `fixtures/providers/voyage/response.json`, `docs/PROVIDERS.md` | Local viam-server config, sanitized Voyage fixture, provider contract notes | yes (SAM section removed) |

Not taken: `segmentation.py`, `exposure.py`, `cutouts.py`, `pipeline.py`, `candidate_store.py`, `webapp.py`,
the Meta SAM fixture and config, `BLOCK_DIMENSIONS_*`, and the processing commands (`rgbd-process`, `rgbd-gallery`,
`rgbd-search`, `rgbd-ui`, `rgbd-delete`, `rgbd-verify-segmentation`).

## What is new on this branch

- `observation_sync.py`: the media -> observation -> artifact_sync stages of SKY-66's `candidate_store.sync`, without
  candidates or review. Same `acks-rgbd-<target>.json` layout and lock file, so a candidate stage can resume later.
- `rgbd_cli.py`: `rgbd-calibration`, `rgbd-overlay`, `rgbd-capture`, `rgbd-status`, `rgbd-verify-embedding`,
  `rgbd-index`, `rgbd-flush` (observations only). `embedding_provider()` is the single constructor for the cached
  Voyage client and is the integration point for any future embedding consumer.
- `cached_embedding.py` (from PR #4) now forwards `uses_text` and keys image-only requests on the image bytes alone;
  text-aware and image-only providers never share a cache entry.
- `tests/test_rgbd_acquisition.py`: observation store, calibration registry, media protocol, Voyage parser and
  image-only request shape through the cache, config gates, CLI gates, observation sync with timeout/retry.

## Cache identity

`ResultCache` (SQLite, `<data-dir>/<mode>/provider-cache/results.sqlite3`) is keyed on exact input bytes (text and
PNG, or PNG only when `uses_text` is false), `encoder_version`, dimension, effective provider settings, endpoint and
adapter version. Mock and live use separate directories. Provider errors and invalid vectors are not cached. Similar
images are not cache hits.

## Reproduce the offline checks

```sh
python3 scripts/team.py check                                   # no installs; RGB-D tests skip without numpy/Pillow
.venv/bin/python -m pip install -e '.[rgbd]'
RGBD_TESTS_REQUIRED=1 .venv/bin/python -m unittest discover -s tests
```

## Limitations

No live capture, provider call or Atlas write was performed from this branch. Concurrent cache misses can invoke
the provider twice; there is no eviction. `rgbd-flush` uploads observations and media only; there is no candidate
record until a processing stage exists on `main`. Reconcile PR #3 (standalone cache) and PR #4 (stacked integration)
with this branch before any of them merge; they overlap in `result_cache.py` and `artifact_store.py`.
