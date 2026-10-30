# CACHE-01: Exact-result cache with RGB-D acquisition and Atlas connection on main

Status and owner: see board.json
Role: data
Branch: cache-integration-on-main (cut from main at f8d9a94)
Supersedes: PR #4 (`codex/cache-rgbd-integration`, stacked on the SKY-66 branch). PR #5 (SKY-66) is not merged;
its acquisition and connection modules are copied here. See docs/CACHE_INTEGRATION.md for the exact split.

## Goal and acceptance

Land the cache primitives and the embedding adapter on `main` together with the parts of SKY-66 needed to capture
RGB-D observations, store them locally with hashes, and upload media and observation records to Atlas, without
carrying the segmentation/exposure/candidate processing or the web UI. Identical embedding requests reuse one
vector; image-only providers are keyed on image bytes alone. Offline tests cover restart, changed settings, invalid
vectors, the media ownership protocol, and observation sync with a simulated timeout and idempotent retry.

## Handoff

Changes: from PR #4, `result_cache.py`, `artifact_store.py`, `cached_embedding.py`, `scripts/cache_demo.py` and their
tests; `cached_embedding.py` gains `uses_text` forwarding and image-only keys. From PR #5, `rgbd.py`, `capture.py`,
`media.py`, `rgbd_mongo.py`, `embeddings.py`, the additive `cli.py` hook, `configs/viam-local-mac.json`, the Voyage
fixture, and a subset of `config.py` / `.env.example` / `pyproject.toml` (`rgbd` extra: numpy, pillow). New:
`observation_sync.py` (media -> observation -> artifact_sync, no candidates), trimmed `rgbd_cli.py` with
`embedding_provider()` as the cached-Voyage constructor, `tests/test_rgbd_acquisition.py`, CI `rgbd` job.

Checks: `python3 scripts/team.py check` 59 tests OK (8 RGB-D tests skip without numpy/Pillow);
`RGBD_TESTS_REQUIRED=1 .venv/bin/python -m unittest discover -s tests` 59 tests OK. No live capture, hosted
provider call or Atlas write was made from this branch.

Limitations: no processing stage on `main`, so `rgbd-flush` uploads observations and media only and no candidate
records exist; `embedding_provider()` has no caller beyond `rgbd-verify-embedding` until a consumer lands.
Concurrent cache misses can compute twice; no eviction. The SKY-66 live evidence (indexes READY, scratch flush,
Voyage/SAM verification) was recorded on that branch and is not re-verified here.

Next: Kris reviews the split against PR #5 and decides whether `rgbd_mongo.upsert_candidate` / `vector_search` stay
or move with the processing slice; retarget PR #4 to this branch or close it; reconcile PR #3; then a designated
operator runs `rgbd-calibration`, `rgbd-capture`, `rgbd-overlay`, `rgbd-index --status`, `rgbd-flush` against the
configured local viam-server and the designated Atlas database.
