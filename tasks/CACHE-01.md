# CACHE-01: SKY-66 pipeline plus exact-result embedding cache, on main

Status and owner: see board.json
Role: data
Branch: cache-integration-on-main (cut from main at f8d9a94)
Supersedes: PR #4 (`codex/cache-rgbd-integration`, stacked on the SKY-66 branch). PR #5 (SKY-66) is not merged;
its modules are copied here explicitly without the web UI. See docs/CACHE_INTEGRATION.md for provenance.

## Goal and acceptance

Land on `main`, in explicit steps rather than by inheritance: the SKY-66 capture/segment/exposure/cutout/embed/upload
pipeline (minus the local web UI) and PR #4's exact-request embedding cache wired into `Pipeline`. Identical embedding
requests reuse one vector across observations and processing versions; image-only providers are keyed on image bytes
alone; candidate identity and the upload protocol are unchanged. Offline tests cover the full pipeline, cache reuse,
restart, invalid vectors, the media ownership protocol, and sync with simulated timeout and idempotent retry.

## Handoff

Changes: from PR #4, `result_cache.py`, `artifact_store.py`, `cached_embedding.py` (now forwards `uses_text`, keys
image-only requests on image bytes), `scripts/cache_demo.py`, their tests. From PR #5, `rgbd.py`, `capture.py`,
`segmentation.py`, `exposure.py`, `cutouts.py`, `embeddings.py`, `media.py`, `candidate_store.py`, `rgbd_mongo.py`,
`rgbd_cli.py` (minus `rgbd-ui`), the additive `cli.py` hook, both provider fixtures, `configs/viam-local-mac.json`,
`tests/test_rgbd_pipeline.py`, and the RGB-D additions to `config.py`, `.env.example`, `pyproject.toml` (`rgbd` extra),
`docs/SETUP.md`, `docs/CONTRACTS.md`, `docs/PROVIDERS.md`, CI. `pipeline.py` re-applies PR #4's hook. Three tests
added to `test_rgbd_pipeline.py`: cache reuse across observations with existing sync retry, reuse across processing
versions, and `uses_text` forwarding through `Pipeline`. One pre-existing SKY-66 test that needed numpy is now skipped
without it.

Checks: `python3 scripts/team.py check` 80 tests OK (26 RGB-D tests skip without numpy/Pillow);
`RGBD_TESTS_REQUIRED=1 .venv/bin/python -m unittest discover -s tests` 80 tests OK with numpy 2.5.4 and
meta-sam-parser installed. No live capture, hosted provider call or Atlas write was made from this branch.

Limitations: the query-time match step (fresh crop -> `$vectorSearch` -> threshold -> pick proposal) does not exist on
any branch; `rgbd-search` only queries by a local candidate. The SKY-66 live evidence (indexes READY, scratch flush,
Voyage/SAM verification) was recorded on that branch and is not re-verified here. Concurrent cache misses can compute
twice; no eviction. No web UI.

Next: Kris reviews the split; retarget PR #4 to this branch or close it; reconcile PR #3; decide whether PR #5 closes
in favour of this branch or keeps the web UI as a follow-up. Then specify the match/proposal record in CONTRACTS.md
and implement it as a data-only step. Live validation by a designated operator per docs/SETUP.md.
