# Cache integration on main: SKY-66 pipeline plus the exact-request embedding cache

This branch (`cache-integration-on-main`) is cut from `main`. It replaces the stacked layout of PR #4, which was
based on the SKY-66 branch (PR #5) and inherited all of it. PR #5 itself is not merged; its modules are copied here
explicitly, with the local web UI left out.

## Provenance

| Source | Files |
| --- | --- |
| PR #4 (`codex/cache-rgbd-integration`) | `result_cache.py`, `artifact_store.py`, `cached_embedding.py`, `scripts/cache_demo.py`, `tests/test_result_cache.py`, `tests/test_artifact_store.py`, `tests/test_cached_embedding.py`, this document, `tasks/CACHE-01.md` |
| PR #5 (`sky-66`), verbatim | `rgbd.py`, `capture.py`, `segmentation.py`, `exposure.py`, `cutouts.py`, `embeddings.py`, `media.py`, `candidate_store.py`, `rgbd_mongo.py`, `cli.py` hook, `configs/viam-local-mac.json`, `fixtures/providers/{meta-sam3,voyage}`, `docs/PROVIDERS.md`, `tests/test_rgbd_pipeline.py` |
| PR #5, edited | `pipeline.py` (PR #4's cache hook re-applied), `rgbd_cli.py` (`rgbd-ui` removed), `config.py` / `.env.example` / `pyproject.toml` / `docs/SETUP.md` / `docs/CONTRACTS.md` / `.github/workflows/check.yml` (RGB-D additions merged into main's versions) |
| Not taken from PR #5 | `webapp.py`, `web/index.html`, the `web` extra, `tasks/SKY-66.md`, the plan file, `docs/INTEGRATIONS.md` evidence rows, the SKY-66 board entry |

## How the two fit

`Pipeline.__init__` wraps the supplied embedding provider in `CachedEmbeddingProvider` backed by
`ResultCache(<data-dir>/<mode>/provider-cache/results.sqlite3)` unless it is already wrapped. The provider
interface, effective settings, processing-version calculation, candidate schema and upload commands are unchanged.
When an embedding checkpoint is absent, the adapter checks the cache before invoking the provider. Old checkpoints
are not backfilled into the cache; it fills on future misses.

Relative to PR #4, `cached_embedding.py` now forwards `uses_text` (added on the SKY-66 tip after PR #4 branched) and
keys image-only requests on the image bytes alone, so differing description text does not split the cache for the
default image-only Voyage configuration. Text-aware and image-only providers never share an entry.

Cache identity: exact UTF-8 text (when `uses_text`), exact PNG bytes, encoder version, dimension, effective provider
settings, endpoint and adapter version. Mock and live use separate directories. Provider errors and invalid vectors
are not cached. Similar images are not cache hits. Identical embedding requests may share a vector, but observations
and candidates retain independent IDs and timestamps.

## The intended execution flow and where it stands

capture -> segment (SAM 3.1) -> exposure gate -> cut -> embed (cached Voyage) -> upload candidates to Atlas: implemented
offline end to end (`rgbd-capture`, `rgbd-process`, `rgbd-flush`), with the SKY-66 live evidence recorded on that
branch and not re-verified here.

compare a fresh crop against the Atlas vector index -> threshold -> pick proposal: not implemented. `rgbd-search`
queries by an existing local candidate; a query-time match step with a similarity threshold and a proposal record is
new work on top of this branch (see docs/CONTRACTS.md, last bullet of the RGB-D section).

## Reproduce the offline checks

```sh
python3 scripts/team.py check                                   # no installs; RGB-D tests skip without numpy/Pillow
.venv/bin/python -m pip install -e '.[rgbd]'
RGBD_TESTS_REQUIRED=1 .venv/bin/python -m unittest discover -s tests
```

## Limitations

No live capture, provider call or Atlas write was performed from this branch. Concurrent cache misses can invoke
the provider twice; there is no eviction. PR #3 (standalone cache) and PR #4 (stacked integration) overlap with this
branch in `result_cache.py` and `artifact_store.py`; reconcile them before any of the three merge.
