# Cache integration with SKY-66

This branch builds on `filesmuggler/sky-66-implement-rgb-d-records-candidate-embeddings-and-atlas`
at `16c8883`. It uses the existing pipeline and uploader; it does not add a second queue.

## Ownership and flow

1. Existing observation store: RGB/depth files, timestamps, calibration and observation IDs.
2. Existing segmentation checkpoints: masks and per-observation segmentation reuse across processing versions.
3. Existing exposure/cutout stages: geometry, crop files and integrity checks.
4. New embedding cache adapter: reuse exact text + context PNG + provider settings.
5. Existing embedding checkpoints: per-candidate vectors, provenance and processing identity.
6. Existing candidate store and sync: candidate records, GridFS media, acknowledgments, Atlas upload/retry.

`Pipeline` wraps its supplied embedding provider in `CachedEmbeddingProvider`.
The provider interface, effective settings, processing version calculation,
candidate schema and upload commands stay unchanged. Existing valid checkpoints
are reused as before. When an embedding checkpoint is absent, the adapter checks
`<data-dir>/<mode>/provider-cache/results.sqlite3` before invoking the provider.
Old checkpoints are not backfilled into the new cache; it fills on future misses.

Cache identity includes exact UTF-8 text, exact PNG bytes, encoder version,
dimension, effective settings, endpoint and adapter version. Mock and live use
separate directories. Provider errors and invalid vectors are not cached. Values
must have the expected dimension and finite numeric entries. The underlying
ResultCache also supports explicit saved-only lookups and registered artifacts;
the embedding adapter uses JSON vectors only, because pipeline files are already
owned and verified by SKY-66.

Identical embedding requests may share a vector, but observations and candidates
retain independent IDs and timestamps. Similar images are not cache hits. After
blocks move, use a fresh observation. This adds no robot commands or grip decisions.

## Scope of reuse

The standalone cache from PR #3 overlaps SKY-66's segmentation and media storage.
Only its reusable storage primitive, offline demo and tests are carried here; the
new integration targets embedding calls. We retain the teammate's segmentation
checkpoint and upload protocols rather than replacing them. No second image copy
is made by the embedding adapter, and its SQLite database is not itself uploaded.

This is an integration review branch, not a deployment. Review it against the
SKY-66 branch first. Do not merge both overlapping implementations blindly into
main; settle this integration with the pipeline owner, then merge in dependency
order. No changes were pushed to the teammate's branch.

## Reproduce the offline checks

In a separate checkout with Python 3.11+:

```sh
git clone --branch codex/cache-rgbd-integration https://github.com/JengaDB/jengadb.git jengadb-cache-integration-test
cd jengadb-cache-integration-test
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[rgbd]'
RGBD_TESTS_REQUIRED=1 .venv/bin/python scripts/team.py check
```

All 73 tests passed locally with Python 3.14, numpy 2.5.4, Pillow 13.0.0 and
meta-sam-parser 0.1.0. Tests include two observations sharing three embedding
computations while retaining six distinct candidate IDs, reuse across processing
versions, restart, changed request/model/settings/dimension/endpoint, and an
upload timeout followed by retry with no duplicates. These use stub providers and
an in-memory remote, not a live Atlas or model-provider connection.

The existing `rgbd-process` / candidate-store / `rgbd-flush` path remains the route
to Atlas. Follow the existing setup instructions for credentials and destinations.
This laptop still needs its runtime MongoDB configuration before a live test.
No real images were sent to a hosted provider or Atlas during this integration.

Limitations: concurrent misses can invoke a provider twice; no eviction or quota;
no new background uploader or automatic retry scheduling. Existing sync is invoked
explicitly and retains its current locking and acknowledgment behavior.
