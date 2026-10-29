# CACHE-01: Integrate exact-result cache with SKY-66

Status and owner: see board.json
Role: data
Branch: codex/cache-rgbd-integration
Base: filesmuggler/sky-66-implement-rgb-d-records-candidate-embeddings-and-atlas at 16c8883

## Goal and acceptance

Reuse exact embedding requests across observations and processing versions without
replacing segmentation checkpoints, changing candidate identity, or adding another
upload queue. Verify the cached outputs pass through the existing sync with a
simulated failure/retry and no duplicates. Review the integration with the pipeline
owner before merging; live Atlas validation is separate from offline evidence.

## Handoff

Changes: ResultCache/ArtifactStore primitives from the standalone branch; an
embedding-provider adapter used by Pipeline; offline regression and combined sync
tests. Existing capture, segmentation, candidate records and uploader unchanged.
See docs/CACHE_INTEGRATION.md for overlap analysis and test instructions.

Checks: all 73 tests pass with RGBD_TESTS_REQUIRED=1 and RGB-D dependencies installed
in an isolated local environment. Two observations produce six candidate records
using three embedding calls. Simulated upload failure then retry is idempotent.

Limits: no live Atlas/model test from this laptop, no NUC access or deployment,
no eviction, no new episode recorder. Concurrent misses can compute twice.
Next: Chris reviews this integration branch and runs it with his configured
providers and designated dataset; reconcile overlapping PR #3 before merging.
