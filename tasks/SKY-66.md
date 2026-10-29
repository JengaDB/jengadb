# SKY-66: RGB-D records, candidate embeddings and Atlas indexes

Status and owner: see board.json (single source of truth)
Role: data
Linear: https://linear.app/skymoondev/issue/SKY-66/implement-rgb-d-records-candidate-embeddings-and-atlas-indexes
Branch: filesmuggler/sky-66-implement-rgb-d-records-candidate-embeddings-and-atlas

## Goal and acceptance (this slice)

Collect RGB-D observations (no trajectories), segment bricks with Meta SAM 3.1, label exposure (exposed / occluded / unknown) from verified-aligned depth, keep pickability unknown, and store per-brick cutouts plus Voyage multimodal embeddings in Atlas behind a filtered vector index. Observations and candidates round-trip with stable IDs, versions and raw assets; attempts are out of scope here.

## Files

src/robot_hackathon/{rgbd,capture,segmentation,exposure,cutouts,embeddings,pipeline,media,candidate_store,rgbd_mongo,rgbd_cli}.py, config.py and cli.py (additive), tests/test_rgbd_pipeline.py, fixtures/providers/, docs/PROVIDERS.md, docs/SETUP.md, docs/INTEGRATIONS.md, docs/CONTRACTS.md, .github/workflows/check.yml (rgbd job), pyproject.toml (`rgbd` extra), .env.example.

## Handoff

Changes: full offline pipeline with content-validated stage checkpoints, effective-config processing identity, deterministic candidate IDs, segmentation cache keyed by image + effective settings, alignment/calibration gate for exposure, depth crops only under verified registration, local candidate store with separate media/observation/candidate acks, GridFS ownership protocol (per-attempt files, SHA-keyed manifest, age-bounded cleanup), Atlas store with unique and vector indexes, CLI commands `rgbd-*`, HTML gallery.

Checks: `python3 scripts/team.py check` 56 tests OK (19 RGB-D tests skip without numpy/Pillow); `RGBD_TESTS_REQUIRED=1 .venv/bin/python -m unittest tests.test_rgbd_pipeline` 30 tests OK. Live evidence 29 Oct 2026: `mongo-ping` OK; `rgbd-index` created unique indexes and `candidate_context_v1` on `hackathon` (READY, queryable); two idempotent flush round trips on scratch database `rgbd_smoke` (dropped afterwards); `rgbd-verify-embedding` returned `voyage-multimodal-3.5`, 1024 dims; `rgbd-verify-segmentation` on a real photo returned 1 SAM 3.1 mask with confidence in 2.2 s (fixture saved).

Limitations: no real capture yet (local viam-server not installed, no RealSense on USB at check time; `capture_live` unexecuted); SAM 3.1 untested on an actual block pile and the `wooden block` prompt is unmeasured; Voyage image path not yet exercised; exposure heuristic untested on real overlapping piles; footprint reasoning disabled; block dimensions unverified; no human label override; `rgbd-search` unexercised until real candidates exist; macOS RealSense support is experimental.

Next: brew-install viam-server, plug in the D435, start `viam-server -config configs/viam-local-mac.json`, record calibration, capture, overlay-check, process, review gallery, flush; replace the SAM fixture with a block-pile response once available.
