# DATA-01: Episode recorder and replay

Status and owner: see board.json (single source of truth)
Role: data
Branch/PR: codex/local-camera-cache (review and integration testing; not merged)

## Goal and acceptance
Persist before/after artifact references, immutable events and distinct planned/measured trajectory samples. Recover interrupted episodes as incomplete. Replay synthetic success, failure and unknown episodes. Test duplicate events, missing artifacts, restart and failed uploads. No hardware or network required.

## Intended files
src/robot_hackathon/episodes.py, src/robot_hackathon/replay.py, tests/test_episodes.py

## Dependencies
Read docs/CONTRACTS.md and docs/INTEGRATIONS.md. Atlas management access is verified, but the application database connection is pending under DATA-02; this does not block offline recorder work. Work against mock interfaces when another track is unavailable. Coordinate shared CLI/schema changes with the affected owner.

## Handoff
Changes: first narrow slice is an independent result cache and camera snapshot demo;
see docs/CACHE_DEMO.md. Chris owns capture/segmentation/crops/embeddings; Kazy owns
cache/storage/sync. Shared episode contracts are unchanged.
Added locally: content-addressed artifact files with integrity checking, explicit
lookup/store APIs, saved-only mode and stage-separated demo keys. New code is not
deployed to the shared NUC while a teammate uses it.
Checks: 37 offline tests pass. Earlier NUC work captured one real JPEG from cam;
separate demo processes showed miss then hit, and changing model version missed.
Limitations: result is explicitly pretend. Full episode recorder/replay acceptance,
automatic retention cleanup, real perception, and upload queue remain unimplemented.
No motion, NUC service changes, package installs or Atlas writes.
Next: connect a real Chris processing callback through the cache interface after
agreeing its inputs, model identity and output format.
