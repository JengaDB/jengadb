# DATA-01: Episode recorder and replay

Status and owner: see board.json (single source of truth)
Role: data
Branch/PR: not assigned

## Goal and acceptance
Persist before/after artifact references, immutable events and distinct planned/measured trajectory samples. Recover interrupted episodes as incomplete. Replay synthetic success, failure and unknown episodes. Test duplicate events, missing artifacts, restart and failed uploads. No hardware or network required.

## Intended files
src/robot_hackathon/episodes.py, src/robot_hackathon/replay.py, tests/test_episodes.py

## Dependencies
Read docs/CONTRACTS.md and docs/INTEGRATIONS.md. Atlas management access is verified, but the application database connection is pending under DATA-02; this does not block offline recorder work. Work against mock interfaces when another track is unavailable. Coordinate shared CLI/schema changes with the affected owner.

## Handoff
Changes: none yet.
Checks: not run for this task.
Limitations: task not implemented.
Next: agree scope, then implement the smallest offline slice.
