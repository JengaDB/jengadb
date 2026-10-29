# VISION-01: Snapshot perception boundary

Status and owner: see board.json (single source of truth)
Role: vision
Branch/PR: not assigned

## Goal and acceptance
Create a provider-neutral snapshot analysis interface with a deterministic fake provider. Preserve observation IDs, calibration versions and crop/resize mapping. Reject stale results. Hosted model integration is a later opt-in step; no images uploaded by default.

## Intended files
src/robot_hackathon/perception.py, fixtures/perception/, tests/test_perception.py

## Dependencies
Read docs/CONTRACTS.md. Work against mock interfaces when another track is unavailable. Coordinate shared CLI/schema changes with the affected owner.

## Handoff
Changes: none yet.
Checks: not run for this task.
Limitations: task not implemented.
Next: agree scope, then implement the smallest offline slice.
