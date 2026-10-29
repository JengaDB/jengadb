# HARDWARE-01: Read-only machine discovery

Status and owner: see board.json (single source of truth)
Role: hardware
Branch/PR: not assigned

## Goal and acceptance
Document verified models, component names and frames from authorized read-only inspection. Extend fake-SDK tests for discovery timeout/cleanup if needed. Leave unavailable settings unknown. No movement, API-key disclosure, obstacle changes or installs on the NUC.

## Intended files
docs/HARDWARE.md, tests/test_viam_discovery.py

## Dependencies
Read docs/CONTRACTS.md. Work against mock interfaces when another track is unavailable. Coordinate shared CLI/schema changes with the affected owner.

## Handoff
Changes: none yet.
Checks: not run for this task.
Limitations: task not implemented.
Next: agree scope, then implement the smallest offline slice.
