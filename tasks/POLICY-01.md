# POLICY-01: Replay the selection workflow on saved observations

Owner: Kazy. Role: vision. Branch: codex/pick-policy-replay.
Status: in_progress; experimental policy boundary needs team review.

Acceptance: package existing observations, reuse segmentation, compare existing
vectors, apply a separately versioned selection policy, and preserve events and
inputs for replay. Do not connect to or move the robot. Inform Kazy before any
future robot access. Keep shared producer/consumer schemas unchanged.

Implemented: pick_policy.py, scripts/replay_pick_policy.py, regression tests and
PICK_POLICY.md. Read-only Atlas comparisons exclude same-observation hits and
filter mode/encoder. Named human labels can support tie-breaking; default labels
cannot. Verified same-observation table-relative height drives top-first ordering.
Proposals are always replay_only and non-executable with no robot pose.

Evidence: all 92 tests pass with RGB-D dependencies, dependency-free baseline also
passes (30 optional skips). Actual saved robot observation yielded 3 candidates and
17/18/17 comparison hits, no qualifying named human reviews; policy abstained due
to unverified geometry. Saved-evidence replay produced an identical proposal with
no network calls. Reports remain under ignored local/policy-runs/.

Limitations: no policy accuracy claim, robot motion, NUC deployment, learned policy,
or successful-grasp evidence. Comparison snapshots and proposal logs are local;
no new Atlas schema/writes. Geometry input is experimental and must be agreed with
Sam/Chris before live integration. Original source ZIP remains unexecuted.

Next: compare Chris's workflow on the same saved captures; Sam supplies verified
geometry with source/calibration IDs. Assess outputs before any execution work.
