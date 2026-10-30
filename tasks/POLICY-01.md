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

## Supervised first-layer attempt (29 October)

User authorized the three-block-layer attempt and explicitly confirmed Sam's
camera-to-arm alignment, 171.5 mm flange table-touch height, and operator readiness.
Two collision-checked moves reached flange world (300, 150, 400) mm; both returned
success with readback position errors below 0.25 mm. No gripper commands, picks,
or placements were executed. Fresh observation obs_82551ee85f5e4f3e yielded three
Meta candidates and three embeddings; repeated processing made no new provider
calls. Independent depth analysis also found three block-sized objects.

Execution stopped before descent: current gripper frame is 172 mm below the
flange, and its configured claw box extends a further 50 mm downward. At the
operator-confirmed physical table-touch flange Z171.5 mm, this collision model
extends to world Z-50.5 mm, inside the configured tabletop (Z0). No obstacle edits
or gripper-table collision exemptions were applied. Sam must reconcile the tool
frame/collision geometry against the physical gripper before another pick attempt.
Capture metadata also still lacks depth intrinsics and verified registration;
operator confirmation was recorded separately rather than rewriting that evidence.
Local evidence: local/first-layer-attempt.json and local/layer-scan-movement.json.

## Follow-up: use Sam's completed workflow with SAM 3 (29 October)

User clarified that the supplied ZIP is Sam's finished motion work and explicitly
requested integrating it, not replacing it. The prior table-geometry conclusion
was revised: Sam deliberately uses the measured 171.5 mm touch height and a scoped
near-table gripper/table allowance. No persistent machine or obstacle configuration
was changed; normal Viam sessions remained enabled.

A local supervised bridge now feeds Meta SAM 3.1 masks and depth-derived geometry
into Sam's frame conversion, centering, orientation, end-grip, placement, and packing
functions. Three blocks were physically picked and set down. A final packing push
assembled a three-block layer; final camera evidence shows nearly parallel blocks,
approximately 25 mm adjacent center spacing, and common yaw roughly -6.5 degrees.
The arm returned raised to approximately world (300,150,400) mm and was stationary.

This trial required software corrections and a deliberate interruption, so it is
not evidence of unattended reliability. Close placements nudged the first block;
Sam's wider staging/packing strategy was necessary. The adapter must verify all
previous placements, restore the camera's absolute wrist heading, and improve
explicit cancellation before reuse. Source geometry must be refreshed for every
new scene. Evidence and limitations: local/sam3-layer/first-layer-result.json,
first-layer-result.png, and EXPERIMENT-NOTES.md. Prototype remains local; no new
PR, main merge, cloud deployment, or NUC installation was performed for this trial.

## Observation desk

Added a loopback-only, read-only viewer with workflow snapshots, block outlines,
event replay, explicit recorded/live status, requested versus completed moves,
and historical evidence provenance. Local experimental bridge now journals new
captures before segmentation, cache timing, actual candidate ranking/checks,
movement events and completion; no robot run was started for this UI work.
Existing 9 run segments and 186 events load from the first trial. Historical
missing decision reasons remain explicitly unavailable. Full suite: 98 tests
pass, including six observer tests. See docs/LIVE_OBSERVER.md for startup and
limitations. New source is local/unpublished; the prototype instrumentation
remains in the ignored local/sam3_layer.py file.


## Layer transfer into the blue square (29 October)

User requested transferring the packed three-block layer together into the marked
square. Added a local supervised stage, `local/transfer_layer_to_square.py`, with
SAM 3.1 re-observation, flat/parallel/spacing checks, shared finger-contact length,
blue-marker target detection, depth-plane conversion, and Sam's bundle grip/place
routines. Observer now labels the plan, grip sensor response, release and verification.
The final outcome requires three masks, center error <=6mm and angle error <=5 degrees.

Physical result: NOT placed. Raised positioning completed, but repeated Viam
connection losses interrupted the attempts before any descent or bundle grasp.
Latest attempt: local/sam3-layer/20261029-192756; result:
local/layer-transfer-result.json. A subsequent read-only check confirmed stationary
flange approximately (300.177,150.040,399.805) mm and responsive camera.

Limitations: local experimental stage, not a published reusable cycle. Marker
detection is specific to this view (three visible corners; fourth obscured);
resume mode references one recorded pre-grasp stop and requires --execute.
No automated movement retries; do not repeat the complete cycle yet. Next: resolve
connection stability, take a fresh plan-only observation, then supervise the first
bundle transfer. No new CNN training is needed for this milestone; gather outcomes
and synchronized robot state/action data before considering learned control.
