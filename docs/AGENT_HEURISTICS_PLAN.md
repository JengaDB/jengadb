# RGB-D retrieval and stacking agent implementation plan

Status: first offline slice implemented on branch `agent-heuristics` as task AGENT-01 (see tasks/AGENT-01.md); contract additions still need team review. No live phase is authorized by this document.
Date: 29 October 2026. Branch inspected: `agent-heuristics`.

## Objective and agreed decisions

Build a workflow for selecting blocks from a table and proposing placements in alternating Jenga layers. Humans play the game. The physical target remains two layers of three blocks; the first software milestone is a reviewable proposal from saved RGB-D, with no robot movement.

- SAM supplies object masks. Voyage supplies visual embeddings. Atlas contains curated, very good pick examples that the team treats as reference ground truth.
- Cosine similarity of at least `0.5` is the initial retrieval threshold. It measures resemblance to reference examples, not the probability of grasp success.
- Retrieval shortlists current candidates. Current depth geometry and finger-clearance checks determine whether a grasp can be proposed.
- Historical positions and trajectories are never used as current grasp coordinates.
- The signed support-surface offset relative to the robot base is configurable in millimeters. Approximately 10 mm is an estimate, not a verified calibration or an assumed sign.
- Sam's scripts are engineering references for calibration, geometry and grasp strategies. Replace their detection method with SAM; review connection and motion behavior separately.
- Scene refresh policy is deferred for a later design decision. A saved-scene proposal is not valid indefinitely for physical execution.

## Existing code and evidence

Reuse the current boundaries rather than starting another pipeline:

| Area | Existing files | Planning implication |
| --- | --- | --- |
| Capture and observation metadata | `src/robot_hackathon/capture.py`, `rgbd.py` | Preserve RGB/depth assets, timing, intrinsics and calibration identity. |
| SAM segmentation | `src/robot_hackathon/segmentation.py` | Existing Meta SAM 3.1 provider; actual block-scene quality still needs evaluation. |
| Crops and exposure | `src/robot_hackathon/cutouts.py`, `exposure.py` | Depth crops require verified registration; exposure alone is not grasp eligibility. |
| Embedding and processing | `src/robot_hackathon/embeddings.py`, `pipeline.py` | Current default is Voyage image-only context crops, including neighboring objects. |
| Storage and retrieval | `src/robot_hackathon/candidate_store.py`, `rgbd_mongo.py` | Existing Atlas cosine index and search adapter; returned evidence needs enrichment. |
| Review UI | `src/robot_hackathon/webapp.py` | Existing labels distinguish explicit human review from `human_default`. |

Task cards and integration notes contain handoff evidence, not a current service health check. Some documents disagree about live verification status; reconcile evidence before claiming readiness. No fresh service calls are needed to review this plan.

Sam's supplied files live outside this repository under `/Users/kris/Downloads/viampythons/`:

- `calibrate_camera.py`: estimates camera XY translation correction using multiple wrist angles; does not establish full extrinsic calibration.
- `jenga.py`: table-plane estimation, pixel/depth-to-world transforms, orientation and finger-clearance reasoning, middle/side/end grasps, and physical movement routines.
- `touch_off.py`: operator-driven claw/table contact calibration. It imports `frame_heights`, absent from the supplied `jenga.py`, and emits constants that version no longer consumes.
- `connect.py`: contains embedded credentials and disables sessions. Do not copy credentials or inherit that connection behavior automatically.

Reading these files does not authorize executing them. Calibration routines themselves move the arm.

## Step 1 — Confirm interfaces and reference assumptions

1. Review this plan with the vision, data and hardware humans before changing shared contracts or CLI behavior.
2. Identify the designated Atlas reference collection and how curated references are distinguished from ordinary captures.
3. Confirm whether existing references use context crops or isolated cutouts, and record their model, dimensions, input type and preprocessing versions.
4. Verify measured block and finger dimensions with the hardware owner; do not silently adopt Sam's constants for another setup.
5. Reconcile the supplied script versions and document which geometry methods have physical validation.

Acceptance: reference dataset, version compatibility, ownership and hardware assumptions are documented. No database contents or machine configuration are changed.

## Step 2 — Define observation and proposal data

Extend the draft contracts through review, preserving existing records:

1. Every stage references `episode_id`, `observation_id`, candidate ID and processing/calibration versions.
2. Observations preserve original RGB and depth, hashes, dimensions, timestamps, timing uncertainty, intrinsics and registration verdict.
3. For a wrist-mounted camera, preserve the camera-to-base transform applicable at capture time and its timing/provenance. Do not transform a saved image using a later arm pose.
4. Retrieval evidence includes reference IDs, curation provenance, encoder identity, raw Atlas score, cosine similarity, threshold and index identity.
5. Geometry includes pixel mapping, pose frame, mm/deg units, table-relative height, valid-depth coverage and uncertainty/rejection reasons.
6. Proposals include grasp style, approach and retreat intent, target layer/slot, evidence IDs and validation status. A proposal remains data, never an executable command.

Acceptance: one synthetic example can be traced from observation through reference matches to a proposal, with no ambiguous units or frames. Synthetic and live data remain separate.

## Step 3 — Establish calibration and configurable offsets

1. Define a signed parameter such as `support_surface_z_in_robot_base_mm`: positive means above the robot-base origin. Leave it unverified until the hardware owner confirms its value and sign.
2. Keep support-surface offset, camera extrinsics, RGB/depth registration and claw-tip geometry as separate calibration quantities.
3. Record calibration source, version and uncertainty. Clarify how the robot-base frame relates to Viam's world frame.
4. Reuse Sam's pinhole conversion and frame-transform ideas, using the correct intrinsics for the registered image.
5. Estimate height relative to a fitted table plane. For a level setup, compare the transformed table surface against the configured offset; a scalar offset cannot correct tilt or camera rotation errors.
6. Define how an operator-established claw-tip calibration will eventually be supplied. Do not run touch-off as part of offline development.

Acceptance: saved fixtures produce expected frame conversions and table-relative heights. Missing, inconsistent or obsolete calibration prevents a grasp proposal.

## Step 4 — Produce SAM candidates from saved RGB-D

1. Begin with saved synthetic fixtures and approved saved scenes: separated blocks first, then touching, overlapping and partially visible blocks.
2. Reuse the existing SAM provider and persist masks, confidence, prompt/model version and original-image mapping.
3. Flag duplicate masks, merged blocks, table/background detections and uncertain boundaries rather than treating every mask as one block.
4. Produce isolated RGB cutouts for inspection and context crops for reference-compatible embedding.
5. Preserve registered depth crops and masks together; a rectangular crop alone includes background pixels. Exclude invalid depth and consider mask erosion to reduce edge contamination.

Acceptance: a reviewer can inspect overlays and trace every crop to the original pixels. Unverified registration yields explicit missing geometry, not fabricated height.

## Step 5 — Embed and retrieve curated good-pick examples

1. Reuse the current Voyage model and preprocessing compatible with the selected reference dataset. Record query/document input-type policy explicitly; any change requires evaluation against the reference encoding policy.
2. Embed context crops initially if that matches the database. Introducing isolated-object embeddings requires a separately versioned reference set and comparison.
3. Filter search by compatible encoder/dimensions and reference-set membership. Add setup filters where appropriate without unintentionally excluding all useful references.
4. Do not accept `pickable=true` alone as proof of curation: current storage can assign it via `human_default`. Use explicit curation provenance for the agreed ground truth.
5. Retrieve a configurable top-k, retain scores and reference IDs, then apply `cosine_similarity >= 0.5`.
6. For the existing cosine Atlas index, convert `cosine = 2 * vectorSearchScore - 1`; the initial cosine threshold corresponds to Atlas score `0.75`. Keep score names distinct.
7. Exclude the query candidate and same-observation copies from reference evidence. Keep evaluation scenes outside the reference set.
8. Return an empty shortlist when nothing qualifies. Do not lower the threshold automatically.

Acceptance: fixture tests cover threshold boundaries, compatible encoders, curated-only references, self-match exclusion and no-match behavior. A later authorized read-only Atlas check verifies actual index/filter behavior.

MongoDB score reference: https://www.mongodb.com/docs/vector-search/indexes/vector-search-type/

## Step 6 — Generate current grasp proposals using Sam's geometry

1. Fit block orientation and dimensions from each current mask and registered depth. Flag cases where one visible face cannot establish a reliable block pose.
2. Deproject valid pixels and transform using capture-time calibration to estimate the current block pose and table-relative height.
3. Adapt Sam's middle, side and end grasp calculations as pure geometry functions first.
4. Inspect the surrounding depth for finger clearance, not just the block's masked pixels. Use measured finger dimensions and explicit margins.
5. Reject uncertain depth, implausible dimensions, insufficient clearance and out-of-workspace candidates with recorded reasons.
6. Rank eligible proposals using retrieval evidence and current geometric quality; preserve both separately for inspection.

Acceptance: saved cases include a similar-looking block rejected for blocked fingers, plus a suitable block with a reproducible grasp proposal. No hardware imports or calls are needed for geometry tests. These checks do not establish full motion-path safety.

## Step 7 — Specify the two-layer placement target

1. Define a calibrated tower origin and orientation, measured block dimensions and six explicit slots.
2. Place three blocks along one orientation and three at 90 degrees, with support heights derived from the layer index and block height.
3. Include held-block offsets, finger opening/release clearance and the growing tower in proposal validation.
4. Flag that Sam's supplied placement routine describes placement on the table; it is not evidence of validated second-layer placement.
5. Keep sweeping, pushing rows together and automatic recovery outside this first workflow.

Acceptance: an offline scene shows six intended slots, alternating orientations and layer heights. The first milestone need only propose one pick and its destination; a six-step simulation does not prove physical feasibility.

## Step 8 — Add OpenAI agent orchestration

1. Define narrow tool contracts for inspecting a scene, retrieving references, requesting validated grasp proposals and inspecting available placement slots.
2. Give the agent candidate/reference IDs, scores, current geometry, uncertainty and rejection reasons. Image inputs, if needed, require approval for the specific data and destination.
3. Use strict structured tool arguments and application-side validation. Verify that all referenced candidates and slots belong to the current planning context.
4. Let the agent select among eligible proposals, request more evidence or abstain. Geometry and calibration calculations remain in deterministic tools.
5. Return a structured pick-and-place proposal with a short rationale and evidence IDs. Do not expose a live motion tool in the offline milestone.
6. Keep OpenAI dependencies lazy and provide a deterministic fake agent for offline replay and tests. Select the model and request budget during implementation review.

Acceptance: the agent cannot select rejected or nonexistent candidates through the application validator. Timeout, malformed output and no eligible candidate produce a reviewable abstention.

OpenAI tool-interface reference: https://developers.openai.com/api/docs/guides/function-calling

## Step 9 — Deliver the offline review milestone

1. Produce an inspectable report showing the source scene, SAM overlays, context/depth crops, matching reference examples, both score types, geometric checks and proposed source/destination poses.
2. Save inputs, intermediate results and proposal evidence so the report can be replayed without network access.
3. Use held-out scenes to review mask quality, retrieval shortlist quality, geometric errors, false eligibility and abstention. Agree numerical tolerances before accepting results.
4. Compare retrieval-assisted selection with a geometry-only baseline. Test whether context crops distinguish useful clearance arrangements instead of merely recognizing wooden blocks.
5. Run `python3 scripts/team.py check` and targeted offline checks for registration, transforms, offsets, similarity thresholds, curation, invalid depth and agent validation.
6. Update the selected implementation task card with changes, checks, limitations and next action; do not mark live integration complete from fixture tests.

Acceptance: a reviewer can follow one saved observation to a supported pick-and-place proposal or a justified abstention. No robot movement is implemented or invoked.

## Step 10 — Review the later live phase separately

Before any live execution implementation:

1. Agree on scene refresh and proposal expiry. Account for both moved objects and a moving wrist camera. This policy remains deliberately unresolved in this plan.
2. Establish the episode recorder, command-versus-outcome distinction, measured telemetry and before/after verification required by the build plan.
3. Review Sam's connection/session behavior, collision allowances and automatic retry paths with the mentor and designated operator. Do not inherit them unchanged.
4. Prepare a specific mentor-approved single-placement plan, including verified frames, obstacle geometry and stop behavior.
5. Validate one supervised placement before expanding to two layers. Do not forward model output directly to the arm or automatically retry motion.

Acceptance: a separate approved live plan exists. This document, credentials and access to the robot do not authorize that phase.

## Review checklist before implementation

- [ ] Confirm this offline-first scope and the six-block target.
- [ ] Identify the curated Atlas reference set and compatible crop/encoder versions (code requires explicit `human` pickable labels; no such labels exist locally yet).
- [x] Define the signed support-surface offset (`support_surface.z_in_robot_base_mm`, positive above the base origin, `verified` flag); measured value and sign still pending hardware verification.
- [ ] Reconcile Sam's script versions and identify validated geometry methods (the plane fit, pose fit and clearance strips are adapted; none is physically validated here).
- [ ] Agree contract additions with affected humans (drafted in docs/CONTRACTS.md).
- [x] First implementation task chosen: AGENT-01 (tasks/AGENT-01.md); synthetic acceptance tolerances used: 1.5 mm position, 1 deg angle, 1 mm height.

Recommended first coding slice after review: saved-scene retrieval evidence and threshold handling, with curated-reference fixtures, followed by pure geometry and the review report. Add the OpenAI coordinator after the underlying proposals can be inspected deterministically.

## Implementation status (29 October 2026)

| Step | State | Where |
| --- | --- | --- |
| 1 Interfaces and reference assumptions | Documented, pending human review | tasks/AGENT-01.md, docs/CONTRACTS.md |
| 2 Observation and proposal data | Implemented offline (context, evidence, proposal, abstention records) | `proposal.py`, `retrieval.py` |
| 3 Calibration and configurable offsets | Implemented: signed offset, extrinsics, claw/block geometry with `verified`; plane-vs-offset comparison; missing calibration blocks proposals | `grasp_geometry.py` (`GeometryCalibrationStore`, `CalibrationBundle`) |
| 4 SAM candidates from saved RGB-D | Reuses SKY-66 masks/cutouts; synthetic ground-truth masks for fixtures; mask erosion and invalid-depth handling | `proposal.observation_scene`, `synthetic_scene.py` |
| 5 Embed and retrieve curated examples | Implemented: Atlas score -> cosine, 0.5 threshold, curated-only, exclusions, no-match; Atlas adapter unexercised live | `retrieval.py`, `rgbd_mongo.vector_search` projection |
| 6 Grasp proposals using Sam's geometry | Implemented as pure functions: pose fit, middle/side/ends, strip clearance, rejections | `grasp_geometry.py` |
| 7 Two-layer placement target | Implemented: six slots, alternating orientation, layer dependencies, second layer flagged unvalidated | `placement.py` |
| 8 OpenAI orchestration | Implemented behind the cloud gate with strict tools, validator and abstentions; deterministic coordinator for offline replay; model/budget placeholders | `coordinator.py` |
| 9 Offline review milestone | Report + JSON sidecars; synthetic cases pass; live capture abstains with reasons | `agent_report.py`, `agent_cli.py` |
| 10 Live phase | Draft plan written for mentor approval; no motion code | docs/LIVE_SINGLE_PLACEMENT_PLAN.md |

## Planning handoff

Changes: this plan plus the AGENT-01 implementation listed above. Machine configuration, Atlas indexes and robot access are unchanged.
Checks: `.venv/bin/python scripts/team.py check` -> 104 tests OK; `python3 scripts/team.py check` without numpy -> 104 run, 43 skipped, OK. Offline evidence only; see tasks/AGENT-01.md for the live-capture abstention details.
Limitations: no fresh Atlas, Voyage, Meta, OpenAI or robot calls; no curated reference set exists yet; geometry constants and the support-surface offset remain unverified for the real setup; no physical validation.
Next action: review the contract additions and the AGENT-01 next action with the vision, data and hardware humans.
