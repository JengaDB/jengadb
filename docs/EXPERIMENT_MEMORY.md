# Jenga experiment memory

Design revision, 29 October 2026. Proposed architecture; not deployed. Supersedes the assumption that cloud perception must operate continuously. Existing code implements only the tested mock attempt logger and optional integration boundaries, not the recorder described here.

## Product objective

Make every supervised attempt produce a reusable episode: what the scene looked like, what the agent proposed, what was commanded, what the robot actually did, and what was observed afterward. Preserve successes, failures, unknowns, aborted attempts and human interventions. The agent's model can remain unchanged while retrieved experience improves its context. This is retrieval-based memory, not demonstrated weight training or guaranteed performance improvement.

## Local-first flow

Observe -> persist snapshot -> optional asynchronous perception -> review candidate -> supervised approved attempt -> record observed trajectory and events -> persist after snapshot/outcome -> sync later.

Internet loss must not lose records or prevent local inspection and labeling. A new scene needing hosted perception may require waiting for reconnection or manual annotation; cached masks do not make novel perception available offline. Actual offline robot connectivity/authentication must be tested separately. No physical action is authorized by this design.

## Organizer tools

- Viam: machine/resource abstraction, calibration/frame metadata, capture on demand, sequences and independent capture/sync; future logic modules can implement recorder and queued inference interfaces.
- MongoDB at the edge: candidate local MongoDB Community deployment for episode manifests, annotations and an outbox, if organizer approval and NUC resources permit. Persist raw media to local files and store checksums/references in MongoDB. A LAN laptop database is an alternative but introduces dependence on that LAN; retain NUC disk capture either way.
- Viam documents mongo_capture_config.uri/database/collection for writing tabular data alongside disk capture. This is not proof that the installed viam-server supports the current documented feature, and it is not automatic replication into the team's Atlas cluster. Verify version and behavior with a mentor before configuring.
- Atlas: searchable long-term episodes, ordinary metadata queries/aggregation first, optional embeddings/vector retrieval later. Application-managed idempotent upload with persisted acknowledgments; immutable episode/event IDs simplify reconciliation. Keep annotation revisions immutable and linked.
- Viam cloud data does not automatically become data in the team's Atlas cluster. Choose and document a supported export or application upload path. Media can use Viam-managed binary storage with stable access-controlled references; a sync-complete episode requires all referenced artifacts to be verified.

MongoDB's official edge page says Atlas Edge Server was removed in 2024 and Device Sync reached EOL in 2025. Do not build against those old products. Ask organizers whether their presentation refers to self-managed MongoDB or a partner integration. Change streams require a replica set/sharded deployment and are not an offline sync solution by themselves; start with an explicit outbox poller rather than adding replica-set complexity solely for notifications.

## Episode contents

Identifiers: episode_id, run_id, robot alias, schema version, experiment/policy version.
Context: task, block ID scoped to a tower-state version, layer/side, camera/robot model, calibration version, frame graph version, environment description, UTC timestamps plus monotonic timing/clock source.
Before: RGB and aligned depth where available, intrinsics, camera pose, joint state, original files and content hashes.
Perception: exact source snapshot ID, provider/model version, prompt, returned masks/labels/scores, pixel-coordinate mapping after resize/crop, request/response times, provenance and review status.
Proposal: proposed grasp/approach, retrieval evidence IDs, intended trajectory, units, coordinate frames and approval status.
Execution: actual timestamped joint samples, end-effector pose, gripper observations, command IDs/statuses, telemetry actually exposed by hardware. Preserve planned and measured paths separately. Do not invent force readings; motor current is not automatically calibrated contact force.
Events: start, approach, contact suspected/observed, grasp, extraction, slip, failure, recovery, human intervention, abort, end. Each label identifies whether human, heuristic or model-generated.
After: immediate and later verification snapshots, extracted-block status, tower stability observation interval, success/failure/unknown, evidence and reviewer. Define the success criterion before trials.
Integrity: expected versus captured samples, gaps/drop counts, clock alignment uncertainty, asset hashes, incomplete/complete status, upload state per asset and event.

Use separate episode summaries, events, trajectory chunks, inference jobs and annotation records rather than one ever-growing document. Keep media out of JSON/BSON records by default. A trajectory means actual recorded samples at a measured rate, not a guarantee of every controller tick.

## Recording policy for a constrained NUC

Start recording before each attempt and finalize after post-attempt verification. Record lightweight joint telemetry at a sustainable measured rate; capture keyframe images at phase boundaries plus lower-rate images during the attempt. A bounded rolling image buffer can preserve the lead-up to a newly detected failure; persist it on an event. Buffer duration and capture rate are tuning parameters, not yet selected. State clearly if pre-event frames can be lost on process failure.

Use a local disk budget and free-space checks. Never silently delete the sole copy of an unsynced episode. If durable logging cannot proceed, inhibit starting a new recorded experiment and report the condition. Keep the episode store separate from Viam directories that may be deleted after successful sync. Upload metadata first for inspection but mark media completeness separately; do not claim a full backup until media is verified. Retain both failure and success evidence under an explicit retention policy.

## Hosted segmentation as a queued job

Use SAM 2/SAM 3 or an interchangeable provider for selected snapshots, not every video frame. A Viam generic logic module can expose proposed commands such as capture_episode, enqueue_analysis, get_analysis and finalize_episode. These names are application interfaces, not existing Viam built-ins. A separate inference worker sends approved images, saves responses and reports job status; module RPCs should not block through a long API call. A vision-service wrapper can later expose detections/segments if the output can truthfully satisfy that interface.

Cache exact-input inference by image bytes hash + provider/model version + prompt + preprocessing version. Store original image and mask coordinate mapping. Reuse for historical analysis of that exact image. Never silently apply historical masks to the current tower; a new scene requires fresh observation and geometric verification. Manual point/box annotation is the offline fallback for new images, not an assumed offline cloud model.

Dream-1 remains an optional action-policy experiment. Separate its outputs from perception results and from authorized Viam execution. Test compatibility and observational predictions first. No automatic command forwarding or second competing robot-control loop.

## Experience retrieval and experiment design

Query similar setup/calibration/object contexts, then compare successful and failed attempts. Add visual embeddings only when they improve retrieval over metadata filters. Return before/after images, actual path, observed outcome and uncertainty to the same agent. An old path provides evidence or a template, never automatic replay against changed geometry.

Use paired trials where possible, change one approved variable at a time, record resets and operator help, and keep a held-out set for comparing later policy versions. Do not claim learning from a growing row count. Useful metrics: verified extraction rate, tower collapse rate, unknown-outcome rate, intervention rate, perception latency, recording completeness and duplicate-free recovery.

## Build order and acceptance

1. Local episode recorder + replay viewer with synthetic episodes and explicit incomplete-record handling.
2. Integrate authorized read-only Viam snapshots/telemetry; profile NUC capture and disk usage.
3. Add MongoDB local metadata/outbox only after confirming deployment suitability; retain local spool fallback.
4. One opt-in hosted segmentation job for an approved image, saved and replayable offline.
5. Atlas upload/read-back for an authorized episode, then replay interrupted uploads without duplication.
6. Supervised physical experiments only after mentor approval; compare outcomes using recorded evidence.

Tests must cover restart during recording, disk full, missing media, inference timeout, duplicate delivery, stale response, changed calibration, unknown outcome, and incomplete sync. Network-disconnected tests must distinguish local recording success from inability to analyze a novel scene through a cloud API.

## Sources checked

- https://docs.viam.com/data/reference/ (MongoDB capture attributes; deletion behavior; resource limits)
- https://docs.viam.com/data/capture-sync/capture-on-demand/
- https://docs.viam.com/data/sequences/
- https://docs.viam.com/data/sync-data-to-your-database/
- https://docs.viam.com/build-modules/write-a-logic-module/
- https://www.mongodb.com/solutions/use-cases/edge-and-mobile
- https://www.mongodb.com/docs/manual/changestreams/
- https://github.com/facebookresearch/sam3
