# JengaDB build plan

## Agreed product

Humans play Jenga; the robot handles setup. Build alternating layers of three blocks from a prepared pickup area. Record each placement as a reusable experiment. Full tower construction is the stretch goal; cleanup of a scattered pile is deferred.

User clarified: JengaDB is the GitHub organization. Viam organization is believed to be named hackathon (exact name and sfhack15 membership unverified). A MongoDB Atlas organization is not currently present/visible to the user; organizer-provided Atlas access remains outstanding. Selected repository: https://github.com/JengaDB/jengadb. Connector reports push access. The starter and coding-agent harness live on branch codex/offline-starter; consult its GitHub PR for publication and review status. No account, machine or cloud changes have been made.

## Current baseline

Completed: local Python mock logger, 20 deterministic attempts, HTML report, 18 passing tests and preparation documentation.
Not implemented: live episode recorder, replay viewer, camera integration, segmentation wrapper, physical placement logic, configured local MongoDB/Atlas integration. Existing optional adapters are not verified against real services.

## Owners (proposed)

Sam + mentor: physical setup, gripper compatibility, frames, obstacle verification, single supervised operator and placement validation.
Chris: camera observations, perception, calibrated pose proposals, snapshot-based hosted segmentation if needed.
Kazy + Codex: episode recording, local durability, MongoDB integration, replay UI and integration coordination.

Only the designated operator/executor issues motion. Parallel work uses saved observations and mock events.

## Milestones and exit criteria

### 0. Confirm the actual setup

Use GitHub repository JengaDB/jengadb; confirm the assigned machine under the Viam hackathon organization, and organizer-provided Atlas access/test database when available. Read-only inspect exact arm/gripper/camera models, frame graph and obstacle geometry. Identify the NUC's available CPU/RAM/disk and installed Viam version. Mentor approves pickup and build zones and validates gripper clearance. Obtain a robot-camera RGB/depth snapshot and verify camera-to-arm transforms. Use actual block dimensions, not dimensions inferred from a photograph.

Exit: known machine and components, designated operator, documented frames/units, approved work area and one usable observation. Account membership alone is not permission to modify a shared machine.

### 1. Build the recorder before motion

Extend the mock logger with immutable episode/event IDs, local media references and hashes, planned versus measured trajectory streams, pre/post snapshots, provenance, outcome labels, incomplete-episode recovery and a queued upload manifest. Build a viewer for an attempt's timeline, imagery and outcome. Keep raw media separate from metadata. Use synthetic episodes and supplied approved images initially.

Exit: simulated success/failure/abort replay correctly; restart and failed upload do not lose or duplicate episodes; partial media is visibly marked incomplete. This is the first software milestone, not a claim of live recording readiness.

### 2. One fixed placement

With mentor approval, validate known pickup and placement poses and one supervised sequence. Keep recording independent of cloud connectivity. Capture actual observed motion and before/after evidence. Define success as block placed within agreed tolerance with stable release; define how it is measured. Record unknown when evidence is insufficient. No automatic motion retries.

Exit: several consecutive supervised placements meet the agreed criterion and each has a complete replayable episode. A command returning successfully is not sufficient.

### 3. Two-layer builder

Parameterize placement relative to a calibrated tower frame using measured block dimensions. Place three blocks, then a second perpendicular layer. Check approach/release clearances for adjacent blocks and the growing tower. Verify placement after each block and replan from observed state when needed.

Exit: two neat stable layers with six complete episode records. Do not extend directly to a full-height tower without checking the changed reach and visibility.

### 4. Variable pickup perception

Introduce separated blocks at varied positions. Use RGB/depth geometry and manual selection as the initial baseline. Evaluate SAM 2/SAM 3 or another pretrained segmentation provider on individual approved snapshots. Wrap analysis in an asynchronous Viam logic-module interface after confirming the provider schema and permissions. Record image ID, calibration/model version, mask coordinate mapping, latency and uncertainty. Reject stale results after scene changes.

Exit: proposed block poses are validated against measured geometry at agreed accuracy. New scenes cannot be analyzed by a cloud model while offline; manual annotation is the initial fallback. Segmentation does not establish graspability or safe movement.

### 5. MongoDB integration and useful memory

Inspect organizer offerings before choosing a local database. Prefer local MongoDB metadata/outbox on the NUC only if supported and resource-appropriate; otherwise keep local durable recording and use an approved LAN host or Atlas asynchronously. Viam's documented MongoDB capture writes tabular data alongside disk capture; verify installed support. Old Atlas Edge Server/Device Sync products are retired.

After explicit authorization for the designated test destination, ping, set indexes, upload one synthetic episode and verify read-back/media completeness; test interruption and repeat upload. Store successes, failures and unknowns. Add metadata-based retrieval first and vector search only when it improves similar-attempt retrieval. Do not replay historical trajectories without current-state validation.

Exit: an episode remains available locally through internet loss and syncs without duplication after reconnection; the viewer can compare prior placements and show what evidence informed a new proposal. MongoDB work can proceed in parallel with milestones 2-4 after the episode schema stabilizes.

## Shared contract

- episode_id / attempt_id: one identifier shared across recording, perception, execution and storage.
- observation: snapshot ID, capture time, RGB/depth refs, intrinsics, camera pose, calibration version.
- proposal: block ID, target pose, frame, position/angle units, uncertainty, strategy/version and approval status.
- execution: planned path, actual samples with timestamps, command status, gripper observations and capture gaps.
- result: success/failure/unknown/aborted lifecycle state, evidence, operator interventions and verifier.
- sync: pending/partial/complete per destination, with asset checksums and acknowledgments.

Keep event data and trajectory chunks separate from the episode summary. Preserve observed outcome independently from execution state; an aborted action may have an unknown physical outcome.

## Demo scope

Minimum: two alternating layers, inspect a recorded placement and a failure/unknown example, demonstrate local recording and later MongoDB synchronization.
Stretch: complete tower, broader pickup positions, retrieval-informed proposals.
Deferred: full scattered-pile cleanup, autonomous Jenga gameplay, training a new action model, Dream-1 on the physical arm before compatibility is established.

## Immediate next work

Use JengaDB/jengadb and confirm authorized Viam resources; inspect configuration read-only. Ask the MongoDB mentor for the event-provided Atlas organization/project or invitation before provisioning anything. Build the local episode recorder/viewer while hardware and perception owners validate their inputs. Prepare proposed configuration changes as a reviewable diff before applying them to shared equipment. No deployment, database writes, hosted-image uploads or motion are implied by this planning document.
