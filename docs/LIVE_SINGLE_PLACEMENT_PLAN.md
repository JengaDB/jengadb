# Live single-placement plan — draft for mentor approval

Status: DRAFT, not approved. No motion code exists in this repository and none may be written until the approval block
at the end is filled in. Credentials, robot access and this document do not authorize movement.
Date: 29 October 2026. Scope owner: Kris. Builds on docs/AGENT_HEURISTICS_PLAN.md step 10 and docs/BUILD_PLAN.md milestone 2.

## 1. What this plan covers

One supervised pick-and-place: the arm picks one Jenga block from the approved pickup zone, chosen by an
`agent-propose` proposal on a fresh capture, and sets it down on the table at slot `L0S0` of the tower target.
One block, first layer, on the table. Repeated up to three times with resets between runs.

Out of scope: second-layer placement, sweeping or pushing blocks, automatic retries, picking from a pile,
anything the proposal did not validate. Those need their own plan after this one has evidence.

## 2. Roles

| Role | Person | Responsibility |
| --- | --- | --- |
| Mentor | TBD | Approves machine, zones, frames, speed limits and this plan; present during runs |
| Designated operator | TBD | The only person who starts motion; hand on the physical e-stop throughout; calls aborts |
| Recorder/observer | TBD | Runs capture/propose, confirms the proposal on screen, labels the outcome, keeps hands clear |

Nobody else touches the workspace, the Viam app or the keyboard while a run is in progress.

## 3. Preconditions (all must hold, with evidence recorded)

Read-only, can be done now:

- [ ] Machine identity and resources confirmed with `viam-check` (names `arm`, `gripper`, `cam`, `table`, `ceiling` or whatever the machine reports). Evidence: resource list in tasks/HARDWARE-01.md.
- [ ] Frame system reviewed with the mentor in the Viam 3D scene: world frame, arm base, gripper offset, camera frame, table and ceiling geometries. The relation between Viam's `world` and the plan's `robot_base` is written down (identical, or the fixed transform).
- [ ] Capture-time camera pose recording added to `rgbd-capture` (one read-only `transform_pose`/frame-system read alongside `get_images`) so every observation carries the camera->world transform that applied at that instant. Without it, geometry rejects with `extrinsics_missing`; that rejection must not be bypassed.
- [ ] RGB/depth registration verified by overlay and recorded (`rgbd-calibration --alignment verified`) for `SETUP_ID=robot-wrist-d435`.
- [ ] Block dimensions measured with calipers (not from a photo) and claw geometry measured: open inner gap, claw thickness, finger width. Recorded with `agent-calibration --file` with `verified: true` only for measured values.
- [ ] Minimum episode recorder available: before-capture, proposal, each command with its acceptance status, measured arm pose at every hold point, after-capture, human outcome label, operator interventions. Local files first; upload later. Milestone 1 says this comes before motion.
- [ ] Sam's script versions reconciled: `touch_off.py` imports `frame_heights`, which the supplied `jenga.py` lacks; whichever calibration helper is used must run as supplied and reviewed, not patched on the day.

Supervised motion, operator-run, mentor present (these move the arm and are part of this plan's Phase A):

- [ ] Camera XY correction (Sam's `calibrate_camera.py` method): arm high above one block, five gripper angles, least-squares correction applied to the camera frame in the config as a reviewed config diff. Accept when scatter <= 3 mm. The config change itself is a mentor-approved edit, not an agent action.
- [ ] Support-surface touch-off (Sam's `touch_off.py` method): closed claws lowered in operator-confirmed steps onto a paper sheet on bare table; records the flange/gripper z at contact. Result becomes `support_surface.z_in_robot_base_mm` with `verified: true`, sign stated explicitly (positive above the base origin), and the claw-tip offset below the gripper frame.

Scene and workspace, checked immediately before each run:

- [ ] Pickup zone: 2-4 blocks lying flat, separated by at least 70 mm edge to edge, nothing else within the camera view of the zone. Tower zone: empty table, marked, at least `footprint_clearance_mm` (about 115 mm with Sam's claw numbers) from any object.
- [ ] Obstacles in the world state: table and ceiling from the machine config plus the tower box at the tower origin; no other objects, hands or cables in the reachable volume.
- [ ] Both zones inside the mentor-approved reach envelope at the planned heights.

## 4. Scene freshness for this test

Capture -> `agent-propose` -> on-screen confirmation -> start of motion must complete within 60 seconds, and nothing
may touch the table in between. If anything does, or the timer lapses, recapture and re-propose. The proposal carries
`context_id`; the executor refuses a context older than 60 s or one whose `observation_id` is not the latest capture.
This is the policy for this test only; a general refresh/expiry policy stays open.

## 5. Run sequence with hold points

Each hold point: the operator reads the printed target pose, compares it with the 3D scene and the physical scene,
and presses a key to continue. Any mismatch, unexpected motion, sound, or doubt -> e-stop first, software second.

| # | Step | Hold / check | Abort if |
| --- | --- | --- | --- |
| 0 | Open gripper, move to the mentor-approved home pose above the pickup zone with the world state loaded | Arm at home, blocks fully in view | Home unreachable or planner error |
| 1 | Capture, propose, show the report: chosen block outlined, grasp style, source pose, slot `L0S0` | Operator confirms the red outline is the intended block and the pose matches the tape measure within 10 mm | Abstention, wrong block, pose off by > 10 mm |
| 2 | Move above the block top centre at travel height (home z), rotate claws to the proposal's closing direction | Gripper visibly centred over the block | Lateral error > 5 mm by eye or by a confirmation capture |
| 3 | Descend straight down with the linear constraint to `tips_above_table_mm`, with the gripper-table collision allowance enabled for this segment only | Claw tips beside the block, not on it | Contact with anything, planner deviation, depth > planned |
| 4 | Close gripper | `grab()` true AND operator sees the block held | Either false: open, lift straight up, end the run as `failure`; no second attempt |
| 5 | Lift 100 mm straight up | Block stays in the claws | Slip: lower, open, lift, end as `failure` |
| 6 | Move above the slot at travel height, rotate to the slot's long-side direction (claw offset from the held-block measurement) | Over the marked slot | Planner path crosses the tower box or leaves the envelope |
| 7 | Descend to `release_z_mm` with the linear constraint and the same limited collision allowance | Block just above the table | Contact before planned height |
| 8 | Open gripper, lift straight up, return to home | Block resting flat in the slot | Block tipped, dragged or pushed |
| 9 | After-capture; human outcome label; episode finalized | — | — |

Success criterion, measured from the after-capture and a ruler: block centre within +/-5 mm of the slot centre,
long side within +/-5 degrees, resting flat, claws fully clear. Anything not clearly observed is `unknown`, not success.

## 6. Decisions that deliberately differ from Sam's supplied scripts

- Sessions stay enabled. Sam disabled them because a missed heartbeat ended runs mid-move; the plan's alternative is to keep
  vision work on a worker thread (as his `detect_blocks` already does) and treat a session expiry as a legitimate stop.
  If that proves unworkable on the day, the mentor decides, not the script.
- The gripper-table collision allowance applies only to steps 3 and 7 (final descent and retreat), not to every near-table move.
- No automatic retry of any motion. `grab()` false ends the run. No "try the next block" loop.
- No sweeping, no pushing rows together, no tower-site search; the tower origin is a measured, marked spot.
- Speeds: the mentor sets the arm velocity/acceleration limits in the machine config before Phase A; the script does not override them.
- Credentials: the operator's own API key from the Viam app; the key pair embedded in `connect.py` is not used.

## 7. Abort and stop procedure

Physical e-stop is the primary stop; a software abort is secondary. After any abort: no automatic recovery. The operator
clears the fault, moves the arm clear by hand or by a mentor-approved jog, the recorder marks the episode `aborted` with
observed outcome `unknown` unless the physical state is clearly seen, and the scene is reset before another run.

## 8. What gets recorded per run

`episode_id`; before/after observations (IDs, hashes, camera pose at capture); `context_id`, `proposal_id`, the proposal
JSON; each motion command with planned target, acceptance status and measured pose at the hold point; gripper result;
timestamps and clock source; operator confirmations and interventions; outcome label with the evidence used and the
labeller; a note of anything that differed from this plan. Command acceptance is never used as the outcome.

## 9. Acceptance for this plan

One placement meets the success criterion with a complete episode record, then two more consecutive runs. Only then is
the two-layer plan drafted. A successful command return without the measured evidence does not count.

## 10. Code that would be written only after approval

1. Capture-time camera pose recording (read-only; can be done before approval since it issues no motion).
2. Minimum episode recorder (no motion; can be done before approval).
3. An operator-run executor module: loads one fresh proposal, enforces the 60 s freshness rule, walks the table in
   section 5 with a keypress at every hold point, uses the motion service with the loaded world state, and has no retry
   path. It will be a separate command gated by `ENABLE_MOTION` plus an explicit `--operator NAME` argument, and
   `Config` currently rejects `ENABLE_MOTION=true`; that rejection is lifted only in the approved change.
4. No changes to Sam's scripts; they remain references.

## Approval

| Item | Value |
| --- | --- |
| Machine alias | |
| Mentor (name, date) | |
| Designated operator | |
| Approved pickup zone / tower origin (mm, frame) | |
| Approved speed limits | |
| Phase A (camera XY correction, touch-off) approved | yes / no |
| Phase B (section 5 runs) approved | yes / no |
