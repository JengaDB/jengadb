# Observation desk

Run from the repository root:

```sh
python -m robot_hackathon.live_observer --runs local/sam3-layer --port 8766
```

Open http://127.0.0.1:8766. The server binds only to loopback, has read-only
endpoints, and imports no camera, robot, model, or database clients. No robot
run starts when this viewer opens. The page polls once per second.

The first trial can be replayed from existing report.json records and frame PNGs.
Older events link to the preceding saved observation by event order, explicitly
labeled as inferred. Missing historical decision reasons are not synthesized.
The result tile describes the whole recorded trial, not success of each sub-run.

For ongoing runs, LiveRunLog writes events.jsonl, explicit observation_frame
links, and a two-second heartbeat. The observer displays a run as active only
while its running heartbeat is less than eight seconds old. A crashed process
therefore becomes stale; this is observer status, not robot connectivity or safety
status. Journal reading tolerates an incomplete last line during append.

The local experimental bridge local/sam3_layer.py is instrumented to emit:

- observation_received: saved RGB/depth, before model processing;
- segmentation_started/completed: Meta SAM 3.1, cache hit/miss and duration;
- policy_evaluated: rule version, ranked candidates, height band, distance,
  checked grip clearance, rejection, and selected candidate;
- move_requested/completed: requested versus read-back world poses;
- placement_verified, stopped, and run_finished.

It uses the existing local camera images. This is a stream of workflow snapshots,
not continuous video. Timestamps represent local recording time; sensor exposure
time is not available in the legacy bridge. A selected block, completed move,
gripper response, and observed layer outcome are different claims.

To instrument another controller without importing the robot implementation:

```python
from robot_hackathon.live_observer import LiveRunLog
log = LiveRunLog(run_directory)
try:
    log.emit('observation_received', folder=str(frame_directory))
    # Perform perception and policy evaluation in the controller.
    log.emit('policy_evaluated', reason='Concise rule-based decision summary',
             candidates=recorded_candidate_checks)
finally:
    log.close('ended')
```

Directories currently use YYYYMMDD-HHMMSS run IDs. Keep frame folders inside the
run folder. Only known observation PNG filenames under the selected root are
served. Never put credentials in event payloads. These logs are an experimental
local observer format; they do not replace the shared episode/candidate schema.

## What controlled the first trial

Codex supervised and edited a Python controller around Sam's supplied jenga.py.
Meta SAM 3.1 segmented images. Depth/PCA estimated block geometry. Deterministic
rules preferred clear end grips, top-height-band candidates, and shorter travel.
Sam's Viam routines executed centering, rotation, calibrated descent, grasp,
placement, and packing. Calibration, thresholds, grip choice, spacing, and step
sequence were configured in code; scene coordinates came from observations.
Atlas/Voyage comparison tests were separate and did not control the physical
picks. No separate Claude or autonomous LLM policy ran the motion loop. Several
software corrections and a supervised interrupted transfer were required.

Verification: six observer tests cover stale activity, incremental writes, legacy
provenance, ended runs, and image path confinement. Full repository checks pass.
