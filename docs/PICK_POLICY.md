# Experimental pick-policy replay

The policy lives in `src/robot_hackathon/pick_policy.py`. It runs on the Mac today;
this change does not deploy it to the NUC. It is a deterministic rule set producing
proposal data. Sam's executor remains a separate component and is not called.
This is a draft adapter for team review, not a replacement for the shared contracts.

## Workflow and ownership

1. Package a saved observation: RGB/depth hashes, timestamp, episode/setup IDs,
   calibration version, and processing version. Emit `observation_loaded`.
2. Reuse saved Meta segmentation and candidate crops for that exact observation
   and processing version. Emit `segmentation_reused`. Live segmentation remains
   owned by the existing pipeline; the replay never captures or calls models.
3. Compare cached Voyage vectors to Atlas with matching mode and encoder version.
   Exclude hits from this same observation, save the returned evidence, and emit
   `comparison_completed`. These are visual neighbors, not proven successful grasps.
4. Apply the versioned policy and emit `decision_proposed`. Either propose a
   candidate or request missing verification. There is no motion command.
5. Store `inputs.json`, `comparisons.json`, `proposal.json`, and `events.jsonl`
   together in one unique replay directory. Failed runs retain `replay_failed`.

The policy exists BEFORE it makes a decision. A decision is its output; we do not
invent a new policy after every pick. Future policy changes receive a new version
and can be compared on the same saved inputs. Execution results and learning from
success/failure are future work; a default pickable label is not an outcome.

## Initial rules: exposed-top-first-v1

- Require matching observation, episode, setup, mode and calibration references.
- Require verified registration, exposed status, valid depth, and verified height
  above the table from the hardware owner's geometry adapter.
- Exclude candidates labelled not_pickable.
- Rank by height above the table, highest first. Distance from camera is NOT height.
- Break equal-height ties using compatible similar examples explicitly marked
  pickable by a named human, then segmentation score, then candidate ID.
- Default human labels and matches to this observation supply no reviewed evidence.
- Always output `executable=false`, `robot_pose=null`, and remaining execution checks.

Optional experimental geometry input is keyed by candidate ID:

```json
{
  "candidate-id": {
    "verified": true,
    "observation_id": "matching-observation-id",
    "setup_id": "matching-setup-id",
    "calibration_version": "matching-calibration-version",
    "height_above_table_mm": 15
  }
}
```

This is an example structure, not real calibration. Do not set verified=true just
to make the policy select a block. Sam's adapter and the team must agree on this
boundary before execution integration.

## Run on saved data

Use the existing virtual environment with RGB-D dependencies. For read-only Atlas
comparison, configure environment variables locally (the script does not load .env):

```sh
ALLOW_CLOUD=true python scripts/replay_pick_policy.py \
  --data-dir <data-root> --mode live --observation <id> \
  --processing-version <version> --atlas --allow-cloud \
  --output-dir local/policy-runs
```

For a fully offline replay, replace `--atlas --allow-cloud` with
`--comparisons <previous-run>/comparisons.json`. Supply `--geometry <file>` only
when verified evidence is available. Keep each run's output; do not mix evidence
from different captures. No credential or robot connection is needed offline.

Atlas retrieves at most 20 neighbors per candidate before removing same-observation
hits; fewer than 20 may remain. This bounded prototype has no learned classifier,
negative-example voting, score calibration, real pose output, or motion integration.

## First real replay

Input: robot snapshot obs_4346ab53cdea4fa3, processing 324e366c1b036730.
Three saved Meta candidates were compared against Atlas. They had 17, 18, and 17
other-observation matches respectively, with no qualifying named human pickable
reviews. The policy requested verification instead of selecting a block: registration,
exposure, depth statistics, and verified table-relative heights were unavailable.

Replaying the saved comparisons in another process produced the identical proposal,
with no model, Atlas or robot calls. All 92 tests pass with dependencies; the baseline
passes with 30 optional tests skipped. This validates replay mechanics, not picking.

## Competing fairly with Chris

Use the same observation IDs, camera setup, and frozen database evidence. Record
policy/model/settings versions for both workflows. Compare false detections,
correct ordering against a human-reviewed reference, abstentions, API calls and
latency. Measure physical success separately, with the same operator and setup.
One snapshot and one timing measurement are not enough to declare a winner.

Tell Kazy BEFORE any robot access. This replay script has no robot access at all.
