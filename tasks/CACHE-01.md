# CACHE-01: Integrate exact-result cache with SKY-66

Status and owner: see board.json
Role: data
Branch: codex/cache-rgbd-integration
Base: filesmuggler/sky-66-implement-rgb-d-records-candidate-embeddings-and-atlas at 1532afb

## Goal and acceptance

Reuse exact embedding requests across observations and processing versions without
replacing segmentation checkpoints, changing candidate identity, or adding another
upload queue. Verify the cached outputs pass through the existing sync with a
simulated failure/retry and no duplicates. Review the integration with the pipeline
owner before merging; live Atlas validation is separate from offline evidence.

## Handoff

Changes: ResultCache/ArtifactStore primitives from the standalone branch; an
embedding-provider adapter used by Pipeline; offline regression and combined sync
tests. Includes Chris's latest capture/viewer changes, image-only cache support,
candidate timestamp preservation on checkpoint repair, and upload metadata retry
repair. Existing candidate schemas and explicit upload selection are preserved.
See docs/CACHE_INTEGRATION.md for overlap analysis and test instructions.

Checks: all 83 tests pass with RGBD_TESTS_REQUIRED=1 and RGB-D dependencies installed
in an isolated local environment. Two observations produce six candidate records
using three embedding calls. Simulated upload failure then retry is idempotent. Shared media retains both
observation associations. Rebuilt checkpoints preserve immutable candidate records.
Dependency-free check also passes (83 discovered, 30 optional tests skipped).

Limits: no live Atlas/model test from this laptop, no NUC access or deployment,
no eviction, no new episode recorder. Concurrent misses can compute twice.
Next: Chris reviews this integration branch and runs it with his configured
providers and designated dataset. Merge PR #4 into the PR #5 feature branch, then
merge #5 into main after combined verification; reconcile overlapping PR #3.
This update is pushed for review, not merged to either destination.

## Mac validation after PR #4 merged

PR #4 is merged into PR #5 at e33b0d5. The local validation branch is
codex/mac-combined-validation, based on that combined commit. All 83 tests pass
with the installed RGB-D, Viam, MongoDB and web dependencies; pip check and SDK/UI
imports pass. Evidence: local/mac-combined-validation.log (ignored local output).

Real integration is pending: this Mac has no saved RGB-D observation, Meta key,
Voyage/hosted-model key or MongoDB database URI configured for this project.
A private ignored .env template is prepared with blank credentials. Chris needs to
copy one observation directory plus its matching calibration directory and supply
configuration locally. See local/MAC_TEST_HANDOFF.md. No live model request,
Atlas write, camera access or robot motion was performed during this validation.

### Hosted embedding connection verified on Kazy's Mac

The supplied model endpoint and Meta key are saved only in private ignored .env.
A text-only request to ai.mongodb.com succeeded with voyage-multimodal-3.5 and
1024 dimensions (4 text tokens, zero image pixels). Meta authentication and the
image embedding path are not yet verified. The remaining inputs are a saved
RGB-D observation/calibration and MONGODB_URI for the intended database.

### Real Atlas/image-cache verification on Kazy's Mac

Database credentials are now saved in private ignored .env. Read-only checks of
hackathon succeeded: 9 observations, 102 candidates, candidate_context_v1 READY
and queryable at 1024 dimensions. Downloaded two existing candidate context crops
from obs_c2a3ee2f1ce34343 and verified their hashes against the stored records.
One real image embedding request returned 1024 dimensions in 0.622 seconds;
Atlas vector search returned 3 matches. A separate Python process, with provider
network calls blocked, returned the identical cached vector in approximately
0.001 seconds with zero provider calls. Changed image and model inputs missed
without sending further requests. These single-run timings are not a benchmark.
Meta segmentation on the downloaded candidate crop succeeded (3 masks); this is
connectivity evidence, not a mask-quality or full-pile perception assessment.
Evidence: local/live-cache-check/*-result.json and local/mongo-connection-check.json.
No Atlas writes or robot/NUC access occurred. Full observation replay still needs
its original calibration files; capture and upload from this Mac remain untested.

### Robot-camera capture through Atlas verified

PR #5 is merged into main at 87557fe. On sfhack15, read-only discovery and camera
capture succeeded; arm.is_moving reported false. Captured RGB and depth at
1280x720 under setup sfhack15-robot-camera, observation obs_4346ab53cdea4fa3.
Processing returned 3 candidates using 1 Meta segmentation and 3 Voyage embedding
calls. Repeat processing made no additional calls. A separate process loaded all
3 exact vectors from the SQLite cache with provider network access blocked.
Uploaded this selected observation to hackathon: 2 media files, 1 observation,
3 candidates. Remote readback matched; repeat sync sent 0 files/observations/
candidates and left no pending candidates. Evidence is in ignored
local/robot-camera-check/processing-result.json and upload-result.json.

Calibration is explicitly unverified. Depth registration and corresponding
intrinsics were not assumed; exposure and grasp eligibility remain unknown.
The uploader still carries its human_default pickable label, which is not a
verified grasp verdict. No robot motion, stop, or robot configuration command
was sent. No software installed on the NUC. Processing and cache ran on this Mac.
Next: Sam verifies camera geometry/table-height mapping before using detections
for picking; local success is not evidence of a safe grasp or completed stacking.
