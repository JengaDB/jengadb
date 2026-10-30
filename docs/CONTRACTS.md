# Shared contracts — draft v1

Discuss changes with affected teammates before implementing competing interfaces. These fields describe the next episode system; they do not replace the existing attempt schema yet.

- Episode: `schema_version`, `episode_id`, `mode` (mock/live), `started_at` UTC, `calibration_version`, `status` (recording/completed/aborted/incomplete), `observed_outcome` (success/failure/unknown), `artifact_refs`.
- Observation: `observation_id`, `episode_id`, capture time, RGB/depth artifact references, camera frame, intrinsics reference, calibration version. Explicitly record missing depth and timing uncertainty.
- Proposal: episode and observation IDs, block label, pose with explicit frame + mm/deg, uncertainty, model/provider version. A proposal is data, never an executable robot command.
- Event: immutable `event_id`, episode ID, timestamp, type, payload, source (human/model/sensor/mock).
- Trajectory: episode ID, planned/measured kind, ordered samples with times, explicit joint units and pose frame/units. A log must reveal missing samples.
- Artifact: ID, relative local reference, SHA-256, media type, capture time and upload status. No embedded images in ordinary metadata documents, secrets or absolute developer-specific paths.

RGB-D observation records (reused from SKY-66; implemented in `rgbd.py`, `observation_sync.py`, `rgbd_mongo.py`):

- Observation v1 adds `setup_id`, `alignment {status: verified|unverified|unknown, method, checked_by, checked_at, calibration_version}`, `timing {clock_source, per_image_timestamps, rgb_depth_skew_ms, uncertainty_note}`, `dimensions_equal` (a fact, not registration), `intrinsics {rgb, depth}`, depth `{unit: mm, scale_m_per_unit, invalid_value, dtype: uint16}`, artifacts with `sha256`/`bytes`/`width`/`height`. Observations are immutable once saved.
- Calibration registry: operator-recorded verdict per `setup_id` and `calibration_version`; software never infers alignment.
- Remote: `observations` insert-only with read-back; `artifact_sync` mutable with `remote_status` moving only pending -> synced; media in GridFS `rgbd_media` under per-attempt ids with the SHA-keyed insert-only `media_manifest` deciding the canonical copy. The `candidates` collection and vector index are created by `rgbd-index` but no candidate producer exists on this branch; the Candidate v1 contract stays with SKY-66.
- Embedding cache: identical request bytes + model + effective settings reuse one vector; the cache never changes observation identity or record schemas.

Data owner supplies recording/replay consumer. Vision supplies observations and proposals. Hardware validates actual resource names and frames. Keep mock fixtures unmistakably synthetic. Runtime schema and serialization tests belong to DATA-01 and VISION-01; this file alone is not validation.
