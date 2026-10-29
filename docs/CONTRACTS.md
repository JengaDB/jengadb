# Shared contracts — draft v1

Discuss changes with affected teammates before implementing competing interfaces. These fields describe the next episode system; they do not replace the existing attempt schema yet.

- Episode: `schema_version`, `episode_id`, `mode` (mock/live), `started_at` UTC, `calibration_version`, `status` (recording/completed/aborted/incomplete), `observed_outcome` (success/failure/unknown), `artifact_refs`.
- Observation: `observation_id`, `episode_id`, capture time, RGB/depth artifact references, camera frame, intrinsics reference, calibration version. Explicitly record missing depth and timing uncertainty.
- Proposal: episode and observation IDs, block label, pose with explicit frame + mm/deg, uncertainty, model/provider version. A proposal is data, never an executable robot command.
- Event: immutable `event_id`, episode ID, timestamp, type, payload, source (human/model/sensor/mock).
- Trajectory: episode ID, planned/measured kind, ordered samples with times, explicit joint units and pose frame/units. A log must reveal missing samples.
- Artifact: ID, relative local reference, SHA-256, media type, capture time and upload status. No embedded images in ordinary metadata documents, secrets or absolute developer-specific paths.

Data owner supplies recording/replay consumer. Vision supplies observations and proposals. Hardware validates actual resource names and frames. Keep mock fixtures unmistakably synthetic. Runtime schema and serialization tests belong to DATA-01 and VISION-01; this file alone is not validation.
