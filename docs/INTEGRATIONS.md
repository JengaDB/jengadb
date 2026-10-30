# Integration status and next steps

Evidence recorded on 29 October 2026. This is a handoff snapshot, not a live health check. Refresh evidence after a successful connection; do not infer readiness from account membership or offline tests. No credentials belong in this file.

## Verified and pending

| Service | Evidence | Remaining work |
| --- | --- | --- |
| GitHub | JengaDB/jengadb; harness PR #1 merged into main; Python 3.11 and 3.12 CI passed | Teammates clone main and create individual branches |
| MongoDB Atlas management API | Service-account authentication succeeded; organization and project discovery succeeded; project JengaDB contains cluster jengadb | Recheck cluster readiness; it reported CREATING with no connection string at the last inspection |
| MongoDB database connection | 29 Oct 2026: `mongo-ping` succeeded against cluster `jengadb` (MongoDB 9.0.4, 3-member replica set) with database user `jenga_db_user` from this laptop; cluster had no application databases | Attempt fixture upload for DATA-02; rotate the database password and service-account secret that were shared in chat |
| MongoDB RGB-D destination (`hackathon.observations`, `hackathon.candidates`) | `rgbd-index` created unique `observation_id`/`candidate_id` indexes and vector index `candidate_context_v1` (1024-d cosine, filters: phase, exposure.status, setup_id, mode, encoder_version, processing_version, segmentation.model_version, calibration_version); status PENDING at creation. Full flush round trip (GridFS media, observation, artifact_sync, 3 candidates) verified twice idempotently on scratch database `rgbd_smoke`, then dropped | Confirm the vector index reaches READY; first real flush waits for a real capture |
| Hosted Voyage embeddings (`ai.mongodb.com`) | `rgbd-verify-embedding` returned `voyage-multimodal-3.5`, 1024 dims with a text-only input | Image path exercised on the first real candidate |
| Meta Model API SAM 3.1 (`api.meta.ai`) | `rgbd-verify-segmentation` on a real system photo returned 1 mask with confidence via streamed SSE and the official `meta_sam_parser`; 2.2 s. Fixture saved | Run on an actual block pile; tune the prompt; measure mask quality on overlapping blocks (SKY-78) |
| Viam (local, development) | Decision: local viam-server on this Mac with the second D435. Not installed yet (`viam-server` absent, no RealSense on USB at check time). Config prepared in configs/viam-local-mac.json; `rgbd-capture` unexecuted | Install via brew, plug in the camera, start viam-server, capture one observation read-only, overlay-check, record calibration |
| Viam (robot sfhack15) | User reports arm, gripper and cam; Connect page was inspected | Later; not needed for dataset collection |
| Linear | Issues exist in Skymoondev/JengaDB (SKY-60..SKY-111). This branch implements SKY-66 | Record evidence on SKY-66; tasks/board.json remains the working status source until migration |

Atlas service-account authentication used the OAuth token endpoint followed by read-only organization, project and cluster queries. No Atlas configuration was changed. Credentials were used in memory for that check and were not saved in the repository or configured in the application. The access token is temporary; this evidence is not a persistent connection.

The organization returned by Atlas was **Krzysztof's Org - 2026-10-29**. The project name is **JengaDB**, and the cluster name is **jengadb**. These are separate from the GitHub organization and Linear workspace.

## MongoDB handoff

A service-account client ID/secret authenticates the Atlas administration API. The current PyMongo adapter reads MONGODB_URI for a database connection; it does not consume that client ID/secret. Do not put management credentials into MONGODB_URI or treat management discovery as a database test.

1. Recheck the existing cluster rather than creating another one.
2. Confirm the intended test database/collection. The values in .env.example are placeholders, not a provisioned destination.
3. Configure a database user scoped to that destination and permit the actual laptop/NUC network as needed. Never open access to every IP as a shortcut.
4. Install the optional MongoDB dependency in the local environment when connecting. Enter the URI at the hidden prompt documented in SETUP.md. The application does not automatically load .env files.
5. Run mongo-ping first. After authorization for the specific synthetic destination, create the unique index, upload the mock fixtures, verify read-back, then repeat the upload to check idempotence. Record destination, commands and outcomes without credentials.
6. Add real episode uploads only after the recorder exists and the media destination is agreed. Mock records remain separate from live records.

Existing commands and credential-entry instructions: [SETUP.md](SETUP.md). Episode implementation: [DATA-01](../tasks/DATA-01.md). Database connection acceptance: [DATA-02](../tasks/DATA-02.md).

## First connected milestone

Read one real camera observation and arm state without movement, save the observation locally, and upload the associated record to the designated MongoDB destination. This milestone has not been completed. The current recorder is a synthetic attempt logger; real episode recording remains DATA-01.

Keep the local record available when the internet is unavailable. No motion, obstacle changes, image-provider uploads or automatic trajectory replay are part of this integration-status update.

## Team coordination

Until Linear issues are created and mapped to these task IDs, tasks/board.json remains the working status source. Skymoondev membership alone does not migrate the board. When that migration is performed, record issue links in task cards and update the agent instructions and helper together to avoid maintaining two competing boards.

## Reference

[MongoDB Atlas API authentication](https://www.mongodb.com/docs/atlas/api/api-authentication/) explains service accounts and management access. Database setup instructions must match the actual cluster and approved deployment.
