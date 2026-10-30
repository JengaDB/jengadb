# Local image and result cache

This small module remembers a JSON processing result for exact input bytes, operation,
model version and settings. It runs on a laptop or NUC with standard Python 3.11+.
It does not depend on a camera, segmentation provider or MongoDB. The latest
artifact storage and saved-only mode are prepared on Kazy's laptop, not deployed
to the shared NUC. They can be tested while someone else uses the robot.

There are two independent modules:

- `ArtifactStore`: saves image/crop bytes as files, named by their content hash.
  Identical bytes share one file. References contain hash, relative path, size
  and media type. Use `read(ref)` to read and verify a file.
- `ResultCache`: saves JSON answers in SQLite, keyed by exact inputs and settings.
  It checks registered files before returning a saved answer.

The default demo stores its database in `data/cache-demo/results.sqlite3` and
files in `data/cache-demo/artifacts/`. These local data files are not Git content.

## Run with any saved image

To test the branch without switching away from ongoing work, make a separate
checkout in a new directory (Python 3.11+; no package installation needed):

```sh
git clone --branch codex/local-camera-cache https://github.com/JengaDB/jengadb.git jengadb-cache-test
cd jengadb-cache-test
python3 scripts/team.py check
```

Chris's integration check: use one saved image with a real processing callback;
confirm the callback runs on the first request, is skipped on the second request
and after restarting, and runs again for changed image bytes or model version.
Confirm saved crops can be read through `cache.artifacts.read(ref)`. A missing crop
should invalidate the answer. Keep this local to the test checkout; no camera,
robot, Atlas credentials or deployment is required for the cache's offline tests.
Real provider testing needs its own configured callback and credentials.

From the repository root, run twice:

```sh
python3 scripts/cache_demo.py /path/to/image.jpg
```

First run: CACHE MISS and a saved **pretend** answer. Second run: CACHE HIT.
Each command starts a new process, so this also checks persistence across program
restarts. Changing `--model-version pretend-v2` creates a different entry.
No real object detection is performed; the count remains null.

Use a saved answer without running processing:

```sh
python3 scripts/cache_demo.py /path/to/image.jpg --lookup-only
```

This exits 0 on a hit or 2 on a miss. `--stage pile`, `--stage assembly`, and
`--stage tower` are separate cache contexts. Changing stage, input bytes, model
version or processing settings creates a different key. Saved-only mode is not
a filesystem read-only mode: it can initialize the database or invalidate a
damaged entry, but never calls a processing callback.

## Earlier NUC demo, 29 October 2026

The earlier result-only demo files are in `/home/intel/jengadb-cache-demo`.
The commands in this section describe that earlier version. No packages were installed,
services changed, or motion requested. An existing Viam SDK 0.84.0 environment
read one JPEG from the configured `cam` resource. This verifies camera access,
not the physical camera model or its calibration.

From an SSH session on the NUC:

```sh
cd /home/intel/jengadb-cache-demo
python3 scripts/cache_demo.py data/snapshots/4d342bc1-e2c5-44c1-9b22-673c889b956e/image.jpg
```

That image was saved with its camera name, source timestamp, request/receive times
and SHA-256. The image is real; its cached answer is explicitly synthetic.
The first run missed, a second separate process hit, and a changed model version
missed. This was a program restart test, not a NUC reboot or power-loss test.

The cache is at `data/cache-demo/results.sqlite3`; the JPEG and capture metadata
are under `data/snapshots/`. Nothing was uploaded to Atlas or a model provider.

To take another snapshot with the existing SDK on this NUC:

```sh
/home/intel/viam-modules/openvino-genai/.venv/bin/python scripts/capture_cache_image.py --address sfhack15-main.ty1cywh7tq.viam.cloud
```

This reads the machine credentials in memory from its existing local configuration;
it never copies them into this project. Use the printed new image path with the
demo. New camera frames normally differ in bytes and should miss the exact cache.

## Chris's integration boundary

`ResultCache.get_or_compute(operation=..., inputs=[image_bytes], model_version=...,
settings={...}, compute=callback)` returns `(result, hit, key)`.

The callback is invoked only on a miss. It can return plain JSON-serializable data
or `CacheOutput(value, artifacts)` when it produces or retains files. All files
needed by the result must be registered in `artifacts`; JSON references alone
are not automatically discovered or checked. No camera capture is built into
either storage module.

Example adapter, with Chris supplying `segment(image_bytes)` and its output format:

```python
from robot_hackathon.result_cache import CacheOutput, ResultCache

cache = ResultCache("data/perception/results.sqlite3")

def cached_segmentation(image_bytes, *, segment, model_version, settings):
    def process():
        # Example agreed adapter output: JSON metadata and encoded crop bytes.
        metadata, crops = segment(image_bytes)
        source = cache.artifacts.put(image_bytes, "image/jpeg")
        crop_refs = [cache.artifacts.put(crop, "image/png") for crop in crops]
        return CacheOutput(
            {"metadata": metadata, "source": source, "crops": crop_refs},
            (source, *crop_refs),
        )

    return cache.get_or_compute(
        operation="vision.segment", inputs=[image_bytes],
        model_version=model_version, settings=settings, compute=process,
    )
```

The example assumes actual JPEG inputs and PNG crops; use the correct media type
for the producer's bytes. It does not implement segmentation or change the team's
episode contract. Cache embeddings independently under a different operation,
using crop bytes, embedding model version and preprocessing settings as inputs.
Segmentation and embedding can therefore be replaced independently.

For already-computed answers use `store(..., value=..., artifacts=...)`.
For saved-only access use `lookup(...)` (entry or None), or omit `compute` from
`get_or_compute` and handle `CacheMiss`. A `CacheEntry` exposes `key`, `value` and
`artifacts`. Decide explicitly whether a miss should run processing, queue work,
or wait; the cache does not choose or contact a provider.

Include prompts, preprocessing versions, crop coordinates and all other relevant
parameters in settings; supply any additional byte inputs in order. Use a real
operation/model identifier distinct from `demo.pretend-detection` and `pretend-v1`.
Provider credentials do not belong in settings or results.

The callback runs outside database transactions. Exceptions are not cached.
Concurrent misses may compute twice; the first committed result is retained.
File writes use a temporary file and atomic rename; results and their registered
references commit together in a database transaction. Missing or damaged files
invalidate a result. Reprocessing can repair the files. A failed operation may
leave an unreferenced complete file, but never a published result referencing a
file that was already known to be missing. Legacy result-only entries still work.

This caches exact inputs, not similar scenes. A new camera frame normally has
different bytes. After any block movement, capture a fresh observation; do not
use an old image's cached answer as proof of the current layout. Include depth,
camera calibration and coordinate-frame versions if they affect a computation.
An artifact hash is not an observation ID; the episode recorder must preserve
each capture's identity and timestamp even when bytes are identical.

## Limits and remaining integration

No eviction, disk quota, cross-machine sync, real segmentation, embeddings or
Atlas upload is included. The filesystem and SQLite are local storage, not a
MongoDB edge service. An Atlas uploader is a separate next module. Full episode
recording/replay is still pending, and this cache does not store or replay robot
commands. It neither opens a camera nor controls the arm.
SQLite uses its default rollback journal and full synchronous writes; WAL is not
enabled. Persistence does not constitute a backup. A corrupt database raises an
error rather than silently deleting all results. Keep the DB and its artifacts
together when moving the cache. There is no automatic retention cleanup yet;
watch disk use before recording a large dataset.

## Validation

`python3 scripts/team.py check`: 37 tests passed, including changed inputs,
operation, model and settings; failed processing/writes; missing/corrupt artifacts;
invalid references; concurrent writers; legacy results; and saved-only access
across separate processes. Tests use synthetic byte fixtures, not real perception.
The earlier three NUC demo runs independently verified miss, hit and
changed-version miss using the real camera JPEG and a synthetic answer.
