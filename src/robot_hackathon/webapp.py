"""Local data-collection UI (FastAPI). Binds to 127.0.0.1 only.

Preview and capture are read-only camera calls. Process sends images to Meta (SAM 3.1) and the hosted Voyage
endpoint; flush writes to Atlas. All of those require ALLOW_CLOUD=true at startup and an explicit button press.
Run: ALLOW_CLOUD=true MODE=live ... .venv/bin/python -m robot_hackathon.webapp
"""
import asyncio
import io
import json
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from .config import Config, ConfigError
from . import rgbd, capture

DATA_DIR = Path("data")
HTML = Path(__file__).parent / "web" / "index.html"


class State:
    def __init__(self):
        self.config = None
        self.client = None
        self.camera = None
        self.camera_error = None
        self.preview = {"at": 0.0, "jpeg": None, "depth_png": None}
        self.busy = None
        self.lock = asyncio.Lock()
        self.last_job = None


state = State()
app = FastAPI(title="JengaDB RGB-D collection")


def _providers():
    from .segmentation import MetaSam3Provider
    from .embeddings import VoyageMultimodalProvider
    return MetaSam3Provider(state.config, allow_cloud=True), VoyageMultimodalProvider(state.config, allow_cloud=True)


def _pipeline():
    from .pipeline import Pipeline
    from .exposure import ExposureParams
    seg, emb = _providers()
    params = ExposureParams(block_dimensions_mm=tuple(state.config.block_dimensions_mm),
                            block_dimensions_verified=state.config.block_dimensions_verified)
    return Pipeline(DATA_DIR, state.config.mode, seg, emb, params), seg, emb


async def _camera():
    if state.camera is not None:
        return state.camera
    from viam.components.camera import Camera
    try:
        state.client = await capture.open_client(state.config)
        state.camera = Camera.from_robot(state.client, state.config.camera_resource)
        state.camera_error = None
    except Exception as exc:  # noqa: BLE001 - sanitized
        state.camera_error = type(exc).__name__
        raise HTTPException(503, f"camera unavailable: {state.camera_error}")
    return state.camera


async def _refresh_preview():
    if time.monotonic() - state.preview["at"] < 0.25 and state.preview["jpeg"]:
        return
    camera = await _camera()
    try:
        rgb, depth, encoded, _, _, _ = await capture.read_frames(camera)
    except Exception as exc:  # noqa: BLE001
        state.camera = None
        raise HTTPException(503, f"frame read failed: {type(exc).__name__}")
    if encoded is None and rgb is not None:
        encoded = rgbd.encode_rgb_png(rgb)
    state.preview = {"at": time.monotonic(), "jpeg": encoded,
                     "depth_png": rgbd.encode_rgb_png(capture.colorize_depth(depth)) if depth is not None else None}


@app.on_event("startup")
async def _startup():
    state.config = Config.from_env()
    if state.config.mode != "live":
        raise ConfigError("The collection UI records live data only; set MODE=live")
    state.config.require_cloud(True)


@app.on_event("shutdown")
async def _shutdown():
    if state.client is not None:
        await state.client.close()


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML.read_text()


@app.get("/api/status")
async def status():
    from .candidate_store import CandidateLocalStore
    cfg = state.config
    registry = rgbd.CalibrationRegistry(DATA_DIR, cfg.mode)
    current = registry.current(cfg.setup_id)
    store = CandidateLocalStore(DATA_DIR, cfg.mode)
    acks = {}
    if store.directory.exists():
        for path in store.directory.glob("acks-rgbd-*.json"):
            d = json.loads(path.read_text())
            acks[path.stem] = {"media": len(d.get("media", {})), "observations": len(d.get("observations", [])), "candidates": len(d.get("candidates", []))}
    return {"mode": cfg.mode, "setup_id": cfg.setup_id, "camera_resource": cfg.camera_resource, "machine": cfg.address,
            "machine_auth": cfg.machine_auth, "camera_connected": state.camera is not None, "camera_error": state.camera_error,
            "calibration": {"current": current, "entry": registry.load(cfg.setup_id, current) if current else None},
            "observations": len(rgbd.ObservationStore(DATA_DIR, cfg.mode).list_ids()), "candidates": len(store.read()),
            "acks": acks, "busy": state.busy, "last_job": state.last_job,
            "providers": {"segmentation": "meta-model-api/sam-3.1", "embedding": "voyage-multimodal-3.5 (image-only vector; description as metadata)", "atlas_db": cfg.database}}


@app.get("/api/preview.jpg")
async def preview_jpg():
    await _refresh_preview()
    media = "image/jpeg" if state.preview["jpeg"][:2] == b"\xff\xd8" else "image/png"
    return Response(state.preview["jpeg"], media_type=media, headers={"Cache-Control": "no-store"})


@app.get("/api/preview-depth.png")
async def preview_depth():
    await _refresh_preview()
    if not state.preview["depth_png"]:
        raise HTTPException(503, "no depth frame")
    return Response(state.preview["depth_png"], media_type="image/png", headers={"Cache-Control": "no-store"})


class CaptureBody(BaseModel):
    episode_id: str | None = None


@app.post("/api/capture")
async def do_capture(body: CaptureBody):
    camera = await _camera()
    try:
        obs = await capture.capture_with_camera(state.config, DATA_DIR, camera, body.episode_id or None)
    except ConfigError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:  # noqa: BLE001
        state.camera = None
        raise HTTPException(500, f"capture failed: {type(exc).__name__}")
    return _obs_summary(obs)


def _obs_summary(obs):
    pipe, _, _ = _pipeline()
    st = pipe.status(obs["observation_id"])
    from .candidate_store import CandidateLocalStore, upload_state, Review
    cands = [c for c in CandidateLocalStore(DATA_DIR, state.config.mode).read() if c["observation_id"] == obs["observation_id"]]
    ignored = Review(DATA_DIR, state.config.mode).ignored(obs["observation_id"])
    return {"observation_id": obs["observation_id"], "upload": upload_state(DATA_DIR, state.config.mode, obs["observation_id"]),
            "candidates_ignored": len([c for c in cands if c["candidate_id"] in ignored]), "episode_id": obs["episode_id"], "captured_at": obs["captured_at"],
            "setup_id": obs["setup_id"], "calibration_version": obs["calibration_version"], "alignment": obs["alignment"]["status"],
            "flags": obs["flags"], "timing": obs["timing"], "stages": st["stages"], "processing_version": st["processing_version"],
            "local_artifacts": st["local_artifacts"], "candidates": len(cands),
            "exposure": {k: sum(1 for c in cands if c["exposure"]["status"] == k) for k in ("exposed", "occluded", "unknown")}}


@app.get("/api/observations")
async def observations():
    store = rgbd.ObservationStore(DATA_DIR, state.config.mode)
    out = []
    for oid in store.list_ids():
        try:
            out.append(_obs_summary(store.load(oid, verify=False)))
        except Exception as exc:  # noqa: BLE001
            out.append({"observation_id": oid, "error": type(exc).__name__})
    return sorted(out, key=lambda o: o.get("captured_at", ""), reverse=True)


@app.get("/api/observations/{oid}/{name}")
async def observation_file(oid: str, name: str):
    store = rgbd.ObservationStore(DATA_DIR, state.config.mode)
    directory = store.directory(oid)
    if name == "overlay.png":
        path = directory / name
        if not path.exists():
            await asyncio.to_thread(capture.render_overlay, DATA_DIR, state.config.mode, oid)
        return FileResponse(path, headers={"Cache-Control": "no-store"})
    if name == "depth-preview.png":
        _, _, depth = await asyncio.to_thread(store.load_arrays, oid)
        return Response(rgbd.encode_rgb_png(capture.colorize_depth(depth)), media_type="image/png")
    if name not in ("rgb.png", "depth.png", "observation.json"):
        raise HTTPException(404)
    return FileResponse(directory / name)


class CalibrationBody(BaseModel):
    calibration_version: str
    alignment: str
    method: str
    checked_by: str
    viewing_pose: str = "unknown"
    notes: str = ""


@app.post("/api/calibration")
async def record_calibration(body: CalibrationBody):
    registry = rgbd.CalibrationRegistry(DATA_DIR, state.config.mode)
    try:
        return registry.record(state.config.setup_id, body.calibration_version, body.alignment, body.method, body.checked_by,
                               viewing_pose=body.viewing_pose, notes=body.notes, block_dimensions_mm=list(state.config.block_dimensions_mm))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


async def _run_job(name, fn, *args):
    if state.busy:
        raise HTTPException(409, f"busy: {state.busy}")
    state.busy = name
    started = time.monotonic()
    try:
        result = await asyncio.to_thread(fn, *args)
        state.last_job = {"name": name, "ok": True, "seconds": round(time.monotonic() - started, 1)}
        return result
    except ConfigError as exc:
        state.last_job = {"name": name, "ok": False, "error": str(exc)}
        raise HTTPException(400, str(exc))
    except Exception as exc:  # noqa: BLE001 - keep provider errors sanitized
        state.last_job = {"name": name, "ok": False, "error": type(exc).__name__}
        raise HTTPException(500, f"{name} failed: {type(exc).__name__}")
    finally:
        state.busy = None


def _process(oid):
    from .candidate_store import CandidateLocalStore
    pipe, seg, emb = _pipeline()
    result = pipe.process(oid)
    store = CandidateLocalStore(DATA_DIR, state.config.mode)
    store.put_many(result["candidates"])
    pruned = store.prune_stale(oid, result["processing_version"])
    return {"observation_id": oid, "processing_version": result["processing_version"], "gated_in": result["gated_in"],
            "stale_candidates_pruned": pruned,
            "gate_reasons": result["gate_reasons"], "candidates": len(result["candidates"]),
            "segmentation_calls": seg.calls, "embedding_calls": emb.calls, "segmentation_info": seg.last_info}


@app.post("/api/process/{oid}")
async def process(oid: str):
    return await _run_job(f"process {oid}", _process, oid)


@app.get("/api/candidates")
async def candidates(observation: str | None = None):
    from .candidate_store import CandidateLocalStore, Review, Acks
    rows = CandidateLocalStore(DATA_DIR, state.config.mode).read()
    if observation:
        rows = [r for r in rows if r["observation_id"] == observation]
    review = Review(DATA_DIR, state.config.mode)
    docs = {}
    uploaded = set()
    store_dir = CandidateLocalStore(DATA_DIR, state.config.mode).directory
    if store_dir.exists():
        for path in store_dir.glob("acks-rgbd-*.json"):
            uploaded |= set(json.loads(path.read_text()).get("candidates", []))
    keep = ("candidate_id", "observation_id", "mask_index", "bbox", "area_px", "segmentation_score", "exposure", "pickability",
            "depth_stats", "depth_crop_reason", "registration_verified", "processing_version", "encoder_version", "description_text", "embedding_input_parts")
    out = []
    for r in rows:
        doc = docs.setdefault(r["observation_id"], review.load(r["observation_id"]))
        out.append({**{k: r.get(k) for k in keep}, "ignored": r["candidate_id"] in set(doc["ignored"]),
                    "uploaded": r["candidate_id"] in uploaded,
                    "pickability_label": review.pickability_label(doc, r["candidate_id"])})
    return out


class ReviewBody(BaseModel):
    ignored: list[str] | None = None
    labels: dict[str, str] | None = None
    reviewed_by: str | None = None


@app.post("/api/review/{oid}")
async def set_review(oid: str, body: ReviewBody):
    """Partial update of one capture's curation: the ignored set and/or human pickability labels.

    Ignored detections are never uploaded; labels default to pickable (label_source human_default) until set."""
    from .candidate_store import CandidateLocalStore, Review
    valid = {c["candidate_id"] for c in CandidateLocalStore(DATA_DIR, state.config.mode).read() if c["observation_id"] == oid}
    review = Review(DATA_DIR, state.config.mode)
    try:
        if body.ignored is not None:
            unknown = set(body.ignored) - valid
            if unknown:
                raise HTTPException(400, f"unknown candidate ids: {sorted(unknown)[:3]}")
            review.set_ignored(oid, body.ignored, body.reviewed_by)
        if body.labels:
            unknown = set(body.labels) - valid
            if unknown:
                raise HTTPException(400, f"unknown candidate ids: {sorted(unknown)[:3]}")
            review.set_labels(oid, body.labels, body.reviewed_by)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return review.load(oid)


class DeleteBody(BaseModel):
    observation_ids: list[str]


@app.post("/api/delete")
async def bulk_delete(body: DeleteBody):
    """Delete several never-uploaded captures. Each id is reported individually; uploaded ones are refused."""
    from .candidate_store import delete_observation as remove
    if state.busy:
        raise HTTPException(409, f"busy: {state.busy}")
    results = []
    for oid in body.observation_ids:
        try:
            results.append({**await asyncio.to_thread(remove, DATA_DIR, state.config.mode, oid), "ok": True})
        except (PermissionError, ValueError, FileNotFoundError) as exc:
            results.append({"observation_id": oid, "ok": False, "error": str(exc)})
    return results


@app.get("/api/candidates/{cid}/{name}")
async def candidate_file(cid: str, name: str):
    from .candidate_store import CandidateLocalStore
    from .pipeline import load_candidate_file
    key = {"context.png": "context_png", "cutout.png": "cutout_png", "mask.png": "mask_png", "depth.png": "depth_crop_png"}.get(name)
    row = next((r for r in CandidateLocalStore(DATA_DIR, state.config.mode).read() if r["candidate_id"] == cid), None)
    if key is None or row is None:
        raise HTTPException(404)
    data = load_candidate_file(DATA_DIR, row, key)
    if data is None:
        raise HTTPException(404, "not available for this candidate")
    return Response(data, media_type="image/png")


def _upload(oid):
    from .rgbd_mongo import RgbdMongoStore
    from .candidate_store import sync
    remote = RgbdMongoStore(state.config)
    try:
        remote.require_indexes()
        return sync(DATA_DIR, state.config.mode, remote, source_alias=f"{state.config.setup_id}-webapp", observation_ids=[oid])
    finally:
        remote.close()


@app.post("/api/upload/{oid}")
async def upload(oid: str):
    """Upload exactly one capture (media, observation, candidates with embeddings). Nothing else leaves the laptop."""
    if not (rgbd.ObservationStore(DATA_DIR, state.config.mode).directory(oid) / "observation.json").exists():
        raise HTTPException(404)
    return await _run_job(f"upload {oid}", _upload, oid)


@app.delete("/api/observations/{oid}")
async def delete_observation(oid: str):
    from .candidate_store import delete_observation as remove
    if state.busy:
        raise HTTPException(409, f"busy: {state.busy}")
    try:
        return await asyncio.to_thread(remove, DATA_DIR, state.config.mode, oid)
    except PermissionError as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/index-status")
async def index_status():
    from .rgbd_mongo import RgbdMongoStore

    def read():
        remote = RgbdMongoStore(state.config)
        try:
            return remote.index_status()
        finally:
            remote.close()
    try:
        return await asyncio.to_thread(read)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": type(exc).__name__}, status_code=503)


def main(port=8765):
    import uvicorn
    print(f"JengaDB collection UI: http://127.0.0.1:{port}  (localhost only; ctrl-C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
