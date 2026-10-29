"""Observation -> candidates orchestrator with content-validated stage checkpoints.

processing_version = sha256(effective_config.json)[:16] over everything that affects results.
candidate_id = "cand_" + sha256(observation_id | processing_version | mask_index)[:20].
A stage is skipped only when its checkpoint parses, matches processing_version and recorded input hashes,
every listed output exists with the recorded sha256, and the checkpoint's own hash verifies.
Anything else is moved to stale/ and recomputed. Segmentation results are keyed by image + settings and
shared across processing versions, so changing exposure parameters never re-calls the provider.
"""
import json
import shutil
from pathlib import Path
from .local_store import atomic_write
from . import rgbd, segmentation, exposure, cutouts, embeddings
from .cached_embedding import CachedEmbeddingProvider
from .result_cache import ResultCache

PIPELINE_CODE_VERSION = "rgbd-pipeline-v1"


def effective_config(seg_provider, exposure_params, cutout_cfg, emb_provider, calibration_sha256, calibration_version,
                     registry_current_version):
    return {"pipeline_code_version": PIPELINE_CODE_VERSION,
            "segmentation": seg_provider.effective_settings(),
            "exposure": exposure_params.effective_settings(),
            "cutouts": cutout_cfg,
            "embedding": emb_provider.effective_settings(),
            "calibration": {"version": calibration_version, "content_sha256": calibration_sha256,
                            "registry_current_version": registry_current_version}}


def processing_version(config_doc):
    return rgbd.sha256_bytes(rgbd.canonical_json(config_doc).encode())[:16]


def candidate_id(observation_id, version, mask_index):
    return "cand_" + rgbd.sha256_bytes(f"{observation_id}|{version}|{mask_index}".encode())[:20]


class StageError(RuntimeError):
    pass


# ---------------------------------------------------------------- checkpoints

def write_checkpoint(path, base_dir, stage, version, inputs, outputs, payload):
    doc = {"stage": stage, "processing_version": version, "inputs": inputs,
           "outputs": [{"path": str(Path(p).relative_to(base_dir)), "sha256": rgbd.sha256_path(p)} for p in outputs],
           "payload": payload, "written_at": rgbd.utc_now()}
    doc["checkpoint_sha256"] = rgbd.sha256_bytes(rgbd.canonical_json(doc).encode())
    atomic_write(path, rgbd.canonical_json(doc))
    return doc


def load_checkpoint(path, base_dir, stage, version, inputs):
    """Return the checkpoint if fully valid; otherwise move it aside and return None."""
    path = Path(path)
    if not path.exists():
        return None
    valid = False
    try:
        doc = json.loads(path.read_text())
        body = {k: v for k, v in doc.items() if k != "checkpoint_sha256"}
        valid = (doc.get("stage") == stage and doc.get("processing_version") == version and doc.get("inputs") == inputs
                 and doc.get("checkpoint_sha256") == rgbd.sha256_bytes(rgbd.canonical_json(body).encode())
                 and all((Path(base_dir) / o["path"]).exists() and rgbd.sha256_path(Path(base_dir) / o["path"]) == o["sha256"]
                         for o in doc.get("outputs", [])))
    except (ValueError, KeyError, TypeError, OSError):
        valid = False
    if valid:
        return doc
    stale = path.parent / "stale"
    stale.mkdir(exist_ok=True)
    shutil.move(str(path), str(stale / f"{path.stem}-{rgbd.utc_now().replace(':', '')}{path.suffix}"))
    return None


# ---------------------------------------------------------------- pipeline

class Pipeline:
    def __init__(self, data_dir, mode, seg_provider, emb_provider, exposure_params=None, cutout_cfg=None):
        self.data_dir, self.mode = Path(data_dir), mode
        self.store = rgbd.ObservationStore(data_dir, mode)
        self.registry = rgbd.CalibrationRegistry(data_dir, mode)
        self.seg = seg_provider
        self.emb = CachedEmbeddingProvider(
            emb_provider, ResultCache(self.data_dir / mode / "provider-cache" / "results.sqlite3"))
        self.exposure_params = exposure_params or exposure.ExposureParams()
        self.cutout_cfg = cutout_cfg or cutouts.cutout_settings()

    def identity(self, obs):
        cal_sha = self.registry.content_sha256(obs["setup_id"], obs["calibration_version"])
        cfg = effective_config(self.seg, self.exposure_params, self.cutout_cfg, self.emb, cal_sha, obs["calibration_version"],
                               self.registry.current(obs["setup_id"]))
        return cfg, processing_version(cfg)

    # -- stage 1: segmentation (shared across processing versions)
    def segment(self, obs, rgb):
        obs_dir = self.store.directory(obs["observation_id"])
        settings = self.seg.effective_settings()
        key = segmentation.segmentation_key(rgbd.artifact(obs, "rgb")["sha256"], settings)
        seg_dir = obs_dir / "segmentation" / key
        inputs = {"rgb_sha256": rgbd.artifact(obs, "rgb")["sha256"], "settings_sha256": rgbd.sha256_bytes(rgbd.canonical_json(settings).encode())}
        ckpt = load_checkpoint(seg_dir / "checkpoint.json", seg_dir, "segmentation", key, inputs)
        if ckpt is None:
            raw = self.seg.segment(rgb)
            masks = segmentation.finalize_masks(raw, rgb.shape[1], rgb.shape[0], settings.get("min_area", 0))
            seg_dir.mkdir(parents=True, exist_ok=True)
            outputs, payload = [], []
            for m in masks:
                png = rgbd.encode_mask_png(m["mask"])
                path = seg_dir / f"mask-{m['mask_index']:03d}.png"
                path.write_bytes(png)
                outputs.append(path)
                payload.append({"mask_index": m["mask_index"], "bbox": m["bbox"], "area": m["area"], "score": m["score"],
                                "mask_sha256": rgbd.sha256_bytes(png), "path": path.name})
            ckpt = write_checkpoint(seg_dir / "checkpoint.json", seg_dir, "segmentation", key, inputs, outputs,
                                    {"settings": settings, "segmentation_key": key, "masks": payload,
                                     "latency_ms": getattr(self.seg, "last_latency_ms", None)})
        masks = []
        for rec in ckpt["payload"]["masks"]:
            array, kind = rgbd.decode_png((seg_dir / rec["path"]).read_bytes())
            masks.append({**rec, "mask": array if kind == "mask" else array > 0})
        return key, seg_dir, ckpt, masks

    def process(self, observation_id):
        obs, rgb, depth = self.store.load_arrays(observation_id)
        cfg, version = self.identity(obs)
        obs_dir = self.store.directory(observation_id)
        proc = obs_dir / "processing" / version
        proc.mkdir(parents=True, exist_ok=True)
        cfg_path = proc / "effective_config.json"
        if not cfg_path.exists():
            atomic_write(cfg_path, rgbd.canonical_json(cfg))
        seg_key, seg_dir, seg_ckpt, masks = self.segment(obs, rgb)
        seg_sha = rgbd.sha256_path(seg_dir / "checkpoint.json")
        depth_sha = rgbd.artifact(obs, "depth")["sha256"]

        # -- stage 2: exposure
        current = self.registry.current(obs["setup_id"])
        cal_entry = self.registry.load(obs["setup_id"], obs["calibration_version"])
        gated_in, gate_reasons = exposure.gate(obs, current, cal_entry)
        inputs = {"segmentation_checkpoint_sha256": seg_sha, "depth_sha256": depth_sha,
                  "calibration_sha256": cfg["calibration"]["content_sha256"]}
        ex_ckpt = load_checkpoint(proc / "exposure.json", proc, "exposure", version, inputs)
        if ex_ckpt is None:
            results = exposure.classify(masks, depth, self.exposure_params, gated_in, gate_reasons,
                                        obs["depth"]["invalid_value"])
            ex_ckpt = write_checkpoint(proc / "exposure.json", proc, "exposure", version, inputs, [],
                                       {"gated_in": gated_in, "gate_reasons": gate_reasons, "results": results})
        results = ex_ckpt["payload"]["results"]
        registration_verified = ex_ckpt["payload"]["gated_in"]

        # -- stage 3: cutouts
        inputs = {"segmentation_checkpoint_sha256": seg_sha, "rgb_sha256": rgbd.artifact(obs, "rgb")["sha256"],
                  "depth_sha256": depth_sha, "registration_verified": registration_verified}
        cut_ckpt = load_checkpoint(proc / "cutouts.json", proc, "cutouts", version, inputs)
        if cut_ckpt is None:
            cut_dir = proc / "cutouts"
            cut_dir.mkdir(exist_ok=True)
            outputs, payload = [], []
            for m in masks:
                c = cutouts.make_cutouts(rgb, depth, m, self.cutout_cfg, registration_verified)
                files = {}
                for name in ("cutout_png", "context_png", "mask_png", "depth_crop_png"):
                    if c[name] is None:
                        files[name] = None
                        continue
                    path = cut_dir / f"{m['mask_index']:03d}-{name.replace('_png', '')}.png"
                    path.write_bytes(c[name])
                    outputs.append(path)
                    files[name] = {"path": str(path.relative_to(proc)), "sha256": rgbd.sha256_bytes(c[name])}
                payload.append({"mask_index": m["mask_index"], "files": files, "crop_transform": c["crop_transform"],
                                "context_transform": c["context_transform"], "cutout_version": c["cutout_version"],
                                "depth_crop_reason": c["depth_crop_reason"]})
            cut_ckpt = write_checkpoint(proc / "cutouts.json", proc, "cutouts", version, inputs, outputs, {"items": payload})
        cut_sha = rgbd.sha256_path(proc / "cutouts.json")

        # -- stage 4: embeddings, one atomic file per candidate
        emb_dir = proc / "embeddings"
        emb_dir.mkdir(exist_ok=True)
        candidates = []
        template_id = self.emb.effective_settings()["text_template_id"]
        for m, res, cut in zip(masks, results, cut_ckpt["payload"]["items"]):
            cid = candidate_id(observation_id, version, m["mask_index"])
            fields = {"setup_id": obs["setup_id"], "exposure": res["exposure"], "depth_stats": res["depth_stats"],
                      "crop_transform": cut["crop_transform"], "registration_verified": registration_verified}
            text = embeddings.geometry_text(fields, template_id)
            context_png = (proc / cut["files"]["context_png"]["path"]).read_bytes()
            inputs = {"cutouts_checkpoint_sha256": cut_sha, "text_sha256": rgbd.sha256_bytes(text.encode()),
                      "context_sha256": cut["files"]["context_png"]["sha256"]}
            e_ckpt = load_checkpoint(emb_dir / f"{cid}.json", proc, "embedding", version, inputs)
            if e_ckpt is None:
                vector = self.emb.embed(text, context_png)
                if len(vector) != self.emb.dimension:
                    raise StageError("Embedding dimension mismatch")
                e_ckpt = write_checkpoint(emb_dir / f"{cid}.json", proc, "embedding", version, inputs, [],
                                          {"candidate_id": cid, "vector": vector, "text": text})
            candidates.append({
                "schema_version": 1, "candidate_id": cid, "observation_id": observation_id, "episode_id": obs["episode_id"],
                "mode": obs["mode"], "setup_id": obs["setup_id"], "phase": "pick", "mask_index": m["mask_index"],
                "bbox": m["bbox"], "area_px": m["area"], "segmentation_score": m["score"], "mask_sha256": m["mask_sha256"],
                "crop_transform": cut["crop_transform"], "context_transform": cut["context_transform"],
                "depth_stats": res["depth_stats"], "depth_crop_reason": cut["depth_crop_reason"],
                "registration_verified": registration_verified, "exposure": res["exposure"], "pickability": res["pickability"],
                "segmentation": {"provider": cfg["segmentation"]["provider"], "model_version": cfg["segmentation"]["model_version"],
                                 "segmentation_key": seg_key},
                "cutout_version": cut["cutout_version"], "encoder_version": self.emb.encoder_version,
                "embedding_dimension": self.emb.dimension, "embedding_ref": str((emb_dir / f"{cid}.json").relative_to(proc)),
                "embedding_text": text, "files": cut["files"], "processing_version": version,
                "processing_dir": str(proc.relative_to(self.data_dir)), "calibration_version": obs["calibration_version"],
                "created_at": e_ckpt["written_at"]})
        atomic_write(proc / "candidates.json", rgbd.canonical_json(candidates))
        return {"observation_id": observation_id, "processing_version": version, "candidates": candidates,
                "gated_in": registration_verified, "gate_reasons": ex_ckpt["payload"]["gate_reasons"],
                "segmentation_key": seg_key}

    def status(self, observation_id):
        obs = self.store.load(observation_id, verify=False)
        cfg, version = self.identity(obs)
        proc = self.store.directory(observation_id) / "processing" / version
        stages = {s: (proc / f"{s}.json").exists() for s in ("exposure", "cutouts", "candidates")}
        stages["segmentation"] = any((self.store.directory(observation_id) / "segmentation").glob("*/checkpoint.json"))
        return {"observation_id": observation_id, "processing_version": version, "stages": stages,
                "local_artifacts": self.store.local_artifact_status(obs),
                "other_processing_versions": sorted(p.name for p in (proc.parent.iterdir() if proc.parent.exists() else []) if p.name != version)}


def load_embedding(data_dir, candidate):
    path = Path(data_dir) / candidate["processing_dir"] / candidate["embedding_ref"]
    return json.loads(path.read_text())["payload"]["vector"]


def load_candidate_file(data_dir, candidate, name):
    ref = candidate["files"].get(name)
    if ref is None:
        return None
    data = (Path(data_dir) / candidate["processing_dir"] / ref["path"]).read_bytes()
    if rgbd.sha256_bytes(data) != ref["sha256"]:
        raise ValueError(f"{name} hash mismatch for {candidate['candidate_id']}")
    return data
