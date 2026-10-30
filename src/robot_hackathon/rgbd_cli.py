"""rgbd-* subcommands. Cloud providers and Atlas writes use the existing ALLOW_CLOUD / --allow-cloud /
--confirm-write gates. No motion; capture is read-only."""
import asyncio
import json
import logging
from html import escape
from pathlib import Path
from .config import ConfigError
from .local_store import atomic_write
from . import rgbd

COMMANDS = ("rgbd-capture", "rgbd-overlay", "rgbd-calibration", "rgbd-process", "rgbd-status", "rgbd-gallery",
            "rgbd-index", "rgbd-flush", "rgbd-verify-segmentation", "rgbd-verify-embedding", "rgbd-search", "rgbd-delete")


def add_arguments(parser):
    parser.add_argument("--observation", default=None, help="Observation ID for rgbd-process/overlay/search")
    parser.add_argument("--all", action="store_true", help="Process every local observation")
    parser.add_argument("--episode", default=None, help="Episode/session ID for rgbd-capture")
    parser.add_argument("--setup-id", default=None, help="Override SETUP_ID for rgbd-calibration")
    parser.add_argument("--calibration-version", default=None)
    parser.add_argument("--alignment", choices=rgbd.ALIGNMENT_STATES, default=None, help="Operator verdict for rgbd-calibration record")
    parser.add_argument("--method", default="", help="How alignment was checked (e.g. overlay inspection of checkerboard edges)")
    parser.add_argument("--checked-by", default="", help="Operator name/alias for the calibration record")
    parser.add_argument("--viewing-pose", default="unknown")
    parser.add_argument("--notes", default="")
    parser.add_argument("--show", action="store_true", help="rgbd-calibration: print current entry")
    parser.add_argument("--status", action="store_true", help="rgbd-index: report vector index state only")
    parser.add_argument("--orphans", action="store_true", help="rgbd-status: list upload attempts past the age bound")
    parser.add_argument("--clean-orphans", action="store_true", help="rgbd-flush: delete orphaned upload attempts")
    parser.add_argument("--candidate", default=None, help="rgbd-search: candidate ID whose vector is the query")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--image", default=None, help="rgbd-verify-segmentation: path to a real photo")
    parser.add_argument("--prompt", default=None, help="rgbd-verify-segmentation: noun phrase to segment")


def _providers(config, allow_cloud):
    from .segmentation import MetaSam3Provider
    from .embeddings import VoyageMultimodalProvider
    return MetaSam3Provider(config, allow_cloud=allow_cloud), VoyageMultimodalProvider(config, allow_cloud=allow_cloud)


def _exposure_params(config):
    from .exposure import ExposureParams
    return ExposureParams(block_dimensions_mm=tuple(config.block_dimensions_mm),
                          block_dimensions_verified=config.block_dimensions_verified)


def run(args, config):
    command = args.command
    mode = config.mode
    if command == "rgbd-calibration":
        registry = rgbd.CalibrationRegistry(args.data_dir, mode)
        setup_id = args.setup_id or config.setup_id
        if args.show or not args.alignment:
            current = registry.current(setup_id)
            print(json.dumps({"setup_id": setup_id, "current": current, "entry": registry.load(setup_id, current) if current else None}, indent=2))
            return 0
        if not (args.calibration_version and args.method and args.checked_by):
            raise ConfigError("rgbd-calibration record needs --calibration-version, --alignment, --method and --checked-by")
        entry = registry.record(setup_id, args.calibration_version, args.alignment, args.method, args.checked_by,
                                viewing_pose=args.viewing_pose, notes=args.notes, block_dimensions_mm=list(config.block_dimensions_mm))
        print(json.dumps(entry, indent=2))
        return 0
    if command == "rgbd-overlay":
        from .capture import render_overlay
        if not args.observation:
            raise ConfigError("--observation is required")
        print(f"Overlay: {render_overlay(args.data_dir, mode, args.observation).resolve()}")
        print("Inspect depth edges (magenta) against RGB edges, then record the verdict with rgbd-calibration.")
        return 0
    if command == "rgbd-status":
        return _status(args, config)
    if command == "rgbd-delete":
        from .candidate_store import delete_observation
        if not args.observation:
            raise ConfigError("--observation ID is required")
        try:
            print(json.dumps(delete_observation(args.data_dir, mode, args.observation)))
        except PermissionError as exc:
            raise ConfigError(str(exc))
        return 0
    if command == "rgbd-gallery":
        path = gallery(args.data_dir, mode, args.report or Path("reports") / f"rgbd-{mode}.html")
        print(f"Gallery: {path.resolve()}")
        return 0

    config.require_cloud(args.allow_cloud)
    logging.disable(logging.CRITICAL)
    if command == "rgbd-capture":
        from .capture import capture_live
        obs = asyncio.run(capture_live(config, args.data_dir, args.episode))
        print(json.dumps({"observation_id": obs["observation_id"], "setup_id": obs["setup_id"], "captured_at": obs["captured_at"],
                          "flags": obs["flags"], "dimensions_equal": obs["dimensions_equal"]}))
        return 0
    if command == "rgbd-verify-segmentation":
        if not args.image:
            raise ConfigError("--image PATH (a real photo) and optionally --prompt are required")
        seg, _ = _providers(config, True)
        if args.prompt:
            seg.prompt = args.prompt
        data = Path(args.image).read_bytes()
        array, kind = rgbd.decode_png(data)
        if kind != "rgb":
            raise ConfigError("--image must decode to an RGB image")
        masks = seg.segment(array)
        print(json.dumps({"url": seg.url, "prompt": seg.prompt, "image_shape": list(array.shape), **seg.last_info,
                          "mask_areas_px": [int(m.sum()) for m, _ in masks], "confidences": [c for _, c in masks]}))
        return 0
    if command == "rgbd-verify-embedding":
        _, emb = _providers(config, True)
        print(json.dumps(emb.verify()))
        return 0
    if command == "rgbd-process":
        from .pipeline import Pipeline
        from .candidate_store import CandidateLocalStore
        seg, emb = _providers(config, True)
        pipe = Pipeline(args.data_dir, mode, seg, emb, _exposure_params(config))
        ids = pipe.store.list_ids() if args.all else ([args.observation] if args.observation else [])
        if not ids:
            raise ConfigError("Give --observation ID or --all")
        store = CandidateLocalStore(args.data_dir, mode)
        for observation_id in ids:
            result = pipe.process(observation_id)
            store.put_many(result["candidates"])
            store.prune_stale(observation_id, result["processing_version"])
            counts = {}
            for c in result["candidates"]:
                counts[c["exposure"]["status"]] = counts.get(c["exposure"]["status"], 0) + 1
            print(json.dumps({"observation_id": observation_id, "processing_version": result["processing_version"],
                              "candidates": len(result["candidates"]), "exposure": counts, "gated_in": result["gated_in"],
                              "gate_reasons": result["gate_reasons"], "segmentation_calls": seg.calls, "embedding_calls": emb.calls}))
        return 0

    from .rgbd_mongo import RgbdMongoStore
    if command in ("rgbd-index", "rgbd-flush") and not args.confirm_write and not args.status:
        raise ConfigError("Database writes require --confirm-write after specific authorization")
    remote = RgbdMongoStore(config)
    try:
        if command == "rgbd-index":
            if args.status:
                print(json.dumps(remote.index_status()))
                return 0
            remote.ensure_indexes()
            print(json.dumps({"unique_indexes": "configured", "vector_index": remote.ensure_vector_index()}))
            return 0
        if command == "rgbd-flush":
            from .candidate_store import sync
            remote.require_indexes()
            if args.clean_orphans:
                print(json.dumps({"orphans_removed": remote.media.clean_orphans()}))
            ids = rgbd.ObservationStore(args.data_dir, mode).list_ids() if args.all else ([args.observation] if args.observation else [])
            if not ids:
                raise ConfigError("rgbd-flush uploads only what you name: give --observation ID or --all")
            result = sync(args.data_dir, mode, remote, source_alias=f"{config.setup_id}-laptop", observation_ids=ids)
            print(json.dumps(result))
            return 1 if result["error"] else 0
        if command == "rgbd-search":
            from .candidate_store import CandidateLocalStore
            from .pipeline import load_embedding
            cands = {c["candidate_id"]: c for c in CandidateLocalStore(args.data_dir, mode).read()}
            if args.candidate not in cands:
                raise ConfigError("--candidate must name a local candidate ID")
            query = cands[args.candidate]
            filters = {"phase": "pick", "encoder_version": query["encoder_version"], "setup_id": query["setup_id"], "mode": mode}
            print(json.dumps(remote.vector_search(load_embedding(args.data_dir, query), filters, args.limit), default=str, indent=2))
            return 0
    finally:
        remote.close()
    raise ConfigError(f"Unknown command {command}")


def _status(args, config):
    from .pipeline import Pipeline
    from .candidate_store import CandidateLocalStore
    if args.orphans:
        config.require_cloud(args.allow_cloud)
        from .rgbd_mongo import RgbdMongoStore
        remote = RgbdMongoStore(config)
        try:
            print(json.dumps({"orphan_attempts": remote.media.orphan_attempts()}, default=str, indent=2))
        finally:
            remote.close()
        return 0
    seg, emb = _providers(config, False)
    pipe = Pipeline(args.data_dir, config.mode, seg, emb, _exposure_params(config))
    store = CandidateLocalStore(args.data_dir, config.mode)
    acks = sorted(store.directory.glob("acks-rgbd-*.json")) if store.directory.exists() else []
    acked = {}
    for path in acks:
        data = json.loads(path.read_text())
        acked[path.stem] = {"media": len(data.get("media", {})), "observations": len(data.get("observations", [])),
                            "candidates": len(data.get("candidates", []))}
    rows = [pipe.status(i) for i in pipe.store.list_ids()]
    print(json.dumps({"mode": config.mode, "observations": rows, "local_candidates": len(store.read()), "acks": acked}, indent=2))
    return 0


def gallery(data_dir, mode, out_path):
    """HTML page of candidate cutouts with exposure status and reasons. Local review aid, not a label editor."""
    import base64
    from .candidate_store import CandidateLocalStore
    from .pipeline import load_candidate_file
    cards = []
    for c in CandidateLocalStore(data_dir, mode).read():
        ctx = load_candidate_file(data_dir, c, "context_png")
        cut = load_candidate_file(data_dir, c, "cutout_png")
        img = lambda b: f'<img src="data:image/png;base64,{base64.b64encode(b).decode()}">' if b else ""
        st = c["exposure"]["status"]
        cards.append(f'<div class="card {st}"><div class="imgs">{img(ctx)}{img(cut)}</div>'
                     f'<strong>{escape(st)}</strong> <span class="pk">pickability: {escape(c["pickability"]["status"])}</span>'
                     f'<p>{escape(c["candidate_id"])}<br>obs {escape(c["observation_id"])} · setup {escape(c["setup_id"])} · pv {escape(c["processing_version"])}'
                     f'<br>{escape(", ".join(map(str, c["exposure"].get("reasons", []))))}'
                     f'<br>depth crop: {escape(str(c["depth_crop_reason"] or "present"))}</p></div>')
    label = "LIVE OBSERVATIONS" if mode == "live" else "MOCK DATA"
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>RGB-D candidates — {mode}</title><style>
body{{font:15px system-ui;background:#101b25;color:#eaf3f5;margin:0;padding:4vw}}h1{{margin:8px 0 24px}}.tag{{color:#75e2c3;font-weight:700;letter-spacing:.1em}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}}.card{{background:#1c2d3b;border-radius:12px;padding:14px;border-left:6px solid #888}}
.card.exposed{{border-color:#75e2c3}}.card.occluded{{border-color:#f28b82}}.card.unknown{{border-color:#f2c94c}}.imgs img{{max-width:48%;margin-right:2%;background:#000;image-rendering:pixelated}}
.pk{{color:#b8c9d3}}p{{color:#b8c9d3;line-height:1.5;word-break:break-all}}</style>
<p class="tag">{label}</p><h1>{len(cards)} candidates</h1><p>Exposure is a depth-geometry heuristic over verified-aligned RGB-D. Pickability is never inferred here.</p>
<div class="grid">{''.join(cards)}</div></html>"""
    atomic_write(out_path, page)
    return Path(out_path)
