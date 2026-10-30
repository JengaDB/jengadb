"""rgbd-* subcommands for acquisition and the Atlas connection. Cloud providers and Atlas writes use the existing
ALLOW_CLOUD / --allow-cloud / --confirm-write gates. No motion; capture is read-only.

Reused from SKY-66 without the processing commands (segmentation, exposure, cutouts, candidates, gallery, UI).
"""
import asyncio
import json
import logging
from pathlib import Path
from .config import ConfigError
from . import rgbd

COMMANDS = ("rgbd-capture", "rgbd-overlay", "rgbd-calibration", "rgbd-status", "rgbd-verify-embedding",
            "rgbd-index", "rgbd-flush")


def add_arguments(parser):
    parser.add_argument("--observation", default=None, help="Observation ID for rgbd-overlay/flush")
    parser.add_argument("--all", action="store_true", help="rgbd-flush: upload every local observation")
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


def embedding_provider(config, data_dir, allow_cloud):
    """The hosted Voyage client wrapped in the exact-request cache. Shared by every future embedding consumer."""
    from .embeddings import VoyageMultimodalProvider
    from .cached_embedding import CachedEmbeddingProvider
    from .result_cache import ResultCache
    cache = ResultCache(Path(data_dir) / config.mode / "provider-cache" / "results.sqlite3")
    return CachedEmbeddingProvider(VoyageMultimodalProvider(config, allow_cloud=allow_cloud), cache)


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
                                viewing_pose=args.viewing_pose, notes=args.notes)
        print(json.dumps(entry, indent=2))
        return 0
    if command == "rgbd-overlay":
        from .capture import render_overlay
        if not args.observation:
            raise ConfigError("--observation is required")
        print(f"Overlay: {render_overlay(args.data_dir, mode, args.observation).resolve()}")
        print("Inspect depth edges (magenta) against RGB edges, then record the verdict with rgbd-calibration.")
        return 0
    if command == "rgbd-status" and not args.orphans:
        store = rgbd.ObservationStore(args.data_dir, mode)
        rows = []
        for obs_id in store.list_ids():
            obs = store.load(obs_id, verify=False)
            rows.append({"observation_id": obs_id, "setup_id": obs["setup_id"], "captured_at": obs["captured_at"],
                         "alignment": obs["alignment"]["status"], "artifacts": store.local_artifact_status(obs)})
        directory = Path(args.data_dir) / mode
        acks = {p.stem.replace("acks-rgbd-", ""): {"media": len(json.loads(p.read_text()).get("media", {})),
                                                   "observations": len(json.loads(p.read_text()).get("observations", []))}
                for p in (sorted(directory.glob("acks-rgbd-*.json")) if directory.exists() else [])}
        print(json.dumps({"mode": mode, "observations": rows, "acks": acks}, indent=2))
        return 0

    config.require_cloud(args.allow_cloud)
    logging.disable(logging.CRITICAL)
    if command == "rgbd-capture":
        from .capture import capture_live
        obs = asyncio.run(capture_live(config, args.data_dir, args.episode))
        print(json.dumps({"observation_id": obs["observation_id"], "setup_id": obs["setup_id"], "captured_at": obs["captured_at"],
                          "flags": obs["flags"], "dimensions_equal": obs["dimensions_equal"]}))
        return 0
    if command == "rgbd-verify-embedding":
        print(json.dumps(embedding_provider(config, args.data_dir, True).provider.verify()))
        return 0

    from .rgbd_mongo import RgbdMongoStore
    if command in ("rgbd-index", "rgbd-flush") and not args.confirm_write and not args.status:
        raise ConfigError("Database writes require --confirm-write after specific authorization")
    remote = RgbdMongoStore(config)
    try:
        if command == "rgbd-status":
            print(json.dumps({"orphan_attempts": remote.media.orphan_attempts()}, default=str, indent=2))
            return 0
        if command == "rgbd-index":
            if args.status:
                print(json.dumps(remote.index_status()))
                return 0
            remote.ensure_indexes()
            print(json.dumps({"unique_indexes": "configured", "vector_index": remote.ensure_vector_index()}))
            return 0
        if command == "rgbd-flush":
            from .observation_sync import sync_observations
            remote.require_indexes()
            if args.clean_orphans:
                print(json.dumps({"orphans_removed": remote.media.clean_orphans()}))
            ids = rgbd.ObservationStore(args.data_dir, mode).list_ids() if args.all else ([args.observation] if args.observation else [])
            if not ids:
                raise ConfigError("rgbd-flush uploads only what you name: give --observation ID or --all")
            result = sync_observations(args.data_dir, mode, remote, source_alias=f"{config.setup_id}-laptop", observation_ids=ids)
            print(json.dumps(result))
            return 1 if result["error"] else 0
    finally:
        remote.close()
    raise ConfigError(f"Unknown command {command}")
