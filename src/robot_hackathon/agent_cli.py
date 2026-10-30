"""agent-* subcommands: offline pick-and-place proposals from saved RGB-D scenes. No motion anywhere here.

agent-propose      build a planning context, decide (deterministic by default) and write a replayable report
agent-calibration  record or show the operator-supplied geometry calibration (extrinsics, support offset, claws)
"""
import json
from pathlib import Path

from .config import ConfigError
from . import rgbd

COMMANDS = ("agent-propose", "agent-calibration")
SYNTHETIC_TOWER_ORIGIN = (160.0, 300.0)


def add_arguments(parser):
    parser.add_argument("--synthetic", default=None, help="agent-propose: synthetic preset (separated|blocked|stacked); MODE=mock only")
    parser.add_argument("--references", choices=("none", "local", "atlas"), default=None,
                        help="agent-propose: reference search (default local for observations, none for synthetic)")
    parser.add_argument("--coordinator", choices=("deterministic", "openai"), default="deterministic")
    parser.add_argument("--tower-origin", default=None, help="agent-propose: 'x,y' mm in robot_base for the tower footprint centre")
    parser.add_argument("--tower-long-side-deg", type=float, default=0.0)
    parser.add_argument("--tower-gap", type=float, default=0.0, help="agent-propose: gap between neighbours in a layer (mm); Sam placed rows at claw+3 mm and pushed them together")
    parser.add_argument("--filled-slots", default="", help="agent-propose: comma-separated slot ids already built, e.g. L0S0,L0S1")
    parser.add_argument("--cosine-threshold", type=float, default=None, help="agent-propose: override the 0.5 default (recorded in evidence)")
    parser.add_argument("--processing-version", default=None, help="agent-propose: candidate processing version to use")
    parser.add_argument("--geometry-version", default=None, help="agent-calibration/agent-propose: geometry calibration version")
    parser.add_argument("--file", default=None, help="agent-calibration: JSON document to record")
    parser.add_argument("--dry-run", action="store_true",
                        help="agent-propose: waive *unverified* calibration so geometry can be inspected; output is a geometry_preview, never a proposal")


def _parse_xy(text):
    try:
        x, y = (float(v) for v in text.split(","))
    except (ValueError, AttributeError):
        raise ConfigError("--tower-origin needs 'x,y' in millimetres") from None
    return (x, y)


def run(args, config):
    if args.command == "agent-calibration":
        return _calibration(args, config)
    if args.command == "agent-propose":
        return _propose(args, config)
    raise ConfigError(f"Unknown command {args.command}")


def _calibration(args, config):
    from .grasp_geometry import GeometryCalibrationStore
    store = GeometryCalibrationStore(args.data_dir, config.mode)
    setup_id = args.setup_id or config.setup_id
    if not args.file:
        current = store.current(setup_id)
        print(json.dumps({"setup_id": setup_id, "current": current, "entry": store.load(setup_id, current) if current else None}, indent=2))
        return 0
    doc = json.loads(Path(args.file).read_text())
    version = args.geometry_version or doc.get("version")
    if not version:
        raise ConfigError("Give --geometry-version or a 'version' field in the document")
    try:
        entry = store.record(setup_id, version, doc)
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    print(json.dumps(entry, indent=2))
    return 0


def _propose(args, config):
    from . import proposal, retrieval, agent_report
    mode = config.mode
    if args.synthetic:
        if mode != "mock":
            raise ConfigError("--synthetic requires MODE=mock; synthetic scenes never mix with live data")
        from . import synthetic_scene as synth
        observation_id = f"obs_synthetic_{args.synthetic}"
        if not (rgbd.ObservationStore(args.data_dir, "mock").directory(observation_id) / "observation.json").exists():
            synth.save_scene(args.data_dir, args.synthetic, observation_id)
        scene = proposal.synthetic_scene(args.data_dir, observation_id)
        references = args.references or "none"
        tower = _parse_xy(args.tower_origin) if args.tower_origin else SYNTHETIC_TOWER_ORIGIN
    elif args.observation:
        observation_id = args.observation
        scene = proposal.observation_scene(args.data_dir, mode, observation_id, args.processing_version, args.geometry_version)
        references = args.references or "local"
        tower = _parse_xy(args.tower_origin) if args.tower_origin else None
    else:
        raise ConfigError("Give --observation ID or --synthetic PRESET")

    policy = retrieval.ReferencePolicy(cosine_threshold=args.cosine_threshold) if args.cosine_threshold is not None else retrieval.ReferencePolicy()
    search, remote = None, None
    try:
        if references == "local":
            docs = retrieval.local_reference_documents(args.data_dir, mode)
            search = retrieval.LocalReferenceSearch(docs, label=f"local-{mode}-candidates") if docs else None
        elif references == "atlas":
            config.require_cloud(args.allow_cloud)
            from .rgbd_mongo import RgbdMongoStore
            remote = RgbdMongoStore(config)
            search = retrieval.AtlasReferenceSearch(remote)
        filled = [s for s in args.filled_slots.split(",") if s]
        context = proposal.build_context(scene, policy, search, tower, filled, args.tower_long_side_deg, dry_run=args.dry_run,
                                         tower_gap_mm=args.tower_gap)
    finally:
        if remote is not None:
            remote.close()

    if args.coordinator == "openai":
        config.require_cloud(args.allow_cloud)
        from .coordinator import OpenAICoordinator
        coordinator = OpenAICoordinator(config, allow_cloud=True)
    else:
        coordinator = proposal.DeterministicCoordinator()
    decision = coordinator.decide(context)

    out_dir = Path(args.data_dir) / mode / "proposals" / observation_id
    ctx_path, dec_path = proposal.save_context(context, decision, out_dir)
    report_path = args.report or Path("reports") / f"agent-{mode}-{observation_id}.html"
    agent_report.render_report(scene, context, decision, report_path)
    summary = {"observation_id": observation_id, "context_id": context["context_id"], "source": context["source"], "dry_run": args.dry_run,
               "candidates": len(context["candidates"]), "eligible": context["eligible_candidate_ids"],
               "calibration_missing": context["calibration"]["missing"], "reference_index": context["reference_index"],
               "decision": decision["kind"], "decided_by": decision["decided_by"],
               "proposal_id": decision.get("proposal_id"), "abstention_reason": decision.get("reason"),
               "context_file": str(ctx_path), "decision_file": str(dec_path), "report": str(Path(report_path).resolve())}
    print(json.dumps(summary, indent=2))
    return 0
