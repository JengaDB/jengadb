import argparse
import asyncio
import json
import logging
from pathlib import Path
from .config import Config, ConfigError
from .local_store import LocalStore
from . import rgbd_cli, agent_cli


def main(argv=None):
    parser = argparse.ArgumentParser(description="Robot attempt logger. Offline by default; no physical actions.")
    parser.add_argument("command", nargs="?", default="mock-demo",
                        choices=("mock-demo", "report", "viam-check", "mongo-ping", "mongo-index", "flush", "live-step", "live-run", "layer-stack", "palletize") + rgbd_cli.COMMANDS + agent_cli.COMMANDS)
    parser.add_argument("--cycles", type=int, default=1, help="palletize: how many layers to build and stack in one go")
    parser.add_argument("--place-spacing-mm", type=float, default=14.0, help="live-run/palletize: extra pitch between consecutive pallet blocks at release; the single-direction push closes it")
    parser.add_argument("--no-sessions", action="store_true", help="live-run/layer-stack/palletize: disable Viam sessions (a missed heartbeat ended runs with SESSION_EXPIRED); a dropped connection then no longer stops a move already executing: the e-stop does")
    parser.add_argument("--align-speed", type=float, default=45.0, help="live-run/layer-stack/palletize: joint speed (deg/s) for the wrist rotations that align the claws")
    parser.add_argument("--approach-speed", type=float, default=40.0, help="live-run/palletize: joint speed (deg/s) for the last 40 mm of every descent to the table")
    parser.add_argument("--push-speed", type=float, default=30.0, help="live-run/palletize: joint speed (deg/s) while pushing the pallet row together")
    parser.add_argument("--grip-retries", type=int, default=1, help="live-run/layer-stack/palletize: repeat the pick movement this many times when the gripper reports an empty grip (grab false, not holding, or opening <= 0.5 mm)")
    parser.add_argument("--max-blocks", type=int, default=3, help="live-run: blocks to place in this run")
    parser.add_argument("--arm-speed", type=float, default=None, help="live-run/live-step: arm joint speed to request via do_command (deg/s)")
    parser.add_argument("--arm-accel", type=float, default=None, help="live-run/live-step: arm joint acceleration to request via do_command (deg/s^2)")
    parser.add_argument("--step", default=None, help="live-step: one motion segment (status|home|hover|descend|grab|lift|to-slot|release|retreat|abort-up|open)")
    parser.add_argument("--operator", default=None, help="live-step: name of the person at the e-stop")
    parser.add_argument("--decision", default=None, help="live-step hover: agent-propose decision json to execute")
    parser.add_argument("--table-z", type=float, default=None, help="live-step: override table top z (mm, world) for the descent")
    parser.add_argument("--push-distance", type=float, default=None, help="live-step push: override the computed slide distance (mm)")
    parser.add_argument("--direct-local", action="store_true", help="layer-stack/live-run: dial viam-server on the LAN (gRPC, no cloud relay); sessions stay on")
    parser.add_argument("--view", default=None, help="layer-stack/live-run: 'x,y,z' flange pose (mm, world) for the overview capture (live-run: moves there first and returns there; default is the current pose and HOME); layer-stack default 300,150,460 (sees ~x 200-545, y -130..490); raise z or shift x,y to cover more table")
    parser.add_argument("--no-pause", action="store_true", help="layer-stack: do not wait for Enter; every check still stops the run. Someone stays at the e-stop")
    parser.add_argument("--grip-lower-mm", type=float, default=0.0, help="layer-stack/live-run: grip this much deeper than the default 8 mm below the block top (limited by the tips floor)")
    parser.add_argument("--keep-out", default=None, help="live-run: 'x,y' mm (world) of a tower that must not be picked from; blocks within the footprint clearance of it are skipped")
    parser.add_argument("--layers", type=int, default=None, help="layer-stack: layers already on the tower; the measured count (top height / block height) must agree")
    parser.add_argument("--tips-floor-mm", type=float, default=None, help="layer-stack/live-run: lowest claw-tip height above the calibrated table for the pick (default 1.0; down to -6.0 when the calibrated table z is known to sit above the real surface)")
    parser.add_argument("--plan-only", action="store_true", help="layer-stack: capture and plan from the home pose; never moves")
    parser.add_argument("--source", default=None, help="layer-stack: 'x,y' mm (world) where the hand-built layer lies when it is outside the overview image; the run hovers there and measures it close up")
    parser.add_argument("--apply-offset", action="store_true", help="layer-stack: apply the offset fitted from earlier verified runs")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--allow-cloud", action="store_true")
    parser.add_argument("--confirm-write", action="store_true", help="Use only after authorization for this test database")
    rgbd_cli.add_arguments(parser)
    agent_cli.add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        config = Config.from_env()
        if args.command in rgbd_cli.COMMANDS:
            return rgbd_cli.run(args, config)
        if args.command in agent_cli.COMMANDS:
            return agent_cli.run(args, config)
        if args.command in ("mock-demo", "report"):
            from .report import render
            store = LocalStore(args.data_dir, config.mode)
            if args.command == "mock-demo":
                if config.mode != "mock":
                    raise ConfigError("mock-demo requires MODE=mock")
                from .mock_robot import MockRobot
                store.put_many(MockRobot().attempts())
            path = args.report or Path("reports") / f"{config.mode}.html"
            result = render(store.read(), path, config.mode)
            print(json.dumps(result))
            print(f"Report: {path.resolve()}")
            return 0
        config.require_cloud(args.allow_cloud)
        if args.command in ("flush", "mongo-index") and not args.confirm_write:
            raise ConfigError("Database writes require --confirm-write after specific authorization")
        # Avoid SDK debug/error logging accidentally disclosing credentials.
        logging.disable(logging.CRITICAL)
        if args.command in ("layer-stack", "palletize"):
            from .layer_stack import run_layer_stack, run_palletize
            from .segmentation import MetaSam3Provider
            from .agent_cli import _parse_xy
            from .live_executor import BLUE_SQUARE_ORIGIN
            logging.disable(logging.NOTSET)
            logging.getLogger().setLevel(logging.WARNING)
            origin = _parse_xy(args.tower_origin) if args.tower_origin else BLUE_SQUARE_ORIGIN
            try:
                if args.command == "palletize":
                    if not args.source:
                        raise ConfigError("palletize needs --source x,y: the pallet spot where the layer is assembled")
                    return asyncio.run(run_palletize(config, args.data_dir, args.operator, args.episode, origin, _parse_xy(args.source),
                                                     MetaSam3Provider(config, allow_cloud=True), cycles=args.cycles, max_blocks=args.max_blocks,
                                                     direct_local=args.direct_local,
                                                     view_pose=[float(v) for v in args.view.split(",")] if args.view else None,
                                                     grip_lower_mm=args.grip_lower_mm, tips_floor_mm=args.tips_floor_mm, expected_layers=args.layers,
                                                     confirm=(lambda msg: print(msg + "[--no-pause: continuing]", flush=True)) if args.no_pause else input,
                                                     apply_offset=args.apply_offset, arm_speed=args.arm_speed, arm_accel=args.arm_accel,
                                                     place_spacing_mm=args.place_spacing_mm, grip_retries=args.grip_retries,
                                                     references=args.references or "none", approach_speed=args.approach_speed,
                                                     push_speed=args.push_speed, disable_sessions=args.no_sessions, align_speed=args.align_speed))
                return asyncio.run(run_layer_stack(config, args.data_dir, args.operator, args.episode, origin,
                                                   MetaSam3Provider(config, allow_cloud=True), direct_local=args.direct_local,
                                                   apply_offset=args.apply_offset, plan_only=args.plan_only, grip_lower_mm=args.grip_lower_mm,
                                                   confirm=(lambda msg: print(msg + "[--no-pause: continuing]", flush=True)) if args.no_pause else input,
                                                   view_pose=[float(v) for v in args.view.split(",")] if args.view else None,
                                                   source_hint_xy=_parse_xy(args.source) if args.source else None,
                                                   tips_floor_mm=args.tips_floor_mm, expected_layers=args.layers, grip_retries=args.grip_retries,
                                                   disable_sessions=args.no_sessions, arm_speed=args.arm_speed, approach_speed=args.approach_speed,
                                                   align_speed=args.align_speed))
            except ConfigError:
                raise
            except Exception as exc:  # noqa: BLE001 - verbatim robot/planner error for the operator; carries no secrets
                print(f"{args.command} failed: {type(exc).__name__}: {exc}")
                print("No automatic retry. Inspect the arm and scene before the next run.")
                return 1
        if args.command in ("live-step", "live-run"):
            from .live_executor import run_step, run_flow
            logging.disable(logging.NOTSET)
            logging.getLogger().setLevel(logging.WARNING)
            try:
                if args.command == "live-run":
                    from .segmentation import MetaSam3Provider
                    from .agent_cli import _parse_xy
                    from .live_executor import BLUE_SQUARE_ORIGIN
                    origin = _parse_xy(args.tower_origin) if args.tower_origin else BLUE_SQUARE_ORIGIN
                    filled = [s for s in (args.filled_slots or "").split(",") if s]
                    return asyncio.run(run_flow(config, args.data_dir, args.operator, args.episode, origin, filled,
                                                args.tower_gap, args.max_blocks, MetaSam3Provider(config, allow_cloud=True),
                                                arm_speed=args.arm_speed, arm_accel=args.arm_accel,
                                                grip_lower_mm=args.grip_lower_mm, tips_floor_mm=args.tips_floor_mm,
                                                keep_out_xy=_parse_xy(args.keep_out) if args.keep_out else None,
                                                view_pose=[float(v) for v in args.view.split(",")] if args.view else None,
                                                direct_local=args.direct_local, place_spacing_mm=args.place_spacing_mm,
                                                grip_retries=args.grip_retries, references=args.references or "none",
                                                approach_speed=args.approach_speed, push_speed=args.push_speed,
                                                disable_sessions=args.no_sessions, align_speed=args.align_speed))
                return asyncio.run(run_step(config, args.data_dir, args.step, args.operator, args.episode, args.decision, args.table_z,
                                            args.push_distance))
            except ConfigError:
                raise
            except Exception as exc:  # noqa: BLE001 - the operator needs the planner/robot error verbatim; it carries no secrets
                print(f"live-step {args.step} failed: {type(exc).__name__}: {exc}")
                print("No automatic retry. Check the arm, then choose the next step (abort-up / home / repeat).")
                return 1
        if args.command == "viam-check":
            from .viam_adapter import ViamAdapter
            print(json.dumps({"resources": asyncio.run(ViamAdapter.discover(config))}))
        else:
            from .mongo_store import MongoStore
            remote = MongoStore(config)
            try:
                if args.command == "mongo-ping":
                    remote.ping()
                    print("Ping succeeded; writes and indexes not tested")
                elif args.command == "mongo-index":
                    remote.ensure_index()
                    print("Unique attempt_id index configured")
                else:
                    remote.require_index()
                    result = LocalStore(args.data_dir, config.mode).flush(remote, remote.target)
                    print(json.dumps(result))
                    return 1 if result["error"] else 0
            finally:
                remote.close()
        return 0
    except ConfigError as exc:
        print(f"Configuration: {exc}")
        return 2
    except ImportError as exc:
        print(f"Optional dependency unavailable ({exc.name}). See docs/SETUP.md; offline demo remains available.")
        return 2
    except Exception:
        print("Operation failed; diagnostic details suppressed to protect secrets. Local records retained.")
        return 1
