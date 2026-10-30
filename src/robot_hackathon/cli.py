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
                        choices=("mock-demo", "report", "viam-check", "mongo-ping", "mongo-index", "flush", "live-step", "live-run") + rgbd_cli.COMMANDS + agent_cli.COMMANDS)
    parser.add_argument("--max-blocks", type=int, default=3, help="live-run: blocks to place in this run")
    parser.add_argument("--arm-speed", type=float, default=None, help="live-run/live-step: arm joint speed to request via do_command (deg/s)")
    parser.add_argument("--arm-accel", type=float, default=None, help="live-run/live-step: arm joint acceleration to request via do_command (deg/s^2)")
    parser.add_argument("--step", default=None, help="live-step: one motion segment (status|home|hover|descend|grab|lift|to-slot|release|retreat|abort-up|open)")
    parser.add_argument("--operator", default=None, help="live-step: name of the person at the e-stop")
    parser.add_argument("--decision", default=None, help="live-step hover: agent-propose decision json to execute")
    parser.add_argument("--table-z", type=float, default=None, help="live-step: override table top z (mm, world) for the descent")
    parser.add_argument("--push-distance", type=float, default=None, help="live-step push: override the computed slide distance (mm)")
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
                                                arm_speed=args.arm_speed, arm_accel=args.arm_accel))
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
