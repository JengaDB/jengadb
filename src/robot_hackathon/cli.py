import argparse
import asyncio
import json
import logging
from pathlib import Path
from .config import Config, ConfigError
from .local_store import LocalStore


def main(argv=None):
    parser = argparse.ArgumentParser(description="Robot attempt logger. Offline by default; no physical actions.")
    parser.add_argument("command", nargs="?", default="mock-demo", choices=("mock-demo", "report", "viam-check", "mongo-ping", "mongo-index", "flush"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--allow-cloud", action="store_true")
    parser.add_argument("--confirm-write", action="store_true", help="Use only after authorization for this test database")
    args = parser.parse_args(argv)
    try:
        config = Config.from_env()
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
    except ImportError:
        print("Optional SDK unavailable. See README; offline demo remains available.")
        return 2
    except Exception:
        print("Operation failed; diagnostic details suppressed to protect secrets. Local records retained.")
        return 1
