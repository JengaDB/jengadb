"""Run twice against the same saved image; processing is explicitly pretend."""
import argparse
import json
import mimetypes
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robot_hackathon.result_cache import CacheMiss, CacheOutput, ResultCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("data/cache-demo/results.sqlite3"))
    parser.add_argument("--model-version", default="pretend-v1")
    parser.add_argument("--stage", choices=("pile", "assembly", "tower"), default="assembly")
    parser.add_argument("--lookup-only", action="store_true", help="Use saved results only; never process")
    args = parser.parse_args()
    image = args.image.read_bytes()
    if not image:
        parser.error("Image file is empty")

    cache = ResultCache(args.cache)

    def pretend():
        print("Running PRETEND processing (no model or object detection).")
        ref = cache.artifacts.put(image, mimetypes.guess_type(args.image.name)[0] or "application/octet-stream")
        return CacheOutput({"synthetic": True, "message": "Cache demonstration; image was not analyzed",
                            "detected_block_count": None, "input_artifact": ref}, (ref,))

    try:
        result, hit, key = cache.get_or_compute(
            operation="demo.pretend-detection", inputs=[image],
            model_version=args.model_version,
            settings={"preprocessing": "none", "stage": args.stage, "mode": "mock"},
            compute=None if args.lookup_only else pretend)
    except CacheMiss:
        print("CACHE MISS — no usable saved answer; processing was not run.")
        return 2
    print("CACHE HIT — loaded saved answer" if hit else "CACHE MISS — saved new answer")
    print(json.dumps({"key": key, "result": result, "cache": str(args.cache)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
