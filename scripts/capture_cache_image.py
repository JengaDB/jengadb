"""Capture one image through an existing Viam SDK; no robot motion or uploads."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4


async def capture(args):
    from viam.robot.client import RobotClient
    from viam.components.camera import Camera
    # Read locally on the machine; never copy or print the machine credentials.
    config = json.loads(args.machine_config.read_text())
    credentials = config["cloud"]["api_key"]
    options = RobotClient.Options.with_api_key(api_key=credentials["key"], api_key_id=credentials["id"])
    robot = await asyncio.wait_for(RobotClient.at_address(args.address, options), 20)
    try:
        cameras = [r.name for r in robot.resource_names if r.subtype == "camera"]
        print("Available cameras:", ", ".join(cameras))
        if args.camera not in cameras:
            raise ValueError("Requested camera is not available")
        started = datetime.now(timezone.utc).isoformat()
        images, metadata = await asyncio.wait_for(Camera.from_robot(robot, args.camera).get_images(timeout=15), 20)
        received = datetime.now(timezone.utc).isoformat()
        rgb = next((im for im in images if str(im.mime_type.value) in ("image/jpeg", "image/png")), None)
        if rgb is None:
            raise ValueError("Camera returned no JPEG or PNG image")
        from PIL import Image
        import io
        with Image.open(io.BytesIO(rgb.data)) as decoded:
            decoded.verify()
        snapshot_id = str(uuid4())
        directory = args.output / snapshot_id
        directory.mkdir(parents=True, exist_ok=False)
        suffix = ".jpg" if rgb.mime_type.value == "image/jpeg" else ".png"
        path = directory / ("image" + suffix)
        with path.open("xb") as handle:
            handle.write(rgb.data)
            handle.flush()
            import os
            os.fsync(handle.fileno())
        record = {"snapshot_id": snapshot_id, "source": "live-camera", "camera": args.camera,
                  "image_source": rgb.name, "requested_at": started, "received_at": received,
                  "captured_at": metadata.captured_at.ToJsonString() if metadata.HasField("captured_at") else None,
                  "sha256": hashlib.sha256(rgb.data).hexdigest(), "image": path.name,
                  "depth_saved": False, "calibration_version": None}
        with (directory / "metadata.json").open("x") as handle:
            json.dump(record, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        print("Saved image:", path)
        print("Image bytes:", len(rgb.data))
    finally:
        await asyncio.wait_for(robot.close(), 5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True)
    parser.add_argument("--camera", default="cam")
    parser.add_argument("--machine-config", type=Path, default=Path("/etc/viam.json"))
    parser.add_argument("--output", type=Path, default=Path("data/snapshots"))
    args = parser.parse_args()
    try:
        asyncio.run(capture(args))
    except Exception as error:
        print("Capture failed (" + type(error).__name__ + "). No motion was requested.", file=sys.stderr)
        raise SystemExit(1)
