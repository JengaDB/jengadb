"""Observation capture through read-only Viam camera.get_images.

Written against viam-sdk 0.85; issues no motion, stop or configuration calls. Works for the robot machine
and for a local viam-server on this Mac (MACHINE_AUTH=none is accepted only for loopback addresses,
enforced in Config.require_viam). Not yet exercised against a camera; see docs/INTEGRATIONS.md.
"""
import asyncio
import uuid
from datetime import datetime, timezone
from .config import ConfigError
from . import rgbd


def _intrinsics(props):
    intr = getattr(props, "intrinsic_parameters", None)
    if intr is None:
        return None
    return {k: getattr(intr, k, None) for k in ("width_px", "height_px", "focal_x_px", "focal_y_px", "center_x_px", "center_y_px")}


async def open_client(config):
    """Read-only connection to the machine (robot address with API key, or local viam-server without one)."""
    config.require_viam()
    from viam.robot.client import RobotClient
    if config.machine_auth == "none":
        from viam.rpc.dial import DialOptions
        options = RobotClient.Options(dial_options=DialOptions(insecure=True, disable_webrtc=True))
    else:
        options = RobotClient.Options.with_api_key(api_key=config.key, api_key_id=config.key_id)
    return await asyncio.wait_for(RobotClient.at_address(config.address, options), timeout=15)


async def read_frames(camera):
    """One read-only get_images: returns (rgb array, depth array, jpeg_or_png bytes, sources, metadata, received_at)."""
    images, metadata = await asyncio.wait_for(camera.get_images(), timeout=15)
    received_at = datetime.now(timezone.utc)
    rgb = depth = encoded = None
    sources = []
    for image in images:
        mime = str(image.mime_type)
        sources.append({"name": image.name, "mime_type": mime, "bytes": len(image.data)})
        if mime == "image/vnd.viam.dep" and depth is None:
            depth = rgbd.decode_viam_depth(image.data)
        elif rgb is None and ("jpeg" in mime or "png" in mime):
            rgb, kind = rgbd.decode_png(image.data)
            encoded = image.data if kind == "rgb" else None
            if kind != "rgb":
                rgb = None
    return rgb, depth, encoded, sources, metadata, received_at


async def read_camera_pose(client, camera_name, destination="world", axis_mm=100.0):
    """Camera frame -> `destination` rigid transform at this instant, read through the frame system only.

    Transforms the camera origin and three axis points (Sam's flange_axes idea), so no orientation-vector maths is
    needed. Returns None with a reason when the camera has no frame. Read-only: no motion, no configuration."""
    from viam.proto.common import Pose, PoseInFrame
    read_at = datetime.now(timezone.utc)
    try:
        frames = {p.frame.reference_frame for p in await asyncio.wait_for(client.get_frame_system_config(), timeout=10)}
        if camera_name not in frames:
            return {"status": "unavailable", "reason": f"no frame named {camera_name} in the frame system", "frames": sorted(frames)}

        async def point(x=0.0, y=0.0, z=0.0):
            q = PoseInFrame(reference_frame=camera_name, pose=Pose(x=x, y=y, z=z, o_z=1))
            p = (await asyncio.wait_for(client.transform_pose(q, destination), timeout=10)).pose
            return (p.x, p.y, p.z)
        o = await point()
        ex, ey, ez = await point(x=axis_mm), await point(y=axis_mm), await point(z=axis_mm)
        cols = [[(a[i] - o[i]) / axis_mm for a in (ex, ey, ez)] for i in range(3)]  # rows of R: columns are camera axes
        return {"status": "read", "frame_from": camera_name, "frame_to": destination, "rotation": cols, "translation_mm": list(o),
                "read_at": read_at.isoformat(), "method": "frame_system.transform_pose on origin and axis points",
                "note": "pose applicable at read time; for a wrist camera it changes whenever the arm moves"}
    except Exception as exc:  # noqa: BLE001 - frame system is optional for capture
        return {"status": "unavailable", "reason": type(exc).__name__}


async def capture_with_camera(config, data_dir, camera, episode_id=None, client=None):
    """Save one observation from an already-connected camera. Read-only."""
    if config.mode != "live":
        raise ConfigError("Live capture requires MODE=live")
    registry = rgbd.CalibrationRegistry(data_dir, "live")
    version = registry.current(config.setup_id)
    if version is None:
        raise ConfigError(f"No calibration recorded for setup {config.setup_id}; run rgbd-calibration record first")
    cal = registry.load(config.setup_id, version)
    intrinsics = {"rgb": None, "depth": None}
    try:
        props = await asyncio.wait_for(camera.get_properties(), timeout=10)
        intr = _intrinsics(props)
        # The RealSense module registers depth to the colour frame (align_color_depth), so the reported colour
        # intrinsics apply to both images. Recorded explicitly; the calibration record carries the operator verdict.
        intrinsics = {"rgb": intr, "depth": intr,
                      "note": "depth registered to colour by the camera module; colour intrinsics applied to depth"}
    except Exception:  # noqa: BLE001 - properties are optional
        pass
    # Camera pose immediately before and after the frames: if they differ the arm moved during capture.
    pose_before = await read_camera_pose(client, config.camera_resource) if client is not None else None
    rgb, depth, _, sources, metadata, received_at = await read_frames(camera)
    pose_after = await read_camera_pose(client, config.camera_resource) if client is not None else None
    return _save(config, data_dir, rgb, depth, sources, metadata, received_at, intrinsics, cal, version, episode_id,
                 camera_pose=_camera_pose_record(pose_before, pose_after))


def _camera_pose_record(before, after):
    if before is None and after is None:
        return {"status": "not_read", "reason": "no robot client (camera-only capture)"}
    if before is None or after is None or before.get("status") != "read" or after.get("status") != "read":
        return {"status": "unavailable", "before": before, "after": after}
    moved = max(abs(a - b) for a, b in zip(before["translation_mm"], after["translation_mm"]))
    moved = max(moved, 100.0 * max(abs(a - b) for ra, rb in zip(before["rotation"], after["rotation"]) for a, b in zip(ra, rb)))
    return {**before, "read_at_after": after["read_at"], "max_change_mm_between_reads": moved,
            "stable": moved < 1.0}


async def capture_live(config, data_dir, episode_id=None):
    """Connect, capture one observation, disconnect. Read-only; no motion or configuration calls."""
    from viam.components.camera import Camera
    client = await open_client(config)
    try:
        return await capture_with_camera(config, data_dir, Camera.from_robot(client, config.camera_resource), episode_id, client=client)
    finally:
        await asyncio.wait_for(client.close(), timeout=5)


def colorize_depth(depth, near_mm=200, far_mm=1500):
    """Preview-only false colour: near = warm, far = cool, invalid = black. Not stored."""
    import numpy as np
    d = depth.astype(np.float32)
    t = np.clip((d - near_mm) / float(far_mm - near_mm), 0, 1)
    out = np.zeros(depth.shape + (3,), dtype=np.uint8)
    out[..., 0] = (255 * (1 - t)).astype(np.uint8)
    out[..., 1] = (255 * (1 - np.abs(2 * t - 1))).astype(np.uint8)
    out[..., 2] = (255 * t).astype(np.uint8)
    out[depth == 0] = 0
    return out


def _save(config, data_dir, rgb, depth, sources, metadata, received_at, intrinsics, cal, version, episode_id, camera_pose=None):
    flags = []
    camera_pose = camera_pose or {"status": "not_read"}
    if camera_pose.get("status") != "read":
        flags.append("camera_pose_not_recorded")
    elif not camera_pose.get("stable", True):
        flags.append("camera_moved_during_capture")
    if rgb is None or depth is None:
        raise RuntimeError("Camera did not return both an RGB image and a depth map")
    captured_at = None
    skew_ms = None
    stamp = getattr(metadata, "captured_at", None)
    if stamp is not None and getattr(stamp, "seconds", 0):
        captured_dt = stamp.ToDatetime().replace(tzinfo=timezone.utc)
        captured_at = captured_dt.isoformat()
        skew_ms = int((received_at - captured_dt).total_seconds() * 1000)
        if abs(skew_ms) > 2000:
            flags.append("captured_at_far_from_local_clock")
    else:
        flags.append("captured_at_missing_from_response")
    flags.append("depth_unit_mm_assumed")
    obs = rgbd.new_observation(
        "live", config.setup_id, episode_id or f"ep_{uuid.uuid4().hex[:8]}", version,
        alignment=dict(cal["alignment"], calibration_version=version),
        camera={"resource_name": config.camera_resource, "serial": config.camera_serial or None, "model": None,
                "sources": sources, "machine_address_kind": "loopback" if config.machine_auth == "none" else "remote",
                "pose_in_world": camera_pose},
        intrinsics=intrinsics, captured_at=captured_at,
        timing={"clock_source": "viam_response_metadata", "per_image_timestamps": "not_provided",
                "rgb_depth_skew_ms": None, "capture_to_local_receive_ms": skew_ms,
                "uncertainty_note": "One ResponseMetadata.captured_at for both images; per-image skew not exposed"},
        flags=flags)
    return rgbd.ObservationStore(data_dir, "live").save(obs, rgb, depth)


def render_overlay(data_dir, mode, observation_id, out_path=None):
    """Depth-edge overlay for the operator's alignment check. Produces a picture, never a verdict."""
    import numpy as np
    store = rgbd.ObservationStore(data_dir, mode)
    obs, rgb, depth = store.load_arrays(observation_id)
    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError("Overlay requires equal dimensions; this observation is flagged as mismatched")
    d = depth.astype(np.float32)
    gx = np.abs(np.diff(d, axis=1, prepend=d[:, :1]))
    gy = np.abs(np.diff(d, axis=0, prepend=d[:1, :]))
    # RealSense depth noise grows roughly with distance squared; use a distance-relative step so the
    # overlay shows object boundaries rather than sensor noise at any working distance.
    threshold = np.maximum(15.0, 0.04 * d)
    valid = depth > 0
    neighbour_valid = valid & np.roll(valid, 1, axis=1) & np.roll(valid, 1, axis=0)
    edges = ((gx > threshold) | (gy > threshold)) & neighbour_valid
    out = rgb.copy()
    out[edges] = (255, 0, 255)
    out[depth == 0] = (out[depth == 0] * 0.4).astype(np.uint8)
    path = out_path or (store.directory(observation_id) / "overlay.png")
    path.write_bytes(rgbd.encode_rgb_png(out))
    return path
