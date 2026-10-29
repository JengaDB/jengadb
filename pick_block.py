"""Reorient the arm over the table, capture an RGBD frame, locate Jenga blocks
in world coordinates, and (optionally, with --grasp) pick the nearest one.

Uses only the VIAM SDK and the resources already configured on the machine:
arm, gripper, cam, and the builtin motion service (with the `table` and
`ceiling` obstacle geometries in the frame system).

Usage:
    python pick_block.py            # reorient + capture + report candidates
    python pick_block.py --grasp    # also descend and grab the nearest block
    python pick_block.py --no-move  # capture from the current pose only
"""

import argparse
import asyncio
import io
import math
import sys
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.proto.service.motion import CollisionSpecification, Constraints, LinearConstraint
from viam.services.motion import MotionClient

from connect import connect

# --- machine constants (from the frame system / camera properties) ---------
FX, FY, CX, CY = 916.5731, 917.2415, 641.1876, 369.5864  # cam intrinsics (px)
TABLE_TOP_Z_NOMINAL = -23.0  # mm, table box centre -123 with 200 mm thickness
BLOCK_HEIGHT_MM = 15.0       # Jenga block lying flat (75 x 25 x 15)

# survey pose for the gripper TCP: above the table, pointing straight down
SURVEY_POSE = Pose(x=400, y=0, z=350, o_x=0, o_y=0, o_z=-1, theta=0)

# segmentation
GRID_STEP = 4                      # subsample depth image by this factor
WORKSPACE_X = (100.0, 800.0)       # mm, world x band to consider
WORKSPACE_Y = (-500.0, 500.0)      # mm
MIN_BLOCK_PX = 40                  # min grid cells for a candidate component

# grasp geometry (heights refer to the FINGERTIPS, not the gripper frame origin)
# The gripper's `claws` collision box extends 50 mm past the gripper frame, but a
# contact test (grab triggered at TCP z=29 but not z=34 with the block top
# measured at +9) shows the REAL fingertips are only ~25 mm past the frame.
# The motion planner still sees the 50 mm box, so descents below TCP z~27 need
# the gripper<->table collision check relaxed (done for descent/lift only).
REAL_FINGERTIP_OFFSET = 25.0       # mm, calibrated by contact
PRE_GRASP_CLEARANCE = 120.0        # mm fingertips above block top
CONTACT_START_ABOVE_TOP = 3.0      # start contact search with fingertips this far above block top
CONTACT_STEP = 3.0                 # mm per step during contact search
EXTRA_AFTER_CONTACT = 8.0          # mm to descend past first contact before the real grasp
MIN_TABLE_CLEARANCE = 2.0          # mm, never model real fingertips closer than this to the table


def ov_to_pose(x, y, z, o_z, theta):
    return Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=o_z, theta=theta)


async def fingertip_offset_mm(gripper):
    """How far the gripper's collision geometry extends past the gripper frame
    origin along +z (the pointing direction). This is where the planner thinks
    the fingertips are, so grasp heights must be expressed relative to it."""
    tip = 0.0
    for g in await gripper.get_geometries():
        cz = g.center.z
        if g.HasField("box"):
            tip = max(tip, cz + g.box.dims_mm.z / 2.0)
        elif g.HasField("sphere"):
            tip = max(tip, cz + g.sphere.radius_mm)
        elif g.HasField("capsule"):
            tip = max(tip, cz + g.capsule.length_mm / 2.0)
    return tip


async def camera_to_world_matrix(machine):
    """Build a 4x4 cam->world transform by passing the camera origin and three
    unit-axis points through the robot's frame system (exact, 4 RPCs)."""
    pts = [(0, 0, 0), (1000, 0, 0), (0, 1000, 0), (0, 0, 1000)]
    out = []
    for px, py, pz in pts:
        q = PoseInFrame(reference_frame="cam", pose=Pose(x=px, y=py, z=pz, o_z=1))
        r = await machine.transform_pose(q, "world")
        out.append(np.array([r.pose.x, r.pose.y, r.pose.z]))
    origin = out[0]
    R = np.stack([(out[1] - origin) / 1000.0,
                  (out[2] - origin) / 1000.0,
                  (out[3] - origin) / 1000.0], axis=1)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = origin
    return T


def deproject_grid(depth, step):
    """Return (us, vs, cam_xyz) for valid depth pixels on a subsampled grid."""
    h, w = depth.shape
    vs, us = np.mgrid[0:h:step, 0:w:step]
    d = depth[vs, us].astype(np.float64)
    valid = (d > 100) & (d < 2000)
    us, vs, d = us[valid], vs[valid], d[valid]
    X = (us - CX) * d / FX
    Y = (vs - CY) * d / FY
    return us, vs, np.stack([X, Y, d], axis=1)


def label_components(mask):
    """4-connected component labelling on a small boolean grid (no scipy)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    cur = 0
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or labels[sy, sx]:
                continue
            cur += 1
            q = deque([(sy, sx)])
            labels[sy, sx] = cur
            while q:
                y, x = q.popleft()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = cur
                        q.append((ny, nx))
    return labels, cur


def find_blocks(world_xyz, us, vs, depth_shape, step, table_z):
    """Segment points sitting on top of the table plane into block candidates."""
    x, y, z = world_xyz[:, 0], world_xyz[:, 1], world_xyz[:, 2]
    in_ws = (x > WORKSPACE_X[0]) & (x < WORKSPACE_X[1]) & (y > WORKSPACE_Y[0]) & (y < WORKSPACE_Y[1])
    above = (z > table_z + 6.0) & (z < table_z + 45.0)
    sel = in_ws & above

    gh, gw = math.ceil(depth_shape[0] / step), math.ceil(depth_shape[1] / step)
    mask = np.zeros((gh, gw), dtype=bool)
    idx = np.full((gh, gw), -1, dtype=np.int64)
    gy, gx = vs // step, us // step
    mask[gy[sel], gx[sel]] = True
    idx[gy, gx] = np.arange(len(us))

    labels, n = label_components(mask)
    blocks = []
    for k in range(1, n + 1):
        cells = np.argwhere(labels == k)
        if len(cells) < MIN_BLOCK_PX:
            continue
        pidx = idx[cells[:, 0], cells[:, 1]]
        pts = world_xyz[pidx]
        xy = pts[:, :2]
        centroid = xy.mean(axis=0)
        top_z = float(np.percentile(pts[:, 2], 90))
        # PCA for long axis / yaw
        cov = np.cov((xy - centroid).T)
        evals, evecs = np.linalg.eigh(cov)
        major = evecs[:, np.argmax(evals)]
        yaw_deg = math.degrees(math.atan2(major[1], major[0]))
        length = 4 * math.sqrt(max(evals)) if max(evals) > 0 else 0
        width = 4 * math.sqrt(min(evals)) if min(evals) > 0 else 0
        px_u = float(us[pidx].mean())
        px_v = float(vs[pidx].mean())
        blocks.append(dict(cx=float(centroid[0]), cy=float(centroid[1]), top_z=top_z,
                           yaw_deg=yaw_deg, length=length, width=width,
                           n=len(cells), px=(px_u, px_v), cells=cells))
    return blocks


def theta_for_block(block_yaw_deg):
    """Jaws close along the gripper's +y axis. With the gripper pointing down
    (o_z=-1) the jaw axis world angle is (90 - theta); verified empirically via
    the frame system. We want the jaw axis perpendicular to the block's long
    axis so the claws close across the 25 mm width:
        90 - theta = yaw + 90  ->  theta = -yaw
    """
    return -block_yaw_deg


async def capture_and_detect(machine, cam, motion, tag):
    """Capture RGBD from the current pose, convert to world, segment blocks.
    Returns (table_z, blocks sorted by distance from the gripper TCP) or None."""
    g = (await motion.get_pose("gripper", "world")).pose
    c = (await motion.get_pose("cam", "world")).pose
    print(f"    gripper world: ({g.x:.1f},{g.y:.1f},{g.z:.1f}) o=({g.o_x:.2f},{g.o_y:.2f},{g.o_z:.2f}) th={g.theta:.1f}")
    print(f"    cam     world: ({c.x:.1f},{c.y:.1f},{c.z:.1f}) th={c.theta:.1f}")

    imgs, _ = await cam.get_images()
    color = next(i for i in imgs if i.name == "color")
    depth_img = next(i for i in imgs if i.name == "depth")
    rgb = Image.open(io.BytesIO(color.data)).convert("RGB")
    depth = np.array(depth_img.bytes_to_depth_array())
    rgb.save(f"/tmp/jenga_{tag}_color.jpg")
    dv = np.clip(depth.astype(np.float32), 0, 800) / 800 * 255
    Image.fromarray(dv.astype(np.uint8)).save(f"/tmp/jenga_{tag}_depth.png")
    ctr = depth[340:380, 620:660]
    print(f"    depth {depth.shape}, valid={float((depth > 0).mean()):.0%}, "
          f"centre median={np.median(ctr[ctr > 0]) if (ctr > 0).any() else 'n/a'} mm")

    T = await camera_to_world_matrix(machine)
    us, vs, cam_xyz = deproject_grid(depth, GRID_STEP)
    world = (T[:3, :3] @ cam_xyz.T).T + T[:3, 3]
    wx, wy, wz = world[:, 0], world[:, 1], world[:, 2]

    in_ws = (wx > WORKSPACE_X[0]) & (wx < WORKSPACE_X[1]) & (wy > WORKSPACE_Y[0]) & (wy < WORKSPACE_Y[1])
    if in_ws.sum() < 100:
        print("    too few points in workspace; camera may not be looking at the table")
        return None
    table_z = float(np.median(wz[in_ws]))
    print(f"    table plane world z: {table_z:.1f} mm (nominal {TABLE_TOP_Z_NOMINAL:.0f}, "
          f"p10={np.percentile(wz[in_ws],10):.1f}, p90={np.percentile(wz[in_ws],90):.1f})")

    blocks = find_blocks(world, us, vs, depth.shape, GRID_STEP, table_z)
    for b in blocks:
        b["dist"] = math.hypot(b["cx"] - g.x, b["cy"] - g.y)
    blocks.sort(key=lambda b: b["dist"])
    print(f"    {len(blocks)} candidate(s), sorted by distance from gripper:")
    for i, b in enumerate(blocks):
        print(f"    #{i}: centre=({b['cx']:.1f},{b['cy']:.1f}) top_z={b['top_z']:.1f} "
              f"(h={b['top_z']-table_z:.1f}) yaw={b['yaw_deg']:.0f}deg "
              f"size~{b['length']:.0f}x{b['width']:.0f}mm cells={b['n']} dist={b['dist']:.0f}")

    draw = ImageDraw.Draw(rgb)
    for i, b in enumerate(blocks):
        for cy_, cx_ in b["cells"]:
            u, v = cx_ * GRID_STEP, cy_ * GRID_STEP
            draw.rectangle([u, v, u + GRID_STEP - 1, v + GRID_STEP - 1],
                           outline=(255, 0, 0) if i == 0 else (255, 200, 0))
        draw.text((b["px"][0] + 6, b["px"][1] - 6), f"#{i}", fill=(0, 255, 0))
    rgb.save(f"/tmp/jenga_{tag}_blocks.jpg")
    print(f"    saved /tmp/jenga_{tag}_color.jpg /tmp/jenga_{tag}_depth.png /tmp/jenga_{tag}_blocks.jpg")
    return table_z, blocks


async def main(args):
    async with await connect() as machine:
        cam = Camera.from_robot(machine, "cam")
        gripper = Gripper.from_robot(machine, "gripper")
        motion = MotionClient.from_robot(machine, "builtin")
        lin = Constraints(linear_constraint=[LinearConstraint(line_tolerance_mm=5, orientation_tolerance_degs=5)])
        # Straight-line descent/lift with the (oversized) gripper geometry allowed
        # to overlap the table obstacle box. All other collision checks stay on.
        allow_table = [
            Constraints(linear_constraint=lin.linear_constraint,
                        collision_specification=[CollisionSpecification(allows=[
                            CollisionSpecification.AllowedFrameCollisions(frame1=f1, frame2=f2)])])
            for f1, f2 in (("gripper", "table"), ("gripper:claws", "table:table"))
        ]

        async def move_to(label, pose, constraints=None):
            """constraints may be a single Constraints or a list of fallbacks."""
            variants = constraints if isinstance(constraints, list) else [constraints]
            last_err = None
            for cons in variants:
                try:
                    ok = await motion.move("gripper", PoseInFrame(reference_frame="world", pose=pose),
                                           constraints=cons, timeout=90)
                except Exception as e:  # planner refusals surface as GRPCError
                    last_err = str(e)[:300]
                    continue
                if ok:
                    return True
                last_err = "plan returned False"
            print(f"    {label} failed: {last_err}")
            return False

        # ---- 1. reorient over the table ---------------------------------
        if not args.no_move:
            print(f"[1] moving gripper to survey pose "
                  f"({SURVEY_POSE.x:.0f},{SURVEY_POSE.y:.0f},{SURVEY_POSE.z:.0f}) pointing down ...")
            if not await move_to("survey", SURVEY_POSE):
                return 1

        # ---- 2/3. capture RGBD, detect blocks ----------------------------
        print("[2] capturing RGBD from survey pose ...")
        res = await capture_and_detect(machine, cam, motion, "survey")
        if res is None or not res[1]:
            print("[3] no block candidates found above the table plane")
            return 1
        table_z, blocks = res
        target = blocks[0]
        modelled_tip = await fingertip_offset_mm(gripper)
        tip = REAL_FINGERTIP_OFFSET
        print(f"[3] target #0 at ({target['cx']:.1f},{target['cy']:.1f}) top_z={target['top_z']:.1f}; "
              f"fingertip offset: modelled {modelled_tip:.0f} mm, using calibrated {tip:.0f} mm")

        if not args.grasp:
            theta = theta_for_block(target["yaw_deg"])
            print(f"    would use theta={theta:.0f}, pre-grasp fingertips z={target['top_z'] + PRE_GRASP_CLEARANCE:.1f}, "
                  f"contact search from fingertips z={target['top_z'] + CONTACT_START_ABOVE_TOP:.1f}")
            print("[4] --grasp not given; stopping before any pick motion")
            return 0

        # ---- 4. hover above target and re-localize from close range -----
        theta = theta_for_block(target["yaw_deg"])
        hover = ov_to_pose(target["cx"], target["cy"], target["top_z"] + PRE_GRASP_CLEARANCE + tip, -1, theta)
        print(f"[4] opening gripper, hovering at TCP ({hover.x:.1f},{hover.y:.1f},{hover.z:.1f}) theta={theta:.0f} ...")
        await gripper.open()
        if not await move_to("hover", hover):
            return 1
        print("    re-localizing from hover pose ...")
        res = await capture_and_detect(machine, cam, motion, "hover")
        if res is not None and res[1] and res[1][0]["dist"] < 80:
            table_z, blocks = res
            target = blocks[0]
            theta = theta_for_block(target["yaw_deg"])
            print(f"    refined target ({target['cx']:.1f},{target['cy']:.1f}) top_z={target['top_z']:.1f} theta={theta:.0f}")
        else:
            print("    re-localization failed; keeping survey estimate")

        pre = ov_to_pose(target["cx"], target["cy"], target["top_z"] + PRE_GRASP_CLEARANCE + tip, -1, theta)
        if not await move_to("pre-grasp", pre):
            return 1

        # ---- 5. contact search, then grasp deeper ------------------------
        # Camera z has a few mm of bias and the fingertip offset is calibrated,
        # so find the block top by touch: step down and try to grab until the
        # gripper reports contact, then open, go EXTRA_AFTER_CONTACT deeper and
        # grasp for real so the fingers hold the block's sides, not its top edge.
        print("[5] contact search ...")
        floor_tip_z = table_z + MIN_TABLE_CLEARANCE
        tip_z = target["top_z"] + CONTACT_START_ABOVE_TOP
        contact_z = None
        descents = [lin] + allow_table
        while tip_z >= floor_tip_z:
            probe = ov_to_pose(target["cx"], target["cy"], tip_z + tip, -1, theta)
            print(f"    fingertips -> z={tip_z:.1f} (TCP z={probe.z:.1f}, {tip_z - table_z:+.1f} mm vs table)")
            if not await move_to("descent", probe, descents):
                print("    planner refused further descent; stopping search")
                break
            grabbed = await gripper.grab()
            status = await gripper.is_holding_something()
            print(f"    grab() -> {grabbed}, holding={status.is_holding_something} meta={dict(status.meta)}")
            await gripper.open()
            await asyncio.sleep(0.5)
            if grabbed or status.is_holding_something:
                contact_z = tip_z
                break
            tip_z -= CONTACT_STEP

        held = False
        if contact_z is None:
            print("    no contact found above the table; nothing to grasp")
        else:
            grasp_tip_z = max(contact_z - EXTRA_AFTER_CONTACT, floor_tip_z)
            grasp = ov_to_pose(target["cx"], target["cy"], grasp_tip_z + tip, -1, theta)
            print(f"    contact at fingertips z={contact_z:.1f}; grasping at z={grasp_tip_z:.1f} "
                  f"(TCP z={grasp.z:.1f}, {grasp_tip_z - table_z:+.1f} mm vs table)")
            if await move_to("grasp descent", grasp, descents):
                grabbed = await gripper.grab()
                status = await gripper.is_holding_something()
                held = bool(grabbed or status.is_holding_something)
                print(f"    grab() -> {grabbed}, holding={status.is_holding_something} meta={dict(status.meta)}")

        print("    lifting ...")
        await move_to("lift", pre, descents)
        await asyncio.sleep(0.5)
        status = await gripper.is_holding_something()
        print(f"[6] done. grasped={held}, after lift is_holding_something -> {status.is_holding_something} meta={dict(status.meta)}")
        return 0 if (held and status.is_holding_something) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--grasp", action="store_true", help="descend and grab the nearest block")
    p.add_argument("--no-move", action="store_true", help="skip the survey move; capture from current pose")
    sys.exit(asyncio.run(main(p.parse_args())))
