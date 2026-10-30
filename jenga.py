import asyncio
import io
import math
import statistics
import struct
from pathlib import Path
from typing import NamedTuple

from PIL import Image, ImageDraw, ImageFilter

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import (Geometry, GeometriesInFrame, Pose, PoseInFrame, RectangularPrism, Vector3,
                               WorldState)
from viam.proto.service.motion import CollisionSpecification, Constraints, LinearConstraint
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

from connect import connect

# --- Robot setup ---
# Where every run starts and scans from: high enough for the depth camera to see the whole pile,
# gripper pointing straight down
# HOME_POSE = Pose(x=400, y=80, z=300, o_x=0, o_y=0, o_z=-1, theta=0)
HOME_POSE = Pose(x=300, y=150, z=400, o_x=0, o_y=0, o_z=-1, theta=0)
SAFE_Z_MM = 150          # arm height that's clear of the blocks, reached straight up before heading home

# --- Blocks ---
BLOCK_LENGTH_MM = 75
BLOCK_WIDTH_MM = 25
BLOCK_HEIGHT_MM = 15

# --- Vision ---
PIXEL_STEP = 4           # only sample every Nth depth pixel, for speed
TABLE_TOLERANCE_MM = 4   # depth noise allowed on the table surface when fitting it
BLOCK_MIN_HEIGHT_MM = 8  # samples at least this far above the table count as block
BLOCK_MAX_HEIGHT_MM = 100
EDGE_SLOPE_MM = 4        # height change over 2 samples that marks the edge between blocks
DARK_EDGE = 15           # how much darker than its surroundings a pixel must be to count as a gap between blocks
MIN_BLOB_SAMPLES = 30
OFF_TABLE_MM = 20        # depth this far below the table plane means the floor past the table's edge
BLOCK_LOOK_LENGTH_MM = (40, 95)  # what a block's top can measure from above to count as a block
BLOCK_LOOK_WIDTH_MM = (8, 28)    # real blocks measure about 17 to 23 mm, after their edges are trimmed
BLOCK_LOOK_MAX_HEIGHT_MM = 45    # three blocks stacked flat

# --- Grasping ---
# The arm's flange z when the closed claw tips touch the table (three touch tests gave 171.0 to 171.5).
# Heights near the table are measured from this, with the camera only saying how tall a block is above
# the table, since the camera's own idea of the table's height drifts by 10 mm or more between sessions.
# Doesn't depend on the gripper or camera frames in the config; measure it again if the table or claws change.
FLANGE_Z_AT_TABLE_TOUCH_MM = 171.5
GRASP_DEPTH_MM = 8       # how far below the block's top surface the claw tips go before closing
ENDS_EXTRA_DEPTH_MM = 10 # the end grip goes this much deeper still
TIPS_MIN_MM = 1          # never send the claw tips lower than this above the table
PICK_LIFT_MM = 100       # how far to raise the block after grabbing it
GRIPPER_OPEN_MM = 104    # gap between the open claws' inner faces: the 120 mm claws box minus two 8 mm claws. Measure it
FINGER_WIDTH_MM = 45     # claw size along the block's long side
CLAW_MARGIN_MM = 5       # gap left beside the block so its own edges don't count as an obstruction
OBSTRUCTION_MIN_MM = 6   # anything lower than this in the claws' path is table noise
GRASP_SIDE_SHIFT_MM = 15  # how far toward one end the side grip holds the block, from its center
ACROSS = 90              # claws closing across the block's narrow sides (middle and side grips)
ALONG = 0                # claws closing on its two ends
# Each grip as (claw angle to the block's long side, [grip points along it from the center])
GRIP_POINTS = {
    "middle": (ACROSS, [0]),
    "side": (ACROSS, [GRASP_SIDE_SHIFT_MM, -GRASP_SIDE_SHIFT_MM]),
    "ends": (ALONG, [0]),
}
SKIP_RADIUS_MM = 15      # a block seen within this of one already given up on is the same block

# --- Placing ---
CLAW_THICKNESS_MM = 8    # one claw, measured across the block
PLACE_CLEARANCE_MM = 3   # extra room on top of a claw's thickness when setting a block down beside another
PLACE_DROP_MM = 1        # let go this far above where the block rests, so it isn't pressed into the table
ROW_LONG_SIDE_DEG = 0    # world direction of the long side of blocks placed in a row
ROW_BLOCKS = 3
# Opening the claws to let go of a block pushes its neighbor away, so a finished row gets pushed together
PUSH_DISTANCE_MM = 60    # how far the closed claws slide, from just outside the last block toward the first
PUSH_START_GAP_MM = 5    # gap between the closed claws and the last block before the push starts
PUSH_TIPS_MM = 5         # claw tips' height above the table while pushing

# --- Sweeping the pile flat ---
FLAT_LAYER_MAX_MM = 20   # anything taller than this isn't a single block lying flat on the table
SWEEP_HEIGHT_MM = 14     # claws' bottom above the table while sweeping: clears flat blocks, hits anything taller
SWEEP_REACH_MM = 70      # sweep from this far on one side of a tall block to this far past it
SWEEP_START_RADIUS_MM = 30  # space the closed claws need to come down at the start of a sweep
SWEEP_ROUNDS = 6

# --- Pile and tower ---
PILE_MARGIN_MM = 50      # a block is part of the pile if it's within this of where a pile block was at the start
TOWER_SIZE_MM = 75       # three blocks side by side
TOWER_CLEARANCE_MM = 75  # nothing within this of the tower's center: the open claws reach 60 mm out when placing
TOWER_MARGIN_MM = 10     # padding around the tower obstacle

# Straight-line moves where the gripper is allowed to touch the table, for reaching down to a block on it
NEAR_TABLE = Constraints(
    linear_constraint=[LinearConstraint()],
    collision_specification=[CollisionSpecification(
        allows=[CollisionSpecification.AllowedFrameCollisions(frame1="gripper", frame2="table")]
    )],
)


class Tower:
    def __init__(self, center: Pose):
        self.center = center  # middle of the tower's footprint, on the table surface
        self.layers = 0

    def geometry(self) -> Geometry:
        # Reserve at least one layer's worth of space before anything is built
        height = max(self.layers, 1) * BLOCK_HEIGHT_MM + TOWER_MARGIN_MM
        size = TOWER_SIZE_MM + 2 * TOWER_MARGIN_MM
        return Geometry(
            center=Pose(x=self.center.x, y=self.center.y, z=self.center.z + height / 2, o_z=1),
            box=RectangularPrism(dims_mm=Vector3(x=size, y=size, z=height)),
            label="tower",
        )


class Pile:
    def __init__(self, centers: list[Pose]):
        self.centers = centers  # where each block was when the pile was designated

    def contains(self, p: Pose) -> bool:
        # Generous enough that blocks nudged by picking their neighbors still count
        return any(math.hypot(p.x - c.x, p.y - c.y) <= PILE_MARGIN_MM for c in self.centers)


# Every run starts fresh: designate_pile() takes whatever blocks are on the table as the pile,
# then choose_tower_site() picks a new spot beside it for the tower
pile: Pile | None = None    # only blocks inside it are ever picked or swept
tower: Tower | None = None  # avoided by every move


# --- Moving ---

def get_obstacles() -> WorldState:
    # The planner already avoids the table and ceiling, since they're in the robot's config with
    # their own geometries (sending them again is rejected as a duplicate). Only the tower is new.
    obstacles = []
    if tower is not None:
        obstacles.append(GeometriesInFrame(reference_frame="world", geometries=[tower.geometry()]))
    return WorldState(obstacles=obstacles)


async def move(machine: RobotClient, arm: Arm, x: float = 0, y: float = 0, z: float = 0,
               rotate: float = 0, constraints: Constraints | None = None):
    # Shift the end effector by (x, y, z) mm in the world frame and spin it by `rotate` degrees
    # about its own axis, letting the motion service plan a path that avoids the obstacles
    motion = MotionClient.from_robot(machine, "builtin")
    current = (await motion.get_pose(arm.name, "world")).pose
    target = Pose(
        x=current.x + x, #+f-b
        y=current.y + y, #+l-r
        z=current.z + z, #up
        o_x=current.o_x,
        o_y=current.o_y,
        o_z=current.o_z,
        theta=current.theta + rotate,
    )
    destination = PoseInFrame(reference_frame="world", pose=target)
    await motion.move(arm.name, destination, world_state=get_obstacles(), constraints=constraints)


async def go_home(machine: RobotClient, arm: Arm, gripper: Gripper, home_pose: Pose = HOME_POSE):
    # Put the robot in the same known state at the start of every run, wherever the last run left it
    await arm.stop()
    await gripper.stop()
    # Open before lifting so a block left in the claws is set down where it is, not dropped from height
    await gripper.open()

    # If the claws are down among the blocks, back straight up before heading anywhere
    motion = MotionClient.from_robot(machine, "builtin")
    current = (await motion.get_pose(arm.name, "world")).pose
    if current.z < SAFE_Z_MM:
        await move(machine, arm, z=SAFE_Z_MM - current.z, constraints=NEAR_TABLE)

    home = PoseInFrame(reference_frame="world", pose=home_pose)
    await motion.move(arm.name, home, world_state=get_obstacles())


async def back_off(machine: RobotClient, arm: Arm, gripper: Gripper, scan_z: float):
    # Straight back up to scanning height, the one direction that's almost always clear
    try:
        grip = await frame_point_in_world(machine, gripper.name)
        await move(machine, arm, z=scan_z - grip.z, constraints=NEAR_TABLE)
    except Exception as e:
        raise RuntimeError("Couldn't back away. The arm may have faulted on a collision: "
                           "clear its error and move it clear by hand") from e


# --- Geometry helpers ---

async def tips_on_table_z(machine: RobotClient) -> float:
    # The gripper frame's z when the closed claw tips touch the table. The gripper points straight down,
    # so its frame sits translation.z below the flange; reading that live keeps this right if it's changed.
    for part in await machine.get_frame_system_config():
        if part.frame.reference_frame == "gripper":
            return FLANGE_Z_AT_TABLE_TOUCH_MM - part.frame.pose_in_observer_frame.pose.z
    raise RuntimeError("No gripper frame in the config")


_intrinsics = {}


async def intrinsics(cam: Camera):
    # Fixed for the camera, so fetch once instead of on every pixel conversion
    if cam.name not in _intrinsics:
        _intrinsics[cam.name] = (await cam.get_properties()).intrinsic_parameters
    return _intrinsics[cam.name]


async def pixel_to_world(machine: RobotClient, cam: Camera, u: float, v: float, depth: float) -> Pose:
    # Pinhole model: pixel + depth -> point in the camera frame, then let the frame system place it in the world
    k = await intrinsics(cam)
    x = (u - k.center_x_px) * depth / k.focal_x_px
    y = (v - k.center_y_px) * depth / k.focal_y_px
    point = PoseInFrame(reference_frame=cam.name, pose=Pose(x=x, y=y, z=depth, o_z=1))
    return (await machine.transform_pose(point, "world")).pose


async def frame_point_in_world(machine: RobotClient, frame: str, x: float = 0, y: float = 0) -> Pose:
    point = PoseInFrame(reference_frame=frame, pose=Pose(x=x, y=y, o_z=1))
    return (await machine.transform_pose(point, "world")).pose


def world_angle(a: Pose, b: Pose) -> float:
    # Direction from a to b on the table, in degrees
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x))


def wrap_90(angle: float) -> float:
    # The gripper is symmetric, so an angle and the same angle +180 are the same grasp
    return (angle + 90) % 180 - 90


# --- Vision ---

class Block(NamedTuple):
    u: float          # pixel center
    v: float
    depth: float      # mm from the camera to the block's top
    height: float     # mm above the table
    angle: float      # direction of the long side in the image, degrees
    length_px: float  # size of the top face in the image
    width_px: float


class Sighting(NamedTuple):
    center: Pose      # block's top-center in the world
    angle: float      # world direction of the block's long side, degrees
    block: Block
    heights: dict     # (u, v) sample -> mm above the table, for the whole frame
    mm_per_px: float  # image scale at the block's depth


async def get_frame(cam: Camera):
    # Returns the depth image and the color JPEG from the same capture.
    # Viam depth format: b"DEPTHMAP", width and height as uint64, then uint16 mm per pixel
    images, _ = await cam.get_images()
    data = next(img.data for img in images if img.data.startswith(b"DEPTHMAP"))
    color = next(img.data for img in images if not img.data.startswith(b"DEPTHMAP"))
    width, height = struct.unpack(">QQ", data[8:24])
    pixels = struct.unpack(f">{width * height}H", data[24:])
    return width, height, pixels, color


def dark_lines(jpeg: bytes):
    # How much darker each pixel is than its surroundings. Blocks lying side by side are the same
    # height, so depth can't split them, but the shadow line between them shows up here.
    img = Image.open(io.BytesIO(jpeg)).convert("L")
    darkest = img.filter(ImageFilter.MinFilter(5)).load()
    background = img.filter(ImageFilter.BoxBlur(12)).load()
    return lambda u, v: background[u, v] - darkest[u, v]


def solve_3x3(a, b):
    # Cramer's rule, enough for a least-squares plane fit
    def det(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    d = det(a)
    result = []
    for i in range(3):
        m = [row[:] for row in a]
        for r in range(3):
            m[r][i] = b[r]
        result.append(det(m) / d)
    return result


def fit_table(samples):
    # The camera is slightly tilted, so the table's depth changes across the image.
    # A flat surface seen through a pinhole camera has 1/depth linear in (u, v), so fit that,
    # then refit on just the points near the plane so the blocks don't skew it.
    points = list(samples.items())[::3]
    inliers = points
    for tolerance in (15, 8, 5, TABLE_TOLERANCE_MM):
        a = [[0.0] * 3 for _ in range(3)]
        b = [0.0] * 3
        for (u, v), d in inliers:
            x = (u / 1000, v / 1000, 1)
            for i in range(3):
                b[i] += x[i] * 1000 / d
                for j in range(3):
                    a[i][j] += x[i] * x[j]
        c = solve_3x3(a, b)
        inliers = [((u, v), d) for (u, v), d in points
                   if abs(1000 / (c[0] * u / 1000 + c[1] * v / 1000 + c[2]) - d) < tolerance]
    return lambda u, v: 1000 / (c[0] * u / 1000 + c[1] * v / 1000 + c[2])


def find_blocks(width: int, height: int, pixels, darkness) -> tuple[list[Block], dict]:
    # Split everything sticking up from the table into separate blocks, highest first.
    # Also returns every sample's height above the table. `darkness` comes from dark_lines().
    samples = {}
    for v in range(0, height, PIXEL_STEP):
        for u in range(0, width, PIXEL_STEP):
            d = pixels[v * width + u]
            if d:
                samples[(u, v)] = d

    total = (width // PIXEL_STEP) * (height // PIXEL_STEP)
    if len(samples) < 0.2 * total:
        raise RuntimeError("Not enough depth data, the camera is probably too close to the table")

    table_depth = fit_table(samples)
    heights = {p: table_depth(*p) - d for p, d in samples.items()}

    # Touching and stacked blocks are joined by sloped edges in the depth image, and blocks lying
    # side by side by a dark line in the color image, so drop both to split them apart
    s = PIXEL_STEP
    mask = {
        p for p, h in heights.items()
        if BLOCK_MIN_HEIGHT_MM < h < BLOCK_MAX_HEIGHT_MM
        and all(abs(heights.get((p[0] + du, p[1] + dv), h) - h) <= EDGE_SLOPE_MM
                for du, dv in ((2 * s, 0), (-2 * s, 0), (0, 2 * s), (0, -2 * s)))
        and darkness(*p) < DARK_EDGE
    }

    blocks, seen = [], set()
    for start in mask:
        if start in seen:
            continue
        blob = [start]
        seen.add(start)
        i = 0
        while i < len(blob):
            u, v = blob[i]
            i += 1
            for n in ((u + s, v), (u - s, v), (u, v + s), (u, v - s)):
                if n in mask and n not in seen:
                    seen.add(n)
                    blob.append(n)
        if len(blob) < MIN_BLOB_SAMPLES:
            continue

        u = sum(p[0] for p in blob) / len(blob)
        v = sum(p[1] for p in blob) / len(blob)
        # Principal axis of the blob's pixels is the block's long side. For a filled rectangle the spread
        # along each axis is (side length)^2 / 12, which gives its length and width.
        cuu = sum((p[0] - u) ** 2 for p in blob) / len(blob)
        cvv = sum((p[1] - v) ** 2 for p in blob) / len(blob)
        cuv = sum((p[0] - u) * (p[1] - v) for p in blob) / len(blob)
        mid, half_gap = (cuu + cvv) / 2, math.hypot((cuu - cvv) / 2, cuv)
        blocks.append(Block(
            u=u,
            v=v,
            depth=statistics.median(samples[p] for p in blob),
            height=statistics.median(heights[p] for p in blob),
            angle=math.degrees(0.5 * math.atan2(2 * cuv, cuu - cvv)),
            length_px=math.sqrt(12 * (mid + half_gap)),
            width_px=math.sqrt(12 * max(mid - half_gap, 0)),
        ))

    return sorted(blocks, key=lambda b: b.height, reverse=True), heights


def looks_like_block(b: Block, focal: float) -> bool:
    # Jenga-block-sized from above and not stacked impossibly high, so mugs, phones and the like are
    # left alone. The ranges allow for trimmed edges, a block standing on its side (a narrower top),
    # and a block partly covered by another one (a shorter visible top).
    scale = b.depth / focal
    return (BLOCK_LOOK_LENGTH_MM[0] <= b.length_px * scale <= BLOCK_LOOK_LENGTH_MM[1]
            and BLOCK_LOOK_WIDTH_MM[0] <= b.width_px * scale <= BLOCK_LOOK_WIDTH_MM[1]
            and b.height <= BLOCK_LOOK_MAX_HEIGHT_MM)


async def detect_blocks(width: int, height: int, pixels, color: bytes) -> tuple[list[Block], dict]:
    # The vision maths is up to a second or so of plain Python. Running it on a worker thread keeps the
    # connection's heartbeats going meanwhile; if they stop for a few seconds the robot ends the session
    # (SESSION_EXPIRED) and stops the arm.
    return await asyncio.to_thread(lambda: find_blocks(width, height, pixels, dark_lines(color)))


async def pile_blocks(machine: RobotClient, cam: Camera):
    # Look at the table and return (blocks, their world centers, heights), keeping only jenga blocks
    # in the pile so the tower and anything else on the table never get picked from or swept
    width, height, pixels, color = await get_frame(cam)
    blocks, heights = await detect_blocks(width, height, pixels, color)
    focal = (await intrinsics(cam)).focal_x_px
    blocks = [b for b in blocks if looks_like_block(b, focal)]
    centers = [await pixel_to_world(machine, cam, b.u, b.v, b.depth) for b in blocks]
    keep = [i for i, c in enumerate(centers)
            if (pile is None or pile.contains(c))
            and (tower is None or math.hypot(c.x - tower.center.x, c.y - tower.center.y) > TOWER_SIZE_MM)]
    return [blocks[i] for i in keep], [centers[i] for i in keep], heights


async def locate_block(machine: RobotClient, cam: Camera, near: Pose | None = None) -> Sighting:
    # Picks the highest block (nothing is resting on it), or the one closest to `near` so
    # repeated looks keep tracking the same block
    blocks, centers, heights = await pile_blocks(machine, cam)
    if not blocks:
        raise RuntimeError("No block found on the table")
    if near is None:
        print("Blocks found, heights above table: " + ", ".join(f"{b.height:.0f} mm" for b in blocks))
        i = 0
    else:
        i = min(range(len(blocks)), key=lambda j: math.hypot(centers[j].x - near.x, centers[j].y - near.y))

    return await make_sighting(machine, cam, blocks[i], centers[i], heights)


async def make_sighting(machine: RobotClient, cam: Camera, block: Block, center: Pose, heights: dict) -> Sighting:
    step = math.radians(block.angle)
    along = await pixel_to_world(machine, cam, block.u + 50 * math.cos(step), block.v + 50 * math.sin(step), block.depth)
    focal = (await intrinsics(cam)).focal_x_px
    return Sighting(center, world_angle(center, along), block, heights, block.depth / focal)


# --- Setting up the pile and tower ---

async def designate_pile(machine: RobotClient, cam: Camera):
    # Take every block on the table right now as this run's pile
    global pile, tower
    pile, tower = None, None
    _, centers, _ = await pile_blocks(machine, cam)
    if not centers:
        raise RuntimeError("No blocks on the table to build with")
    pile = Pile(centers)
    print(f"Pile has {len(centers)} blocks")


def find_tower_site(width: int, height: int, heights: dict, mm_per_px: float, clearance_mm: float = TOWER_CLEARANCE_MM):
    # Find the open patch of table nearest the pile, with no blocks, table edge or unseen
    # area within clearance_mm. Returns its pixel (u, v), or None if there isn't one.
    s = PIXEL_STEP
    g = 4 * s  # search on a coarser grid to keep this fast
    bad_samples, raised = {}, []
    for v in range(0, height, s):
        for u in range(0, width, s):
            h = heights.get((u, v))
            if h is None or h < -OFF_TABLE_MM or h > BLOCK_MIN_HEIGHT_MM:
                cell = (u // g, v // g)
                bad_samples[cell] = bad_samples.get(cell, 0) + 1
            if h is not None and h > BLOCK_MIN_HEIGHT_MM:
                raised.append((u, v))
    # A cell with only a stray bad sample or two is depth noise, not an object
    blocked_cells = {cell for cell, n in bad_samples.items() if n >= 3}

    # Grow every blocked cell by the clearance radius; whatever is left is far enough from everything
    r = clearance_mm / mm_per_px / g
    disk = [(du, dv) for du in range(-math.ceil(r), math.ceil(r) + 1)
            for dv in range(-math.ceil(r), math.ceil(r) + 1) if du * du + dv * dv <= r * r]
    too_close = {(cu + du, cv + dv) for cu, cv in blocked_cells for du, dv in disk}

    edge = r * g  # the clearance circle has to fit inside the image, since we can't see past it
    open_cells = [(cu, cv) for cu in range(width // g) for cv in range(height // g)
                  if (cu, cv) not in too_close
                  and edge <= cu * g + g / 2 <= width - edge and edge <= cv * g + g / 2 <= height - edge]
    if not open_cells:
        return None

    # Closest to the pile keeps the moves between pile and tower short
    if raised:
        pile_u = sum(p[0] for p in raised) / len(raised)
        pile_v = sum(p[1] for p in raised) / len(raised)
    else:
        pile_u, pile_v = width / 2, height / 2
    cu, cv = min(open_cells, key=lambda c: (c[0] * g + g / 2 - pile_u) ** 2 + (c[1] * g + g / 2 - pile_v) ** 2)
    return cu * g + g / 2, cv * g + g / 2


async def find_clear_spot(machine: RobotClient, cam: Camera, clearance_mm: float) -> Pose:
    # The open patch of table nearest the pile with nothing within clearance_mm, on the table surface
    width, height, pixels, color = await get_frame(cam)
    _, heights = await detect_blocks(width, height, pixels, color)
    focal = (await intrinsics(cam)).focal_x_px
    table_depth = statistics.median(pixels[v * width + u] for (u, v), h in heights.items() if abs(h) < TABLE_TOLERANCE_MM)

    site = await asyncio.to_thread(find_tower_site, width, height, heights, table_depth / focal, clearance_mm)
    if site is None:
        raise RuntimeError(f"No clear spot {2 * clearance_mm:.0f} mm across in view. "
                           "Push the blocks together or raise the camera")

    u, v = site
    nearby = [pixels[y * width + x] for x in range(int(u) - 8, int(u) + 9, 4)
              for y in range(int(v) - 8, int(v) + 9, 4) if pixels[y * width + x]]
    return await pixel_to_world(machine, cam, u, v, statistics.median(nearby))


async def choose_tower_site(machine: RobotClient, cam: Camera):
    # Pick where to build this run's tower and register it as an obstacle for every move after this
    global tower
    center = await find_clear_spot(machine, cam, TOWER_CLEARANCE_MM)
    tower = Tower(center)
    print(f"Building the tower at ({center.x:.0f}, {center.y:.0f}), table height {center.z:.0f} mm")


# --- Picking up a block ---

def claw_obstruction(sighting: Sighting, shift_mm: float = 0, claws_deg: float = ACROSS) -> float | None:
    # Look where the two open claws come down beside the block, gripping `shift_mm` along its long side
    # from the center, with the claws closing at claws_deg to its long side. Only those strips have to be
    # clear; something lower than the claws in between gets pushed toward the block as they close.
    # Returns the height of whatever is in the way, or None if both strips are clear.
    b = sighting.block
    scale = 1 / sighting.mm_per_px
    a = math.radians(b.angle)
    c = math.radians(b.angle + claws_deg)
    close_u, close_v = math.cos(c), math.sin(c)  # the way the claws close
    face_u, face_v = -close_v, close_u           # across each claw's face

    # Anything taller than the bottom of the claws at grasp height would be hit
    limit = max(grip_tips_mm(b.height, claws_deg), OBSTRUCTION_MIN_MM)
    # Each open claw's inner face is GRIPPER_OPEN_MM / 2 out, and it's CLAW_THICKNESS_MM thick. The margin
    # allows for the grip being a few mm off; it never reaches in over the block's own edge.
    half_block = (BLOCK_LENGTH_MM if claws_deg == ALONG else BLOCK_WIDTH_MM) / 2
    inner = max(half_block + CLAW_MARGIN_MM, GRIPPER_OPEN_MM / 2 - CLAW_MARGIN_MM)
    outer = GRIPPER_OPEN_MM / 2 + CLAW_THICKNESS_MM + CLAW_MARGIN_MM
    s = PIXEL_STEP
    cu = b.u + shift_mm * scale * math.cos(a)
    cv = b.v + shift_mm * scale * math.sin(a)

    in_the_way = []
    for side in (1, -1):
        t = -FINGER_WIDTH_MM / 2
        while t <= FINGER_WIDTH_MM / 2:
            r = inner
            while r <= outer:
                closing, facing = side * r * scale, t * scale
                u = cu + closing * close_u + facing * face_u
                v = cv + closing * close_v + facing * face_v
                h = sighting.heights.get((round(u / s) * s, round(v / s) * s))
                if h is not None and h > limit:
                    in_the_way.append(h)
                r += s / scale
            t += s / scale

    # A few samples over the limit can be noise; more than that is a real object
    if len(in_the_way) < 3:
        return None
    return statistics.median(in_the_way)


async def center_over_block(machine: RobotClient, arm: Arm, cam: Camera, gripper: Gripper,
                            near: Pose | None = None, passes: int = 3, tolerance_mm: float = 2) -> Pose:
    # Re-detect after each move so small calibration errors get corrected
    for _ in range(passes):
        near = (await locate_block(machine, cam, near)).center
        grip = await frame_point_in_world(machine, gripper.name)

        dx, dy = near.x - grip.x, near.y - grip.y
        print(f"Block at ({near.x:.1f}, {near.y:.1f}), gripper off by ({dx:.1f}, {dy:.1f}) mm")
        if abs(dx) < tolerance_mm and abs(dy) < tolerance_mm:
            break
        await move(machine, arm, x=dx, y=dy)
    return near


async def align_with_block(machine: RobotClient, arm: Arm, cam: Camera, gripper: Gripper,
                           near: Pose, claws_deg: float = ACROSS, passes: int = 3, tolerance_deg: float = 3) -> Pose:
    # Spin the gripper so its claws close at claws_deg to the block's long side
    sign, last_error = 1, None
    for _ in range(passes):
        near, block_angle, *_ = await locate_block(machine, cam, near)
        # The claws open and close along the gripper's y axis
        close_angle = world_angle(
            await frame_point_in_world(machine, gripper.name),
            await frame_point_in_world(machine, gripper.name, y=100),
        )
        error = wrap_90(block_angle + claws_deg - close_angle)
        print(f"Block long side at {block_angle:.1f} deg, gripper off by {error:.1f} deg")
        if abs(error) < tolerance_deg:
            break
        # If the last turn made things worse, theta turns the other way on this arm
        if last_error is not None and abs(error) > abs(last_error):
            sign = -sign
        last_error = error
        await move(machine, arm, rotate=-sign * error)
    return near


def grip_tips_mm(block_height: float, claws_deg: float) -> float:
    # How far above the table the claw tips go to grip a block this tall
    depth = GRASP_DEPTH_MM + (ENDS_EXTRA_DEPTH_MM if claws_deg == ALONG else 0)
    return max(block_height - depth, TIPS_MIN_MM)


class Grip(NamedTuple):
    tips_mm: float    # how far above the table the claw tips were when they closed on the block
    held_x: float     # where the block's center sits in the gripper frame, so placing can put the
    held_y: float     # block itself (not the gripper) on the spot
    claws_deg: float  # angle between the claws' closing direction and the block's long side


async def try_grasp(machine: RobotClient, arm: Arm, gripper: Gripper, sighting: Sighting, shift_mm: float,
                    claws_deg: float) -> Grip | None:
    # One attempt from directly above the block. Returns how it's held, or None if it wasn't picked up.
    a = math.radians(sighting.angle)
    if shift_mm:
        await move(machine, arm, x=shift_mm * math.cos(a), y=shift_mm * math.sin(a))
    c = sighting.center
    held = (await machine.transform_pose(
        PoseInFrame(reference_frame="world", pose=Pose(x=c.x, y=c.y, z=c.z, o_z=1)), gripper.name)).pose

    # The block's height comes from up here, since the depth camera can't see it once we get close
    grip = await frame_point_in_world(machine, gripper.name)
    try:
        tips_mm = grip_tips_mm(sighting.block.height, claws_deg)
        target = await tips_on_table_z(machine) + tips_mm
        await move(machine, arm, z=target - grip.z, constraints=NEAR_TABLE)
    except Exception as e:
        print(f"Couldn't reach down to the block: {e}")
        return None

    if not await gripper.grab():
        print("Gripper closed but isn't holding anything")
        await gripper.open()
        return None
    await move(machine, arm, z=PICK_LIFT_MM, constraints=NEAR_TABLE)
    return Grip(tips_mm, held.x, held.y, claws_deg)


def grip_fits(sighting: Sighting, grip_points) -> bool:
    # Whether the claws have room for at least one of grip_points, a (claws_deg, [shift_mm, ...]) pair
    claws_deg, shifts_mm = grip_points
    return any(claw_obstruction(sighting, shift, claws_deg) is None for shift in shifts_mm)


async def choose_block(machine: RobotClient, cam: Camera, skip: list[Pose] = (), grips=None) -> Sighting | None:
    # The highest block, not one in `skip`, that at least one of `grips` has room for as seen from here
    # (any grip if not given). None if every block has something in the way.
    grips = grips or list(GRIP_POINTS.values())
    blocks, centers, heights = await pile_blocks(machine, cam)
    for block, center in zip(blocks, centers):  # highest first
        if any(math.hypot(center.x - s.x, center.y - s.y) < SKIP_RADIUS_MM for s in skip):
            continue
        sighting = await make_sighting(machine, cam, block, center, heights)
        if any(grip_fits(sighting, g) for g in grips):
            return sighting
        print(f"Skipping the block at ({center.x:.0f}, {center.y:.0f}): something is in the claws' way")
    return None


async def grab_with(machine: RobotClient, arm: Arm, cam: Camera, gripper: Gripper, block: Pose,
                    claws_deg: float, shifts_mm: list[float]) -> Grip | None:
    # Go over the block at `block` and pick it up with the claws closing at claws_deg to its long side,
    # gripping at the first of shifts_mm (along it from the center) that the claws have room for.
    # Returns how it's held, or None if none of them work, leaving the arm at scanning height.
    block = await center_over_block(machine, arm, cam, gripper, block)
    block = await align_with_block(machine, arm, cam, gripper, block, claws_deg)
    scan_z = (await frame_point_in_world(machine, gripper.name)).z

    for shift in shifts_mm:
        block = await center_over_block(machine, arm, cam, gripper, block)
        sighting = await locate_block(machine, cam, block)
        obstruction = claw_obstruction(sighting, shift, claws_deg)
        if obstruction is not None:
            print(f"Something {obstruction:.0f} mm tall is in the claws' way gripping at {shift:+} mm")
            continue

        grip = await try_grasp(machine, arm, gripper, sighting, shift, claws_deg)
        if grip is not None:
            return grip
        await back_off(machine, arm, gripper, scan_z)
    return None


async def grab_middle(machine: RobotClient, arm: Arm, cam: Camera, gripper: Gripper, block: Pose) -> Grip | None:
    # Claws across the block's narrow sides, at its center
    return await grab_with(machine, arm, cam, gripper, block, *GRIP_POINTS["middle"])


async def grab_side(machine: RobotClient, arm: Arm, cam: Camera, gripper: Gripper, block: Pose) -> Grip | None:
    # Claws across the block's narrow sides, toward one end (whichever end has room)
    return await grab_with(machine, arm, cam, gripper, block, *GRIP_POINTS["side"])


async def grab_ends(machine: RobotClient, arm: Arm, cam: Camera, gripper: Gripper, block: Pose) -> Grip | None:
    # Claws on the block's two short ends, turned 90 degrees from the middle grip
    if GRIPPER_OPEN_MM / 2 <= BLOCK_LENGTH_MM / 2 + CLAW_MARGIN_MM:
        raise RuntimeError(f"GRIPPER_OPEN_MM is {GRIPPER_OPEN_MM:g}, too narrow to grip a {BLOCK_LENGTH_MM} mm "
                           "block by its ends")
    return await grab_with(machine, arm, cam, gripper, block, *GRIP_POINTS["ends"])


GRAB_STYLES = {"1": grab_middle, "2": grab_side, "3": grab_ends}
GRAB_GRIP_POINTS = {grab_middle: GRIP_POINTS["middle"], grab_side: GRIP_POINTS["side"], grab_ends: GRIP_POINTS["ends"]}


async def pick_up_block(machine: RobotClient, arm: Arm, cam: Camera, gripper: Gripper,
                        first: Sighting | None = None, grab=grab_middle) -> Grip:
    # Expects the arm to be high enough above the table for the depth camera to see the blocks. Goes for
    # `first` (or the best block from choose_block), and if that turns out to be blocked, comes back here
    # and tries the next best. `grab` is one of grab_middle, grab_side or grab_ends. Returns how the
    # block it picked up is held.
    await gripper.open()
    start = await frame_point_in_world(machine, gripper.name)
    grips = [GRAB_GRIP_POINTS[grab]]
    skipped = []
    target = first or await choose_block(machine, cam, grips=grips)
    while target is not None:
        grip = await grab(machine, arm, cam, gripper, target.center)
        if grip is not None:
            return grip
        skipped.append(target.center)
        # Back to where the whole pile was in view, to choose again
        grip = await frame_point_in_world(machine, gripper.name)
        await move(machine, arm, x=start.x - grip.x, y=start.y - grip.y)
        target = await choose_block(machine, cam, skipped, grips)
        if target is not None:
            print(f"Trying a different block, at ({target.center.x:.0f}, {target.center.y:.0f}). "
                  f"The picture is updated: {await save_target_picture(cam, target)}")
    raise RuntimeError("Every block has something in the claws' way. Spread them out a little")


# --- Placing a block ---

async def turn_claws_to(machine: RobotClient, arm: Arm, gripper: Gripper, close_deg: float,
                        passes: int = 3, tolerance_deg: float = 1):
    # Spin the gripper until its claws close along world direction close_deg (either way round)
    sign, last_error = 1, None
    for _ in range(passes):
        # The claws open and close along the gripper's y axis
        close_angle = world_angle(
            await frame_point_in_world(machine, gripper.name),
            await frame_point_in_world(machine, gripper.name, y=100),
        )
        error = wrap_90(close_deg - close_angle)
        if abs(error) < tolerance_deg:
            return
        # If the last turn made things worse, theta turns the other way on this arm
        if last_error is not None and abs(error) > abs(last_error):
            sign = -sign
        last_error = error
        await move(machine, arm, rotate=-sign * error)


async def place_block(machine: RobotClient, arm: Arm, gripper: Gripper, spot: Pose, long_side_deg: float,
                      held: Grip):
    # Set the held block down on the table centered on `spot`, its long side along long_side_deg.
    # Travels at the height it was lifted to after the pick, clear of the flat blocks.
    grip = await frame_point_in_world(machine, gripper.name)
    await move(machine, arm, x=spot.x - grip.x, y=spot.y - grip.y)
    await turn_claws_to(machine, arm, gripper, long_side_deg + held.claws_deg)

    # Put the block's center on the spot, which isn't under the gripper if it was gripped off-center
    block = await frame_point_in_world(machine, gripper.name, x=held.held_x, y=held.held_y)
    await move(machine, arm, x=spot.x - block.x, y=spot.y - block.y)
    grip = await frame_point_in_world(machine, gripper.name)
    travel_z = grip.z
    # The block is held at the same height in the claws as when it was picked up
    target = await tips_on_table_z(machine) + held.tips_mm + PLACE_DROP_MM
    await move(machine, arm, z=target - grip.z, constraints=NEAR_TABLE)
    await gripper.open()
    await move(machine, arm, z=travel_z - target, constraints=NEAR_TABLE)


async def push_together(machine: RobotClient, arm: Arm, gripper: Gripper, last: Pose, first: Pose,
                        long_side_deg: float):
    # Bring the closed claws down just outside the block at `last`, on the side away from `first`, and
    # slide them PUSH_DISTANCE_MM toward `first`, closing up the gaps left in the row
    d = math.hypot(first.x - last.x, first.y - last.y)
    ux, uy = (first.x - last.x) / d, (first.y - last.y) / d  # toward the first block
    back = BLOCK_WIDTH_MM / 2 + CLAW_THICKNESS_MM + PUSH_START_GAP_MM  # closed claws are two claws thick
    start_x, start_y = last.x - ux * back, last.y - uy * back

    await gripper.grab()  # closed claws make a solid pusher
    grip = await frame_point_in_world(machine, gripper.name)
    travel_z = grip.z
    await move(machine, arm, x=start_x - grip.x, y=start_y - grip.y)
    # Push with the claws' flat outer face, which runs along the blocks' long sides when they close across them
    await turn_claws_to(machine, arm, gripper, long_side_deg + 90)
    grip = await frame_point_in_world(machine, gripper.name)
    await move(machine, arm, x=start_x - grip.x, y=start_y - grip.y)  # in case turning shifted it

    target = await tips_on_table_z(machine) + PUSH_TIPS_MM
    await move(machine, arm, z=target - grip.z, constraints=NEAR_TABLE)
    await move(machine, arm, x=ux * PUSH_DISTANCE_MM, y=uy * PUSH_DISTANCE_MM, constraints=NEAR_TABLE)
    await move(machine, arm, z=travel_z - target, constraints=NEAR_TABLE)
    await gripper.open()


# --- Sweeping the pile flat ---

def tallest_near(heights: dict, u: float, v: float, radius_px: float) -> float:
    # Height of the tallest thing within radius_px of (u, v), ignoring a couple of noisy samples
    found = sorted((h for (pu, pv), h in heights.items() if (pu - u) ** 2 + (pv - v) ** 2 <= radius_px ** 2),
                   reverse=True)
    return found[2] if len(found) > 2 else 0


def crowding(heights: dict, b: Block, du: float, dv: float, scale: float) -> int:
    # How many raised samples sit where block b would land if pushed in direction (du, dv)
    a = math.radians(b.angle)
    count = 0
    for (u, v), h in heights.items():
        if h <= BLOCK_MIN_HEIGHT_MM:
            continue
        ahead = ((u - b.u) * du + (v - b.v) * dv) * scale
        beside = ((u - b.u) * math.cos(a) + (v - b.v) * math.sin(a)) * scale
        if BLOCK_WIDTH_MM / 2 + CLAW_MARGIN_MM < ahead < SWEEP_REACH_MM and abs(beside) < 40:
            count += 1
    return count


def plan_sweep(blocks: list[Block], heights: dict, focal: float):
    # Pick a block that isn't lying flat and a straight sweep that knocks it over: across its
    # short side, toward whichever side has more empty table so it doesn't land on other blocks.
    # Returns (block, start_px, end_px), or None if everything is already flat.
    tall = [b for b in blocks if b.height > FLAT_LAYER_MAX_MM]
    if not tall:
        return None

    for b in tall:
        scale = b.depth / focal
        a = math.radians(b.angle + 90)
        directions = [(math.cos(a), math.sin(a)), (-math.cos(a), -math.sin(a))]
        directions.sort(key=lambda d: crowding(heights, b, *d, scale))

        reach = SWEEP_REACH_MM / scale
        for du, dv in directions:
            start = (b.u - reach * du, b.v - reach * dv)
            end = (b.u + reach * du, b.v + reach * dv)
            # The closed claws come down at the start, so nothing there can be taller than they'll be
            if tallest_near(heights, *start, SWEEP_START_RADIUS_MM / scale) < SWEEP_HEIGHT_MM:
                return b, start, end
    raise RuntimeError("Every block that isn't flat has something tall where a sweep would start")


async def flatten_pile(machine: RobotClient, arm: Arm, cam: Camera, gripper: Gripper):
    # Sweep the closed gripper through anything stacked or standing on its edge until all the
    # blocks lie in one flat layer. Expects the arm at scanning height.
    scan_z = (await frame_point_in_world(machine, gripper.name)).z
    focal = (await intrinsics(cam)).focal_x_px
    await gripper.grab()  # closed claws are a smaller, sturdier pusher

    for _ in range(SWEEP_ROUNDS):
        blocks, _, heights = await pile_blocks(machine, cam)
        plan = await asyncio.to_thread(plan_sweep, blocks, heights, focal)
        if plan is None:
            print("All blocks are lying flat")
            return

        b, (su, sv), (eu, ev) = plan
        print(f"Sweeping a block {b.height:.0f} mm tall")
        start = await pixel_to_world(machine, cam, su, sv, b.depth)
        end = await pixel_to_world(machine, cam, eu, ev, b.depth)

        try:
            grip = await frame_point_in_world(machine, gripper.name)
            await move(machine, arm, x=start.x - grip.x, y=start.y - grip.y)
            target = await tips_on_table_z(machine) + SWEEP_HEIGHT_MM
            await move(machine, arm, z=target - grip.z, constraints=NEAR_TABLE)
            await move(machine, arm, x=end.x - start.x, y=end.y - start.y, constraints=NEAR_TABLE)
        except Exception as e:
            print(f"Sweep didn't finish: {e}")
        await back_off(machine, arm, gripper, scan_z)

    raise RuntimeError(f"Blocks still aren't flat after {SWEEP_ROUNDS} sweeps")


async def ask(prompt: str) -> str:
    return (await asyncio.to_thread(input, prompt)).strip().lower()


async def full_run(machine: RobotClient, arm: Arm, cam: Camera, gripper: Gripper):
    await go_home(machine, arm, gripper)
    await flatten_pile(machine, arm, cam, gripper)
    await designate_pile(machine, cam)
    await choose_tower_site(machine, cam)
    await pick_up_block(machine, arm, cam, gripper)


async def save_target_picture(cam: Camera, sighting: Sighting) -> Path:
    # Outline the block about to be picked on a fresh photo, so it can be checked before anything moves.
    # The arm hasn't moved since the sighting, so its pixel position still lines up.
    _, _, _, color = await get_frame(cam)
    img = Image.open(io.BytesIO(color)).convert("RGB")
    b = sighting.block
    a = math.radians(b.angle)
    half_long = BLOCK_LENGTH_MM / 2 / sighting.mm_per_px
    half_short = BLOCK_WIDTH_MM / 2 / sighting.mm_per_px
    corners = [(b.u + sl * half_long * math.cos(a) - ss * half_short * math.sin(a),
                b.v + sl * half_long * math.sin(a) + ss * half_short * math.cos(a))
               for sl, ss in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    draw = ImageDraw.Draw(img)
    draw.polygon(corners, outline=(255, 0, 0), width=6)
    draw.ellipse([b.u - 6, b.v - 6, b.u + 6, b.v + 6], fill=(255, 0, 0))
    path = Path(__file__).resolve().with_name("target.jpg")
    img.save(path)
    return path


async def test_pick(machine: RobotClient, arm: Arm, cam: Camera, gripper: Gripper):
    # Find one block, pick it up, then put it back down where it was
    if await ask("The arm will move to its home pose. Type 'y' to go: ") != "y":
        return
    await go_home(machine, arm, gripper)

    sighting = await choose_block(machine, cam)
    if sighting is None:
        raise RuntimeError("Every block has something in the claws' way. Spread them out a little")
    print(f"Going for the block at ({sighting.center.x:.0f}, {sighting.center.y:.0f}), "
          f"{sighting.block.height:.0f} mm tall")
    print(f"It's outlined in red here: {await save_target_picture(cam, sighting)}")
    if await ask("Check that's a jenga block and keep a hand on the e-stop. Type 'y' to pick it up: ") != "y":
        return
    await pick_up_block(machine, arm, cam, gripper, sighting)

    await ask("Holding the block. Press Enter to put it back down: ")
    await move(machine, arm, z=-PICK_LIFT_MM, constraints=NEAR_TABLE)
    await gripper.open()
    await move(machine, arm, z=PICK_LIFT_MM, constraints=NEAR_TABLE)
    await go_home(machine, arm, gripper)


async def test_place_row(machine: RobotClient, arm: Arm, cam: Camera, gripper: Gripper, count: int = ROW_BLOCKS):
    # Pick up `count` blocks and set them down side by side, one claw's thickness apart (plus a little room).
    # They go in one after another from one end, so only one claw ever has to fit between two blocks.
    if await ask("The arm will move to its home pose. Type 'y' to go: ") != "y":
        return
    await go_home(machine, arm, gripper)
    await designate_pile(machine, cam)  # so the placed blocks don't get picked up again

    # Room for the whole row, plus the claws opening beside the blocks at either end
    spacing = BLOCK_WIDTH_MM + CLAW_THICKNESS_MM + PLACE_CLEARANCE_MM
    offsets = [(i - (count - 1) / 2) * spacing for i in range(count)]
    spot = await find_clear_spot(machine, cam, offsets[-1] + GRIPPER_OPEN_MM / 2 + CLAW_THICKNESS_MM + CLAW_MARGIN_MM)
    across = math.radians(ROW_LONG_SIDE_DEG + 90)  # side by side means offset across the long sides
    slots = [Pose(x=spot.x + d * math.cos(across), y=spot.y + d * math.sin(across), z=spot.z) for d in offsets]
    print(f"Placing {count} blocks around ({spot.x:.0f}, {spot.y:.0f}), {spacing:.0f} mm apart center to center")

    # Each round starts from home, where the whole pile is in view, so the target is chosen fresh
    # from what the table looks like after the last block was taken
    for i, slot in enumerate(slots, start=1):
        sighting = await choose_block(machine, cam)
        while True:
            if sighting is None:
                raise RuntimeError("Every block has something in the claws' way. Spread them out a little")
            print(f"Block {i}: going for the one at ({sighting.center.x:.0f}, {sighting.center.y:.0f}), "
                  f"{sighting.block.height:.0f} mm tall")
            print(f"It's outlined in red here: {await save_target_picture(cam, sighting)}")
            grab = GRAB_STYLES.get(await ask(
                f"Keep a hand on the e-stop. To pick it up and place it at ({slot.x:.0f}, {slot.y:.0f}), grip it "
                "1) in the middle, 2) on one side or 3) by the ends. Type 1, 2 or 3 (anything else stops): "))
            if grab is None:
                return
            if grip_fits(sighting, GRAB_GRIP_POINTS[grab]):
                break
            # Show the block it'll really go for before moving, rather than switching partway through
            print("That grip doesn't have room on this block, so finding one it fits")
            other = await choose_block(machine, cam, grips=[GRAB_GRIP_POINTS[grab]])
            if other is None:
                print("No block has room for that grip. Pick a different one")
            else:
                sighting = other
        held = await pick_up_block(machine, arm, cam, gripper, sighting, grab)
        await place_block(machine, arm, gripper, slot, ROW_LONG_SIDE_DEG, held)
        await go_home(machine, arm, gripper)

    if count > 1:
        await push_together(machine, arm, gripper, slots[-1], slots[0], ROW_LONG_SIDE_DEG)
        await go_home(machine, arm, gripper)


async def main():
    async with await connect() as machine:
        arm: Arm = Arm.from_robot(machine, "arm")
        gripper: Gripper = Gripper.from_robot(machine, "gripper")
        cam: Camera = Camera.from_robot(machine, "cam")

        await test_place_row(machine, arm, cam, gripper)  # test_pick() for one block, full_run() later


if __name__ == '__main__':
    asyncio.run(main())
