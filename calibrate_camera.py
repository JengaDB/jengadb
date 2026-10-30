import asyncio
import math

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose

from connect import connect
from jenga import frame_point_in_world, go_home, locate_block, move

# Camera calibration: finds how far the camera's x and y in the config are from where it really sits on
# the arm. With the arm high above one block that stays put, it turns the gripper to a few angles and looks
# at the block each time. If the config has the camera in the wrong place, the block seems to move in a
# circle as the camera swings around the gripper's axis; the circle's size and direction give the correction.
# Where the calibration starts and looks from. Keep the gripper pointing straight down (o_z=-1), high
# enough for the depth camera to see the table, with the block in view.
HOME_POSE = Pose(x=300, y=100, z=300, o_x=0, o_y=0, o_z=-1, theta=0)
TURNS_DEG = [0, 45, 90, -45, -90]  # gripper angles to look from, relative to where it starts
GOOD_FIT_MM = 3                    # leftover scatter above this means the camera's angle is probably off too


async def ask(prompt: str) -> str:
    return (await asyncio.to_thread(input, prompt)).strip().lower()


async def flange_axes(machine, gripper: Gripper):
    # World xy directions of the flange's x and y axes, which turn with it (the gripper frame is fixed to it)
    o = await frame_point_in_world(machine, gripper.name)
    ex = await frame_point_in_world(machine, gripper.name, x=100)
    ey = await frame_point_in_world(machine, gripper.name, y=100)
    return ((ex.x - o.x) / 100, (ex.y - o.y) / 100), ((ey.x - o.x) / 100, (ey.y - o.y) / 100)


def solve(a, b):
    # Gaussian elimination with partial pivoting, for the small least-squares system below
    n = len(b)
    m = [row[:] + [v] for row, v in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in reversed(range(n)):
        x[r] = (m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))) / m[r][r]
    return x


def fit(sightings):
    # Each look: seen = block - R * error, where R turns the flange's x/y axes into the world's. Least squares
    # for the block's true position and the camera's error, both in mm; the error is in the flange's x/y,
    # the same axes as the camera's translation in the config.
    rows, rhs = [], []
    for (sx, sy), (ex, ey) in sightings:
        rows.append([1, 0, -ex[0], -ey[0]])
        rhs.append(sx)
        rows.append([0, 1, -ex[1], -ey[1]])
        rhs.append(sy)
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(4)] for i in range(4)]
    atb = [sum(r[i] * v for r, v in zip(rows, rhs)) for i in range(4)]
    solution = solve(ata, atb)
    residuals = [v - sum(c * s for c, s in zip(r, solution)) for r, v in zip(rows, rhs)]
    scatter = math.sqrt(sum(e * e for e in residuals) / len(residuals))
    px, py, dx, dy = solution
    return (px, py), (dx, dy), scatter


async def main():
    async with await connect() as machine:
        arm: Arm = Arm.from_robot(machine, "arm")
        gripper: Gripper = Gripper.from_robot(machine, "gripper")
        cam: Camera = Camera.from_robot(machine, "cam")

        print("Put a single jenga block on the table, away from anything else, and leave it there throughout.")
        if await ask(f"The arm will move to ({HOME_POSE.x:g}, {HOME_POSE.y:g}, {HOME_POSE.z:g}). Type 'yes' to go: ") != "yes":
            return
        await go_home(machine, arm, gripper, HOME_POSE)

        block = (await locate_block(machine, cam)).center
        print(f"Block seen at ({block.x:.1f}, {block.y:.1f})")
        if await ask(f"The gripper will move over it and turn to {len(TURNS_DEG)} angles, staying at this height. "
                     "Type 'yes' to go: ") != "yes":
            return
        grip = await frame_point_in_world(machine, gripper.name)
        await move(machine, arm, x=block.x - grip.x, y=block.y - grip.y)

        sightings, turned = [], 0
        for angle in TURNS_DEG:
            if angle != turned:
                await move(machine, arm, rotate=angle - turned)
                turned = angle
            block = (await locate_block(machine, cam, block)).center
            sightings.append(((block.x, block.y), await flange_axes(machine, gripper)))
            print(f"  turned {angle:+} deg: block seen at ({block.x:.1f}, {block.y:.1f})")
        if turned:
            await move(machine, arm, rotate=-turned)

        (px, py), (dx, dy), scatter = fit(sightings)
        cam_frame = next(p.frame.pose_in_observer_frame.pose for p in await machine.get_frame_system_config()
                         if p.frame.reference_frame == cam.name)

        print(f"\nThe block is really at ({px:.1f}, {py:.1f}).")
        print(f"The camera is {math.hypot(dx, dy):.1f} mm from where the config says.")
        print(f"In the Viam app, set the cam frame's translation to x = {cam_frame.x + dx:.1f}, "
              f"y = {cam_frame.y + dy:.1f} (it's {cam_frame.x:g}, {cam_frame.y:g} now). Leave z as it is.")
        if scatter > GOOD_FIT_MM:
            print(f"\nThe looks don't fit a clean circle ({scatter:.1f} mm of scatter left over). The camera's "
                  "angle in the config may be off as well, or the block moved. Apply the correction and run "
                  "this again.")
        else:
            print(f"Good fit ({scatter:.1f} mm of scatter). Run this again after changing the config: the "
                  "correction should come out near 0.")


if __name__ == '__main__':
    asyncio.run(main())
