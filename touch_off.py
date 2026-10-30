import asyncio

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame

import jenga
from connect import connect
from jenga import NEAR_TABLE, choose_tower_site, frame_heights, frame_point_in_world, go_home, move

# Touch-off: lower the closed claws onto bare table in small steps you confirm one at a time, to find
# where the claw tips really are. Nothing moves below the first stop without you pressing a key.
START_CLEARANCE_MM = 160  # first stop: gripper frame this far above the camera's table. The real tips sit
                          # about 90 mm below the gripper frame (much more than the config says), so this
                          # leaves about 70 mm of air.
BACK_OFF_MM = 60          # how far above the first stop to retreat when done or on any failure
MAX_STEP_MM = 10


async def ask(prompt: str) -> str:
    return (await asyncio.to_thread(input, prompt)).strip().lower()


def reach_along_z(geometries) -> float:
    # How far a frame's boxes reach along its +z axis: toward the tips for the gripper, up for the table
    return max(g.center.z + g.box.dims_mm.z / 2 for g in geometries)


async def lower_until_touching(machine, arm: Arm, gripper: Gripper, table_z: float) -> float | None:
    # Returns the gripper frame's z when the tips touch, or None if you quit
    step = 5.0
    while True:
        z = (await frame_point_in_world(machine, gripper.name)).z
        cmd = await ask(f"gripper z = {z:.1f} ({z - table_z:.1f} above the camera's table). "
                        f"[Enter] down {step:g} mm, [number] set step, [u] up, [t] touching, [q] quit: ")
        if cmd == "t":
            return z
        if cmd == "q":
            return None
        if cmd == "u":
            await move(machine, arm, z=step, constraints=NEAR_TABLE)
        elif cmd:
            try:
                step = min(abs(float(cmd)), MAX_STEP_MM)
            except ValueError:
                print("Didn't understand that")
        else:
            await move(machine, arm, z=-step, constraints=NEAR_TABLE)


async def main():
    async with await connect() as machine:
        arm: Arm = Arm.from_robot(machine, "arm")
        gripper: Gripper = Gripper.from_robot(machine, "gripper")
        cam: Camera = Camera.from_robot(machine, "cam")

        if await ask("The arm will move to its home pose above the table. Type 'yes' to go: ") != "yes":
            return
        await go_home(machine, arm, gripper)

        # Borrow the tower-site search to find bare table, then drop it so it isn't an obstacle here
        await choose_tower_site(machine, cam)
        spot = jenga.tower.center
        jenga.tower = None
        print(f"Bare table at ({spot.x:.0f}, {spot.y:.0f}). The camera puts its surface at z = {spot.z:.1f} mm.")

        if await ask("Check that spot is clear and keep a hand on the e-stop. Type 'yes' to go: ") != "yes":
            return

        await gripper.grab()  # closed, so the tips are together and lowest
        grip = await frame_point_in_world(machine, gripper.name)
        await move(machine, arm, x=spot.x - grip.x, y=spot.y - grip.y)
        start_z = spot.z + START_CLEARANCE_MM
        await move(machine, arm, z=start_z - grip.z, constraints=NEAR_TABLE)

        print("Slide a sheet of paper under the claw tips. Step down, going to 1 mm steps as you get close,")
        print("and press [t] as soon as the paper is pinned.")
        try:
            touch_z = await lower_until_touching(machine, arm, gripper, spot.z)
        finally:
            # Always retreat above the first stop, even if something failed
            grip = await frame_point_in_world(machine, gripper.name)
            await move(machine, arm, z=start_z + BACK_OFF_MM - grip.z, constraints=NEAR_TABLE)
        if touch_z is None:
            return

        offset = touch_z - spot.z
        print(f"\nClaw tips touch with the gripper frame at z = {touch_z:.1f}.")
        frames = {name: round(z, 1) for name, z in (await frame_heights(machine)).items()}
        print("Put these in jenga.py:")
        print(f"    CLAW_TIP_OFFSET_MM = {offset:.1f}")
        print(f"    CALIBRATED_FRAMES = {frames}")

        # Compare with the config, to see whether the table or the gripper geometry is off
        config_tip = reach_along_z(await gripper.get_geometries())
        table_top = reach_along_z(await Gripper.from_robot(machine, "table").get_geometries())
        config_table_z = (await machine.transform_pose(
            PoseInFrame(reference_frame="table", pose=Pose(z=table_top, o_z=1)), "world")).pose.z
        print(f"\nThe config says the claws reach {config_tip:.0f} mm below the gripper frame, which would put")
        print(f"the table at z = {touch_z - config_tip:.1f}. The config's table top is z = {config_table_z:.1f}")
        print(f"and the camera measured z = {spot.z:.1f}. Whichever is far from the others is the one to fix.")


if __name__ == '__main__':
    asyncio.run(main())
