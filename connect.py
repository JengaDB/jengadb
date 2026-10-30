import asyncio
import os

from viam.robot.client import RobotClient
from viam.components.gripper import Gripper
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.services.generic import Generic as GenericService

def load_env(path='.env'):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())
    except FileNotFoundError:
        pass

load_env()

async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ['API_KEY'],
        api_key_id=os.environ['API_KEY_ID']
    )
    return await RobotClient.at_address(os.environ['MACHINE_ADDRESS'], opts)

async def main():
    async with await connect() as machine:
        print('Resources:')
        print(machine.resource_names)
        
        # table
        table: Gripper = Gripper.from_robot(machine, "table")
        table_return_value = await table.is_moving()
        print(f"table is_moving return value: {table_return_value}")

        # ceiling
        ceiling: Gripper = Gripper.from_robot(machine, "ceiling")
        ceiling_return_value = await ceiling.is_moving()
        print(f"ceiling is_moving return value: {ceiling_return_value}")

        # arm
        arm: Arm = Arm.from_robot(machine, "arm")
        arm_return_value = await arm.get_end_position()
        print(f"arm get_end_position return value: {arm_return_value}")

        # cam
        cam: Camera = Camera.from_robot(machine, "cam")
        cam_return_value = await cam.get_images()
        print(f"cam get_images return value: {cam_return_value}")

        # gripper
        gripper: Gripper = Gripper.from_robot(machine, "gripper")
        gripper_return_value = await gripper.is_moving()
        print(f"gripper is_moving return value: {gripper_return_value}")

if __name__ == '__main__':
    asyncio.run(main())
