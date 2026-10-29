import asyncio

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.services.generic import Generic as GenericService

async def connect():
    opts = RobotClient.Options.with_api_key(
         
        api_key='srob2lh4us4k7qtt5506u3dl11ur6dxj',
        
        api_key_id='44a265ab-6fd2-4f6b-ba70-49fbfd7c81ea'
    )
    
    return await RobotClient.at_address('sfhack15-main.ty1cywh7tq.viam.cloud', opts)

async def main():
    async with await connect() as machine:
        print('Resources:')
        print(machine.resource_names)
        
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
