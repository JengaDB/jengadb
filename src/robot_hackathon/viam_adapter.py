import asyncio
from .interfaces import DisabledActionAdapter

class ViamAdapter(DisabledActionAdapter):
    @staticmethod
    async def discover(config):
        config.require_viam()
        from viam.robot.client import RobotClient
        client = None
        try:
            options = RobotClient.Options.with_api_key(api_key=config.key, api_key_id=config.key_id)
            client = await asyncio.wait_for(RobotClient.at_address(config.address, options), timeout=15)
            return [resource.name for resource in client.resource_names]
        finally:
            if client is not None:
                await asyncio.wait_for(client.close(), timeout=5)
