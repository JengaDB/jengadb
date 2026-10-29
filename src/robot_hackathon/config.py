from dataclasses import dataclass, field
import os
import re
from urllib.parse import urlsplit, parse_qs

class ConfigError(ValueError):
    pass

def boolean(env, key):
    value = env.get(key, "false").lower().strip()
    if value not in ("true", "false"):
        raise ConfigError(f"{key} must be true or false")
    return value == "true"

@dataclass(frozen=True)
class Config:
    mode: str = "mock"
    allow_cloud: bool = False
    address: str = field(default="", repr=False)
    key_id: str = field(default="", repr=False)
    key: str = field(default="", repr=False)
    uri: str = field(default="", repr=False)
    database: str = "hackathon"
    collection: str = "attempts"

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        mode = env.get("MODE", "mock")
        if mode not in ("mock", "live"):
            raise ConfigError("MODE must be mock or live")
        if boolean(env, "ENABLE_MOTION"):
            raise ConfigError("Motion is not implemented; ENABLE_MOTION must remain false")
        return cls(mode, boolean(env, "ALLOW_CLOUD"), env.get("MACHINE_ADDRESS", ""),
                   env.get("API_KEY_ID", ""), env.get("API_KEY", ""),
                   env.get("MONGODB_URI", ""), env.get("MONGODB_DATABASE", "hackathon"),
                   env.get("MONGODB_COLLECTION", "attempts"))

    def require_cloud(self, explicit):
        if not explicit or not self.allow_cloud:
            raise ConfigError("Requires ALLOW_CLOUD=true and --allow-cloud after authorization")

    def require_viam(self):
        if not all((self.address, self.key_id, self.key)):
            raise ConfigError("Set MACHINE_ADDRESS, API_KEY_ID and API_KEY locally")

    def require_mongo(self):
        if not self.uri:
            raise ConfigError("Set MONGODB_URI locally")
        try:
            parsed = urlsplit(self.uri)
            options = {k.lower(): v for k, v in parse_qs(parsed.query).items()}
        except ValueError:
            raise ConfigError("Invalid MONGODB_URI") from None
        if parsed.scheme not in ("mongodb", "mongodb+srv") or not parsed.hostname:
            raise ConfigError("Invalid MONGODB_URI")
        for key, values in options.items():
            if ((key in ("tls", "ssl") and any(v.lower() != "true" for v in values))
                or key in ("tlsinsecure", "tlsallowinvalidcertificates", "tlsallowinvalidhostnames")):
                raise ConfigError("Insecure TLS options are prohibited")
        for value in (self.database, self.collection):
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
                raise ConfigError("Use a simple designated database and collection name")
