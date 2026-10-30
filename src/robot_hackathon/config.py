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
    # RGB-D collection pipeline (SKY-66). All optional; defaults keep mock mode dependency-free.
    machine_auth: str = "apikey"
    setup_id: str = "mock"
    camera_resource: str = "cam"
    camera_serial: str = ""
    meta_api_key: str = field(default="", repr=False)
    meta_base_url: str = "https://api.meta.ai"
    voyage_key: str = field(default="", repr=False)
    voyage_base_url: str = "https://ai.mongodb.com"
    observations_collection: str = "observations"
    candidates_collection: str = "candidates"
    block_dimensions_mm: tuple = (75.0, 25.0, 15.0)
    block_dimensions_verified: bool = False
    media_attempt_max_age_hours: float = 24.0

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        mode = env.get("MODE", "mock")
        if mode not in ("mock", "live"):
            raise ConfigError("MODE must be mock or live")
        if boolean(env, "ENABLE_MOTION"):
            raise ConfigError("Motion is not implemented; ENABLE_MOTION must remain false")
        machine_auth = env.get("MACHINE_AUTH", "apikey")
        if machine_auth not in ("apikey", "none"):
            raise ConfigError("MACHINE_AUTH must be apikey or none")
        try:
            dims = tuple(float(x) for x in env.get("BLOCK_DIMENSIONS_MM", "75,25,15").split(","))
            if len(dims) != 3 or any(d <= 0 for d in dims):
                raise ValueError
            max_age = float(env.get("MEDIA_ATTEMPT_MAX_AGE_HOURS", "24"))
            if max_age <= 0:
                raise ValueError
        except ValueError:
            raise ConfigError("BLOCK_DIMENSIONS_MM needs three positive numbers; MEDIA_ATTEMPT_MAX_AGE_HOURS must be positive") from None
        meta_base = (env.get("META_API_BASE_URL") or "https://api.meta.ai").strip().rstrip("/")
        if urlsplit(meta_base).scheme != "https" or not urlsplit(meta_base).hostname:
            raise ConfigError("META_API_BASE_URL must be an https URL")
        # MongoDB's hosted Voyage models endpoint; accept either the VOYAGE_* or MONGO_DB_MODELS_ENDPOINT_* names.
        base = (env.get("MONGO_DB_MODELS_ENDPOINT_URL") or "https://ai.mongodb.com").strip().rstrip("/")
        if "://" not in base:
            base = "https://" + base
        if urlsplit(base).scheme != "https" or not urlsplit(base).hostname:
            raise ConfigError("MONGO_DB_MODELS_ENDPOINT_URL must be an https host")
        return cls(mode, boolean(env, "ALLOW_CLOUD"), env.get("MACHINE_ADDRESS", ""),
                   env.get("API_KEY_ID", ""), env.get("API_KEY", ""),
                   env.get("MONGODB_URI", ""), env.get("MONGODB_DATABASE", "hackathon"),
                   env.get("MONGODB_COLLECTION", "attempts"),
                   machine_auth=machine_auth, setup_id=env.get("SETUP_ID", "mock"),
                   camera_resource=env.get("CAMERA_RESOURCE", "cam"), camera_serial=env.get("CAMERA_SERIAL", ""),
                   meta_api_key=env.get("META_API_KEY", ""), meta_base_url=meta_base,
                   voyage_key=env.get("VOYAGE_API_KEY") or env.get("MONGO_DB_MODELS_ENDPOINT_KEY", ""),
                   voyage_base_url=base,
                   observations_collection=env.get("RGBD_OBSERVATIONS_COLLECTION", "observations"),
                   candidates_collection=env.get("RGBD_CANDIDATES_COLLECTION", "candidates"),
                   block_dimensions_mm=dims, block_dimensions_verified=boolean(env, "BLOCK_DIMENSIONS_VERIFIED"),
                   media_attempt_max_age_hours=max_age)

    def require_cloud(self, explicit):
        if not explicit or not self.allow_cloud:
            raise ConfigError("Requires ALLOW_CLOUD=true and --allow-cloud after authorization")

    def require_viam(self):
        if self.machine_auth == "none":
            host = urlsplit("//" + self.address).hostname if self.address else None
            if host not in ("localhost", "127.0.0.1", "::1"):
                raise ConfigError("MACHINE_AUTH=none is allowed only for a loopback MACHINE_ADDRESS")
            return
        if not all((self.address, self.key_id, self.key)):
            raise ConfigError("Set MACHINE_ADDRESS, API_KEY_ID and API_KEY locally")

    def require_meta(self):
        if not self.meta_api_key:
            raise ConfigError("Set META_API_KEY locally")

    def require_voyage(self):
        if not self.voyage_key:
            raise ConfigError("Set VOYAGE_API_KEY or MONGO_DB_MODELS_ENDPOINT_KEY locally")

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
        for value in (self.database, self.collection, self.observations_collection, self.candidates_collection):
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
                raise ConfigError("Use a simple designated database and collection name")
