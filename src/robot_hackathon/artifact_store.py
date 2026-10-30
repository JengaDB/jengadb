"""Content-addressed local files, independent of cameras and model providers."""
import hashlib
import os
from pathlib import Path
import re
import tempfile


class ArtifactStore:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, ref):
        digest = ref.get("sha256", "")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid artifact hash")
        relative = f"{digest[:2]}/{digest}"
        if ref.get("path") != relative:
            raise ValueError("Invalid artifact path")
        path = self.directory / relative
        if not path.resolve().is_relative_to(self.directory):
            raise ValueError("Artifact path escapes storage directory")
        return path

    def read(self, ref):
        data = self.path_for(ref).read_bytes()
        if len(data) != ref.get("size_bytes") or hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("Artifact contents do not match reference")
        return data

    def valid(self, ref):
        try:
            self.read(ref)
            return True
        except (OSError, ValueError, TypeError, AttributeError):
            return False

    def put(self, data, media_type="application/octet-stream"):
        if not isinstance(data, bytes) or not isinstance(media_type, str) or not media_type:
            raise ValueError("Artifact requires bytes and a media type")
        digest = hashlib.sha256(data).hexdigest()
        ref = {"sha256": digest, "path": f"{digest[:2]}/{digest}",
               "size_bytes": len(data), "media_type": media_type}
        if self.valid(ref):
            return ref
        path = self.path_for(ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            # Sync the entry and the directory that contains its hash prefix.
            for directory in (path.parent, self.directory):
                directory_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return ref
