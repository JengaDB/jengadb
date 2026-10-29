from typing import Protocol

class AttemptSource(Protocol):
    def attempts(self) -> list[dict]: ...

class RemoteStore(Protocol):
    def upsert(self, attempt: dict) -> None: ...

class DisabledActionAdapter:
    def execute(self, *args, **kwargs):
        raise RuntimeError("Physical actions are not implemented in this preparation project")
