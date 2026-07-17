from .base import Sandbox
from .docker import DockerSandbox
from .local import LocalSandbox


def create_sandbox(name: str) -> Sandbox:
    if name == "docker":
        return DockerSandbox()
    if name == "local":
        return LocalSandbox()
    raise ValueError(f"Unknown sandbox: {name}")


__all__ = ["Sandbox", "DockerSandbox", "LocalSandbox", "create_sandbox"]
