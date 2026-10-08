"""Checksummed, atomic specialist artifacts in reserved namespaces."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch


REPO_ROOT = Path(os.environ.get("LLM_SPECIALIST_ROOT", Path.cwd())).resolve()


def file_sha256(path: Path) -> str:
    """Hash a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value: object) -> str:
    """Hash canonical JSON metadata."""
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def namespace_path(path: str | Path, namespace: str) -> Path:
    """Resolve a path strictly below a reserved specialist namespace."""
    target = Path(path).resolve()
    root = (REPO_ROOT / namespace / "specialist").resolve()
    # Reject symlinks that redirect the reserved namespace outside the repository.
    if root != REPO_ROOT / namespace / "specialist":
        raise ValueError("Specialist namespace must not be redirected by a symlink.")
    if target == root or not target.is_relative_to(root):
        raise ValueError(f"Output must be below {root}.")
    return target


def run_path(run_name: str, namespace: str) -> Path:
    """Return an artifact location for one path-safe new specialist run."""
    if not isinstance(run_name, str) or not run_name or run_name in (".", ".."):
        raise ValueError("run_name must be a nonempty path-safe name.")
    if Path(run_name).name != run_name or "\\" in run_name:
        raise ValueError("run_name must be a nonempty path-safe name.")
    return namespace_path(REPO_ROOT / namespace / "specialist" / run_name, namespace)


def atomic_json(path: Path, value: object, *, replace: bool = False) -> None:
    """Write strict JSON atomically, refusing unrequested replacement."""
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_bytes(path, payload.encode("utf-8"), replace=replace)


def atomic_bytes(path: Path, payload: bytes, *, replace: bool = False) -> None:
    """Publish bytes atomically with exclusive creation by default."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".specialist-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Replace an owned run checkpoint atomically and write its checksum."""
    import io

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_bytes(path, buffer.getvalue(), replace=True)
    atomic_bytes(path.with_suffix(".pt.sha256"), (file_sha256(path) + "\n").encode(), replace=True)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Verify a specialist checkpoint before safe, CPU-only deserialization."""
    target = Path(path)
    digest_path = target.with_suffix(".pt.sha256")
    if not digest_path.is_file() or digest_path.read_text().strip() != file_sha256(target):
        raise ValueError(f"Checkpoint checksum missing or mismatched: {target}.")
    value = torch.load(target, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("format") != "specialist-v1":
        raise ValueError("Not a specialist-v1 checkpoint.")
    required = {"config", "kind", "model_state", "prepared_identity", "encoder_identity", "epoch", "step"}
    if not required <= value.keys():
        raise ValueError("Specialist checkpoint is missing required fields.")
    return value
