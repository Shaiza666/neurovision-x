"""Copy or download a checkpoint and check its SHA-256.

Used by scripts/fetch_checkpoints.py. A mismatch deletes the destination
so a bad file is not left where training or the dashboard would load it.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Any


class CheckpointFetchError(RuntimeError):
    """The checkpoint could not be placed or its digest did not match."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_local(path_value: str, repo_root: Path) -> Path:
    source = Path(path_value)
    if not source.is_absolute():
        source = repo_root / source
    return source


def materialize_checkpoint(
    spec: dict[str, Any],
    target: Path,
    *,
    repo_root: Path,
) -> str:
    """Place ``spec`` at ``target``.

    Returns ``present`` when the file is already there and the digest matches,
    ``downloaded`` when ``url`` was fetched, or ``copied`` when ``path`` was
    copied from the repo. ``sha256`` is required for a successful place.
    """
    expected = spec.get("sha256")
    if not expected or not isinstance(expected, str):
        raise CheckpointFetchError("Checkpoint spec is missing sha256")
    target = Path(target)
    if target.is_file() and file_sha256(target) == expected:
        return "present"

    url = spec.get("url") or None
    local = spec.get("path") or None
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        partial.unlink()
    try:
        if url:
            urllib.request.urlretrieve(str(url), partial)
            kind = "downloaded"
        elif local:
            source = _resolve_local(str(local), repo_root)
            if not source.is_file():
                raise CheckpointFetchError(f"Local checkpoint missing: {source}")
            shutil.copyfile(source, partial)
            kind = "copied"
        else:
            raise CheckpointFetchError("Checkpoint spec has neither url nor path")
        partial.replace(target)
    except CheckpointFetchError:
        if partial.exists():
            partial.unlink()
        raise
    except Exception as exc:
        if partial.exists():
            partial.unlink()
        raise CheckpointFetchError(f"Could not place {target.name}: {exc}") from exc

    actual = file_sha256(target)
    if actual != expected:
        target.unlink(missing_ok=True)
        raise CheckpointFetchError(
            f"SHA-256 mismatch for {target.name}: expected {expected}, got {actual}"
        )
    return kind
