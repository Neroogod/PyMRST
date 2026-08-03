"""Digest and version provenance for MATLAB/MRST reference artifacts."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def sha256_file(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_mrst_version(mrst_root: Path) -> str:
    """Infer a reviewable MRST label without executing MATLAB."""
    root = Path(mrst_root)
    match = re.search(r"mrst[-_]?([0-9]{4}[a-z])", root.name, re.IGNORECASE)
    return f"MRST {match.group(1)}" if match else root.name


def matlab_release_from_output(output: str) -> str:
    match = re.search(r"MATLAB_RELEASE=([^\r\n]+)", output)
    return match.group(1).strip() if match else "unknown"


__all__ = ["infer_mrst_version", "matlab_release_from_output", "sha256_file"]
