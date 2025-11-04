"""
io.py
------
Utility functions for robust file and I/O operations.

Purpose
    Small, composable helpers for safe file handling used across the project.
    Functions here prevent data corruption (atomic writes), enforce predictable
    outputs (JSON/TSV writers), and support resumable workflows.

Used by
    - build_zarr.py and related data ingestion scripts
    - scripts/download_* modules (NLCD/LCMS, GEE/MRLC)

Design notes
    * Favor idempotency: do not overwrite silently; create dirs as needed.
    * Prefer atomic writes: write to a temp file, then rename into place.
    * Keep zero heavy deps so these utilities work in lightweight scripts.

Assistant guidance
    Preserve atomic write semantics and logging via utils.log.log/fail. Do not
    introduce side-effects or hidden global state.
"""

from __future__ import annotations

import os
import json
import hashlib
import tempfile
from pathlib import Path
from typing import Any, Iterable, Tuple, Callable
from utils.log import log, fail

# ---------------------------------------------------------------------
# Basic path checks
# ---------------------------------------------------------------------

def ensure_file(path: str, purpose: str = "input"):
    """Ensure a file exists; raise fatal if not."""
    if not os.path.isfile(path):
        fail(f"Missing {purpose}: {path}")


def ensure_dir(path: str, verbose: bool = True):
    """Ensure a directory exists (create if missing)."""
    os.makedirs(path, exist_ok=True)
    if not os.path.isdir(path):
        fail(f"Unable to create directory: {path}")
    if verbose:
        log("Ensured directory exists: %s", path)
    return path

# ---------------------------------------------------------------------
# Filename utilities
# ---------------------------------------------------------------------

def safe_basename(s: str) -> str:
    """Return a filesystem-safe base name (no spaces or slashes)."""
    return os.path.basename(s).replace(" ", "_").replace("/", "_")

# ---------------------------------------------------------------------
# JSON / TSV writers
# ---------------------------------------------------------------------

def write_json(path: str, obj: Any, indent: int = 2):
    """Write a Python object to JSON (UTF-8)."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)
    log("Wrote JSON: %s", path)


def write_tsv(path: str, rows: Iterable[Tuple], header: Tuple[str, ...] | None = None):
    """Write a tab-separated file with optional header."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write("	".join(header) + " 
            ")
        for row in rows:
            f.write("	".join(map(str, row)) + " 
            ")
    log("Wrote TSV: %s", path)

# ---------------------------------------------------------------------
# Integrity helpers
# ---------------------------------------------------------------------

def md5sum(path: str, block_size: int = 8192) -> str:
    """Compute MD5 checksum for a file."""
    ensure_file(path, "checksum input")
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path: str, expected: str):
    """Compare a file's checksum against an expected hash."""
    actual = md5sum(path)
    if actual != expected:
        fail(f"Checksum mismatch for {path}: expected {expected}, got {actual}")
    log("Checksum verified for %s", path)

# ---------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------

def atomic_write(final_path: str | Path,
                 write_fn: Callable[[Path], None],
                 tmpdir: str | Path | None = None) -> None:
    """Safely write a file atomically.

    Writes to a temporary file and renames it onto the destination. The rename
    is atomic on a single filesystem, preventing partial files from appearing
    when a process crashes mid-write.

    Args:
        final_path: Destination path to write to (str or Path).
        write_fn: Function that accepts a Path and writes the full payload to it.
        tmpdir: Optional directory for the temp file (defaults to final_path.parent).

    Example:
        >>> atomic_write("out/nlcd_2019.tif", lambda p: save_raster(p, arr))
    """
    final = Path(final_path)
    tmp_parent = Path(tmpdir) if tmpdir else final.parent
    ensure_dir(str(tmp_parent), verbose=False)

    # Use NamedTemporaryFile with delete=False so we can control rename/unlink.
    with tempfile.NamedTemporaryFile(dir=tmp_parent, delete=False, suffix=".tmp") as tf:
        tmp_path = Path(tf.name)
        log("Writing temporary file: %s", tmp_path)

    try:
        write_fn(tmp_path)
        # Ensure destination directory exists before replacement
        ensure_dir(str(final.parent), verbose=False)
        log("Replacing %s with %s", final, tmp_path)
        tmp_path.replace(final)
    except Exception as e:
        # Best effort cleanup of tmp and propagate the error
        try:
            if tmp_path.exists():
                os.unlink(tmp_path)
        except Exception:
            pass
        fail(f"Atomic write failed for {final}: {e}")
    """
