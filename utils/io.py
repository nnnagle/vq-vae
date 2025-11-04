import os
import json
import hashlib
from typing import Any, Iterable, Tuple
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

def write_tsv(path: str, rows: Iterable[Tuple], header: Tuple[str, ...] = None):
    """Write a tab-separated file with optional header."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(map(str, row)) + "\n")
    log("Wrote TSV: %s", path)

# ---------------------------------------------------------------------
# Integrity helpers (optional but handy for data verification)
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
