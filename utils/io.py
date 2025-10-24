import os
from utils.log import log, fail

def ensure_file(path: str, purpose: str = "input"):
    """Check file existence."""
    if not os.path.isfile(path):
        fail(f"Missing {purpose}: {path}")

def ensure_dir(path: str):
    """Make directory if missing."""
    os.makedirs(path, exist_ok=True)
    if not os.path.isdir(path):
        fail(f"Unable to create directory: {path}")
    log("Ensured directory exists: %s", path)

