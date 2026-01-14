#!/usr/bin/env python3
"""
Print a compact directory tree for the project.

Why:
    When debugging with ChatGPT, you often need to show the project layout
    without dumping full files. This script prints a clean tree:
        - Only relevant source directories
        - Ignores venv, checkpoints, data, etc.
        - Shows file sizes for quick orientation
"""

from pathlib import Path

# Folders to skip (editable)
SKIP_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    "venv",
    ".venv",
    "data",
    "runs",
    "checkpoints",
    ".mypy_cache",
    ".pytest_cache",
}

# File extensions worth showing (editable)
SHOW_EXT = {".py", ".yaml", ".yml", ".json"}


def print_tree(root: Path, prefix: str = ""):
    entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))

    # Filter noise
    entries = [e for e in entries if e.name not in SKIP_DIRS]

    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "

        if entry.is_dir():
            print(prefix + connector + entry.name + "/")
            extension_prefix = "    " if i == len(entries) - 1 else "│   "
            print_tree(entry, prefix + extension_prefix)

        else:
            if entry.suffix in SHOW_EXT:
                size = entry.stat().st_size
                print(prefix + connector + f"{entry.name}  ({size} bytes)")


def main():
    project_root = Path(__file__).resolve().parents[1]
    print(f"[Project Root] {project_root}\n")
    print_tree(project_root)


if __name__ == "__main__":
    main()
