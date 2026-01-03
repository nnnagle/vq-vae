#!/usr/bin/env python3

from pathlib import Path
from typing import Iterable, Optional
import argparse
import sys


def print_project_tree(
    root: Path,
    *,
    max_depth: Optional[int] = None,
    include_files: bool = True,
    ignore_dirs: Iterable[str] = (".git", "__pycache__", ".venv"),
    ignore_files: Iterable[str] = (),
    file_suffixes: Optional[Iterable[str]] = None,
    _prefix: str = "",
    _depth: int = 0,
):
    if _depth == 0:
        print(root.resolve().name)

    if max_depth is not None and _depth >= max_depth:
        return

    try:
        entries = sorted(
            root.iterdir(),
            key=lambda p: (p.is_file(), p.name.lower()),
        )
    except PermissionError:
        print(f"{_prefix}└── [permission denied]")
        return

    entries = [
        e for e in entries
        if not (
            (e.is_dir() and e.name in ignore_dirs)
            or (e.is_file() and e.name in ignore_files)
            or (e.is_file() and file_suffixes and e.suffix not in file_suffixes)
        )
    ]

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "

        if entry.is_dir():
            print(f"{_prefix}{connector}{entry.name}/")
            extension = "    " if is_last else "│   "
            print_project_tree(
                entry,
                max_depth=max_depth,
                include_files=include_files,
                ignore_dirs=ignore_dirs,
                ignore_files=ignore_files,
                file_suffixes=file_suffixes,
                _prefix=_prefix + extension,
                _depth=_depth + 1,
            )
        elif include_files:
            print(f"{_prefix}{connector}{entry.name}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Pretty-print a project directory tree (deterministic, ML-friendly)."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Root directory (default: current directory)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum directory depth",
    )
    parser.add_argument(
        "--dirs-only",
        action="store_true",
        help="Show directories only",
    )
    parser.add_argument(
        "--suffixes",
        type=str,
        default=None,
        help="Comma-separated list of file suffixes to include (e.g. .yaml,.pt,.log)",
    )
    parser.add_argument(
        "--ignore-dirs",
        type=str,
        default=".git,__pycache__,.venv",
        help="Comma-separated directory names to ignore",
    )
    parser.add_argument(
        "--ignore-files",
        type=str,
        default="",
        help="Comma-separated file names to ignore",
    )

    args = parser.parse_args(argv)

    root = Path(args.path)
    if not root.exists():
        sys.exit(f"Path does not exist: {root}")

    suffixes = (
        {s.strip() for s in args.suffixes.split(",") if s.strip()}
        if args.suffixes
        else None
    )

    ignore_dirs = tuple(s for s in args.ignore_dirs.split(",") if s)
    ignore_files = tuple(s for s in args.ignore_files.split(",") if s)

    print_project_tree(
        root,
        max_depth=args.max_depth,
        include_files=not args.dirs_only,
        ignore_dirs=ignore_dirs,
        ignore_files=ignore_files,
        file_suffixes=suffixes,
    )


if __name__ == "__main__":
    main()
