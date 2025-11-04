# utils/argyaml.py
from __future__ import annotations
import argparse, sys
from pathlib import Path

try:
    import yaml
except Exception as e:
    raise RuntimeError("PyYAML is required. pip install pyyaml") from e


def parse_args_with_yaml(parser: argparse.ArgumentParser, section: str):
    """
    Read --config <file>, load YAML[section], and inject those as argv flags.
    Precedence: YAML < CLI (CLI wins).
    Works even when your parser uses 'required=True'.
    """
    # Parse only --config first to locate the file
    prelim = argparse.ArgumentParser(add_help=False)
    prelim.add_argument("--config", type=str, default=None,
                        help="Path to YAML with a top-level section (e.g., 'build_zarr').")
    prelim_args, _ = prelim.parse_known_args()

    yaml_argv: list[str] = []

    if prelim_args.config:
        cfg_path = Path(prelim_args.config)
        if not cfg_path.exists():
            parser.error(f"--config file not found: {cfg_path}")

        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        if section not in cfg or not isinstance(cfg[section], dict):
            parser.error(f"--config missing section '{section}'. "
                         f"Top-level keys: {list(cfg.keys())}")

        # Build mapping: arg dest -> preferred long option (e.g., '--features_csv')
        dest2long = {}
        dest2action = {}
        for a in parser._actions:
            if not a.option_strings:
                continue
            long = None
            # prefer the longest-looking flag (usually the long one)
            for s in sorted(a.option_strings, key=len, reverse=True):
                if s.startswith("--"):
                    long = s
                    break
            if long is None:
                long = a.option_strings[-1]  # fallback to whatever exists
            dest2long[a.dest] = long
            dest2action[a.dest] = a

        # Turn YAML key/values into argv tokens
        for k, v in cfg[section].items():
            if k not in dest2long:
                # Unknown key in YAML; ignore but keep going
                continue

            opt = dest2long[k]
            action = dest2action[k]

            # Booleans need to respect action type (store_true/store_false)
            if isinstance(action, argparse._StoreTrueAction):
                if bool(v):
                    yaml_argv.append(opt)
                # if false, omit
            elif isinstance(action, argparse._StoreFalseAction):
                if not bool(v):
                    yaml_argv.append(opt)
                # if true, omit
            else:
                # Lists expand as multiple values after one flag
                if isinstance(v, (list, tuple)):
                    yaml_argv.extend([opt] + [str(x) for x in v])
                else:
                    yaml_argv.extend([opt, str(v)])

    # Final parse: YAML first, then real CLI; later (CLI) wins.
    args = parser.parse_args(yaml_argv + sys.argv[1:])
    setattr(args, "_config_path", prelim_args.config)
    return args
