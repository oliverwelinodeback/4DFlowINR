#!/usr/bin/env python3
"""
Command-line entry point for 4DFlowINR training.
--------
Example; From the repository root:
    python src/train.py --config configs/paper/inr.py
"""

from __future__ import annotations
import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
import wandb

# Repository paths
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent

# Ensure both current and future config layouts are importable.
for path in (SRC_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from trainer import train

def parse_args() -> argparse.Namespace:
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a 4DFlowINR model from a Python configuration file."
    )
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to the Python configuration file (e.g. configs/paper/inr.py)"
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Optional run-name override for non-sweep runs"
        ),
    )
    parser.add_argument(
        "--sweep-id",
        default=None,
        help=(
            "Optional existing W&B sweep ID"
            "If omitted and config.sweep=True, a new sweep is created"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate the configuration without starting training",
    )
    return parser.parse_args()


def resolve_config_path(config_arg: str) -> Path:

    supplied_path = Path(config_arg).expanduser()
    candidates = []

    if supplied_path.is_absolute():
        candidates.append(supplied_path)
    else:
        candidates.extend(
            [
                Path.cwd() / supplied_path,
                REPO_ROOT / supplied_path,
                SRC_DIR / supplied_path,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not find configuration file '{config_arg}'.\n"
        f"Searched:\n{searched}"
    )


def load_config_module(config_path: Path) -> ModuleType:
    """Import a Python configuration file from an arbitrary path."""
    module_name = f"_4dflowinr_config_{config_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, config_path)

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create an import specification for {config_path}"
        )

    module = importlib.util.module_from_spec(spec)

    # Register before execution so imports/dataclasses can resolve the module.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "get_config"):
        raise AttributeError(
            f"Configuration file '{config_path}' must define get_config()."
        )

    return module


def main() -> None:
    # Run training using the selected configuration

    args = parse_args()

    config_path = resolve_config_path(args.config)
    config_module = load_config_module(config_path)
    config = config_module.get_config()

    print("="*20)
    print("4DFlowINR")
    print(f"Configuration: {config_path}")
    print(f"Network name:  {config.network_name}")
    print(f"Sweep:         {config.sweep}")
    print("="*20)

    # Validate sweep configuration
    if config.sweep and not hasattr(config_module, "get_sweep_config"):
        raise AttributeError(
            f"Configuration '{config_path}' has config.sweep=True "
            "but does not define get_sweep_config()."
        )

    if args.dry_run:
        if config.sweep:
            config_module.get_sweep_config()

        print("Configuration loaded successfully. Dry run complete.")
        return

    os.chdir(SRC_DIR)

    if config.sweep:
        sweep_project = config.wandb.project

        if args.sweep_id is not None:
            print(f"Joining existing W&B sweep: {args.sweep_id}")

            wandb.agent(
                args.sweep_id,
                project=sweep_project,
                function=lambda: train(
                    config=config,
                    use_sweep=True,
                ),
            )

        else:
            sweep_config = config_module.get_sweep_config()

            sweep_id = wandb.sweep(
                sweep=sweep_config,
                project=sweep_project,
            )

            print(f"Created W&B sweep: {sweep_id}")
            wandb.agent(
                sweep_id,
                function=lambda: train(
                    config=config,
                    use_sweep=True,
                ),
            )

    else:
        run_name = (
            args.run_name
            if args.run_name is not None
            else str(config.network_name)
        )
        train(
            config=config,
            run_name=run_name,
            use_sweep=False,
        )

if __name__ == "__main__":
    main()