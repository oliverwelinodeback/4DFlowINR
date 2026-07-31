"""
Prediction with a trained 4DFlowINR model.

From the repository root:
    python src/predict.py \
        --config configs/paper/inr.py \
        --checkpoint models/.../checkpoints/paper_inr_it008000.pth \
        --data-file data/healthy/example_lr.h5 \
        --output-dir predictions/example \
        --spatial-factor 2 \
        --temporal-factor 2
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path
import numpy as np
import torch
# Repository paths
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
for path in (SRC_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import networks
from utils.checkpoints import load_model_weights
from utils.data_io import load_data
from utils.prediction_utils import predict_superresolved_grid

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate an arbitrary-resolution prediction from a trained 4DFlowINR checkpoint."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Python configuration file used for the model, "
            "e.g. configs/paper/inr.py."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help=(
            "Optional input HDF5 override. If omitted, config.data_file is used"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory in which prediction outputs are written",
    )
    parser.add_argument(
        "--spatial-factor",
        type=int,
        default=None,
        help=(
            "Optional spatial upsampling-factor override. "
            "If omitted, config.predictions.spatial_factor is used"
        ),
    )
    parser.add_argument(
        "--temporal-factor",
        type=int,
        default=None,
        help=(
            "Optional temporal upsampling-factor override. "
            "If omitted, config.predictions.temporal_factor is used"
        ),
    )

    return parser.parse_args()


# Configuration helpers
def resolve_existing_path(path_arg, description):
    # Resolve a path supplied relative to the current, repository, or src directory
    supplied_path = Path(path_arg).expanduser()

    if supplied_path.is_absolute():
        candidates = [supplied_path]
    else:
        candidates = [REPO_ROOT / supplied_path, SRC_DIR / supplied_path,
        ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = "\n".join(
        "  - {}".format(path)
        for path in candidates
    )

    raise FileNotFoundError(
        "Could not find {} '{}'.\nSearched:\n{}".format(
            description, path_arg, searched,
        )
    )


def load_config(config_path):
    """Load a Python configuration file defining get_config()."""

    module_name = "_4dflowinr_predict_config_{}".format(
        config_path.stem
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(config_path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            "Could not load configuration file: {}".format(config_path)
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "get_config"):
        raise AttributeError(
            "Configuration file '{}' must define get_config().".format(config_path)
        )

    return module.get_config()


# Prediction
def main():

    args = parse_args()

    # Resolve paths before changing working directory
    config_path = resolve_existing_path(
        args.config,
        "configuration file",
    )

    checkpoint_path = resolve_existing_path(
        args.checkpoint,
        "checkpoint",
    )

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # predict_superresolved_grid() currently writes SR_final.h5.
    output_h5 = output_dir / "SR_final.h5"

    if output_h5.exists():
        raise FileExistsError(
            "Prediction file already exists: {}"
            .format(output_h5)
        )

    # Load configuration
    config = load_config(config_path)
    config.sweep = False

    # Optional input-data override
    if args.data_file is not None:
        data_file = resolve_existing_path(
            args.data_file,
            "input data file",
        )
        config.data_file = str(data_file)

    # Optional output-resolution overrides
    if args.spatial_factor is not None:
        if args.spatial_factor < 1:
            raise ValueError(
                "--spatial-factor must be >= 1."
            )
        config.predictions.spatial_factor = (
            args.spatial_factor
        )

    if args.temporal_factor is not None:
        if args.temporal_factor < 1:
            raise ValueError(
                "--temporal-factor must be >= 1."
            )
        config.predictions.temporal_factor = (
            args.temporal_factor
        )

    # prediction_utils uses config.log_dir as its output location
    config.log_dir = str(output_dir)

    # Keep the same relative-path convention as train.py/config files.
    os.chdir(SRC_DIR)

    # Load input data
    print("="*30)
    print("4DFlowINR prediction")
    print("Configuration:    {}".format(config_path))
    print("Checkpoint:       {}".format(checkpoint_path))
    print("Input data:       {}".format(config.data_file))
    print("Output directory: {}".format(output_dir))
    print(
        "SR factor:       {}x spatial, {}x temporal".format(
            config.predictions.spatial_factor, config.predictions.temporal_factor,
        )
    )
    print("="*30)

    u, v, w, p, px, py, pz, mask, config = load_data(config)
    input_U_max = max(u.max(), v.max(), w.max())

    # Build network and restore checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device: {}".format(device))

    model = networks.build_model(config).to(device)

    try:
        checkpoint = load_model_weights(
            checkpoint_path, model, device,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not load the checkpoint into the configured network, "
            "Ensure that --config matches the configuration used to train this checkpoint"
        ) from exc

    # Prefer the normalization factor stored with the trained model.
    checkpoint_U_max = checkpoint.get("U_max", None)

    if checkpoint_U_max is not None:
        U_max = float(checkpoint_U_max)
    else:
        U_max = float(input_U_max)

    iteration = int(checkpoint.get("iteration", 0) or 0)

    print("Checkpoint iteration: {}".format(iteration))

    # Evaluate continuous INR on requested output grid
    predict_superresolved_grid(
        config=config, model=model, device=device,
        it=iteration, u=u, mask=mask,
        U_max=U_max, save_pred=True,
    )

    print("="*30)
    print("Prediction complete.")
    print("HDF5 output: {}".format(output_h5))
    print("="*30)

if __name__ == "__main__":
    main()