import argparse
import importlib.util
import os
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = Path(SRC_DIR).parent

sys.path.insert(0, SRC_DIR)

import networks
from utils.ntk import ntk_eigendecomposition
from utils.prepare_data import create_and_normalize_coords

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute and visualize the empirical neural tangent kernel "
            "of an INR at initialization."
        )
    )

    parser.add_argument(
        "--architecture",
        choices=["WIRE", "SIREN"],
        default="WIRE",
        help="INR architecture",
    )
    parser.add_argument(
        "--config",
        default="configs/paper/inr.py",
        help=(
            "Configuration defining the coordinate normalization. "
        ),
    )
    parser.add_argument(
        "--data-file",
        required=True,
        help=(
            "LR HDF5 file whose dimensions define the coordinate domain. "
        ),
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=20.0,
        help="Frequency parameter omega_0",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=20.0,
        help="WIRE scale parameter sigma_0. Ignored for SIREN",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="Resolution of the 2D NTK grid",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for block-wise NTK computation",
    )
    parser.add_argument(
        "--num-eigenpairs",
        type=int,
        default=200,
        help="Number of NTK eigenpairs to compute",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for network initialization",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to src/tools/ntk_results",
    )
 
    return parser.parse_args()

def load_config(config_path):
    """Load a 4DFlowINR Python configuration file."""

    path = Path(config_path).expanduser()

    if not path.is_absolute():
        path = Path(SRC_DIR) / path

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find configuration file: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        "_4dflowinr_ntk_config",
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load configuration: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "get_config"):
        raise AttributeError(
            f"{path} does not define get_config()."
        )

    return module.get_config(), path

def build_model(config, args, device):
    """Construct the selected INR architecture."""

    if args.architecture == "WIRE":
        model = networks.WIRE(
            in_dim=config.network.in_dim,
            out_dim=config.network.out_dim,
            depth=config.network.depth,
            hidden_features=config.network.hidden_features,
            first_omega_0=args.omega,
            hidden_omega_0=args.omega,
            scale=args.sigma,
            complex=config.network.complex,
        )

    elif args.architecture == "SIREN":
        model = networks.SIREN(
            in_dim=config.network.in_dim,
            out_dim=config.network.out_dim,
            depth=config.network.depth,
            hidden_features=config.network.hidden_features,
            first_omega_0=args.omega,
            hidden_omega_0=args.omega,
        )
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")

    return model.to(device)

def create_ntk_grid(
    config,
    data_file,
    resolution,
    device,
):
    """
    Create a regular x-y NTK grid in the same normalized coordinate
    space used during 4DFlowINR training.

    The x/y domain is determined from the normalized coordinates of
    the supplied LR dataset. Time and z are fixed to actual normalized
    coordinates from that dataset.
    """

    # Training interprets data paths relative to src/.
    data_path = Path(data_file).expanduser()

    if not data_path.is_absolute():
        data_path = Path(SRC_DIR) / data_path

    data_path = data_path.resolve()

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Could not find data file: {data_path}"
        )

    # We only need the dimensions; no velocity data need to be loaded.
    with h5py.File(data_path, "r") as f:
        if "u" not in f:
            raise KeyError(f"'u' dataset not found in {data_path}")
        shape = f["u"].shape

    if len(shape) != 4:
        raise ValueError(f"Expected u with shape (T, X, Y, Z), got {shape}")

    t_len, x_len, y_len, z_len = shape

    (t_normalized, x_normalized, y_normalized, z_normalized, _) = create_and_normalize_coords(
        config, t_len, x_len, y_len, z_len)

    t_value = 0.5 * (float(t_normalized.min()) + float(t_normalized.max()))

    z_value = 0.5 * (float(z_normalized.min()) + float(z_normalized.max()))

    # Regular visualization grid spanning the actual normalized
    # x/y domain seen by this INR.
    xs = np.linspace(
        x_normalized.min(),
        x_normalized.max(),
        resolution,
    )

    ys = np.linspace(
        y_normalized.min(),
        y_normalized.max(),
        resolution,
    )

    grid_x, grid_y = np.meshgrid(xs,ys,indexing="xy")

    coords_np = np.stack(
        [
            np.full(resolution * resolution, t_value),
            grid_x.ravel(),
            grid_y.ravel(),
            np.full(resolution * resolution, z_value),
        ],
        axis=1,
    ).astype(np.float32)

    info = {
        "data_path": data_path,
        "data_shape": shape,
        "t_value": float(t_value),
        "z_value": float(z_value),
        "t_range": (float(t_normalized.min()),float(t_normalized.max())),
        "x_range": (float(x_normalized.min()),float(x_normalized.max())),
        "y_range": (float(y_normalized.min()),float(y_normalized.max())),
        "z_range": (float(z_normalized.min()),float(z_normalized.max())),
    }

    return torch.tensor(
        coords_np,
        dtype=torch.float32,
        device=device,
    ), info

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, config_path = load_config(args.config)

    # Reproducible network initialization
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    coords, grid_info = create_ntk_grid(
        config=config,
        data_file=args.data_file,
        resolution=args.resolution,
        device=device,
    )

    n_points = coords.shape[0]
    if args.num_eigenpairs >= n_points:
        raise ValueError(
            f"--num-eigenpairs must be smaller than the number of "
            f"grid points ({n_points})."
        )

    model = build_model(config, args, device)
    model.eval()

    # Descriptive run name.
    if args.architecture == "WIRE":
        name = (
            f"WIRE_omega{args.omega:g}_sigma{args.sigma:g}_seed{args.seed}"
        )
    else:
        name = (
            f"SIREN_omega{args.omega:g}_seed{args.seed}"
        )

    output_root = (
        args.output_dir
        if args.output_dir is not None
        else os.path.join(os.path.dirname(__file__), "ntk_results")
    )

    run_dir = os.path.join(output_root, name)
    os.makedirs(run_dir, exist_ok=True)

    print("="*30)
    print("Empirical NTK analysis")
    print("="*30)
    print(f"Architecture:      {args.architecture}")
    print(f"omega_0:           {args.omega}")
    if args.architecture == "WIRE":
        print(f"sigma_0:           {args.sigma}")
    print(f"Seed:              {args.seed}")
    print(
        f"Grid:              "
        f"{args.resolution} x {args.resolution} ({n_points} points)"
    )
    print(f"Batch size:        {args.batch_size}")
    print(f"Eigenpairs:        {args.num_eigenpairs}")
    print(f"Device:            {device}")

    print(f"Config:            {config_path}")
    print(f"Data:              {grid_info['data_path']}")
    print(f"Data shape:        {grid_info['data_shape']}")

    print("="*30)

    eigvals, eigvecs, ntk_matrix = ntk_eigendecomposition(
        model,
        coords,
        k=args.num_eigenpairs,
        batch_size=args.batch_size,
    )

    eigs = eigvals.numpy()
    eigvecs_np = eigvecs.numpy()
    ntk_np = ntk_matrix.cpu().numpy()

    cond = (
        eigs[0] / eigs[-1]
        if eigs[-1] != 0
        else float("inf")
    )

    print(
        f"lambda_max={eigs[0]:.4f}, "
        f"lambda_min={eigs[-1]:.4f}, "
        f"kappa={cond:.2f}"
    )

    # Save numerical results
    np.save(os.path.join(run_dir, "eigvals.npy"), eigs)
    np.save(os.path.join(run_dir, "eigvecs.npy"), eigvecs_np)
    np.save(os.path.join(run_dir, "ntk_matrix.npy"), ntk_np)

    # Eigenvector heatmaps
    for i in range(min(5, args.num_eigenpairs)):
        eigenfunction = eigvecs_np[:, i].reshape(
            args.resolution,
            args.resolution,
        )

        plt.figure(figsize=(4, 4))
        plt.imshow(
            eigenfunction,
            cmap="viridis",
            origin="lower",
        )
        plt.axis("off")
        plt.tight_layout(pad=0)

        plt.savefig(
            os.path.join(
                run_dir,
                f"{name}_eigvec_{i}.svg",
            ),
            format="svg",
            bbox_inches="tight",
        )
        plt.close()

    # Eigenvalue spectrum
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(eigs)
    axes[0].set_title("Raw eigenvalues")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Eigenvalue")

    axes[1].plot(eigs / eigs[0])
    axes[1].set_title("Normalized eigenvalues")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel(r"$\lambda / \lambda_{\max}$")

    fig.suptitle(name.replace("_", " "))
    fig.tight_layout()

    fig.savefig(
        os.path.join(run_dir, f"{name}_eigvals.svg"),
        format="svg",
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Saved to {run_dir}/")


if __name__ == "__main__":
    main()