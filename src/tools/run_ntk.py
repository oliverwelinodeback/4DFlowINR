import argparse
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Allow imports from src/
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC_DIR)

import networks
from utils.ntk import ntk_eigendecomposition

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
        "--depth",
        type=int,
        default=6,
        help="Network depth",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=128,
        help="Number of hidden features",
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


def build_model(args, device):
    """Construct the selected INR architecture."""

    if args.architecture == "WIRE":
        model = networks.WIRE(
            in_dim=4,
            out_dim=3,
            depth=args.depth,
            hidden_features=args.hidden,
            first_omega_0=args.omega,
            hidden_omega_0=args.omega,
            scale=args.sigma,
            complex=False,
        )

    elif args.architecture == "SIREN":
        model = networks.SIREN(
            in_dim=4,
            out_dim=3,
            depth=args.depth,
            hidden_features=args.hidden,
            first_omega_0=args.omega,
            hidden_omega_0=args.omega,
        )

    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")

    return model.to(device)

def create_ntk_grid(resolution, device):
    """Create a regular x-y grid with fixed t=0.5 and z=0.5."""

    xs = np.linspace(0, 1, resolution)
    ys = np.linspace(0, 1, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)

    coords_np = np.stack(
        [
            np.full(resolution * resolution, 0.5),  # t
            grid_x.ravel(),                         # x
            grid_y.ravel(),                         # y
            np.full(resolution * resolution, 0.5),  # z
        ],
        axis=1,
    ).astype(np.float32)

    return torch.tensor(coords_np, device=device)


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Reproducible network initialization
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    coords = create_ntk_grid(
        args.resolution,
        device,
    )

    n_points = coords.shape[0]
    if args.num_eigenpairs >= n_points:
        raise ValueError(
            f"--num-eigenpairs must be smaller than the number of "
            f"grid points ({n_points})."
        )

    model = build_model(args, device)
    model.eval()

    # Descriptive run name.
    if args.architecture == "WIRE":
        name = (
            f"WIRE_omega{args.omega:g}_sigma{args.sigma:g}_"
            f"depth{args.depth}_hidden{args.hidden}_seed{args.seed}"
        )
    else:
        name = (
            f"SIREN_omega{args.omega:g}_"
            f"depth{args.depth}_hidden{args.hidden}_seed{args.seed}"
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
    print(f"Depth:             {args.depth}")
    print(f"Hidden features:   {args.hidden}")
    print(f"Seed:              {args.seed}")
    print(
        f"Grid:              "
        f"{args.resolution} x {args.resolution} ({n_points} points)"
    )
    print(f"Batch size:        {args.batch_size}")
    print(f"Eigenpairs:        {args.num_eigenpairs}")
    print(f"Device:            {device}")
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