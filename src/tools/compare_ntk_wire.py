import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import networks
from utils.ntk import ntk_eigendecomposition

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIGS   = [(5, 5), (20, 20), (60, 30), (90, 90)]  # (omega_0, sigma_0)
K_EIGS    = 200
BATCH     = 128
DEPTH     = 6
HIDDEN    = 64
SEED      = 42
RES       = 64   # RES x RES = 4096 points, same idea as paper's pixel grid

torch.manual_seed(SEED)
np.random.seed(SEED)

# Regular 2D grid in (x, y) at fixed t=0.5, z=0.5 — analogous to the paper's pixel grid
t_val, z_val = 0.5, 0.5
xs = np.linspace(0, 1, RES)
ys = np.linspace(0, 1, RES)
grid_x, grid_y = np.meshgrid(xs, ys)
coords_np = np.stack([
    np.full(RES * RES, t_val),
    grid_x.ravel(),
    grid_y.ravel(),
    np.full(RES * RES, z_val),
], axis=1).astype(np.float32)  # (RES*RES, 4)
coords = torch.tensor(coords_np, device=DEVICE)

outdir = os.path.join(os.path.dirname(__file__), "ntk_results")
os.makedirs(outdir, exist_ok=True)

for omega, sigma in CONFIGS:
    name = f"WIRE_omega{omega}_sigma{sigma}_depth{DEPTH}_hidden{HIDDEN}"
    print(f"\n--- {name} ---")
    model = networks.WIRE(
        in_dim=4, out_dim=3, depth=DEPTH, hidden_features=HIDDEN,
        first_omega_0=omega, hidden_omega_0=omega,
        scale=sigma, complex=False
    ).to(DEVICE)
    model.eval()
    with torch.no_grad():
        eigvals, eigvecs, ntk_matrix = ntk_eigendecomposition(model, coords, k=K_EIGS, batch_size=BATCH)
    eigs = eigvals.numpy()
    cond = eigs[0] / eigs[-1] if eigs[-1] != 0 else float('inf')
    print(f"  λ_max={eigs[0]:.4f}  λ_min={eigs[-1]:.4f}  κ={cond:.2f}")

    run_dir = os.path.join(outdir, name)
    os.makedirs(run_dir, exist_ok=True)
    np.save(os.path.join(run_dir, "eigvals.npy"), eigs)
    np.save(os.path.join(run_dir, "eigvecs.npy"), eigvecs.numpy())
    np.save(os.path.join(run_dir, "ntk_matrix.npy"), ntk_matrix.cpu().numpy())

    # Eigenvector heatmaps — direct reshape, no interpolation (same as paper)
    for i in range(5):
        v_i = eigvecs[:, i].numpy().reshape(RES, RES)
        plt.figure(figsize=(4, 4))
        plt.imshow(v_i, cmap='viridis', origin='lower')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(run_dir, f"{name}_eigvec_{i}.svg"), format="svg", bbox_inches="tight")
        plt.close()

    # Eigenvalue spectrum
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(eigs)
    axes[0].set_title("Raw eigenvalues")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Eigenvalue")
    axes[1].plot(eigs / eigs[0])
    axes[1].set_title("Normalized eigenvalues (λ / λ_max)")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Normalized eigenvalue")
    plt.suptitle(name.replace("_", " "))
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"{name}_eigvals.svg"), format="svg", bbox_inches="tight")
    plt.close()

    print(f"  Saved to {run_dir}/")
