# --- Minimal PDE-residual plotting script -------------------------------------
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

from configs.tunings_260409_SAPINN_LBFGS.Config_260409_SAPINN_sweepN_factorial import get_config
from utils.prepare_data import prepare_data, load_data
import networks
from utils.loss_utils import navier_stokes_loss
from utils.prepare_data import sample_collocation_points


# --------- Helpers ------------------------------------------------------------
def as_torch_cloud(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device=device, dtype=torch.float32)


def make_plot_cloud(xyz_data, mask, max_points=2_000_000):
    """Return a big fluid-only cloud as numpy (for plotting only)."""
    if isinstance(xyz_data, torch.Tensor):
        xyz_np = xyz_data.detach().cpu().numpy()
    else:
        xyz_np = xyz_data
    fluid_idx = np.where(mask == 1)[0]
    if fluid_idx.size > max_points:
        fluid_idx = np.random.choice(fluid_idx, size=max_points, replace=False)
    fluid_idx.sort()
    return xyz_np[fluid_idx]


def build_model(config, device):
    if config.network.arch == "SIREN":
        model = networks.SIREN(
            in_dim=config.network.in_dim,
            out_dim=config.network.out_dim,
            depth=config.network.depth,
            hidden_features=config.network.hidden_features,
            first_omega_0=config.network.omega_0,
            hidden_omega_0=config.network.omega_0
        )
    elif config.network.arch == "WIRE":
        model = networks.WIRE(
            in_dim=config.network.in_dim,
            out_dim=config.network.out_dim,
            depth=config.network.depth,
            hidden_features=config.network.hidden_features,
            first_omega_0=config.network.omega_0,
            hidden_omega_0=config.network.omega_0,
            scale=config.network.sigma_0,
            complex=config.network.complex
        )
    elif config.network.arch == "FFN":
        model = networks.FFN(
            input_dim=config.network.in_dim,
            output_dim=config.network.out_dim,
            depth=config.network.depth,
            hidden_dim=config.network.hidden_features,
            fourier_mapping_size=config.network.fourier_mapping_size,
            scale=config.network.fourier_scale
        )
    else:
        raise ValueError(f"Unknown network arch: {config.network.arch}")
    return model.to(device)


def plot_residual_distribution(
    config, model, xyz_collocation, standardization_factors,
    *,
    time_mode="all",        # "all" | "phase" | "bins"
    phase_idx=None,         # if time_mode=="phase"
    time_bins=4,            # if time_mode=="bins"
    scale="log",            # "log" | "linear"
    cmap="YlOrRd",          # red/yellow = higher residuals
    n_points=300_000,       # points per image (None = all)
    chunk=200_000,          # eval chunk size (OOM safety)
    out_prefix="residuals/out",
    vmin=None,              # manual colorbar min (optional)
    vmax=None,              # manual colorbar max (optional)
):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    xyz_col = as_torch_cloud(xyz_collocation, DEVICE).detach()
    xyz_col.requires_grad_(True)

    t_all = xyz_col[:, 0].detach().cpu().numpy()

    def select_time_sets():
        if time_mode == "all":
            return [np.arange(xyz_col.shape[0])], ["all"]
        if time_mode == "phase":
            assert phase_idx is not None
            uniq = np.unique(t_all)
            target = uniq[np.argmin(np.abs(uniq - phase_idx))]
            return [np.where(t_all == target)[0]], [f"t{int(phase_idx)}"]
        if time_mode == "bins":
            qs = np.quantile(t_all, np.linspace(0, 1, time_bins + 1))
            idxs, labels = [], []
            for b in range(time_bins):
                lo, hi = qs[b], qs[b+1]
                sel = np.where((t_all >= lo) & (t_all < hi))[0] if b < time_bins - 1 \
                      else np.where((t_all >= lo) & (t_all <= hi))[0]
                if sel.size:
                    idxs.append(sel); labels.append(f"bin{b+1}")
            return idxs, labels
        raise ValueError("time_mode must be 'all' | 'phase' | 'bins'")

    sets, labels = select_time_sets()

    def make_norm(vals):
        if vmin is not None and vmax is not None:
            return mpl.colors.LogNorm(vmin=vmin, vmax=vmax) if scale == "log" \
                   else mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        if scale == "log":
            eps = 1e-12
            vmin_a = float(np.min(vals[vals > eps])) if np.any(vals > eps) else eps
            vmax_a = float(np.max(vals))
            return mpl.colors.LogNorm(vmin=max(vmin_a, eps), vmax=max(vmax_a, vmin_a + 1e-9))
        return mpl.colors.Normalize(vmin=0.0, vmax=float(np.max(vals)))

    for i, (ids, lab) in enumerate(zip(sets, labels), 1):
        print(f"[{i}/{len(sets)}] Evaluating {lab} (candidates={ids.size})")
        sel = ids if (n_points is None or ids.size <= n_points) \
              else np.sort(np.random.choice(ids, size=n_points, replace=False))
        print(f"   -> plotting {sel.size} points...")

        cloud = xyz_col[sel]
        resid_chunks = []
        start = 0
        while start < cloud.shape[0]:
            end = min(start + chunk, cloud.shape[0])
            chunk_xyz = cloud[start:end].clone().detach().requires_grad_(True)
            with torch.enable_grad():
                uvw_pred = model(chunk_xyz)
                per_point, _, _ = navier_stokes_loss(
                    uvw_pred, chunk_xyz, standardization_factors, config,
                    return_per_point=True
                )
            resid_chunks.append(per_point.detach().cpu().numpy())
            start = end

        resid = np.concatenate(resid_chunks, axis=0)
        xyz_np = cloud.detach().cpu().numpy()
        x, y, z = xyz_np[:, 1], xyz_np[:, 2], xyz_np[:, 3]

        finite = np.isfinite(resid)
        if not np.all(finite):
            x, y, z, resid = x[finite], y[finite], z[finite], resid[finite]

        # Print percentiles to help calibrate vmin/vmax
        pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
        vals = np.percentile(resid, pcts)
        print(f"   Residual stats [{lab}]: min={resid.min():.3e}  max={resid.max():.3e}  mean={resid.mean():.3e}")
        print(f"   Percentiles: " + "  ".join(f"p{p:.0f}={v:.3e}" for p, v in zip(pcts, vals)))

        resid_norm = (resid - resid.min()) / (resid.max() - resid.min() + 1e-12)
        alpha_vals = 0.05 + 0.95 * resid_norm

        norm_obj = make_norm(resid)
        cmap_obj = plt.get_cmap(cmap)
        rgba = cmap_obj(norm_obj(resid)).astype(np.float32)
        rgba[:, 3] = alpha_vals

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=rgba, s=3, edgecolors='none')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f"PDE residuals ({scale}) | {lab} | N={resid.size}")
        sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm_obj)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02).set_label("Residual magnitude")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{lab}.png", dpi=200)
        plt.close(fig)
        print(f"   -> saved '{out_prefix}_{lab}.png'")


def plot_c_weights(c_weights, out_path, iteration, n_bins=100):
    """
    Plot the distribution of SA collocation weights at a given iteration.
    Shows histogram + summary stats (mean, std, % clipped at min/max).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    w = np.asarray(c_weights, dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: histogram ---
    ax = axes[0]
    ax.hist(w, bins=n_bins, color='steelblue', edgecolor='none', alpha=0.85)
    ax.axvline(w.mean(), color='red',    lw=1.5, linestyle='--', label=f'mean={w.mean():.3f}')
    ax.axvline(np.median(w), color='orange', lw=1.5, linestyle=':', label=f'median={np.median(w):.3f}')
    ax.set_xlabel('c_weight'); ax.set_ylabel('Count')
    ax.set_title(f'c_weights distribution — iter {iteration:,}')
    ax.legend(fontsize=9)

    # % clipped at lower / upper bounds
    w_min, w_max = w.min(), w.max()
    pct_lo = 100.0 * np.mean(w == w_min)
    pct_hi = 100.0 * np.mean(w == w_max)
    ax.text(0.98, 0.97, f'min={w_min:.3f} ({pct_lo:.1f}%)\nmax={w_max:.3f} ({pct_hi:.1f}%)\nstd={w.std():.3f}',
            transform=ax.transAxes, va='top', ha='right', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # --- Right: CDF ---
    ax2 = axes[1]
    sorted_w = np.sort(w)
    cdf = np.arange(1, len(sorted_w) + 1) / len(sorted_w)
    ax2.plot(sorted_w, cdf, color='steelblue', lw=1.5)
    ax2.set_xlabel('c_weight'); ax2.set_ylabel('CDF')
    ax2.set_title(f'c_weights CDF — iter {iteration:,}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"   -> saved '{out_path}'")


def plot_c_weights_evolution(iterations_weights, out_path):
    """
    Multi-panel summary: one subplot per checkpoint showing c_weight histogram.
    iterations_weights: list of (iteration, c_weights_array) tuples.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = len(iterations_weights)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for idx, (it, w) in enumerate(iterations_weights):
        w = np.asarray(w, dtype=np.float32)
        ax = axes_flat[idx]
        ax.hist(w, bins=60, color='steelblue', edgecolor='none', alpha=0.85)
        ax.axvline(w.mean(), color='red', lw=1.2, linestyle='--')
        ax.set_title(f'iter {it:,}  μ={w.mean():.3f}  σ={w.std():.3f}', fontsize=9)
        ax.set_xlabel('c_weight', fontsize=8); ax.set_ylabel('Count', fontsize=8)
        ax.tick_params(labelsize=7)

    # hide unused subplots
    for idx in range(len(iterations_weights), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle('SA c_weights evolution across checkpoints', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"   -> saved '{out_path}'")


def plot_c_weights_spatial(
    c_weights, xyz_collocation,
    *,
    iteration,
    time_mode="bins",
    time_bins=6,
    n_points=300_000,
    cmap="RdYlBu_r",
    out_prefix="residuals/out_cweights",
    vmin=None,
    vmax=None,
):
    """
    Plot SA collocation weights in 3D space.
    Red = high weight (hard region the SA is focusing on).
    Blue = low weight (easy region).
    Alpha mapped to weight magnitude so low-weight points are transparent.
    """
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    w = np.asarray(c_weights, dtype=np.float32)
    xyz_np = np.asarray(xyz_collocation, dtype=np.float32)  # (N, 4): [t, x, y, z]
    t_all = xyz_np[:, 0]

    if time_mode == "bins":
        qs = np.quantile(t_all, np.linspace(0, 1, time_bins + 1))
        sets, labels = [], []
        for b in range(time_bins):
            lo, hi = qs[b], qs[b + 1]
            sel = np.where((t_all >= lo) & (t_all < hi))[0] if b < time_bins - 1 \
                  else np.where((t_all >= lo) & (t_all <= hi))[0]
            if sel.size:
                sets.append(sel); labels.append(f"bin{b + 1}")
    else:  # "all"
        sets = [np.arange(len(w))]; labels = ["all"]

    # Fix colorbar range across all bins for comparability
    w_vmin = float(w.min()) if vmin is None else vmin
    w_vmax = float(w.max()) if vmax is None else vmax
    norm_obj = mpl.colors.Normalize(vmin=w_vmin, vmax=w_vmax)
    cmap_obj = plt.get_cmap(cmap)

    for ids, lab in zip(sets, labels):
        sel = ids if ids.size <= n_points \
              else np.sort(np.random.choice(ids, size=n_points, replace=False))

        w_sel = w[sel]
        xyz_sel = xyz_np[sel]
        x, y, z = xyz_sel[:, 1], xyz_sel[:, 2], xyz_sel[:, 3]

        # alpha: high weight = opaque, low weight = transparent
        w_norm = (w_sel - w_sel.min()) / (w_sel.max() - w_sel.min() + 1e-12)
        alpha_vals = 0.05 + 0.95 * w_norm

        rgba = cmap_obj(norm_obj(w_sel)).astype(np.float32)
        rgba[:, 3] = alpha_vals

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=rgba, s=3, edgecolors='none')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f"SA c_weights | iter {iteration:,} | {lab} | N={sel.size}")

        sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm_obj)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02).set_label("c_weight")
        plt.tight_layout()
        out = f"{out_prefix}_{lab}.png"
        plt.savefig(out, dpi=200); plt.close(fig)
        print(f"   -> saved '{out}'")


# --------- Main ---------------------------------------------------------------
if __name__ == "__main__":

    print("Starting script")

    MODEL_DIR = "../models/260409_SAPINN_sweepN_factorial"

    RUNS = {
        "R1": os.path.join(MODEL_DIR, "SA_PINN_ICAD48_sv13_sweepN_R1_20260414-1634_dx6dn672"),
        "R2": os.path.join(MODEL_DIR, "SA_PINN_ICAD48_sv13_sweepN_R2_20260414-1634_mngdbct0"),
        "R3": os.path.join(MODEL_DIR, "SA_PINN_ICAD48_sv13_sweepN_R3_20260417-1506_7innh3bu"),
        "R4": os.path.join(MODEL_DIR, "SA_PINN_ICAD48_sv13_sweepN_R4_20260417-1506_k71hill6"),
    }

    # Checkpoints saved every 2500 iters up to 40k
    CHECKPOINTS = list(range(2500, 40001, 2500))

    # ==========================================
    # Load data once (shared across runs)
    # ==========================================
    config = get_config()
    config.data_file    = "../data/stenosis_50/ICAD48_05mm3_20ms_LR_sv13_tSNR10_newMask.h5"
    config.data_file_ref = "../data/stenosis_50/ICAD48_05mm3_20ms.h5"
    config.constants.venc = 1.3
    config.predictions.peak_flow_idx = 14

    u, v, w, p, px, py, pz, mask, config = load_data(config)
    uvw_data, xyz_data, mask_flat, boundary_mask_flat, standardization_factors, U_max = \
        prepare_data(config, u, v, w, p, px, py, pz, mask)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_flat = mask_flat.astype(np.uint8)

    config.collocation_points = 1_500_000
    xyz_collocation = sample_collocation_points(config, xyz_data, mask_flat)

    # ==========================================
    # Per-run loop
    # ==========================================
    for run_name, run_dir in RUNS.items():
        print(f"\n{'='*60}")
        print(f"Processing {run_name}: {run_dir}")
        print('='*60)

        CKPT_DIR   = os.path.join(run_dir, "checkpoints")
        RESULTS_DIR = f"residuals/R_tests/{run_name}"
        os.makedirs(RESULTS_DIR, exist_ok=True)

        iterations_weights = []

        for ckpt_iter in CHECKPOINTS:
            ckpt_name = f"260409_SAPINN_sweepN_it{ckpt_iter:06d}.pth"
            ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
            if not os.path.exists(ckpt_path):
                print(f"[SKIP] {ckpt_path} not found")
                continue

            print(f"\n=== {run_name} iter {ckpt_iter:,} ===")
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            c_weights = ckpt.get("c_weights", None)

            if c_weights is not None:
                c_weights = np.asarray(c_weights, dtype=np.float32)
                print(f"   c_weights: shape={c_weights.shape}  min={c_weights.min():.4f}  "
                      f"max={c_weights.max():.4f}  mean={c_weights.mean():.4f}  std={c_weights.std():.4f}")
                iterations_weights.append((ckpt_iter, c_weights))
           
                plot_c_weights(
                    c_weights,
                    out_path=os.path.join(RESULTS_DIR, f"c_weights_it{ckpt_iter:06d}.png"),
                    iteration=ckpt_iter,
                )
           
                plot_c_weights_spatial(
                    c_weights, xyz_collocation,
                    iteration=ckpt_iter,
                    time_mode="bins", time_bins=6,
                    n_points=300_000,
                    cmap="RdYlBu_r",
                    out_prefix=os.path.join(RESULTS_DIR, f"c_weights_spatial_it{ckpt_iter:06d}"),
                    vmin=0.95,
                    vmax=1.25,
                )
            else:
                print(f"   [INFO] c_weights is None (SA not active at this checkpoint)")

            # PDE residual scatter — final checkpoint only (expensive)
            if ckpt_iter == CHECKPOINTS[-1]:
                model = build_model(config, DEVICE)
                model.load_state_dict(ckpt['model_state_dict'])

                plot_residual_distribution(
                    config, model,
                    xyz_collocation=xyz_collocation,
                    standardization_factors=standardization_factors,
                    time_mode="bins", time_bins=6,
                    scale="log",
                    cmap="YlOrRd",
                    n_points=200_000,
                    chunk=10_000,
                    out_prefix=os.path.join(RESULTS_DIR, f"residuals_it{ckpt_iter:06d}"),
                    vmin=5e-4,
                    vmax=1e-2
                )

        if iterations_weights:
            plot_c_weights_evolution(
                iterations_weights,
                out_path=os.path.join(RESULTS_DIR, "c_weights_evolution.png"),
            )
        else:
            print(f"[INFO] No c_weights found for {run_name} — SA was not active.")
