# --- Minimal PDE-residual plotting script -------------------------------------
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

from configs.tunings_251106.Config_251031_sweep_WIRE_momentum_SA_sampling_sv import get_config
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

    # to torch, on device
    xyz_col = as_torch_cloud(xyz_collocation, DEVICE).detach()
    xyz_col.requires_grad_(True)

    # ----- time selection -----
    t_all = xyz_col[:, 0].detach().cpu().numpy()

    def select_time_sets():
        if time_mode == "all":
            return [np.arange(xyz_col.shape[0])], ["all"]

        if time_mode == "phase":
            assert phase_idx is not None, "Provide phase_idx for time_mode='phase'."
            uniq = np.unique(t_all)
            target = uniq[np.argmin(np.abs(uniq - phase_idx))]
            return [np.where(t_all == target)[0]], [f"t{int(phase_idx)}"]

        if time_mode == "bins":
            qs = np.quantile(t_all, np.linspace(0, 1, time_bins + 1))
            idxs, labels = [], []
            for b in range(time_bins):
                lo, hi = qs[b], qs[b+1]
                if b < time_bins - 1:
                    sel = np.where((t_all >= lo) & (t_all < hi))[0]
                else:
                    sel = np.where((t_all >= lo) & (t_all <= hi))[0]
                if sel.size:
                    idxs.append(sel)
                    labels.append(f"bin{b+1}")
            return idxs, labels

        raise ValueError("time_mode must be 'all' | 'phase' | 'bins'")

    sets, labels = select_time_sets()

    # ----- color normalization -----
    def make_norm(vals):
        # Use manual override if provided
        if vmin is not None and vmax is not None:
            if scale == "log":
                return mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                return mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        # Otherwise auto-scale
        if scale == "log":
            eps = 1e-12
            vmin_auto = float(np.min(vals[vals > eps])) if np.any(vals > eps) else eps
            vmax_auto = float(np.max(vals))
            return mpl.colors.LogNorm(vmin=max(vmin_auto, eps), vmax=max(vmax_auto, vmin_auto + 1e-9))
        else:
            return mpl.colors.Normalize(vmin=0.0, vmax=float(np.max(vals)))

    # ----- per set plotting -----
    for i, (ids, lab) in enumerate(zip(sets, labels), 1):
        print(f"[{i}/{len(sets)}] Evaluating {lab} (candidates={ids.size})")

        # subsample uniformly for plotting
        if n_points is None or n_points < 0 or ids.size <= n_points:
            sel = ids
        else:
            sel = np.random.choice(ids, size=n_points, replace=False)
            sel.sort()

        print(f"   -> plotting {sel.size} points...")

        cloud = xyz_col[sel]

        # chunked residual eval
        resid_chunks = []
        start = 0
        while start < cloud.shape[0]:
            end = min(start + chunk, cloud.shape[0])
            chunk_xyz = cloud[start:end]
            chunk_xyz.requires_grad_(True)
            with torch.enable_grad():
                uvw_pred = model(chunk_xyz)
                per_point,_,_ = navier_stokes_loss(
                    uvw_pred, chunk_xyz, standardization_factors, config,
                    return_per_point=True
                )
            resid_chunks.append(per_point.detach().cpu().numpy())
            start = end

        resid = np.concatenate(resid_chunks, axis=0)

        # plot x,y,z
        xyz_np = cloud.detach().cpu().numpy()
        x, y, z = xyz_np[:, 1], xyz_np[:, 2], xyz_np[:, 3]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # --- scatter with per-point alpha based on residual magnitude (robust) ---

        # drop any NaN/Inf residuals to keep shapes aligned
        finite = np.isfinite(resid)
        if not np.all(finite):
            x, y, z = x[finite], y[finite], z[finite]
            resid = resid[finite]

        # normalize residuals to [0,1] for alpha mapping
        resid_norm = (resid - resid.min()) / (resid.max() - resid.min() + 1e-12)

        # opacity mapping (low residual = transparent, high = opaque)
        alpha_min = 0.05
        alpha_max = 1.0
        alpha_vals = alpha_min + (alpha_max - alpha_min) * resid_norm

        # build RGBA via colormap + norm, then inject alpha
        norm_obj = make_norm(resid)
        cmap_obj = plt.get_cmap(cmap)
        rgba = cmap_obj(norm_obj(resid)).astype(np.float32)   # (N,4)
        rgba[:, 3] = alpha_vals                               # set alpha channel

        # render using explicit per-point colors (no cmap/norm here)
        sc = ax.scatter(
            x, y, z,
            c=rgba,                 # pass full RGBA per point
            s=3,
            edgecolors='none'
        )

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f"PDE residuals ({scale}) | {lab} | N={resid.size}")
        # Create a fake mappable so the colorbar matches the colormap & scaling
        sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm_obj)
        sm.set_array([])  # required for matplotlib < 3.7
        cbar = plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label("Residual magnitude")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{lab}.png", dpi=200)
        plt.close(fig)

        print(f"   -> saved '{out_prefix}_{lab}.png'")



# --------- Main ---------------------------------------------------------------
if __name__ == "__main__":
    
    print("Starting script")
    config = get_config()

    #network_path = "../models/251125_WIRE_MOMENTUM_ALL_SV_SAPINN/251125_WIRE_MOMENTUM_ALL_SV_NewMask_SA_HV01_sv17_20251127-0926/checkpoints/251125_WIRE_MOMENTUM_ALL_SV_SAPINN_it040000.pth"
    #results_directory = "residuals/HV01"
    #config.predictions.peak_flow_idx = 12
    #config.data_file = "../data/healthy/HV01_05mm3_20ms_LR_sv17_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/healthy/HV01_05mm3_20ms.h5"
    #res_case = 'HV01'
    #network_path = "../models/251125_WIRE_MOMENTUM_ALL_SV_SAPINN/251125_WIRE_MOMENTUM_ALL_SV_NewMask_SA_HV03_sv13_20251126-1657/checkpoints/251125_WIRE_MOMENTUM_ALL_SV_SAPINN_it040000.pth"
    #results_directory = "residuals/HV03"
    #res_case = 'HV03'
    #config.predictions.peak_flow_idx = 4
    #config.data_file = "../data/healthy/HV03_05mm3_20ms_LR_sv13_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/healthy/HV03_05mm3_20ms.h5"
    #network_path = "../models/251125_WIRE_MOMENTUM_ALL_SV_SAPINN/251125_WIRE_MOMENTUM_ALL_SV_NewMask_SA_HV06_sv12_20251126-1657/checkpoints/251125_WIRE_MOMENTUM_ALL_SV_SAPINN_it040000.pth"
    #results_directory = "residuals/HV06"
    #res_case = 'HV06'
    #config.predictions.peak_flow_idx = 2
    #config.data_file = "../data/healthy/HV06_05mm3_20ms_LR_sv12_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/healthy/HV06_05mm3_20ms.h5"
    #network_path = "../models/251125_WIRE_MOMENTUM_ALL_SV_SAPINN/251125_WIRE_MOMENTUM_ALL_SV_NewMask_SA_ICAD17_sv41_20251127-0701/checkpoints/251125_WIRE_MOMENTUM_ALL_SV_SAPINN_it040000.pth"
    #results_directory = "residuals/ICAD17"
    #res_case = 'ICAD17'
    #config.predictions.peak_flow_idx = 8
    #config.data_file = "../data/stenosis_70/ICAD17_05mm3_20ms_LR_sv41_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_70/ICAD17_05mm3_20ms.h5"
    #network_path = "../models/251125_WIRE_MOMENTUM_ALL_SV_SAPINN/251125_WIRE_MOMENTUM_ALL_SV_NewMask_SA_ICAD21_sv26_20251127-0704/checkpoints/251125_WIRE_MOMENTUM_ALL_SV_SAPINN_it040000.pth"
    #results_directory = "residuals/ICAD21"
    #res_case = 'ICAD21'
    #config.predictions.peak_flow_idx = 12
    #config.data_file = "../data/stenosis_70/ICAD21_05mm3_20ms_LR_sv26_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_70/ICAD21_05mm3_20ms.h5"
    #network_path = "../models/251125_WIRE_MOMENTUM_ALL_SV_SAPINN/251125_WIRE_MOMENTUM_ALL_SV_NewMask_SA_ICAD28_sv13_20251126-1657/checkpoints/251125_WIRE_MOMENTUM_ALL_SV_SAPINN_it040000.pth"
    #results_directory = "residuals/ICAD28"
    #res_case = 'ICAD28'
    #config.predictions.peak_flow_idx = 2
    #config.data_file = "../data/stenosis_50/ICAD28_05mm3_20ms_LR_sv13_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_50/ICAD28_05mm3_20ms.h5"
    #network_path = "../models/251125_WIRE_MOMENTUM_ALL_SV_SAPINN/251125_WIRE_MOMENTUM_ALL_SV_NewMask_SA_ICAD48_sv13_20251126-1657/checkpoints/251125_WIRE_MOMENTUM_ALL_SV_SAPINN_it040000.pth"
    #results_directory = "residuals/ICAD48"
    #res_case = 'ICAD48'
    #config.predictions.peak_flow_idx = 14
    #config.data_file = "../data/stenosis_50/ICAD48_05mm3_20ms_LR_sv13_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_50/ICAD48_05mm3_20ms.h5"
    #network_path = "../models/251125_WIRE_MOMENTUM_ALL_SV_SAPINN/251125_WIRE_MOMENTUM_ALL_SV_NewMask_SA_ICAD98_sv51_20251126-1657/checkpoints/251125_WIRE_MOMENTUM_ALL_SV_SAPINN_it040000.pth"
    #results_directory = "residuals/ICAD98"
    #res_case = 'ICAD98'
    #config.predictions.peak_flow_idx = 12
    #config.data_file = "../data/stenosis_50/ICAD98_05mm3_20ms_LR_sv51_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_50/ICAD98_05mm3_20ms.h5"
    network_path = "../models/251125_WIRE_MOMENTUM_ALL_SV_SAPINN/251125_WIRE_MOMENTUM_ALL_SV_NewMask_SA_ICAD146_sv17_20251127-0705/checkpoints/251125_WIRE_MOMENTUM_ALL_SV_SAPINN_it040000.pth"
    results_directory = "residuals/ICAD146"
    res_case = 'ICAD146'
    config.predictions.peak_flow_idx = 8
    config.data_file = "../data/stenosis_70/ICAD146_05mm3_20ms_LR_sv17_tSNR10_newMask.h5"
    config.data_file_ref = "../data/stenosis_70/ICAD146_05mm3_20ms.h5"

    os.makedirs(results_directory, exist_ok=True)

    # Load data
    u, v, w, p, px, py, pz, mask, config = load_data(config)

    # Prepare data (gets xyz_data, mask_flat, and standardization_factors)
    uvw_data, xyz_data, mask_flat, boundary_mask_flat, standardization_factors, U_max = \
        prepare_data(config, u, v, w, p, px, py, pz, mask)

    # Init network
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.network.arch == "SIREN":
        model = networks.SIREN(
            in_dim=config.network.in_dim,
            out_dim=config.network.out_dim,
            depth=config.network.depth,
            hidden_features=config.network.hidden_features,
            first_omega_0=config.network.omega_0,
            hidden_omega_0=config.network.omega_0
        ).to(DEVICE)
    elif config.network.arch == "FF_SIREN":
        model = networks.FF_SIREN(
            in_dim=config.network.in_dim,
            out_dim=config.network.out_dim,
            depth=config.network.depth,
            hidden_features=config.network.hidden_features,
            first_omega_0=config.network.omega_0,
            hidden_omega_0=config.network.omega_0,
            fourier_mapping_size=config.network.fourier_mapping_size,
            scale=config.network.fourier_scale
        ).to(DEVICE)
    elif config.network.arch == "FFN":
        model = networks.FFN(
            input_dim=config.network.in_dim,
            output_dim=config.network.out_dim,
            depth=config.network.depth,
            hidden_dim=config.network.hidden_features,
            fourier_mapping_size=config.network.fourier_mapping_size,
            scale=config.network.fourier_scale
        ).to(DEVICE)
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
        ).to(DEVICE)
    else:
        raise ValueError("Unknown network.")

    # Load checkpoint
    checkpoint = torch.load(network_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    if config.network.arch == "FFN":
        model.fourier_encoder.B = checkpoint['fourier_B']

    # Build big visualization cloud (fluid only) and plot
    mask_flat = mask_flat.astype(np.uint8)
    #xyz_plot_cloud = make_plot_cloud(xyz_data, mask_flat, max_points=2_000_000)
    config.collocation_points = 1_500_000
    xyz_collocation = sample_collocation_points(config, xyz_data, mask_flat)

    plot_residual_distribution(
        config, model,
        xyz_collocation=xyz_collocation,
        standardization_factors=standardization_factors,
        time_mode="bins", time_bins=6,        # or "phase", phase_idx=8, or "all"
        scale="linear",                          # or "linear"
        cmap="inferno",
        n_points=300_000,
        chunk=1_000,
        out_prefix=os.path.join(results_directory, f"{res_case}_residuals_0.01"),
        vmin=0,   # manually fix colorbar lower bound
        vmax=0.01,   # manually fix colorbar upper bound
    )
