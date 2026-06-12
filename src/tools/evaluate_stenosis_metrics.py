# --- Stenosis-region metrics: R1 vs R2 ----------------------------------------
#
# Computes velocity error, pressure gradient error, and PDE residual metrics
# for all runs in two regions: whole fluid domain and stenosis bounding box.
# Both whole-domain and stenosis metrics are averaged over all T timesteps,
# matching the W&B evaluation path in trainer.py.
#
# Metrics reported:
#   - Relative speed error              [%]
#   - Pressure gradient relative error  [%]
#   - Median PDE residual
#   - 95th / 99th percentile PDE residual
#
# Run from src/:  python tools/evaluate_stenosis_metrics.py
# ---------------------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from configs.tunings_260409_SAPINN_LBFGS.Config_260409_SAPINN_sweepN_factorial import get_config
from utils.prepare_data import (prepare_data, load_data, load_ref_data,
                                 prepare_ref_data, sample_collocation_points)
from utils.loss_utils import navier_stokes_loss
import networks


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MODEL_DIR = "../models/260409_SAPINN_sweepN_factorial"
RUNS = {
    "R1": os.path.join(MODEL_DIR, "SA_PINN_ICAD48_sv13_sweepN_R1_20260414-1634_dx6dn672"),
    "R2": os.path.join(MODEL_DIR, "SA_PINN_ICAD48_sv13_sweepN_R2_20260414-1634_mngdbct0"),
    "R3": os.path.join(MODEL_DIR, "SA_PINN_ICAD48_sv13_sweepN_R3_20260417-1506_7innh3bu"),
    "R4": os.path.join(MODEL_DIR, "SA_PINN_ICAD48_sv13_sweepN_R4_20260417-1506_k71hill6"),
}
CKPT_ITER  = 40_000
PEAK_T_IDX = 14

# Stenosis bounding box in normalized coordinates (verified visually)
X_LO, X_HI = -1.65, -1.3
Y_LO, Y_HI = -1.55, -1.4
Z_LO, Z_HI = -1.4, -1.3

# Residual evaluation
N_RESID = 200_000
CHUNK   = 10_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_model(config, device):
    return networks.WIRE(
        in_dim=config.network.in_dim,
        out_dim=config.network.out_dim,
        depth=config.network.depth,
        hidden_features=config.network.hidden_features,
        first_omega_0=config.network.omega_0,
        hidden_omega_0=config.network.omega_0,
        scale=config.network.sigma_0,
        complex=config.network.complex,
    ).to(device)


def run_inference(model, xyz_np, device, batch=20_000):
    model.eval()
    chunks = []
    for start in range(0, xyz_np.shape[0], batch):
        xb = torch.from_numpy(xyz_np[start:start+batch]).float().to(device)
        with torch.no_grad():
            chunks.append(model(xb).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def collect_residuals_with_coords(config, model, xyz_coll, standardization_factors,
                                   device, n=N_RESID, chunk=CHUNK):
    """Returns (resid, xyz) for n sampled collocation points."""
    model.eval()
    N = xyz_coll.shape[0]
    sel = np.sort(np.random.choice(N, size=min(n, N), replace=False))
    xyz_sel = torch.from_numpy(xyz_coll[sel].astype(np.float32)).to(device)

    resid_chunks = []
    for start in range(0, len(sel), chunk):
        end = min(start + chunk, len(sel))
        xb = xyz_sel[start:end].clone().detach().requires_grad_(True)
        with torch.enable_grad():
            uvw = model(xb)
            per_pt, _, _ = navier_stokes_loss(
                uvw, xb, standardization_factors, config, return_per_point=True
            )
        resid_chunks.append(per_pt.detach().cpu().numpy())

    resid = np.concatenate(resid_chunks).astype(np.float32)
    xyz_out = xyz_coll[sel].astype(np.float32)
    finite = np.isfinite(resid)
    return resid[finite], xyz_out[finite]


def stenosis_filter(xyz):
    """Boolean mask: True if point is inside the stenosis bounding box."""
    return (
        (xyz[:, 1] >= X_LO) & (xyz[:, 1] <= X_HI) &
        (xyz[:, 2] >= Y_LO) & (xyz[:, 2] <= Y_HI) &
        (xyz[:, 3] >= Z_LO) & (xyz[:, 3] <= Z_HI)
    )


def resid_stats(r):
    return dict(median=np.median(r), p95=np.percentile(r, 95),
                p99=np.percentile(r, 99), mean=np.mean(r))


def grad_rel_error(px_p, py_p, pz_p, px_r, py_r, pz_r):
    """Matches calculate_gradient_relative_error() from utils/evaluation_utils.py."""
    error_mag = np.sqrt((px_p - px_r)**2 + (py_p - py_r)**2 + (pz_p - pz_r)**2)
    ref_mag   = np.sqrt(px_r**2 + py_r**2 + pz_r**2)
    rel = error_mag / (ref_mag + 1e-6)
    return 100 * rel.mean()


def vector_rel_error(u_p, v_p, w_p, u_r, v_r, w_r):
    """Matches calculate_tanh_relative_error() from utils/evaluation_utils.py exactly."""
    eps = 1e-5
    diff_speed   = np.sqrt((u_p - u_r)**2 + (v_p - v_r)**2 + (w_p - w_r)**2)
    actual_speed = np.sqrt(u_r**2 + v_r**2 + w_r**2)
    rel = np.tanh(diff_speed / (actual_speed + eps))   # tanh, not clip
    rel = np.where(actual_speed != 0, rel, diff_speed)
    rel = np.round(rel * 1e4) / 1e4
    return rel.sum() / (len(rel) + 1) * 100


def print_table(results):
    header = (f"{'Run':<6} {'Region':<12} {'Rel [%]':>8} "
              f"{'GradRel[%]':>11} "
              f"{'ResidMed':>10} {'Resid p95':>10} {'Resid p99':>10}")
    print("\n" + header)
    print("-" * len(header))
    for run_name, regions in results.items():
        for region, m in regions.items():
            print(f"{run_name:<6} {region:<12} "
                  f"{m['rel']:>8.2f} "
                  f"{m['grad_rel']:>11.2f} "
                  f"{m['resid_median']:>10.4e} {m['resid_p95']:>10.4e} {m['resid_p99']:>10.4e}")


# ---------------------------------------------------------------------------
# Load shared data
# ---------------------------------------------------------------------------
print("Loading config and LR data...")
config = get_config()
config.data_file     = "../data/stenosis_50/ICAD48_05mm3_20ms_LR_sv13_tSNR10_newMask.h5"
config.data_file_ref = "../data/stenosis_50/ICAD48_05mm3_20ms.h5"
config.constants.venc = 1.3
config.predictions.peak_flow_idx = PEAK_T_IDX

u_lr, v_lr, w_lr, p_lr, px_lr, py_lr, pz_lr, mask, config = load_data(config)
_, xyz_data, mask_flat, _, standardization_factors, U_max = \
    prepare_data(config, u_lr, v_lr, w_lr, p_lr, px_lr, py_lr, pz_lr, mask)

print("Loading HR reference data...")
u_hr, v_hr, w_hr, p_hr, px_hr, py_hr, pz_hr, mask_hr = load_ref_data(config)
uvw_ref, xyz_ref, mask_flat_ref, _ = prepare_ref_data(
    config, u_lr, u_hr, v_hr, w_hr, p_hr, px_hr, py_hr, pz_hr, mask_hr, U_max
)

T_ref, X_ref, Y_ref, Z_ref = u_hr.shape
U = config.constants.U   # 2.0 m/s characteristic scale

# Standardization factors for pressure gradient denormalization
# layout: [mean_t, std_t, mean_x, std_x, mean_y, std_y, mean_z, std_z]
_, _, _, std_x, _, std_y, _, std_z = standardization_factors

# HR reference u/v/w and pressure gradients at peak timestep (fluid only)
t = PEAK_T_IDX
mfr_4d   = mask_flat_ref.reshape(T_ref, X_ref, Y_ref, Z_ref)
fluid_3d = mfr_4d[t].ravel().astype(bool)
u_ref_peak  = u_hr[t].ravel()[fluid_3d]
v_ref_peak  = v_hr[t].ravel()[fluid_3d]
w_ref_peak  = w_hr[t].ravel()[fluid_3d]
px_ref_peak = px_hr[t].ravel()[fluid_3d]   # Pa/m (loaded × 1000 by load_ref_data)
py_ref_peak = py_hr[t].ravel()[fluid_3d]
pz_ref_peak = pz_hr[t].ravel()[fluid_3d]

# xyz_ref coords for fluid points at peak timestep
t_vals   = np.unique(xyz_ref[:, 0])
t_target = t_vals[t] if t < len(t_vals) else t_vals[-1]
t_mask   = xyz_ref[:, 0] == t_target
fluid_flat = mask_flat_ref == 1
peak_fluid = t_mask & fluid_flat
xyz_peak   = xyz_ref[peak_fluid]   # (N_fluid, 4)

# Stenosis mask on the HR grid fluid points
in_box_hr = stenosis_filter(xyz_peak)
print(f"\nStenosis box: {in_box_hr.sum():,} HR fluid points "
      f"({100*in_box_hr.mean():.1f}% of fluid at t={t})")

# Collocation cloud for residual evaluation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mask_flat_uint8 = mask_flat.astype(np.uint8)
config.collocation_points = 1_500_000
xyz_coll = sample_collocation_points(config, xyz_data, mask_flat_uint8)

# ---------------------------------------------------------------------------
# Per-run evaluation
# ---------------------------------------------------------------------------
results = {}

for run_name, run_dir in RUNS.items():
    print(f"\n{'='*55}")
    print(f"  {run_name}: {run_dir}")
    print('='*55)

    ckpt_name = f"260409_SAPINN_sweepN_it{CKPT_ITER:06d}.pth"
    ckpt_path = os.path.join(run_dir, "checkpoints", ckpt_name)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model = build_model(config, DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])

    # --- Velocity error on HR grid ---
    print("  Running inference on HR fluid points...")
    fluid_idx_ref = mask_flat_ref == 1
    xyz_fluid_ref = xyz_ref[fluid_idx_ref]
    uvw_pred_fluid = run_inference(model, xyz_fluid_ref, DEVICE)   # normalized

    # Reshape predictions into flat (T*X*Y*Z, n_out) array
    n_out = uvw_pred_fluid.shape[1]
    uvw_full = np.zeros((len(mask_flat_ref), n_out), dtype=np.float32)
    uvw_full[fluid_idx_ref] = uvw_pred_fluid

    rho = config.constants.rho
    L   = config.constants.L

    # --- Average over ALL timesteps for both whole domain and stenosis box ---
    vel_rels, grad_rels = [], []
    svel_rels, sgrad_rels = [], []

    for ti in range(T_ref):
        t_target_ti  = t_vals[ti] if ti < len(t_vals) else t_vals[-1]
        peak_fluid_ti = (xyz_ref[:, 0] == t_target_ti) & (mask_flat_ref == 1)
        fluid_3d_ti  = mfr_4d[ti].ravel().astype(bool)

        u_r  = u_hr[ti].ravel()[fluid_3d_ti]
        v_r  = v_hr[ti].ravel()[fluid_3d_ti]
        w_r  = w_hr[ti].ravel()[fluid_3d_ti]
        px_r = px_hr[ti].ravel()[fluid_3d_ti]
        py_r = py_hr[ti].ravel()[fluid_3d_ti]
        pz_r = pz_hr[ti].ravel()[fluid_3d_ti]

        uvw_p  = uvw_full[peak_fluid_ti, :3] * U
        grad_p = uvw_full[peak_fluid_ti, 3:6].copy()
        grad_p[:, 0] *= rho * U**2 / L / std_x
        grad_p[:, 1] *= rho * U**2 / L / std_y
        grad_p[:, 2] *= rho * U**2 / L / std_z

        # Stenosis mask for this timestep
        xyz_ti  = xyz_ref[peak_fluid_ti]   # (N_fluid_ti, 4)
        in_box_ti = stenosis_filter(xyz_ti)

        vel_rels.append(vector_rel_error(uvw_p[:,0], uvw_p[:,1], uvw_p[:,2], u_r, v_r, w_r))
        grad_rels.append(grad_rel_error(grad_p[:,0], grad_p[:,1], grad_p[:,2], px_r, py_r, pz_r))

        if in_box_ti.sum() > 0:
            svel_rels.append(vector_rel_error(uvw_p[in_box_ti,0], uvw_p[in_box_ti,1],
                                              uvw_p[in_box_ti,2], u_r[in_box_ti],
                                              v_r[in_box_ti], w_r[in_box_ti]))
            sgrad_rels.append(grad_rel_error(grad_p[in_box_ti,0], grad_p[in_box_ti,1],
                                              grad_p[in_box_ti,2], px_r[in_box_ti],
                                              py_r[in_box_ti], pz_r[in_box_ti]))

    sp_whole = np.mean(vel_rels)
    gr_whole = np.mean(grad_rels)
    sp_sten  = np.mean(svel_rels)  if svel_rels  else np.nan
    gr_sten  = np.mean(sgrad_rels) if sgrad_rels else np.nan

    # --- PDE residuals on collocation points ---
    print("  Evaluating PDE residuals...")
    resid, xyz_resid = collect_residuals_with_coords(
        config, model, xyz_coll, standardization_factors, DEVICE
    )
    in_box_coll = stenosis_filter(xyz_resid)
    print(f"  Residual points in stenosis box: {in_box_coll.sum():,} / {len(resid):,}")

    rs_whole = resid_stats(resid)
    rs_sten  = resid_stats(resid[in_box_coll]) if in_box_coll.sum() > 10 \
               else dict(median=np.nan, p95=np.nan, p99=np.nan, mean=np.nan)

    results[run_name] = {
        "whole":    dict(rel=sp_whole, grad_rel=gr_whole,
                         resid_median=rs_whole['median'],
                         resid_p95=rs_whole['p95'], resid_p99=rs_whole['p99']),
        "stenosis": dict(rel=sp_sten,  grad_rel=gr_sten,
                         resid_median=rs_sten['median'],
                         resid_p95=rs_sten['p95'], resid_p99=rs_sten['p99']),
    }

    # c_weights stats for SA runs (R2, R3)
    if run_name in ("R2", "R3"):
        cw = ckpt.get("c_weights", None)
        if cw is not None:
            cw = np.asarray(cw, dtype=np.float32)
            cw_sel = cw[np.sort(np.random.choice(len(cw), size=min(N_RESID, len(cw)),
                                                   replace=False))]
            in_box_cw = stenosis_filter(xyz_resid)
            print(f"\n  c_weights ({run_name}):")
            print(f"    Whole domain : mean={cw.mean():.4f}  std={cw.std():.4f}  "
                  f"max={cw.max():.4f}")
            if in_box_cw.sum() > 0:
                cw_box = cw_sel[in_box_cw]
                print(f"    Stenosis box : mean={cw_box.mean():.4f}  std={cw_box.std():.4f}  "
                      f"max={cw_box.max():.4f}  N={in_box_cw.sum()}")

# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------
print_table(results)

# Delta summary — three comparisons that tell the story
PAIRS = [
    ("R4", "R1", "LBFGS earlier vs later (no SA)"),
    ("R3", "R2", "SA-Adam vs SA+LBFGS"),
    ("R4", "R3", "long LBFGS vs SA-Adam (head-to-head)"),
]
print("\n--- Delta (B - A, negative = B better) ---")
for run_b, run_a, label in PAIRS:
    if run_a not in results or run_b not in results:
        continue
    print(f"\n  {run_b} - {run_a}  [{label}]")
    for region in ["whole", "stenosis"]:
        a = results[run_a][region]
        b = results[run_b][region]
        drel      = b['rel']          - a['rel']
        dgrad_rel = b['grad_rel']     - a['grad_rel']
        dmed      = b['resid_median'] - a['resid_median']
        dp95      = b['resid_p95']    - a['resid_p95']
        print(f"    {region:<12}: ΔRel={drel:+.2f}%  "
              f"ΔGradRel={dgrad_rel:+.2f}%  "
              f"ΔResidMed={dmed:+.4e}  ΔResidP95={dp95:+.4e}")
