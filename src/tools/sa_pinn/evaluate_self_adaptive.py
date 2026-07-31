# --- Stenosis ROI evaluation: metrics + residuals + c_weights -----------
#
# Evaluates self-adaptive PINN runs:
#   - Velocity relative error          [%]  — whole domain + stenosis
#   - Pressure gradient relative error [%]  — whole domain + stenosis
#   - PDE residual statistics          — whole domain + stenosis
#   - c_weights spatial scatter        — SA runs only
#   - Residual 3D scatter              — all runs
#
# Workflow:
#   1. Run plot_stenosis_roi.py to find stenosis box coords
#   2. Fill in the stenosis_box dicts below with the confirmed coordinates
#   3. Run: python src/tools/sa_pinn/evaluate_self_adaptive.py
#
# Run from 4DFlowINR/:  python src/tools/evaluate_self_adaptive.py

# ---------------------------------------------------------------------------
import os
import sys
from pathlib import Path
SCRIPT_PATH = Path(__file__).resolve()
SRC_DIR = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
import csv
import numpy as np
import torch
import matplotlib as mpl
mpl.use('Agg')
from configs.extensions.sa_pinn.sa_adam_lbfgs import get_config
from utils.data_io import load_data, load_ref_data
from utils.prepare_data import prepare_data, prepare_ref_data, sample_collocation_points
from utils.loss_utils import navier_stokes_loss
import networks
# Import plotting helpers from existing scripts (importable as modules)
from tools.sa_pinn.plotting import plot_c_weights_spatial, plot_residual_distribution

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MODEL_DIR = "models/extensions"
N_RESID   = 200_000   # reduced to fit in 40 GB GPU alongside model + inference
CHUNK     = 5_000     # small chunks — each needs a full backward graph in memory
OUT_DIR   = "residuals/sa_pinn_evaluation22"

# ---------------------------------------------------------------------------
# Case definitions — grouped by (case, dtype)
# ---------------------------------------------------------------------------
# stenosis_box: fill in after running plot_stenosis_roi.py.
# One box per case — the same normalised coords apply to both LR and HRLR
# (fixed template normalization). HV01 has no stenosis — leave as None.
#
# dtype_meta: per-dtype data file + resolution settings (loaded separately
# because LR and HRLR use different input files and ref_spatial_factor).
x_sebjo/SRFlow/models/260505_SAPINN_factorial/SA_PINN_HV01_HRLR2_SA+LBFGS/checkpoints/260505_Vincent_it030000.pth
CASE_RUNS = {
    "HV01": {
        "dtype_meta": {
            "LR": dict(
                data_file     = "data/healthy/HV01_05mm3_20ms_LR_sv17_tSNR10_newMask.h5",
                data_file_ref = "data/healthy/HV01_05mm3_20ms.h5",
                venc=1.7, peak_flow_idx=12,
                ref_spatial_factor=2, ref_temporal_factor=2,
                dx=0.001, dt=0.04,
            ),
            "HRLR": dict(
                data_file     = "data/healthy/HV01_05mm3_20ms_HRLR_sv17_tSNR10.h5",
                data_file_ref = "data/healthy/HV01_05mm3_20ms.h5",
                venc=1.7, peak_flow_idx=12,
                ref_spatial_factor=1, ref_temporal_factor=1,
                dx=0.0005, dt=0.02,
            ),
        },
        "stenosis_box": None,   # healthy — no stenosis region
        "runs": {
            "LR_LBFGS":     ("LR", os.path.join(MODEL_DIR,
            "pinn_adam_lbfgs/stilted-sweep-1_20260728-1619/checkpoints/pinn_adam_lbfgs_it030000.pth")),
            "LR_SA-Adam":   ("LR", os.path.join(MODEL_DIR,
            "sa_pinn/vivid-sweep-1_20260728-1616/checkpoints/sa_pinn_adam_it030000.pth")),
            "LR_SA+LBFGS":  ("LR", os.path.join(MODEL_DIR,
            "sa_pinn_adam_lbfgs/robust-sweep-1_20260728-1617/checkpoints/sa_pinn_adam_lbfgs_it030000.pth")),

            "HRLR10_LBFGS":    ("HRLR", os.path.join(MODEL_DIR,   
            "pinn_adam_lbfgs/dauntless-sweep-2_20260728-2042/checkpoints/pinn_adam_lbfgs_it030000.pth")),
            "HRLR10_SA-Adam":  ("HRLR", os.path.join(MODEL_DIR,
            "sa_pinn/electric-sweep-2_20260728-2057/checkpoints/sa_pinn_adam_it030000.pth")),
            "HRLR10_SA+LBFGS": ("HRLR", os.path.join(MODEL_DIR,
            "sa_pinn_adam_lbfgs/deft-sweep-2_20260728-2116/checkpoints/sa_pinn_adam_lbfgs_it030000.pth")),

            "HRLR2_LBFGS":     ("HRLR", os.path.join(MODEL_DIR,
            "pinn_adam_lbfgs/upbeat-sweep-3_20260729-0104/checkpoints/pinn_adam_lbfgs_it030000.pth")),
            "HRLR2_SA-Adam":   ("HRLR", os.path.join(MODEL_DIR,
            "sa_pinn/soft-sweep-3_20260729-0137/checkpoints/sa_pinn_adam_it030000.pth")),
            "HRLR2_SA+LBFGS":  ("HRLR", os.path.join(MODEL_DIR,
            "sa_pinn_adam_lbfgs/deft-sweep-3_20260729-0210/checkpoints/sa_pinn_adam_lbfgs_it030000.pth")),

        },
    },
    #"ICAD48": {
    #    "dtype_meta": {
    #        "LR": dict(
    #            data_file     = "data/stenosis_50/ICAD48_05mm3_20ms_LR_sv13_tSNR10_newMask.h5",
    #            data_file_ref = "data/stenosis_50/ICAD48_05mm3_20ms.h5",
    #            venc=1.3, peak_flow_idx=14,
    #            ref_spatial_factor=2, ref_temporal_factor=2,
    #            dx=0.001, dt=0.04,
    #        ),
    #        "HRLR": dict(
    #            data_file     = "data/stenosis_50/ICAD48_05mm3_20ms_HRLR_sv13_tSNR10.h5",
    #            data_file_ref = "data/stenosis_50/ICAD48_05mm3_20ms.h5",
    #            venc=1.3, peak_flow_idx=14,
    #            ref_spatial_factor=1, ref_temporal_factor=1,
    #            dx=0.0005, dt=0.02,
    #        ),
    #    },
    #    # Same box for LR and HRLR — fill in after running plot_stenosis_mask_check_vincent.py
    #    "stenosis_box": dict(X_LO=-1.65, X_HI=-1.3, Y_LO=-1.55, Y_HI=-1.4, Z_LO=-1.4, Z_HI=-1.3),
    #    "runs": {
    #        "LR_LBFGS":        (os.path.join(MODEL_DIR, "SA_PINN_ICAD48_LR_LBFGS"),        "LR"),
    #        "LR_SA-Adam":      (os.path.join(MODEL_DIR, "SA_PINN_ICAD48_LR_SA-Adam"),       "LR"),
    #        "LR_SA+LBFGS":     (os.path.join(MODEL_DIR, "SA_PINN_ICAD48_LR_SA+LBFGS"),     "LR"),
    #        "HRLR10_LBFGS":    (os.path.join(MODEL_DIR, "SA_PINN_ICAD48_HRLR10_LBFGS"),    "HRLR"),
    #        "HRLR10_SA-Adam":  (os.path.join(MODEL_DIR, "SA_PINN_ICAD48_HRLR10_SA-Adam"),  "HRLR"),
    #        "HRLR10_SA+LBFGS": (os.path.join(MODEL_DIR, "SA_PINN_ICAD48_HRLR10_SA+LBFGS"), "HRLR"),
    #        "HRLR2_LBFGS":     (os.path.join(MODEL_DIR, "SA_PINN_ICAD48_HRLR2_LBFGS"),    "HRLR"),
    #        "HRLR2_SA-Adam":   (os.path.join(MODEL_DIR, "SA_PINN_ICAD48_HRLR2_SA-Adam"),   "HRLR"),
    #        "HRLR2_SA+LBFGS":  (os.path.join(MODEL_DIR, "SA_PINN_ICAD48_HRLR2_SA+LBFGS"), "HRLR"),
    #    },
    #},
    #"ICAD21": {
    #    "dtype_meta": {
    #        "LR": dict(
    #            data_file     = "data/stenosis_70/ICAD21_05mm3_20ms_LR_sv26_tSNR10_newMask.h5",
    #            data_file_ref = "data/stenosis_70/ICAD21_05mm3_20ms.h5",
    #            venc=2.6, peak_flow_idx=12,
    #            ref_spatial_factor=2, ref_temporal_factor=2,
    #            dx=0.001, dt=0.04,
    #        ),
    #        "HRLR": dict(
    #            data_file     = "data/stenosis_70/ICAD21_05mm3_20ms_HRLR_sv26_tSNR10.h5",
    #            data_file_ref = "data/stenosis_70/ICAD21_05mm3_20ms.h5",
    #            venc=2.6, peak_flow_idx=12,
    #            ref_spatial_factor=1, ref_temporal_factor=1,
    #            dx=0.0005, dt=0.02,
    #        ),
    #    },
    #    # Same box for LR and HRLR — fill in after running plot_stenosis_mask_check_vincent.py
    #    "stenosis_box": dict(X_LO=-1.7, X_HI=-1.4, Y_LO=-1.55, Y_HI=-1.35, Z_LO=-1.4, Z_HI=-1.3),
    #    "runs": {
    #        "LR_LBFGS":        (os.path.join(MODEL_DIR, "SA_PINN_ICAD21_LR_LBFGS"),        "LR"),
    #        "LR_SA-Adam":      (os.path.join(MODEL_DIR, "SA_PINN_ICAD21_LR_SA-Adam"),       "LR"),
    #        "LR_SA+LBFGS":     (os.path.join(MODEL_DIR, "SA_PINN_ICAD21_LR_SA+LBFGS"),     "LR"),
    #        "HRLR10_LBFGS":    (os.path.join(MODEL_DIR, "SA_PINN_ICAD21_HRLR10_LBFGS"),    "HRLR"),
    #        "HRLR10_SA-Adam":  (os.path.join(MODEL_DIR, "SA_PINN_ICAD21_HRLR10_SA-Adam"),  "HRLR"),
    #        "HRLR10_SA+LBFGS": (os.path.join(MODEL_DIR, "SA_PINN_ICAD21_HRLR10_SA+LBFGS"), "HRLR"),
    #        "HRLR2_LBFGS":     (os.path.join(MODEL_DIR, "SA_PINN_ICAD21_HRLR2_LBFGS"),    "HRLR"),
    #        "HRLR2_SA-Adam":   (os.path.join(MODEL_DIR, "SA_PINN_ICAD21_HRLR2_SA-Adam"),   "HRLR"),
    #        "HRLR2_SA+LBFGS":  (os.path.join(MODEL_DIR, "SA_PINN_ICAD21_HRLR2_SA+LBFGS"), "HRLR"),
    #    },
    #},
}


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
        xb = torch.from_numpy(xyz_np[start:start + batch]).float().to(device)
        with torch.no_grad():
            chunks.append(model(xb).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def collect_residuals_with_coords(config, model, xyz_coll, standardization_factors,
                                   device, n=N_RESID, chunk=CHUNK):
    model.eval()
    N = xyz_coll.shape[0]
    sel = np.sort(np.random.choice(N, size=min(n, N), replace=False))
    xyz_sel_np = xyz_coll[sel].astype(np.float32)

    resid_chunks = []
    for start in range(0, len(sel), chunk):
        end = min(start + chunk, len(sel))
        xb = torch.from_numpy(xyz_sel_np[start:end]).to(device).requires_grad_(True)
        with torch.enable_grad():
            uvw = model(xb)
            per_pt, _, _ = navier_stokes_loss(
                uvw, xb, standardization_factors, config, return_per_point=True
            )
        resid_chunks.append(per_pt.detach().cpu().numpy())
        del xb, uvw, per_pt
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    resid  = np.concatenate(resid_chunks).astype(np.float32)
    xyz_out = xyz_sel_np
    finite  = np.isfinite(resid)
    return resid[finite], xyz_out[finite]


def make_stenosis_filter(box):
    def stenosis_filter(xyz):
        return (
            (xyz[:, 1] >= box["X_LO"]) & (xyz[:, 1] <= box["X_HI"]) &
            (xyz[:, 2] >= box["Y_LO"]) & (xyz[:, 2] <= box["Y_HI"]) &
            (xyz[:, 3] >= box["Z_LO"]) & (xyz[:, 3] <= box["Z_HI"])
        )
    return stenosis_filter


def resid_stats(r):
    return dict(median=np.median(r), p95=np.percentile(r, 95),
                p99=np.percentile(r, 99), mean=np.mean(r))


def grad_rel_error(px_p, py_p, pz_p, px_r, py_r, pz_r):
    error_mag = np.sqrt((px_p - px_r)**2 + (py_p - py_r)**2 + (pz_p - pz_r)**2)
    ref_mag   = np.sqrt(px_r**2 + py_r**2 + pz_r**2)
    return 100 * (error_mag / (ref_mag + 1e-6)).mean()


def vector_rel_error(u_p, v_p, w_p, u_r, v_r, w_r):
    eps = 1e-5
    diff_speed   = np.sqrt((u_p - u_r)**2 + (v_p - v_r)**2 + (w_p - w_r)**2)
    actual_speed = np.sqrt(u_r**2 + v_r**2 + w_r**2)
    rel = np.tanh(diff_speed / (actual_speed + eps))
    rel = np.where(actual_speed != 0, rel, diff_speed)
    rel = np.round(rel * 1e4) / 1e4
    return rel.sum() / (len(rel) + 1) * 100


def print_table(results, case_name, has_stenosis):
    cols = f"{'Run':<22} {'Region':<12} {'Rel [%]':>8} {'GradRel[%]':>11} {'ResidMed':>10} {'Resid p95':>10} {'Resid p99':>10}"
    print(f"\n=== {case_name} ===")
    print(cols)
    print("-" * len(cols))
    for run_name, regions in results.items():
        for region, m in regions.items():
            print(f"{run_name:<22} {region:<12} "
                  f"{m['rel']:>8.2f} {m['grad_rel']:>11.2f} "
                  f"{m['resid_median']:>10.4e} {m['resid_p95']:>10.4e} {m['resid_p99']:>10.4e}")


def save_csv(results, case_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"results_{case_name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "region", "rel_pct", "gradrel_pct",
                    "resid_median", "resid_p95", "resid_p99", "resid_mean"])
        for run_name, regions in results.items():
            for region, m in regions.items():
                w.writerow([run_name, region,
                             f"{m['rel']:.4f}", f"{m['grad_rel']:.4f}",
                             f"{m['resid_median']:.6e}", f"{m['resid_p95']:.6e}",
                             f"{m['resid_p99']:.6e}", f"{m['resid_mean']:.6e}"])
    print(f"\nSaved: {path}")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

for case_name, case_cfg in CASE_RUNS.items():
    print(f"\n{'#'*65}")
    print(f"#  CASE: {case_name}")
    print(f"{'#'*65}")

    dtype_metas  = case_cfg["dtype_meta"]
    box          = case_cfg["stenosis_box"]   # None for HV01, single dict for stenosis cases
    runs         = case_cfg["runs"]           # {run_name: (run_dir, dtype)}
    has_stenosis = box is not None
    stenosis_filter = make_stenosis_filter(box) if has_stenosis else None
    results      = {}

    # ------------------------------------------------------------------
    # Load data once per (case, dtype) — reuse across runs of same dtype
    # ------------------------------------------------------------------
    loaded = {}   # dtype → (config, u_hr, v_hr, w_hr, px_hr, py_hr, pz_hr,
                  #           xyz_ref, mask_flat_ref, standardization_factors,
                  #           xyz_coll, t_vals, mfr_4d)

    for dtype, cm in dtype_metas.items():
        print(f"\nLoading {dtype} data ({case_name})...")
        config = get_config()
        config.data_file              = cm["data_file"]
        config.data_file_ref          = cm["data_file_ref"]
        config.constants.venc         = cm["venc"]
        config.predictions.peak_flow_idx = cm["peak_flow_idx"]
        config.ref_spatial_factor     = cm["ref_spatial_factor"]
        config.ref_temporal_factor    = cm["ref_temporal_factor"]
        config.resolution.from_file   = False
        config.resolution.dx = config.resolution.dy = config.resolution.dz = cm["dx"]
        config.resolution.dt          = cm["dt"]

        u_lr, v_lr, w_lr, p_lr, px_lr, py_lr, pz_lr, mask, config = load_data(config)
        _, xyz_data, mask_flat, _, standardization_factors, U_max = \
            prepare_data(config, u_lr, v_lr, w_lr, p_lr, px_lr, py_lr, pz_lr, mask)

        print(f"Loading HR reference data ({case_name}, {dtype})...")
        u_hr, v_hr, w_hr, p_hr, px_hr, py_hr, pz_hr, mask_hr = load_ref_data(config)
        _, xyz_ref, mask_flat_ref, _ = prepare_ref_data(
            config, u_lr, u_hr, v_hr, w_hr, p_hr, px_hr, py_hr, pz_hr, mask_hr, U_max
        )

        T_ref, X_ref, Y_ref, Z_ref = u_hr.shape
        t_vals = np.unique(xyz_ref[:, 0])
        mfr_4d = mask_flat_ref.reshape(T_ref, X_ref, Y_ref, Z_ref)

        mask_flat_uint8 = mask_flat.astype(np.uint8)
        config.collocation_points = 1_500_000
        xyz_coll = sample_collocation_points(config, xyz_data, mask_flat_uint8)
        print(f"  Collocation cloud: {len(xyz_coll):,} points")

        if has_stenosis:
            n_box = stenosis_filter(xyz_coll).sum()
            print(f"  Stenosis coll pts [{dtype}]: {n_box:,} ({100*n_box/len(xyz_coll):.1f}%)")

        loaded[dtype] = dict(
            config=config,
            u_hr=u_hr, v_hr=v_hr, w_hr=w_hr,
            px_hr=px_hr, py_hr=py_hr, pz_hr=pz_hr,
            xyz_ref=xyz_ref, mask_flat_ref=mask_flat_ref,
            standardization_factors=standardization_factors,
            xyz_coll=xyz_coll,
            t_vals=t_vals, mfr_4d=mfr_4d,
            T_ref=T_ref, X_ref=X_ref, Y_ref=Y_ref, Z_ref=Z_ref,
        )

    # ------------------------------------------------------------------
    # Per-run evaluation
    # ------------------------------------------------------------------
    for run_name, (dtype, checkpoint_filename) in runs.items():
        print(f"\n{'='*55}")
        print(f"  {case_name} / {checkpoint_filename}  [{dtype}]")
        print('='*55)

        ckpt_path = checkpoint_filename
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] checkpoint not found: {ckpt_path}")
            continue

        ld = loaded[dtype]
        cfg              = ld["config"]
        u_hr, v_hr, w_hr = ld["u_hr"], ld["v_hr"], ld["w_hr"]
        px_hr, py_hr, pz_hr = ld["px_hr"], ld["py_hr"], ld["pz_hr"]
        xyz_ref          = ld["xyz_ref"]
        mask_flat_ref    = ld["mask_flat_ref"]
        sf_factors       = ld["standardization_factors"]
        xyz_coll         = ld["xyz_coll"]
        t_vals           = ld["t_vals"]
        mfr_4d           = ld["mfr_4d"]
        T_ref            = ld["T_ref"]

        U   = cfg.constants.U
        rho = cfg.constants.rho
        L   = cfg.constants.L
        _, _, _, std_x, _, std_y, _, std_z = sf_factors

        run_has_stenosis = has_stenosis

        ckpt  = torch.load(ckpt_path, map_location=DEVICE)
        model = build_model(cfg, DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])

        run_out = os.path.join(OUT_DIR, case_name, run_name)
        os.makedirs(run_out, exist_ok=True)

        # --- Inference on all HR fluid points ---
        print("  Running inference on HR fluid points...")
        fluid_idx_ref  = mask_flat_ref == 1
        xyz_fluid_ref  = xyz_ref[fluid_idx_ref]
        uvw_pred_fluid = run_inference(model, xyz_fluid_ref, DEVICE)

        n_out    = uvw_pred_fluid.shape[1]
        uvw_full = np.zeros((len(mask_flat_ref), n_out), dtype=np.float32)
        uvw_full[fluid_idx_ref] = uvw_pred_fluid

        # --- Average metrics over all T timesteps ---
        vel_rels, grad_rels   = [], []
        svel_rels, sgrad_rels = [], []

        for ti in range(T_ref):
            t_target_ti   = t_vals[ti] if ti < len(t_vals) else t_vals[-1]
            peak_fluid_ti = (xyz_ref[:, 0] == t_target_ti) & (mask_flat_ref == 1)
            fluid_3d_ti   = mfr_4d[ti].ravel().astype(bool)

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

            vel_rels.append(vector_rel_error(uvw_p[:,0], uvw_p[:,1], uvw_p[:,2], u_r, v_r, w_r))
            grad_rels.append(grad_rel_error(grad_p[:,0], grad_p[:,1], grad_p[:,2], px_r, py_r, pz_r))

            if run_has_stenosis:
                xyz_ti    = xyz_ref[peak_fluid_ti]
                in_box_ti = stenosis_filter(xyz_ti)
                if in_box_ti.sum() > 0:
                    svel_rels.append(vector_rel_error(
                        uvw_p[in_box_ti,0], uvw_p[in_box_ti,1], uvw_p[in_box_ti,2],
                        u_r[in_box_ti], v_r[in_box_ti], w_r[in_box_ti]))
                    sgrad_rels.append(grad_rel_error(
                        grad_p[in_box_ti,0], grad_p[in_box_ti,1], grad_p[in_box_ti,2],
                        px_r[in_box_ti], py_r[in_box_ti], pz_r[in_box_ti]))

        sp_whole = np.mean(vel_rels)
        gr_whole = np.mean(grad_rels)
        sp_sten  = np.mean(svel_rels)  if svel_rels  else np.nan
        gr_sten  = np.mean(sgrad_rels) if sgrad_rels else np.nan

        print(f"  Whole:    Rel={sp_whole:.2f}%  GradRel={gr_whole:.2f}%")
        if run_has_stenosis:
            print(f"  Stenosis: Rel={sp_sten:.2f}%  GradRel={gr_sten:.2f}%")

        # --- PDE residuals ---
        print("  Evaluating PDE residuals...")
        resid, xyz_resid = collect_residuals_with_coords(
            cfg, model, xyz_coll, sf_factors, DEVICE
        )

        in_box_coll = stenosis_filter(xyz_resid) if run_has_stenosis else np.zeros(len(resid), bool)
        if run_has_stenosis:
            print(f"  Residual pts in stenosis: {in_box_coll.sum():,} / {len(resid):,}")

        rs_whole = resid_stats(resid)
        rs_sten  = resid_stats(resid[in_box_coll]) if run_has_stenosis and in_box_coll.sum() > 10 \
                   else dict(median=np.nan, p95=np.nan, p99=np.nan, mean=np.nan)

        results[run_name] = {
            "whole": dict(rel=sp_whole, grad_rel=gr_whole,
                          resid_median=rs_whole['median'],
                          resid_p95=rs_whole['p95'], resid_p99=rs_whole['p99'],
                          resid_mean=rs_whole['mean']),
        }
        if run_has_stenosis:
            results[run_name]["stenosis"] = dict(
                rel=sp_sten, grad_rel=gr_sten,
                resid_median=rs_sten['median'],
                resid_p95=rs_sten['p95'], resid_p99=rs_sten['p99'],
                resid_mean=rs_sten['mean'],
            )

        # --- Residual 3D scatter (all runs) ---
        print("  Plotting residuals...")
        plot_residual_distribution(
            cfg, model,
            xyz_collocation=xyz_coll,
            standardization_factors=sf_factors,
            time_mode="all",
            scale="log",
            cmap="YlOrRd",
            n_points=200_000,
            chunk=CHUNK,
            out_prefix=os.path.join(run_out, "residuals"),
            vmin=5e-4,
            vmax=1e-2,
        )

        # --- c_weights spatial scatter (SA runs only) ---
        is_sa = "SA" in run_name
        if is_sa:
            cw = ckpt.get("c_weights", None)
            if cw is not None:
                cw = np.asarray(cw, dtype=np.float32)
                print(f"  c_weights: mean={cw.mean():.4f}  std={cw.std():.4f}  "
                      f"max={cw.max():.4f}")
                plot_c_weights_spatial(
                    cw, xyz_coll,
                    time_mode="all",
                    n_points=300_000,
                    cmap="RdYlBu_r",
                    out_prefix=os.path.join(run_out, "c_weights_spatial"),
                    vmin=0.95,
                    vmax=1.25,
                )
            else:
                print("  [INFO] c_weights not found in checkpoint")

        del model, ckpt
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Summary table + CSV
    # ------------------------------------------------------------------
    print_table(results, case_name, has_stenosis)
    save_csv(results, case_name, OUT_DIR)
