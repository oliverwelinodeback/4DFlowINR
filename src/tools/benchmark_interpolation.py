import os
import sys
import time
import h5py
import numpy as np
import pandas as pd

# Script will be located alongside your metrics script
sys.path.append("../")
from utils import evaluation_utils as e_utils

# -----------------------------
# Fast-ish 3D cubic resize
# -----------------------------
def _cubic_resize_3d_to_shape(vol_3d: np.ndarray, out_shape_xyz):
    """
    Cubic 3D resize of a single 3D volume to out_shape_xyz = (X,Y,Z).
    Uses scipy.ndimage.zoom if available; otherwise falls back to skimage.
    Returns float32.
    """
    in_shape = vol_3d.shape
    out_shape = tuple(int(x) for x in out_shape_xyz)

    if in_shape == out_shape:
        return vol_3d.astype(np.float32, copy=False)

    # Try scipy (fast)
    try:
        from scipy.ndimage import zoom

        zoom_factors = (
            out_shape[0] / in_shape[0],
            out_shape[1] / in_shape[1],
            out_shape[2] / in_shape[2],
        )
        out = zoom(vol_3d, zoom=zoom_factors, order=3, mode="nearest", prefilter=True)

        # Fix potential off-by-one due to rounding in zoom
        out_fixed = out
        for ax in range(3):
            if out_fixed.shape[ax] > out_shape[ax]:
                sl = [slice(None)] * 3
                sl[ax] = slice(0, out_shape[ax])
                out_fixed = out_fixed[tuple(sl)]
            elif out_fixed.shape[ax] < out_shape[ax]:
                pad_width = [(0, 0), (0, 0), (0, 0)]
                pad_width[ax] = (0, out_shape[ax] - out_fixed.shape[ax])
                out_fixed = np.pad(out_fixed, pad_width, mode="edge")

        return out_fixed.astype(np.float32, copy=False)

    except Exception:
        # Fallback: skimage (exact shape)
        from skimage.transform import resize

        out = resize(
            vol_3d,
            out_shape,
            order=3,
            mode="edge",
            preserve_range=True,
            anti_aliasing=False,
        )
        return out.astype(np.float32, copy=False)


def _temporal_interp_x2_insert_between(frames: np.ndarray):
    """
    Temporal x2 interpolation by inserting midpoints between consecutive frames.

    Input:  frames shape (T, X, Y, Z)
    Output: shape (2*(T-1)+1, X, Y, Z)

    Example:
      LR: 1,3,5  -> out: 1,2,3,4,5  (no extrapolated last frame)
    """
    T = frames.shape[0]
    if T <= 1:
        return frames

    out_T = 2 * (T - 1) + 1
    out = np.empty((out_T,) + frames.shape[1:], dtype=np.float32)

    out[0::2] = frames
    out[1::2] = 0.5 * (frames[:-1] + frames[1:])

    return out


def _hr_indices_for_lr(T_hr: int, T_lr: int):
    """
    LR frames correspond to HR frames at indices 0,2,4,... (temporal downsample x2).
    Returns hr_idx_use of length min(T_lr, ceil(T_hr/2)).
    """
    hr_idx = np.arange(0, T_hr, 2, dtype=int)
    T_compare = min(T_lr, len(hr_idx))
    return hr_idx[:T_compare], T_compare


def _ensure_mask_3d(mask):
    if mask.ndim == 4:
        return mask[0]
    return mask


def compute_metrics_over_time(
    u_pred, v_pred, w_pred,
    u_ref, v_ref, w_ref,
    mask, boundary_mask, core_mask, nf_mask,
    peak_flow_idx_used: int,
):
    """
    Computes the same set of metrics as your script over time dimension.
    Returns: dict with aggregates + peak + optional bookkeeping.
    """
    T = u_pred.shape[0]

    tanh_rel_err = np.zeros((T, 3), dtype=np.float32)
    rel_err = np.zeros((T, 3), dtype=np.float32)
    abs_err = np.zeros((T, 4), dtype=np.float32)
    rmse = np.zeros((T, 4), dtype=np.float32)
    vnrmse = np.zeros((T, 4), dtype=np.float32)
    d_error = np.zeros((T, 4), dtype=np.float32)

    Ks = np.zeros((T, 3, 3), dtype=np.float32)
    Ms = np.zeros((T, 3, 3), dtype=np.float32)
    Rs = np.zeros((T, 3, 3), dtype=np.float32)

    for t in range(T):
        rel_err[t, 0] = e_utils.calculate_relative_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], mask)
        rel_err[t, 1] = e_utils.calculate_relative_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask)
        rel_err[t, 2] = e_utils.calculate_relative_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], core_mask)

        tanh_rel_err[t, 0] = e_utils.calculate_tanh_relative_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], mask)
        tanh_rel_err[t, 1] = e_utils.calculate_tanh_relative_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask)
        tanh_rel_err[t, 2] = e_utils.calculate_tanh_relative_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], core_mask)

        abs_err[t, 0] = e_utils.calculate_absolute_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], mask)
        abs_err[t, 1] = e_utils.calculate_absolute_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask)
        abs_err[t, 2] = e_utils.calculate_absolute_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], core_mask)
        abs_err[t, 3] = e_utils.calculate_absolute_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], nf_mask)

        rmse[t, 0] = e_utils.calculate_rmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], mask)
        rmse[t, 1] = e_utils.calculate_rmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask)
        rmse[t, 2] = e_utils.calculate_rmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], core_mask)
        rmse[t, 3] = e_utils.calculate_rmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], nf_mask)

        vnrmse[t, 0] = e_utils.calculate_vnrmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], mask)
        vnrmse[t, 1] = e_utils.calculate_vnrmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask)
        vnrmse[t, 2] = e_utils.calculate_vnrmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], core_mask)
        vnrmse[t, 3] = e_utils.calculate_vnrmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], nf_mask)

        d_error[t, 0] = e_utils.calculate_directional_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], mask)
        d_error[t, 1] = e_utils.calculate_directional_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask)
        d_error[t, 2] = e_utils.calculate_directional_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], core_mask)
        d_error[t, 3] = e_utils.calculate_directional_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], nf_mask)

        Ks[t][0][0], Ms[t][0][0], Rs[t][0][0] = e_utils.linreg(u_pred[t], u_ref[t], mask)
        Ks[t][1][0], Ms[t][1][0], Rs[t][1][0] = e_utils.linreg(v_pred[t], v_ref[t], mask)
        Ks[t][2][0], Ms[t][2][0], Rs[t][2][0] = e_utils.linreg(w_pred[t], w_ref[t], mask)

        Ks[t][0][1], Ms[t][0][1], Rs[t][0][1] = e_utils.linreg(u_pred[t], u_ref[t], boundary_mask)
        Ks[t][1][1], Ms[t][1][1], Rs[t][1][1] = e_utils.linreg(v_pred[t], v_ref[t], boundary_mask)
        Ks[t][2][1], Ms[t][2][1], Rs[t][2][1] = e_utils.linreg(w_pred[t], w_ref[t], boundary_mask)

        Ks[t][0][2], Ms[t][0][2], Rs[t][0][2] = e_utils.linreg(u_pred[t], u_ref[t], core_mask)
        Ks[t][1][2], Ms[t][1][2], Rs[t][1][2] = e_utils.linreg(v_pred[t], v_ref[t], core_mask)
        Ks[t][2][2], Ms[t][2][2], Rs[t][2][2] = e_utils.linreg(w_pred[t], w_ref[t], core_mask)

    # Aggregates
    rel_err_tot = np.mean(rel_err, axis=0)
    tanh_rel_err_tot = np.mean(tanh_rel_err, axis=0)
    abs_err_tot = np.mean(abs_err, axis=0)
    rmse_tot = np.mean(rmse, axis=0)
    vnrmse_tot = np.mean(vnrmse, axis=0)
    d_error_tot = np.mean(d_error, axis=0)
    Rs_tot = np.mean(Rs, axis=0)
    Ks_tot = np.mean(Ks, axis=0)

    # Peak (safe)
    peak_flow_idx_used = int(np.clip(peak_flow_idx_used, 0, T - 1))

    out = {
        # totals
        "Relative error [Fluid]": rel_err_tot[0],
        "Relative error [Bound]": rel_err_tot[1],
        "Relative error [Core]": rel_err_tot[2],
        "tanh Relative error [Fluid]": tanh_rel_err_tot[0],
        "tanh Relative error [Bound]": tanh_rel_err_tot[1],
        "tanh Relative error [Core]": tanh_rel_err_tot[2],
        "Absolute error [Fluid]": abs_err_tot[0],
        "Absolute error [Bound]": abs_err_tot[1],
        "Absolute error [Core]": abs_err_tot[2],
        "Absolute error [Non-F]": abs_err_tot[3],
        "R.M.S. error [Fluid]": rmse_tot[0],
        "R.M.S. error [Bound]": rmse_tot[1],
        "R.M.S. error [Core]": rmse_tot[2],
        "R.M.S. error [Non-F]": rmse_tot[3],
        "vNRMSE error [Fluid]": vnrmse_tot[0],
        "vNRMSE error [Bound]": vnrmse_tot[1],
        "vNRMSE error [Core]": vnrmse_tot[2],
        "vNRMSE error [Non-F]": vnrmse_tot[3],
        "Directional error [Fluid]": d_error_tot[0],
        "Directional error [Bound]": d_error_tot[1],
        "Directional error [Core]": d_error_tot[2],
        "Directional error [Non-F]": d_error_tot[3],
        "U R2 [Fluid]": Rs_tot[0][0],
        "U R2 [Bound]": Rs_tot[0][1],
        "U R2 [Core]": Rs_tot[0][2],
        "V R2 [Fluid]": Rs_tot[1][0],
        "V R2 [Bound]": Rs_tot[1][1],
        "V R2 [Core]": Rs_tot[1][2],
        "W R2 [Fluid]": Rs_tot[2][0],
        "W R2 [Bound]": Rs_tot[2][1],
        "W R2 [Core]": Rs_tot[2][2],
        "U K [Fluid]": Ks_tot[0][0],
        "U K [Bound]": Ks_tot[0][1],
        "U K [Core]": Ks_tot[0][2],
        "V K [Fluid]": Ks_tot[1][0],
        "V K [Bound]": Ks_tot[1][1],
        "V K [Core]": Ks_tot[1][2],
        "W K [Fluid]": Ks_tot[2][0],
        "W K [Bound]": Ks_tot[2][1],
        "W K [Core]": Ks_tot[2][2],
        # peak
        "PEAK INDEX USED": peak_flow_idx_used,
        "Relative error [Fluid] Peak": rel_err[peak_flow_idx_used][0],
        "Relative error [Bound] Peak": rel_err[peak_flow_idx_used][1],
        "Relative error [Core] Peak": rel_err[peak_flow_idx_used][2],
        "tanh Relative error [Fluid] Peak": tanh_rel_err[peak_flow_idx_used][0],
        "tanh Relative error [Bound] Peak": tanh_rel_err[peak_flow_idx_used][1],
        "tanh Relative error [Core] Peak": tanh_rel_err[peak_flow_idx_used][2],
        "Absolute error [Fluid] Peak": abs_err[peak_flow_idx_used][0],
        "Absolute error [Bound] Peak": abs_err[peak_flow_idx_used][1],
        "Absolute error [Core] Peak": abs_err[peak_flow_idx_used][2],
        "Absolute error [Non-F] Peak": abs_err[peak_flow_idx_used][3],
        "R.M.S. error [Fluid] Peak": rmse[peak_flow_idx_used][0],
        "R.M.S. error [Bound] Peak": rmse[peak_flow_idx_used][1],
        "R.M.S. error [Core] Peak": rmse[peak_flow_idx_used][2],
        "R.M.S. error [Non-F] Peak": rmse[peak_flow_idx_used][3],
        "vNRMSE error [Fluid] Peak": vnrmse[peak_flow_idx_used][0],
        "vNRMSE error [Bound] Peak": vnrmse[peak_flow_idx_used][1],
        "vNRMSE error [Core] Peak": vnrmse[peak_flow_idx_used][2],
        "vNRMSE error [Non-F] Peak": vnrmse[peak_flow_idx_used][3],
        "Directional error [Fluid] Peak": d_error[peak_flow_idx_used][0],
        "Directional error [Bound] Peak": d_error[peak_flow_idx_used][1],
        "Directional error [Core] Peak": d_error[peak_flow_idx_used][2],
        "Directional error [Non-F] Peak": d_error[peak_flow_idx_used][3],
    }

    return out


if __name__ == "__main__":

    # ---------------- SETTINGS ----------------
    # HR reference
    data_dir = "../../data/healthy"
    hr_filename = "HV01_05mm3_20ms.h5"

    # Typically these are your LR simulation/acquisition files.

    peak_flow_idx_hr = 12  # defined in HR time

    lr_data_dir = data_dir  # adjust if needed
    lr_filename_stems = [
        # HV01 - H1
        'HV01_05mm3_20ms_LR_sv17_tSNR10_newMask',
        # ...
    ]

    # Where to save metrics
    prediction_dir = "interpolation_results"
    method_folder = "/Interp_Cubic_x2" 
    results_subdir = "results"

    # Metrics behavior
    do_temporal_interp = True   # True: tempospatial; False: spatial-only (time maps LR k -> HR 2*k)

    # ------------------------------------------

    ground_truth_file = f"{data_dir}/{hr_filename}"

    # Create output results dir
    method_dir = f"{prediction_dir}{method_folder}"
    results_dir = os.path.join(method_dir, results_subdir)
    os.makedirs(results_dir, exist_ok=True)

    print("Start time:", time.ctime())
    print(f"HR file: {ground_truth_file}")
    print(f"LR dir:  {lr_data_dir}")
    print(f"Method:  {method_folder}  (temporal_interp={do_temporal_interp})")
    print(f"Results: {results_dir}")

    # -------------------------------
    # Load HR (reference) + masks
    # -------------------------------
    with h5py.File(ground_truth_file, "r") as hf:
        u_hr_all = np.asarray(hf["u"], dtype=np.float32)
        v_hr_all = np.asarray(hf["v"], dtype=np.float32)
        w_hr_all = np.asarray(hf["w"], dtype=np.float32)
        T_hr = u_hr_all.shape[0]

        mask = _ensure_mask_3d(np.asarray(hf["mask"]))
        mask = mask.astype(np.float32)

        nf_mask = 1.0 - mask
        boundary_mask, core_mask = e_utils.create_boundary_and_core_masks(mask, 0.1, "voxels")
        boundary_mask = boundary_mask.astype(np.float32)
        core_mask = core_mask.astype(np.float32)

        X, Y, Z = mask.shape
        cov_a = np.sum(mask) / (X * Y * Z)
        cov_b = np.sum(boundary_mask) / (X * Y * Z)
        cov_c = np.sum(core_mask) / (X * Y * Z)
        ratio_b = np.sum(boundary_mask) / np.sum(mask)
        ratio_c = np.sum(core_mask) / np.sum(mask)

        print(" ")
        print(f"Coverage: {100*cov_a:.3f} %")
        print(f"Boundary --- cov: {100*cov_b:.3f} %, ratio: {100*ratio_b:.3f} %")
        print(f"Core --- cov: {100*cov_c:.3f} %, ratio: {100*ratio_c:.3f} %")

        hr_shape_xyz = mask.shape

    # -------------------------------
    # Loop over LR inputs
    # -------------------------------
    for lr_stem in lr_filename_stems:
        lr_file = f"{lr_data_dir}/{lr_stem}.h5"
        print("\n" + "=" * 60)
        print(f"LR FILE: {lr_file}")
        print("Start:", time.ctime())
        t0 = time.time()

        with h5py.File(lr_file, "r") as lf:
            u_lr = np.asarray(lf["u"], dtype=np.float32)
            v_lr = np.asarray(lf["v"], dtype=np.float32)
            w_lr = np.asarray(lf["w"], dtype=np.float32)

        T_lr = u_lr.shape[0]
        print(f"T_lr={T_lr}, LR shape={u_lr.shape[1:]}, HR shape={hr_shape_xyz}")

        # -------------------------------
        # Spatial x2 (cubic) to HR shape
        # -------------------------------
        u_sp = np.empty((T_lr,) + hr_shape_xyz, dtype=np.float32)
        v_sp = np.empty((T_lr,) + hr_shape_xyz, dtype=np.float32)
        w_sp = np.empty((T_lr,) + hr_shape_xyz, dtype=np.float32)

        for t in range(T_lr):
            u_sp[t] = _cubic_resize_3d_to_shape(u_lr[t], hr_shape_xyz)
            v_sp[t] = _cubic_resize_3d_to_shape(v_lr[t], hr_shape_xyz)
            w_sp[t] = _cubic_resize_3d_to_shape(w_lr[t], hr_shape_xyz)

        # -------------------------------
        # Temporal handling + reference alignment
        # -------------------------------
        if do_temporal_interp:
            # tempospatial: build 1,2,3,4,5... by inserting between frames
            u_pred = _temporal_interp_x2_insert_between(u_sp)
            v_pred = _temporal_interp_x2_insert_between(v_sp)
            w_pred = _temporal_interp_x2_insert_between(w_sp)

            T_pred = u_pred.shape[0]

            # Compare to first T_pred of HR (truncate if HR shorter)
            T_compare = min(T_pred, T_hr)
            u_pred = u_pred[:T_compare]
            v_pred = v_pred[:T_compare]
            w_pred = w_pred[:T_compare]

            u_ref = u_hr_all[:T_compare]
            v_ref = v_hr_all[:T_compare]
            w_ref = w_hr_all[:T_compare]

            # Peak index: HR-defined index corresponds to same index in this compare (if within range)
            peak_used = int(np.clip(peak_flow_idx_hr, 0, T_compare - 1))

            temporal_note = "tempospatial_x2_insert_between; compare vs HR[0:T_compare]"

        else:
            # spatial-only: LR frames correspond to HR at 0,2,4,...
            hr_idx_use, T_compare = _hr_indices_for_lr(T_hr=T_hr, T_lr=T_lr)

            u_pred = u_sp[:T_compare]
            v_pred = v_sp[:T_compare]
            w_pred = w_sp[:T_compare]

            u_ref = u_hr_all[hr_idx_use]
            v_ref = v_hr_all[hr_idx_use]
            w_ref = w_hr_all[hr_idx_use]

            # Peak index: HR-defined peak maps to approx floor(peak/2) in this sequence
            peak_used = int(np.clip(peak_flow_idx_hr // 2, 0, T_compare - 1))

            temporal_note = "spatial_only; compare vs HR[0,2,4,...]"

        print(f"Comparison timesteps: {T_compare}")
        print(f"Temporal mode: {temporal_note}")
        print(f"Peak used index: {peak_used}")

        # -------------------------------
        # Metrics
        # -------------------------------
        metrics_core = compute_metrics_over_time(
            u_pred=u_pred, v_pred=v_pred, w_pred=w_pred,
            u_ref=u_ref, v_ref=v_ref, w_ref=w_ref,
            mask=mask, boundary_mask=boundary_mask, core_mask=core_mask, nf_mask=nf_mask,
            peak_flow_idx_used=peak_used,
        )

        # Add bookkeeping + rename keys to avoid collision
        mode_tag = "tempospatial" if do_temporal_interp else "spatial"
        metrics_name = f"metrics_interp_cubic_x2_{mode_tag}_{lr_stem}"

        metrics = {
            "method": "cubic_interp_x2",
            "mode": mode_tag,
            "lr_filename": f"{lr_stem}.h5",
            "hr_filename": hr_filename,
            "note": temporal_note,
            "T_hr": T_hr,
            "T_lr": T_lr,
            "T_compare": T_compare,
            "PEAK FLOW INDEX (HR)": peak_flow_idx_hr,
            "Coverage [%]": 100 * cov_a,
            "Boundary Coverage [%]": 100 * cov_b,
            "Core Coverage [%]": 100 * cov_c,
            "Ratio Boundary/Core [%]": 100 * ratio_c,
        }

        # Prefix metric names so you can distinguish easily
        for k, v in metrics_core.items():
            metrics[f"Interp2x {k}"] = v

        # Save CSV
        metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        metrics_filename = os.path.join(results_dir, f"{metrics_name}.csv")
        metrics_df.to_csv(metrics_filename, index=False)

        dt = time.time() - t0
        print(f"Saved: {metrics_filename}")
        print(f"Done in {dt:.2f} s")

    print("\nAll done.")
    print("End time:", time.ctime())
