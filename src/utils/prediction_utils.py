import h5py
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import zoom
from utils.evaluation_utils import (
    create_boundary_and_core_masks, 
    calculate_tanh_relative_error, 
    calculate_absolute_error, 
    calculate_rmse, 
    calculate_absolute_error_pressure, 
    calculate_rmse_pressure, 
    calculate_divergence,
    calculate_directional_error,
    calculate_vnrmse,
    linreg,
    calculate_gradient_absolute_error,
    calculate_gradient_relative_error,
    calculate_gradient_directional_error,
    calculate_gradient_nrmse
)
from utils.loss_utils import vector_potential_fn
from utils.prepare_data import create_and_normalize_coords, upsample_1d
from utils.preprocessing_utils import compute_outer_boundary_mask


def save_to_h5(output_filepath, col_name, dataset, expand_dim=True):
    
    if expand_dim:
        dataset = np.expand_dims(dataset, axis=0)

    # convert float64 to float32 to save space
    if dataset.dtype == 'float64':
        dataset = np.array(dataset, dtype='float32')
    
    with h5py.File(output_filepath, 'a') as hf:    
        if col_name not in hf:
            datashape = (None, )
            if (dataset.ndim > 1):
                datashape = (None, ) + dataset.shape[1:]
            hf.create_dataset(col_name, data=dataset, maxshape=datashape)
        else:
            hf[col_name].resize((hf[col_name].shape[0]) + dataset.shape[0], axis = 0)
            hf[col_name][-dataset.shape[0]:] = dataset

def save_prediction_metadata(
    h5_filename,
    config,
    spatial_factor=1,
    temporal_factor=1,
):
    # Store spatial and temporal resolution metadata in a prediction HDF5 file 

    spacing = np.asarray(
        [
            config.resolution.dx / spatial_factor,
            config.resolution.dy / spatial_factor,
            config.resolution.dz / spatial_factor,
        ],
        dtype=np.float32,
    )

    dt = config.resolution.dt
    if config.setup.include_time:
        dt = dt / temporal_factor

    with h5py.File(h5_filename, "a") as hf:
        hf.attrs["spacing"] = spacing
        hf.attrs["dt"] = float(dt)
        hf.attrs["origin"] = np.asarray(
            [0.0, 0.0, 0.0],
            dtype=np.float32,
        )

        # Store units
        hf.attrs["velocity_units"] = "m/s"

        if "p" in hf:
            hf.attrs["pressure_units"] = "Pa"

        if all(key in hf for key in ("p_x", "p_y", "p_z")):
            hf.attrs["pressure_gradient_units"] = "Pa/m"

def save_h5_predictions(config, model, device, it, xyz_ref, u_ref, v_ref, w_ref,
                        p_ref, px_ref, py_ref, pz_ref, mask_ref, mask_flat_ref,
                        U_max, standardization_factors):

    # Lightweight h5 save: only forward pass + denormalize + write h5, no metrics
    model.eval()
    xyz_ref_t = torch.from_numpy(xyz_ref).float().to(device)
    xyz_ref_t.requires_grad = config.training.use_vector_potential

    if config.training.use_vector_potential:
        with torch.set_grad_enabled(True):
            uvw_pred = model(xyz_ref_t)
            uvw_pred = vector_potential_fn(uvw_pred, xyz_ref_t)
            uvw_pred = uvw_pred.detach().cpu().numpy()
    else:
        with torch.no_grad():
            uvw_pred = model(xyz_ref_t)
            uvw_pred = uvw_pred.cpu().numpy()

    if config.predictions.fluid_region:
        fluid_indices = mask_flat_ref == 1
        uvw_pred_full = np.zeros((len(mask_flat_ref), uvw_pred.shape[1]))
        uvw_pred_full[fluid_indices] = uvw_pred
        uvw_pred = uvw_pred_full

    # Denormalize
    if config.predictions.denormalize:
        if config.vel_normalization == "characteristic":
            uvw_pred[:, 0] *= config.constants.U
            uvw_pred[:, 1] *= config.constants.U
            uvw_pred[:, 2] *= config.constants.U
            if config.setup.include_pressure and config.training.reference_gradients:
                _, _, _, std_x, _, std_y, _, std_z = standardization_factors
                uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_x
                uvw_pred[:, 4] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_y
                uvw_pred[:, 5] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_z
            elif config.setup.include_pressure and not config.training.reference_gradients:
                uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2)
        elif config.vel_normalization == "max_velocity":
            uvw_pred[:, 0] *= U_max
            uvw_pred[:, 1] *= U_max
            uvw_pred[:, 2] *= U_max
            if config.setup.include_pressure and config.training.reference_gradients:
                _, _, _, std_x, _, std_y, _, std_z = standardization_factors
                uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_x
                uvw_pred[:, 4] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_y
                uvw_pred[:, 5] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_z
            elif config.setup.include_pressure and not config.training.reference_gradients:
                uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2)

    # Reshape
    if config.setup.include_time:
        T, X, Y, Z = u_ref.shape
    else:
        X, Y, Z = u_ref.shape
        T = 1
    D_pred = uvw_pred.shape[1]
    uvw_pred = uvw_pred.reshape(T, X, Y, Z, D_pred)

    u_pred = uvw_pred[:, :, :, :, 0]
    v_pred = uvw_pred[:, :, :, :, 1]
    w_pred = uvw_pred[:, :, :, :, 2]

    # Save h5
    h5_filename = f"{config.log_dir}/SR_it{it:06d}.h5"
    save_to_h5(h5_filename, "u", u_pred, expand_dim=False)
    save_to_h5(h5_filename, "v", v_pred, expand_dim=False)
    save_to_h5(h5_filename, "w", w_pred, expand_dim=False)

    if config.setup.include_pressure and not config.training.reference_gradients:
        p_pred = uvw_pred[:, :, :, :, 3]
        save_to_h5(h5_filename, "p", p_pred, expand_dim=False)
    elif config.training.reference_gradients:
        save_to_h5(h5_filename, "p_x", uvw_pred[:, :, :, :, 3], expand_dim=False)
        save_to_h5(h5_filename, "p_y", uvw_pred[:, :, :, :, 4], expand_dim=False)
        save_to_h5(h5_filename, "p_z", uvw_pred[:, :, :, :, 5], expand_dim=False)
    if config.include_ref:
        save_to_h5(h5_filename, "mask", mask_ref, expand_dim=False)

    save_prediction_metadata(
        h5_filename,
        config,
        spatial_factor=config.ref_spatial_factor,
        temporal_factor=config.ref_temporal_factor,
    )

    print(f"[Iteration {it}] Saved h5 predictions to {h5_filename}")

def evaluate_predictions(config, model, device, it, xyz_ref, 
    u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, 
    mask_ref, mask_flat_ref, U_max, standardization_factors, 
    save_pred=False):

    # Create directory
    directory = os.path.join(
        config.log_dir,
        "evaluation",
        f"iter_{it:06d}",
    )
    os.makedirs(directory, exist_ok=True)

    # Predict reference coordinates
    model.eval()
    xyz_ref = torch.from_numpy(xyz_ref).float().to(device)
    xyz_ref.requires_grad = config.training.use_vector_potential

    if config.training.use_vector_potential:
        with torch.set_grad_enabled(True):
            uvw_pred = model(xyz_ref)
            uvw_pred = vector_potential_fn(uvw_pred, xyz_ref)
            uvw_pred = uvw_pred.detach().cpu().numpy()
    else:
        with torch.no_grad():
            uvw_pred = model(xyz_ref)
            uvw_pred = uvw_pred.cpu().numpy()

    if config.predictions.fluid_region:
        fluid_indices = mask_flat_ref==1
        uvw_pred_full = np.zeros(((len(mask_flat_ref), len(uvw_pred[0]))))
        uvw_pred_full[fluid_indices] = uvw_pred
        uvw_pred = uvw_pred_full

    # Denormalize predictions
    if config.predictions.denormalize:
        if config.vel_normalization == "characteristic":
            uvw_pred[:, 0] *= config.constants.U  # u
            uvw_pred[:, 1] *= config.constants.U  # v
            uvw_pred[:, 2] *= config.constants.U  # w
            #if config.setup.include_pressure:
            #    uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2)  # p
            if (config.setup.include_pressure and config.training.reference_gradients):

                _, _, _, std_x, _, std_y, _, std_z = standardization_factors

                uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_x # px
                uvw_pred[:, 4] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_y # py
                uvw_pred[:, 5] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_z # pz
            elif config.setup.include_pressure and not config.training.reference_gradients:
                uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2)  # p

        elif config.vel_normalization == "max_velocity":
            uvw_pred[:, 0] *= U_max  # u
            uvw_pred[:, 1] *= U_max  # v
            uvw_pred[:, 2] *= U_max  # w
            #if config.setup.include_pressure:
            #    uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2)  # p
            if (config.setup.include_pressure and config.training.reference_gradients):
                _, _, _, std_x, _, std_y, _, std_z = standardization_factors

                uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_x # px
                uvw_pred[:, 4] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_y # py
                uvw_pred[:, 5] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_z # pz
                
            elif config.setup.include_pressure and not config.training.reference_gradients:
                uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2)  # p

    # Define dimensions based on include_time
    if config.setup.include_time:
        T, X, Y, Z = u_ref.shape
    else:
        X, Y, Z = u_ref.shape
        T = 1  # Single time step

        u_ref = np.expand_dims(u_ref, axis=0)
        v_ref = np.expand_dims(v_ref, axis=0)
        w_ref = np.expand_dims(w_ref, axis=0)
        p_ref = np.expand_dims(p_ref, axis=0) if config.setup.include_pressure else None
        px_ref = np.expand_dims(px_ref, axis=0) if config.training.reference_gradients else None
        py_ref = np.expand_dims(py_ref, axis=0) if config.training.reference_gradients else None
        pz_ref = np.expand_dims(pz_ref, axis=0) if config.training.reference_gradients else None
    
    D_pred = uvw_pred.shape[1]

    uvw_pred = uvw_pred.reshape(T, X, Y, Z, D_pred)

    u_pred = uvw_pred[:, :, :, :, 0]
    v_pred = uvw_pred[:, :, :, :, 1]
    w_pred = uvw_pred[:, :, :, :, 2]
    p_pred_x = uvw_pred[:, :, :, :, 3] if config.training.reference_gradients else None
    p_pred_y = uvw_pred[:, :, :, :, 4] if config.training.reference_gradients else None
    p_pred_z = uvw_pred[:, :, :, :, 5] if config.training.reference_gradients else None
    p_pred = uvw_pred[:, :, :, :, 3] if (config.setup.include_pressure and not config.training.reference_gradients) else None 

    # Get metrics
    T = len(u_pred)
    nf_mask = 1.0 - mask_ref
    boundary_mask, core_mask = create_boundary_and_core_masks(mask_ref, 0.1, 'voxels')

    rel_err = np.zeros((T,3))
    abs_err = np.zeros((T,7))
    rmse = np.zeros((T,7))

    vnrmse = np.zeros((T,4))
    d_error = np.zeros((T,4))
    div_pred = np.zeros((T,4))
    div_ref = np.zeros((T,4))

    Ks = np.zeros((T,3,3))
    Ms = np.zeros((T,3,3))
    Rs = np.zeros((T,3,3))

    Ks_pgrad = np.zeros((T,3,3))
    Ms_pgrad = np.zeros((T,3,3))
    Rs_pgrad = np.zeros((T,3,3))

    grad_abs_err = np.zeros((T,3))
    grad_rel_err = np.zeros((T,3))
    grad_dir_err = np.zeros((T,3))
    grad_nrmse =   np.zeros((T,3))

    reference_spacing = (
        config.resolution.dx/ config.ref_spatial_factor, 
        config.resolution.dy/config.ref_spatial_factor,
        config.resolution.dz/config.ref_spatial_factor
    )

    for t in range(T):
        rel_err[t,0] = (calculate_tanh_relative_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
        rel_err[t,1] = (calculate_tanh_relative_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
        rel_err[t,2] = (calculate_tanh_relative_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], core_mask))

        abs_err[t,0] = (calculate_absolute_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
        abs_err[t,1] = (calculate_absolute_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
        abs_err[t,2] = (calculate_absolute_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], core_mask))
        abs_err[t,3] = (calculate_absolute_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], nf_mask))

        rmse[t,0] = (calculate_rmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
        rmse[t,1] = (calculate_rmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
        rmse[t,2] = (calculate_rmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], core_mask))
        rmse[t,3] = (calculate_rmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], nf_mask))

        vnrmse[t,0] = (calculate_vnrmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
        vnrmse[t,1] = (calculate_vnrmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
        vnrmse[t,2] = (calculate_vnrmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], core_mask))
        vnrmse[t,3] = (calculate_vnrmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], nf_mask))

        d_error[t,0] = (calculate_directional_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
        d_error[t,1] = (calculate_directional_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
        d_error[t,2] = (calculate_directional_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], core_mask))
        d_error[t,3] = (calculate_directional_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], nf_mask))

        div_pred[t,0] = (calculate_divergence([u_pred[t], v_pred[t], w_pred[t]], reference_spacing, mask_ref))
        div_pred[t,1] = (calculate_divergence([u_pred[t], v_pred[t], w_pred[t]], reference_spacing, boundary_mask))
        div_pred[t,2] = (calculate_divergence([u_pred[t], v_pred[t], w_pred[t]], reference_spacing, core_mask))
        div_pred[t,3] = (calculate_divergence([u_pred[t], v_pred[t], w_pred[t]], reference_spacing, nf_mask))

        div_ref[t,0] = (calculate_divergence([u_ref[t], v_ref[t], w_ref[t]], reference_spacing, mask_ref))
        div_ref[t,1] = (calculate_divergence([u_ref[t], v_ref[t], w_ref[t]], reference_spacing, boundary_mask))
        div_ref[t,2] = (calculate_divergence([u_ref[t], v_ref[t], w_ref[t]], reference_spacing, core_mask))
        div_ref[t,3] = (calculate_divergence([u_ref[t], v_ref[t], w_ref[t]], reference_spacing, nf_mask))

        Ks[t][0][0], Ms[t][0][0], Rs[t][0][0] = linreg(u_pred[t], u_ref[t], mask_ref)
        Ks[t][1][0], Ms[t][1][0], Rs[t][1][0] = linreg(v_pred[t], v_ref[t], mask_ref)
        Ks[t][2][0], Ms[t][2][0], Rs[t][2][0] = linreg(w_pred[t], w_ref[t], mask_ref)

        Ks[t][0][1], Ms[t][0][1], Rs[t][0][1] = linreg(u_pred[t], u_ref[t], boundary_mask)
        Ks[t][1][1], Ms[t][1][1], Rs[t][1][1] = linreg(v_pred[t], v_ref[t], boundary_mask)
        Ks[t][2][1], Ms[t][2][1], Rs[t][2][1] = linreg(w_pred[t], w_ref[t], boundary_mask)

        Ks[t][0][2], Ms[t][0][2], Rs[t][0][2] = linreg(u_pred[t], u_ref[t], core_mask)
        Ks[t][1][2], Ms[t][1][2], Rs[t][1][2] = linreg(v_pred[t], v_ref[t], core_mask)
        Ks[t][2][2], Ms[t][2][2], Rs[t][2][2] = linreg(w_pred[t], w_ref[t], core_mask)

        if config.training.reference_gradients:
            # px
            Ks_pgrad[t][0][0], Ms_pgrad[t][0][0], Rs_pgrad[t][0][0] = linreg(p_pred_x[t], px_ref[t], mask_ref)
            Ks_pgrad[t][0][1], Ms_pgrad[t][0][1], Rs_pgrad[t][0][1] = linreg(p_pred_x[t], px_ref[t], boundary_mask)
            Ks_pgrad[t][0][2], Ms_pgrad[t][0][2], Rs_pgrad[t][0][2] = linreg(p_pred_x[t], px_ref[t], core_mask)

            # py
            Ks_pgrad[t][1][0], Ms_pgrad[t][1][0], Rs_pgrad[t][1][0] = linreg(p_pred_y[t], py_ref[t], mask_ref)
            Ks_pgrad[t][1][1], Ms_pgrad[t][1][1], Rs_pgrad[t][1][1] = linreg(p_pred_y[t], py_ref[t], boundary_mask)
            Ks_pgrad[t][1][2], Ms_pgrad[t][1][2], Rs_pgrad[t][1][2] = linreg(p_pred_y[t], py_ref[t], core_mask)

            # pz
            Ks_pgrad[t][2][0], Ms_pgrad[t][2][0], Rs_pgrad[t][2][0] = linreg(p_pred_z[t], pz_ref[t], mask_ref)
            Ks_pgrad[t][2][1], Ms_pgrad[t][2][1], Rs_pgrad[t][2][1] = linreg(p_pred_z[t], pz_ref[t], boundary_mask)
            Ks_pgrad[t][2][2], Ms_pgrad[t][2][2], Rs_pgrad[t][2][2] = linreg(p_pred_z[t], pz_ref[t], core_mask)

            # Pressure gradient errors
            grad_abs_err[t, 0] = calculate_gradient_absolute_error(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], mask_ref)
            grad_abs_err[t, 1] = calculate_gradient_absolute_error(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], boundary_mask)
            grad_abs_err[t, 2] = calculate_gradient_absolute_error(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], core_mask)

            grad_rel_err[t, 0] = calculate_gradient_relative_error(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], mask_ref)
            grad_rel_err[t, 1] = calculate_gradient_relative_error(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], boundary_mask)
            grad_rel_err[t, 2] = calculate_gradient_relative_error(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], core_mask)

            grad_dir_err[t, 0] = calculate_gradient_directional_error(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], mask_ref)
            grad_dir_err[t, 1] = calculate_gradient_directional_error(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], boundary_mask)
            grad_dir_err[t, 2] = calculate_gradient_directional_error(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], core_mask)

            grad_nrmse[t, 0] = calculate_gradient_nrmse(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], mask_ref)
            grad_nrmse[t, 1] = calculate_gradient_nrmse(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], boundary_mask)
            grad_nrmse[t, 2] = calculate_gradient_nrmse(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], core_mask)

    # Total average
    rel_err_tot = np.mean(rel_err, axis=0)
    abs_err_tot = np.mean(abs_err, axis=0)
    rmse_tot = np.mean(rmse, axis=0)
    vnrmse_tot = np.mean(vnrmse, axis=0)
    d_error_tot = np.mean(d_error, axis=0)

    div_pred_tot = np.nanmean(div_pred, axis=0)
    div_ref_tot = np.nanmean(div_ref, axis=0)
    
    Ks_tot = np.mean(Ks, axis=0)
    Rs_tot = np.mean(Rs, axis=0)

    if config.training.reference_gradients:
        grad_abs_err_tot = np.mean(grad_abs_err, axis=0)
        grad_rel_err_tot = np.mean(grad_rel_err, axis=0)
        grad_nrmse_tot =   np.mean(grad_nrmse, axis=0)
        grad_dir_err_tot = np.mean(grad_dir_err, axis=0)
        Ks_pgrad_tot =     np.mean(Ks_pgrad, axis=0)
        Rs_pgrad_tot =     np.mean(Rs_pgrad, axis=0)

    # Save metrics to csv
    metrics = {
        'Relative error [Fluid]': rel_err_tot[0],
        'Relative error [Bound]': rel_err_tot[1],
        'Relative error [Core]':  rel_err_tot[2],

        'Absolute error [Fluid]': abs_err_tot[0],
        'Absolute error [Bound]': abs_err_tot[1],
        'Absolute error [Core]':  abs_err_tot[2],
        #'Absolute error [Non-F]': abs_err_tot[3],

        'R.M.S. error [Fluid]': rmse_tot[0],
        'R.M.S. error [Bound]': rmse_tot[1],
        'R.M.S. error [Core]':  rmse_tot[2],
        #'R.M.S. error [Non-F]': rmse_tot[3],

        'VNRMSE [Fluid]': vnrmse_tot[0],
        'VNRMSE [Bound]': vnrmse_tot[1],
        'VNRMSE [Core]':  vnrmse_tot[2],
        #'VNRMSE [Non-F]': vnrmse_tot[3],

        'Directional error [Fluid]': d_error_tot[0],
        'Directional error [Bound]': d_error_tot[1],
        'Directional error [Core]':  d_error_tot[2],

        'Divergence prediction [Fluid]': div_pred_tot[0],
        'Divergence prediction [Bound]': div_pred_tot[1],
        'Divergence prediction [Core]':  div_pred_tot[2],

        'Divergence reference [Fluid]': div_ref_tot[0],
        'Divergence reference [Bound]': div_ref_tot[1],
        'Divergence reference [Core]':  div_ref_tot[2],

        'U R2 [Fluid]': Rs_tot[0][0],
        'U R2 [Bound]': Rs_tot[0][1],
        'U R2 [Core]':  Rs_tot[0][2],
        'V R2 [Fluid]': Rs_tot[1][0],
        'V R2 [Bound]': Rs_tot[1][1],
        'V R2 [Core]':  Rs_tot[1][2],
        'W R2 [Fluid]': Rs_tot[2][0],
        'W R2 [Bound]': Rs_tot[2][1],
        'W R2 [Core]':  Rs_tot[2][2],

        'U K [Fluid]': Ks_tot[0][0],
        'U K [Bound]': Ks_tot[0][1],
        'U K [Core]':  Ks_tot[0][2],
        'V K [Fluid]': Ks_tot[1][0],
        'V K [Bound]': Ks_tot[1][1],
        'V K [Core]':  Ks_tot[1][2],
        'W K [Fluid]': Ks_tot[2][0],
        'W K [Bound]': Ks_tot[2][1],
        'W K [Core]':  Ks_tot[2][2],

        'PEAK FLOW INDEX:': config.predictions.peak_flow_idx,

        'U K [Core] Peak':  Ks[config.predictions.peak_flow_idx][0][2],
        'U R2 [Core] Peak': Rs[config.predictions.peak_flow_idx][0][2],
        'V K [Core] Peak':  Ks[config.predictions.peak_flow_idx][1][2],
        'V R2 [Core] Peak': Rs[config.predictions.peak_flow_idx][1][2],
        'W K [Core] Peak':  Ks[config.predictions.peak_flow_idx][2][2],
        'W R2 [Core] Peak': Rs[config.predictions.peak_flow_idx][2][2],
    }

    if config.training.reference_gradients:
        metrics.update({
            'Absolute error Pressure Gradient [Fluid]': grad_abs_err_tot[0],
            'Absolute error Pressure Gradient [Bound]': grad_abs_err_tot[1],
            'Absolute error Pressure Gradient [Core]':  grad_abs_err_tot[2],

            'Relative error Pressure Gradient (%) [Fluid]': grad_rel_err_tot[0]*100,
            'Relative error Pressure Gradient (%) [Bound]': grad_rel_err_tot[1]*100,
            'Relative error Pressure Gradient (%) [Core]':  grad_rel_err_tot[2]*100,

            'Pressure Gradient NRMSE (%) [Fluid]': grad_nrmse_tot[0]*100,
            'Pressure Gradient NRMSE (%) [Bound]': grad_nrmse_tot[1]*100,
            'Pressure Gradient NRMSE (%) [Core]':  grad_nrmse_tot[2]*100,

            'Pressure Gradient Directional Error [Fluid]': grad_dir_err_tot[0],
            'Pressure Gradient Directional Error [Bound]': grad_dir_err_tot[1],
            'Pressure Gradient Directional Error [Core]':  grad_dir_err_tot[2],

            'PX K    [Fluid]': Ks_pgrad_tot[0][0],
            'PX K    [Bound]': Ks_pgrad_tot[0][1],
            'PX K    [Core]':  Ks_pgrad_tot[0][2],
            'PY K    [Fluid]': Ks_pgrad_tot[1][0],
            'PY K    [Bound]': Ks_pgrad_tot[1][1],
            'PY K    [Core]':  Ks_pgrad_tot[1][2],
            'PZ K    [Fluid]': Ks_pgrad_tot[2][0],
            'PZ K    [Bound]': Ks_pgrad_tot[2][1],
            'PZ K    [Core]':  Ks_pgrad_tot[2][2],

            'PX R2    [Fluid]': Rs_pgrad_tot[0][0],
            'PX R2    [Bound]': Rs_pgrad_tot[0][1],
            'PX R2    [Core]':  Rs_pgrad_tot[0][2],
            'PY R2    [Fluid]': Rs_pgrad_tot[1][0],
            'PY R2    [Bound]': Rs_pgrad_tot[1][1],
            'PY R2    [Core]':  Rs_pgrad_tot[1][2],
            'PZ R2    [Fluid]': Rs_pgrad_tot[2][0],
            'PZ R2    [Bound]': Rs_pgrad_tot[2][1],
            'PZ R2    [Core]':  Rs_pgrad_tot[2][2],

            'PEAK FLOW INDEX:': config.predictions.peak_flow_idx,

            'PX K [Core] Peak':  Ks_pgrad[config.predictions.peak_flow_idx][0][2],
            'PX M [Core]':       Ms_pgrad[config.predictions.peak_flow_idx][0][2],
            'PX R2 [Core] Peak': Rs_pgrad[config.predictions.peak_flow_idx][0][2],
            'PY K [Core] Peak':  Ks_pgrad[config.predictions.peak_flow_idx][1][2],
            'PY M [Core]':       Ms_pgrad[config.predictions.peak_flow_idx][1][2],
            'PY R2 [Core] Peak': Rs_pgrad[config.predictions.peak_flow_idx][1][2],
            'PZ K [Core] Peak':  Ks_pgrad[config.predictions.peak_flow_idx][2][2],
            'PZ M [Core]':       Ms_pgrad[config.predictions.peak_flow_idx][2][2],
            'PZ R2 [Core] Peak': Rs_pgrad[config.predictions.peak_flow_idx][2][2],

        })

    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    metrics_filename = f"{directory}/metrics.csv"
    metrics_df.to_csv(metrics_filename, index=False)

    def print_metric_summary(metrics):
        print(
            "Evaluation | "
            f"MRE={metrics['Relative error [Fluid]']:.2f}% | "
            f"vNRMSE={metrics['VNRMSE [Fluid]']:.4f} | "
            f"W slope={metrics['W K [Core]']:.3f} | "
            f"W R2={metrics['W R2 [Core]']:.3f}"
        )

    print_metric_summary(metrics)

    if save_pred:
        
        # Save ref predictions to results directory
        h5_filename = f"{config.log_dir}/SR_it{it:06d}.h5"

        save_to_h5(h5_filename, "u", u_pred, expand_dim=False)
        save_to_h5(h5_filename, "v", v_pred, expand_dim=False)
        save_to_h5(h5_filename, "w", w_pred, expand_dim=False)

        if (config.setup.include_pressure and not config.training.reference_gradients):
            save_to_h5(h5_filename, "p", p_pred, expand_dim=False)
        elif config.training.reference_gradients:
            save_to_h5(h5_filename, "p_x", p_pred_x, expand_dim=False)
            save_to_h5(h5_filename, "p_y", p_pred_y, expand_dim=False)
            save_to_h5(h5_filename, "p_z", p_pred_z, expand_dim=False)

        if config.include_ref:
            save_to_h5(h5_filename, "mask", mask_ref, expand_dim=False)

        save_prediction_metadata(
            h5_filename,
            config,
            spatial_factor=config.ref_spatial_factor,
            temporal_factor=config.ref_temporal_factor,
        )        

    return metrics

def _plot_field_comparison(
    rows,
    column_titles,
    output_path,
    symmetric=False,
):
    n_rows = len(rows)
    n_cols = len(column_titles)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, (row_label, fields) in enumerate(rows):

        if symmetric:
            limit = max(
                np.nanmax(np.abs(field))
                for field in fields
            )
            vmin, vmax = -limit, limit
        else:
            vmin = min(np.nanmin(field) for field in fields)
            vmax = max(np.nanmax(field) for field in fields)

        for col_idx, field in enumerate(fields):

            ax = axes[row_idx, col_idx]

            image = ax.imshow(
                field.T,
                origin="lower",
                cmap="RdBu_r" if symmetric else "viridis",
                vmin=vmin,
                vmax=vmax,
            )

            if row_idx == 0:
                ax.set_title(column_titles[col_idx])

            if col_idx == 0:
                ax.set_ylabel(row_label)

            ax.set_xticks([])
            ax.set_yticks([])

        fig.colorbar(
            image,
            ax=axes[row_idx, :].tolist(),
            shrink=0.8,
        )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_reference_comparison(config, model, device, it, xyz_ref, 
    u_lr, v_lr, w_lr, p_lr, u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, 
    mask_ref, mask_flat_ref, U_max, standardization_factors
    ):

    # Create directory
    directory = os.path.join(
        config.log_dir,
        "evaluation",
        f"iter_{it:06d}",
    )
    os.makedirs(directory, exist_ok=True)

    # Predict reference coordinates
    model.eval()
    xyz_ref = torch.from_numpy(xyz_ref).float().to(device)
    xyz_ref.requires_grad = config.training.use_vector_potential

    if config.training.use_vector_potential:
        with torch.set_grad_enabled(True):
            uvw_pred = model(xyz_ref)
            uvw_pred = vector_potential_fn(uvw_pred, xyz_ref)
            uvw_pred = uvw_pred.detach().cpu().numpy()
    else:
        with torch.no_grad():
            uvw_pred = model(xyz_ref)
            uvw_pred = uvw_pred.cpu().numpy()

    if config.predictions.fluid_region:
        fluid_indices = mask_flat_ref==1
        uvw_pred_full = np.zeros(((len(mask_flat_ref), len(uvw_pred[0]))))
        uvw_pred_full[fluid_indices] = uvw_pred
        uvw_pred = uvw_pred_full

    # Denormalize predictions
    if config.predictions.denormalize:
        if config.vel_normalization == "characteristic":
            uvw_pred[:, 0] *= config.constants.U  # u
            uvw_pred[:, 1] *= config.constants.U  # v
            uvw_pred[:, 2] *= config.constants.U  # w
            #if config.setup.include_pressure:
            #    uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2)  # p
            if (config.setup.include_pressure and config.training.reference_gradients):
                _, _, _, std_x, _, std_y, _, std_z = standardization_factors

                uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_x # px
                uvw_pred[:, 4] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_y # py
                uvw_pred[:, 5] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_z # pz

            elif config.setup.include_pressure and not config.training.reference_gradients:
                uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2)  # p
        elif config.vel_normalization == "max_velocity":
            uvw_pred[:, 0] *= U_max  # u
            uvw_pred[:, 1] *= U_max  # v
            uvw_pred[:, 2] *= U_max  # w
            #if config.setup.include_pressure:
            #    uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2)  # p
            if (config.setup.include_pressure and config.training.reference_gradients):
                _, _, _, std_x, _, std_y, _, std_z = standardization_factors

                uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_x # px
                uvw_pred[:, 4] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_y # py
                uvw_pred[:, 5] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_z # pz
            elif config.setup.include_pressure and not config.training.reference_gradients:
                uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2)  # p

    # Define dimensions based on include_time
    if config.setup.include_time:
        T, X, Y, Z = u_ref.shape
    else:
        X, Y, Z = u_ref.shape
        T = 1  # Single time step

    D_pred = uvw_pred.shape[1]
    uvw_pred = uvw_pred.reshape(T, X, Y, Z, D_pred)

    # Visualization indices are defined on the LR input grid.
    t_lr = config.visualization.time_index_lr if config.setup.include_time else 0
    z_lr = config.visualization.z_index_lr

    # Map the selected LR indices to the reference/SR grid.
    t_ref = t_lr * config.ref_temporal_factor if config.setup.include_time else 0
    z_ref = z_lr * config.ref_spatial_factor

    # Extract prediction at the selected reference-grid time point.
    u_pred = uvw_pred[t_ref, :, :, :, 0]
    v_pred = uvw_pred[t_ref, :, :, :, 1]
    w_pred = uvw_pred[t_ref, :, :, :, 2]

    px_pred = (uvw_pred[t_ref, :, :, :, 3] if config.training.reference_gradients else None)
    py_pred = (uvw_pred[t_ref, :, :, :, 4] if config.training.reference_gradients else None)
    pz_pred = (uvw_pred[t_ref, :, :, :, 5] if config.training.reference_gradients else None)
    p_pred = (
        uvw_pred[t_ref, :, :, :, 3]
        if (
            config.setup.include_pressure
            and not config.training.reference_gradients
        )
        else None
    )

    # Create LR vs HR vs SR plots
    velocity_rows = [
        ("u [m/s]",[u_lr[t_lr, :, :, z_lr], 
        u_ref[t_ref, :, :, z_ref], 
        u_pred[:, :, z_ref],],
        ),
        ("v [m/s]",[v_lr[t_lr, :, :, z_lr],
        v_ref[t_ref, :, :, z_ref],
        v_pred[:, :, z_ref],],
        ),
        ("w [m/s]",[w_lr[t_lr, :, :, z_lr],
        w_ref[t_ref, :, :, z_ref],
        w_pred[:, :, z_ref],
        ],),
    ]

    _plot_field_comparison(
        velocity_rows, ["LR", "Reference", "Prediction"],
        os.path.join(directory, "velocity_comparison.png"),
        symmetric=False,
    )

    if config.setup.include_pressure and not config.training.reference_gradients:

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.title('Reference p')
        plt.imshow(p_ref[t_ref, :, :, z_ref].T, origin='lower', cmap='coolwarm')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.title('Predicted p')
        plt.imshow(p_pred[:, :, z_ref].T, origin='lower', cmap='coolwarm')
        plt.colorbar()

        plt.savefig(os.path.join(directory, f"prediction_vs_reference_p.png"))
        plt.close()
    
    if config.training.reference_gradients:

        gradient_rows = [
        ("dp/dx [Pa/m]", [px_ref[t_ref, :, :, z_ref], px_pred[:, :, z_ref],],),
        ("dp/dy [Pa/m]", [py_ref[t_ref, :, :, z_ref], py_pred[:, :, z_ref],],),
        ("dp/dz [Pa/m]", [pz_ref[t_ref, :, :, z_ref], pz_pred[:, :, z_ref],],),
        ]

        _plot_field_comparison(
            gradient_rows,["Reference", "Prediction"],
            os.path.join(
                directory, "pressure_gradient_comparison.png",), 
            symmetric=False,
        )

    return

def predict_superresolved_grid(config, model, device, it, u, mask, 
    U_max, save_pred=False):
    
    directory = os.path.join(
        config.log_dir,
        "evaluation",
        f"iter_{it:06d}",
    )
    os.makedirs(directory, exist_ok=True)

    # Extract boundaries
    if config.predictions.expand_mask:
        boundary_mask = compute_outer_boundary_mask(mask)
        mask = mask + boundary_mask

    if config.setup.include_time:
        t_len, x_len, y_len, z_len = u.shape
    else:
        x_len, y_len, z_len = u.shape
        t_len = 1

    t_normalized, x_normalized, y_normalized, z_normalized, standardization_factors = create_and_normalize_coords(config, t_len, x_len, y_len, z_len)

    # Upsample each coordinate
    t_ups = upsample_1d(t_normalized, config.predictions.temporal_factor, config.predictions.temporal_upsampling_mode) if config.setup.include_time else []
    x_ups = upsample_1d(x_normalized, config.predictions.spatial_factor, mode=config.predictions.spatial_upsampling_mode)
    y_ups = upsample_1d(y_normalized, config.predictions.spatial_factor, mode=config.predictions.spatial_upsampling_mode)
    z_ups = upsample_1d(z_normalized, config.predictions.spatial_factor, mode=config.predictions.spatial_upsampling_mode)

    # Select the SR-grid coordinates closest to the visualization
    # location defined on the LR input grid.
    t_lr = (
        config.visualization.time_index_lr
        if config.setup.include_time
        else 0
    )
    z_lr = config.visualization.z_index_lr

    if config.setup.include_time:
        t_pred = int(
            np.argmin(
                np.abs(t_ups - t_normalized[t_lr])
            )
        )
    else:
        t_pred = 0

    z_pred = int(
        np.argmin(
            np.abs(z_ups - z_normalized[z_lr])
        )
    )

    if config.setup.include_time:
        grids = np.meshgrid(t_ups, x_ups, y_ups, z_ups, indexing='ij')
    else:
        grids = np.meshgrid(x_ups, y_ups, z_ups, indexing='ij')
    
    flat_coords = [grid.ravel() for grid in grids]
    xyz_plot_full = np.stack(flat_coords, axis=-1) 

    if config.predictions.fluid_region:
        # Upsample mask
        mask_plot = zoom(mask, zoom=config.predictions.spatial_factor, order=0, grid_mode=True, mode='nearest')
        mask_plot_flat = np.tile(mask_plot.ravel(), len(t_ups)) if config.setup.include_time else mask_plot.ravel()
        fluid_indices = mask_plot_flat == 1

        xyz_plot = xyz_plot_full[fluid_indices]
    else:
        xyz_plot = xyz_plot_full  
    
    # Predict fluid data poinst grid
    model.eval()
    xyz_plot = torch.from_numpy(xyz_plot).float().to(device)
    xyz_plot.requires_grad = config.training.use_vector_potential

    if config.training.use_vector_potential:
        with torch.set_grad_enabled(True):
            uvw_pred_plot = model(xyz_plot)
            uvw_pred_plot = vector_potential_fn(uvw_pred_plot, xyz_plot)
            uvw_pred_plot = uvw_pred_plot.detach().cpu().numpy()
    else:
        with torch.no_grad():
            uvw_pred_plot = model(xyz_plot)
            uvw_pred_plot = uvw_pred_plot.cpu().numpy()

    if config.predictions.fluid_region:
        uvw_pred_full = np.zeros(((len(xyz_plot_full), len(uvw_pred_plot[0])))) + config.predictions.non_fluid_value
        uvw_pred_full[fluid_indices] = uvw_pred_plot
        uvw_pred_plot = uvw_pred_full

    if config.setup.include_time:
        uvw_pred_plot = uvw_pred_plot.reshape(len(t_ups), len(x_ups), len(y_ups), len(z_ups), len(uvw_pred_plot[0]))
    else:
        uvw_pred_plot = uvw_pred_plot.reshape(1, len(x_ups), len(y_ups), len(z_ups), len(uvw_pred_plot[0]))

    # Denormalize
    if config.setup.include_time:
        u_pred = uvw_pred_plot[:, :, :, :, 0]
        v_pred = uvw_pred_plot[:, :, :, :, 1]
        w_pred = uvw_pred_plot[:, :, :, :, 2]

        px_pred = uvw_pred_plot[:, :, :, :, 3] if config.training.reference_gradients else None
        py_pred = uvw_pred_plot[:, :, :, :, 4] if config.training.reference_gradients else None
        pz_pred = uvw_pred_plot[:, :, :, :, 5] if config.training.reference_gradients else None
        
        p_pred = uvw_pred_plot[:, :, :, :, 3] if (config.setup.include_pressure and not config.training.reference_gradients) else None 

        p_pred = uvw_pred_plot[:, :, :, :, 3] if config.setup.include_pressure else None
    else:
        u_pred = uvw_pred_plot[0, :, :, :, 0]
        v_pred = uvw_pred_plot[0, :, :, :, 1]
        w_pred = uvw_pred_plot[0, :, :, :, 2]
        p_pred = uvw_pred_plot[0, :, :, :, 3] if config.setup.include_pressure else None

    # Denormalize predictions
    if config.predictions.denormalize:
        if config.vel_normalization == "characteristic":
            u_pred = u_pred*config.constants.U
            v_pred = v_pred*config.constants.U
            w_pred = w_pred*config.constants.U

            #if config.setup.include_pressure:
            #    uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2)  # p
            if (config.setup.include_pressure and config.training.reference_gradients):
                _, _, _, std_x, _, std_y, _, std_z = standardization_factors

                px_pred *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_x # px
                py_pred *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_y # py
                pz_pred *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_z # pz

            elif config.setup.include_pressure and not config.training.reference_gradients:
                p_pred *= config.constants.rho * (config.constants.U ** 2)  # p

        elif config.vel_normalization == "max_velocity":
            u_pred = u_pred*U_max
            v_pred = v_pred*U_max
            w_pred = w_pred*U_max
            p_pred = p_pred*(config.constants.rho*(config.constants.U**2)) if config.setup.include_pressure else None
    
    # Plotting (example using matplotlib)
    plt.figure(figsize=(12, 6))


    plt.subplot(2, 2, 1)
    plt.title('Predicted u')
    plt.imshow(u_pred[t_pred, :, :, z_pred].T, origin='lower', extent=[x_ups.min(), x_ups.max(), y_ups.min(), y_ups.max()])
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title('Predicted v')
    plt.imshow(v_pred[t_pred, :, :, z_pred].T, origin='lower', extent=[x_ups.min(), x_ups.max(), y_ups.min(), y_ups.max()])
    plt.colorbar()
    
    plt.subplot(2, 2, 3)
    plt.title('Predicted w')
    plt.imshow(w_pred[t_pred, :, :, z_pred].T, origin='lower', extent=[x_ups.min(), x_ups.max(), y_ups.min(), y_ups.max()])
    plt.colorbar()

    plt.savefig(os.path.join(directory, f"predictions.png"))
    plt.close()

    if save_pred:
        
        # Save ref predictions to results directory
        h5_filename = f"{config.log_dir}/SR_final.h5"
        save_to_h5(h5_filename, "u", u_pred, expand_dim=False)
        save_to_h5(h5_filename, "v", v_pred, expand_dim=False)
        save_to_h5(h5_filename, "w", w_pred, expand_dim=False)

        if (config.setup.include_pressure and not config.training.reference_gradients):
            save_to_h5(f"{config.log_dir}/SR_final.h5", "p", p_pred, expand_dim=False)
        #elif config.training.reference_gradients:
        #    save_to_h5(f"{config.log_dir}/SR_final.h5", "p_x", px_pred, expand_dim=False)
        #    save_to_h5(f"{config.log_dir}/SR_final.h5", "p_y", py_pred, expand_dim=False)
        #    save_to_h5(f"{config.log_dir}/SR_final.h5", "p_z", pz_pred, expand_dim=False)

        save_prediction_metadata(
            h5_filename,
            config,
            spatial_factor=config.predictions.spatial_factor,
            temporal_factor=config.predictions.temporal_factor,
        )

    return