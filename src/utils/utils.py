import os
import shutil
import h5py
import torch
import random
import numpy as np
import pandas as pd
from utils.prepare_data import create_and_normalize_coords, upsample_1d, extract_fluid_region, compute_outer_boundary_mask
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
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

#import vtk
#from vtk.util import numpy_support as ns

def copy_cource_code(model_dir, directory_to_backup=["."], folder_name="backup_source"):

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    print("Copying source code to model directory...")

    # Copy all the source file to the model dir for backup
    for directory in directory_to_backup:
        files = os.listdir(directory)
        for fname in files:
            if fname.endswith(".py"):
                dest_fpath = os.path.join(model_dir, folder_name, directory, fname)
                os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
                shutil.copy2(f"{directory}/{fname}", dest_fpath)

    return 

def save_to_h5(output_filepath, col_name, dataset):
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

def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
def save_checkpoint(model, iter, config, final=False, extra_data=None):

    checkpoint = {
        'iteration': iter,
        'model_state_dict': model.state_dict(),
    }

    if extra_data is not None:
        checkpoint.update(extra_data)

    # Specify save path
    if not final:
        directory = f'{config.log_dir}/checkpoints'
        if not os.path.exists(directory):
            os.makedirs(directory)
        save_path = os.path.join(directory, f"{config.network_name}_it{iter:03d}.pth")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
    else:
        save_path = os.path.join(config.log_dir, f"{config.network_name}_final.pth")
        torch.save(checkpoint, save_path)
        print(f"Final model saved to {save_path}")

def sample_to_device(config, xyz_train, xyz_collocation, xyz_boundary, uvw_train, mask_flat, device):
    
    # Data / Fluid Points
    if config.training.data_points_per_batch is not None:
        data_indices = np.random.choice(len(xyz_train), size=config.training.data_points_per_batch, replace=False)
        xyz_data_batch = xyz_train[data_indices]
        uvw_data_batch = uvw_train[data_indices]
        mask_batch = mask_flat[data_indices]
    else:
        xyz_data_batch = xyz_train
        uvw_data_batch = uvw_train
        mask_batch = mask_flat

    xyz_data_batch = torch.from_numpy(xyz_data_batch).float().to(device)
    uvw_data_batch = torch.from_numpy(uvw_data_batch).float().to(device)
    mask_batch = torch.from_numpy(mask_batch).float().to(device)
    mask_batch = mask_batch.view(-1, 1)

    # Collocation Points
    xyz_collocation_batch = None
    if config.sample_collocation:
        if config.training.coll_points_per_batch is not None:
            coll_indices = np.random.choice(len(xyz_collocation), size=config.training.coll_points_per_batch, replace=False)
            xyz_collocation_batch = xyz_collocation[coll_indices]
        else:
            xyz_collocation_batch = xyz_collocation

        xyz_collocation_batch = torch.from_numpy(xyz_collocation_batch).float().to(device)

    # Boundary Points
    xyz_boundary_batch = None
    if config.sample_boundary:
        if config.training.boundary_points_per_batch is not None:
            boundary_indices = np.random.choice(len(xyz_boundary), size=config.training.boundary_points_per_batch, replace=False)
            xyz_boundary_batch = xyz_boundary[boundary_indices]
        else:
            xyz_boundary_batch = xyz_boundary

        xyz_boundary_batch = torch.from_numpy(xyz_boundary_batch).float().to(device)

    return xyz_data_batch, uvw_data_batch, mask_batch, xyz_collocation_batch, xyz_boundary_batch


def sample_ref_to_device(config, xyz_train, uvw_train, mask_flat, device):
    
    # Data / Fluid Points
    if config.training.data_points_per_batch is not None:
        data_indices = np.random.choice(len(xyz_train), size=config.training.data_points_per_batch, replace=False)
        xyz_data_batch = xyz_train[data_indices]
        uvw_data_batch = uvw_train[data_indices]
        mask_batch = mask_flat[data_indices]
    else:
        xyz_data_batch = xyz_train
        uvw_data_batch = uvw_train
        mask_batch = mask_flat

    xyz_data_batch = torch.from_numpy(xyz_data_batch).float().to(device)
    uvw_data_batch = torch.from_numpy(uvw_data_batch).float().to(device)
    mask_batch = torch.from_numpy(mask_batch).float().to(device)
    mask_batch = mask_batch.view(-1, 1)

    return xyz_data_batch, uvw_data_batch, mask_batch


def evaluate_predictions(config, model, device, it, xyz_ref, u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask_ref, mask_flat_ref, U_max, standardization_factors):

    # Create directory
    directory = f'{config.log_dir}/errors/iter_{it}'
    if not os.path.exists(directory):
        os.makedirs(directory)

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

    if config.plot.fluid_region:
        fluid_indices = mask_flat_ref==1
        uvw_pred_full = np.zeros(((len(mask_flat_ref), len(uvw_pred[0]))))
        uvw_pred_full[fluid_indices] = uvw_pred
        uvw_pred = uvw_pred_full

    # Denormalize predictions
    if config.plot.denormalize:
        if config.vel_normalization == "characteristic":
            uvw_pred[:, 0] *= config.constants.U  # u
            uvw_pred[:, 1] *= config.constants.U  # v
            uvw_pred[:, 2] *= config.constants.U  # w
            #if config.setup.include_pressure:
            #    uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2)  # p
            if (config.setup.include_pressure and config.training.reference_gradients):

                _, _, _, std_x, _, std_y, _, std_z = standardization_factors
                print(std_x, std_y, std_z)

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
                print(std_x, std_y, std_z)

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

        div_pred[t,0] = (calculate_divergence([u_pred[t], v_pred[t], w_pred[t]], [config.resolution.dx, config.resolution.dy, config.resolution.dz], mask_ref))
        div_pred[t,1] = (calculate_divergence([u_pred[t], v_pred[t], w_pred[t]], [config.resolution.dx, config.resolution.dy, config.resolution.dz], boundary_mask))
        div_pred[t,2] = (calculate_divergence([u_pred[t], v_pred[t], w_pred[t]], [config.resolution.dx, config.resolution.dy, config.resolution.dz], core_mask))
        div_pred[t,3] = (calculate_divergence([u_pred[t], v_pred[t], w_pred[t]], [config.resolution.dx, config.resolution.dy, config.resolution.dz], nf_mask))

        div_ref[t,0] = (calculate_divergence([u_ref[t], v_ref[t], w_ref[t]], [config.resolution.dx, config.resolution.dy, config.resolution.dz], mask_ref))
        div_ref[t,1] = (calculate_divergence([u_ref[t], v_ref[t], w_ref[t]], [config.resolution.dx, config.resolution.dy, config.resolution.dz], boundary_mask))
        div_ref[t,2] = (calculate_divergence([u_ref[t], v_ref[t], w_ref[t]], [config.resolution.dx, config.resolution.dy, config.resolution.dz], core_mask))
        div_ref[t,3] = (calculate_divergence([u_ref[t], v_ref[t], w_ref[t]], [config.resolution.dx, config.resolution.dy, config.resolution.dz], nf_mask))

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

    print('Total avg')
    rel_err_tot = np.mean(rel_err, axis=0)
    print(f'Relative error [Fluid] {rel_err_tot[0]:.1f}')
    print(f'Relative error [Bound] {rel_err_tot[1]:.1f}')
    print(f'Relative error [Core] {rel_err_tot[2]:.1f}')

    abs_err_tot = np.mean(abs_err, axis=0)
    print(f'Absolute error [Fluid] {abs_err_tot[0]:.4f}')
    print(f'Absolute error [Bound] {abs_err_tot[1]:.4f}')
    print(f'Absolute error [Core] {abs_err_tot[2]:.4f}')
    print(f'Absolute error [Non-F] {abs_err_tot[3]:.4f}')

    rmse_tot = np.mean(rmse, axis=0)
    print(f'R.M.S.   error [Fluid] {rmse_tot[0]:.4f}')
    print(f'R.M.S.   error [Bound] {rmse_tot[1]:.4f}')
    print(f'R.M.S.   error [Core] {rmse_tot[2]:.4f}')
    print(f'R.M.S.   error [Non-F] {rmse_tot[3]:.4f}')

    print(' ')
    print(config.predictions.peak_flow_idx, 'Peak')
    print(f'U [Fluid] k: {Ks[config.predictions.peak_flow_idx][0][0]:.4f} \t m: {Ms[config.predictions.peak_flow_idx][0][0]:.4f} \t r^2: {Rs[config.predictions.peak_flow_idx][0][0]:.4f}')
    print(f'  [Bound] k: {Ks[config.predictions.peak_flow_idx][0][1]:.4f} \t m: {Ms[config.predictions.peak_flow_idx][0][1]:.4f} \t r^2: {Rs[config.predictions.peak_flow_idx][0][1]:.4f}')
    print(f'  [Core] k: {Ks[config.predictions.peak_flow_idx][0][2]:.4f} \t m: {Ms[config.predictions.peak_flow_idx][0][2]:.4f} \t r^2: {Rs[config.predictions.peak_flow_idx][0][2]:.4f}')

    print(' ')
    print(f'V [Fluid] k: {Ks[config.predictions.peak_flow_idx][1][0]:.4f} \t m: {Ms[config.predictions.peak_flow_idx][1][0]:.4f} \t r^2: {Rs[config.predictions.peak_flow_idx][1][0]:.4f}')
    print(f'  [Bound] k: {Ks[config.predictions.peak_flow_idx][1][1]:.4f} \t m: {Ms[config.predictions.peak_flow_idx][1][1]:.4f} \t r^2: {Rs[config.predictions.peak_flow_idx][1][1]:.4f}')
    print(f'  [Core] k: {Ks[config.predictions.peak_flow_idx][1][2]:.4f} \t m: {Ms[config.predictions.peak_flow_idx][1][2]:.4f} \t r^2: {Rs[config.predictions.peak_flow_idx][1][2]:.4f}')

    print(' ')
    print(f'W [Fluid] k: {Ks[config.predictions.peak_flow_idx][2][0]:.4f} \t m: {Ms[config.predictions.peak_flow_idx][2][0]:.4f} \t r^2: {Rs[config.predictions.peak_flow_idx][2][0]:.4f}')
    print(f'  [Bound] k: {Ks[config.predictions.peak_flow_idx][2][1]:.4f} \t m: {Ms[config.predictions.peak_flow_idx][2][1]:.4f} \t r^2: {Rs[config.predictions.peak_flow_idx][2][1]:.4f}')
    print(f'  [Core] k: {Ks[config.predictions.peak_flow_idx][2][2]:.4f} \t m: {Ms[config.predictions.peak_flow_idx][2][2]:.4f} \t r^2: {Rs[config.predictions.peak_flow_idx][2][2]:.4f}')

    if config.training.reference_gradients:
        print(' ')
        print(f'PX [Fluid] k: {Ks_pgrad[config.predictions.peak_flow_idx][0][0]:.4f} \t m: {Ms_pgrad[config.predictions.peak_flow_idx][0][0]:.4f} \t r^2: {Rs_pgrad[config.predictions.peak_flow_idx][0][0]:.4f}')
        print(f'   [Bound] k: {Ks_pgrad[config.predictions.peak_flow_idx][0][1]:.4f} \t m: {Ms_pgrad[config.predictions.peak_flow_idx][0][1]:.4f} \t r^2: {Rs_pgrad[config.predictions.peak_flow_idx][0][1]:.4f}')
        print(f'   [Core] k: {Ks_pgrad[config.predictions.peak_flow_idx][0][2]:.4f} \t m: {Ms_pgrad[config.predictions.peak_flow_idx][0][2]:.4f} \t r^2: {Rs_pgrad[config.predictions.peak_flow_idx][0][2]:.4f}')

        print(' ')
        print(f'PY [Fluid] k: {Ks_pgrad[config.predictions.peak_flow_idx][1][0]:.4f} \t m: {Ms_pgrad[config.predictions.peak_flow_idx][1][0]:.4f} \t r^2: {Rs_pgrad[config.predictions.peak_flow_idx][1][0]:.4f}')
        print(f'   [Bound] k: {Ks_pgrad[config.predictions.peak_flow_idx][1][1]:.4f} \t m: {Ms_pgrad[config.predictions.peak_flow_idx][1][1]:.4f} \t r^2: {Rs_pgrad[config.predictions.peak_flow_idx][1][1]:.4f}')
        print(f'   [Core] k: {Ks_pgrad[config.predictions.peak_flow_idx][1][2]:.4f} \t m: {Ms_pgrad[config.predictions.peak_flow_idx][1][2]:.4f} \t r^2: {Rs_pgrad[config.predictions.peak_flow_idx][1][2]:.4f}')

        print(' ')
        print(f'PZ [Fluid] k: {Ks_pgrad[config.predictions.peak_flow_idx][2][0]:.4f} \t m: {Ms_pgrad[config.predictions.peak_flow_idx][2][0]:.4f} \t r^2: {Rs_pgrad[config.predictions.peak_flow_idx][2][0]:.4f}')
        print(f'   [Bound] k: {Ks_pgrad[config.predictions.peak_flow_idx][2][1]:.4f} \t m: {Ms_pgrad[config.predictions.peak_flow_idx][2][1]:.4f} \t r^2: {Rs_pgrad[config.predictions.peak_flow_idx][2][1]:.4f}')
        print(f'   [Core] k: {Ks_pgrad[config.predictions.peak_flow_idx][2][2]:.4f} \t m: {Ms_pgrad[config.predictions.peak_flow_idx][2][2]:.4f} \t r^2: {Rs_pgrad[config.predictions.peak_flow_idx][2][2]:.4f}')

        print(' ')

        grad_abs_err_tot = np.mean(grad_abs_err, axis=0)
        print(f'Absolute error Pressure Gradient [Fluid] {grad_abs_err_tot[0]:.4f}')
        print(f'Absolute error Pressure Gradient [Bound] {grad_abs_err_tot[1]:.4f}')
        print(f'Absolute error Pressure Gradient [Core] {grad_abs_err_tot[2]:.4f}')

        grad_rel_err_tot = np.mean(grad_rel_err, axis=0)
        print(f'Relative error Pressure Gradient [Fluid] {grad_rel_err_tot[0]*100:.4f} %')
        print(f'Relative error Pressure Gradient [Bound] {grad_rel_err_tot[1]*100:.4f} %')
        print(f'Relative error Pressure Gradient [Core] {grad_rel_err_tot[2]*100:.4f} %')

        grad_nrmse_tot = np.mean(grad_nrmse, axis=0)
        print(f'Pressure Gradient NRMSE [Fluid] {grad_nrmse_tot[0]*100:.2f} %')
        print(f'Pressure Gradient NRMSE [Bound] {grad_nrmse_tot[1]*100:.2f} %')
        print(f'Pressure Gradient NRMSE [Core] {grad_nrmse_tot[2]*100:.2f} %')

        grad_dir_err_tot = np.mean(grad_dir_err, axis=0)
        print(f'Pressure Gradient Directional Error [Fluid] {grad_dir_err_tot[0]:.2f} deg')
        print(f'Pressure Gradient Directional Error [Bound] {grad_dir_err_tot[1]:.2f} deg')
        print(f'Pressure Gradient Directional Error [Core] {grad_dir_err_tot[2]:.2f} deg')

    # Save metrics to csv
    metrics = {
        'Relative error [Fluid]': rel_err_tot[0],
        'Relative error [Bound]': rel_err_tot[1],
        'Relative error [Core]': rel_err_tot[2],

        'Absolute error [Fluid]': abs_err_tot[0],
        'Absolute error [Bound]': abs_err_tot[1],
        'Absolute error [Core]': abs_err_tot[2],
        'Absolute error [Non-F]': abs_err_tot[3],

        'R.M.S. error [Fluid]': rmse_tot[0],
        'R.M.S. error [Bound]': rmse_tot[1],
        'R.M.S. error [Core]': rmse_tot[2],
        'R.M.S. error [Non-F]': rmse_tot[3],

        'VNRMSE [Fluid]': vnrmse[0,0],
        'VNRMSE [Bound]': vnrmse[0,1],
        'VNRMSE [Core]': vnrmse[0,2],
        'VNRMSE [Non-F]': vnrmse[0,3],

        'Directional error [Fluid]': d_error[0,0],
        'Directional error [Bound]': d_error[0,1],
        'Directional error [Core]': d_error[0,2],
        'Directional error [Non-F]': d_error[0,3],

        'Divergence prediction [Fluid]': div_pred[0,0],
        'Divergence prediction [Bound]': div_pred[0,1],
        'Divergence prediction [Core]': div_pred[0,2],
        'Divergence prediction [Non-F]': div_pred[0,3],

        'Divergence reference [Fluid]': div_ref[0,0],
        'Divergence reference [Bound]': div_ref[0,1],
        'Divergence reference [Core]': div_ref[0,2],
        'Divergence reference [Non-F]': div_ref[0,3],

        'U k [Core]': Ks[config.predictions.peak_flow_idx][0][2],
        'U m [Core]': Ms[config.predictions.peak_flow_idx][0][2],
        'U r^2 [Core]': Rs[config.predictions.peak_flow_idx][0][2],
        'V k [Core]': Ks[config.predictions.peak_flow_idx][1][2],
        'V m [Core]': Ms[config.predictions.peak_flow_idx][1][2],
        'V r^2 [Core]': Rs[config.predictions.peak_flow_idx][1][2],
        'W k [Core]': Ks[config.predictions.peak_flow_idx][2][2],
        'W m [Core]': Ms[config.predictions.peak_flow_idx][2][2],
        'W r^2 [Core]': Rs[config.predictions.peak_flow_idx][2][2],
    }

    if config.training.reference_gradients:
        metrics.update({
            'Absolute error Pressure Gradient [Fluid]': grad_abs_err_tot[0],
            'Absolute error Pressure Gradient [Bound]': grad_abs_err_tot[1],
            'Absolute error Pressure Gradient [Core]': grad_abs_err_tot[2],

            'Relative error Pressure Gradient (%) [Fluid]': grad_rel_err_tot[0]*100,
            'Relative error Pressure Gradient (%) [Bound]': grad_rel_err_tot[1]*100,
            'Relative error Pressure Gradient (%) [Core]': grad_rel_err_tot[2]*100,

            'Pressure Gradient NRMSE (%) [Fluid]': grad_nrmse_tot[0]*100,
            'Pressure Gradient NRMSE (%) [Bound]': grad_nrmse_tot[1]*100,
            'Pressure Gradient NRMSE (%) [Core]': grad_nrmse_tot[2]*100,

            'Pressure Gradient Directional Error [Fluid]': grad_dir_err_tot[0],
            'Pressure Gradient Directional Error [Bound]': grad_dir_err_tot[1],
            'Pressure Gradient Directional Error [Core]': grad_dir_err_tot[2],

            'PX k [Core]': Ks_pgrad[config.predictions.peak_flow_idx][0][2],
            'PX m [Core]': Ms_pgrad[config.predictions.peak_flow_idx][0][2],
            'PX r^2 [Core]': Rs_pgrad[config.predictions.peak_flow_idx][0][2],
            'PY k [Core]': Ks_pgrad[config.predictions.peak_flow_idx][1][2],
            'PY m [Core]': Ms_pgrad[config.predictions.peak_flow_idx][1][2],
            'PY r^2 [Core]': Rs_pgrad[config.predictions.peak_flow_idx][1][2],
            'PZ k [Core]': Ks_pgrad[config.predictions.peak_flow_idx][2][2],
            'PZ m [Core]': Ms_pgrad[config.predictions.peak_flow_idx][2][2],
            'PZ r^2 [Core]': Rs_pgrad[config.predictions.peak_flow_idx][2][2],
        })

    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    metrics_filename = f"{directory}/metrics.csv"
    metrics_df.to_csv(metrics_filename, index=False)

    return metrics

def plot_predictions_vs_reference(config, model, device, it, xyz_ref, u_lr, v_lr, w_lr, p_lr, u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask_ref, mask_flat_ref, U_max, standardization_factors):

    # Create directory
    directory = f'{config.log_dir}/errors/iter_{it}'
    if not os.path.exists(directory):
        os.makedirs(directory)

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

    if config.plot.fluid_region:
        fluid_indices = mask_flat_ref==1
        uvw_pred_full = np.zeros(((len(mask_flat_ref), len(uvw_pred[0]))))
        uvw_pred_full[fluid_indices] = uvw_pred
        uvw_pred = uvw_pred_full

    # Denormalize predictions
    if config.plot.denormalize:
        if config.vel_normalization == "characteristic":
            uvw_pred[:, 0] *= config.constants.U  # u
            uvw_pred[:, 1] *= config.constants.U  # v
            uvw_pred[:, 2] *= config.constants.U  # w
            #if config.setup.include_pressure:
            #    uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2)  # p
            if (config.setup.include_pressure and config.training.reference_gradients):
                _, _, _, std_x, _, std_y, _, std_z = standardization_factors
                print(std_x, std_y, std_z)

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
                print(std_x, std_y, std_z)

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

    if config.setup.include_time:
        u_pred = uvw_pred[config.plot.t_step, :, :, :, 0]
        v_pred = uvw_pred[config.plot.t_step, :, :, :, 1]
        w_pred = uvw_pred[config.plot.t_step, :, :, :, 2]
        #p_pred = uvw_pred[config.plot.t_stepr, :, :, :, 3] if config.setup.include_pressure else None
        px_pred = uvw_pred[config.plot.t_step, :, :, :, 3] if config.training.reference_gradients else None
        py_pred = uvw_pred[config.plot.t_step, :, :, :, 4] if config.training.reference_gradients else None
        pz_pred = uvw_pred[config.plot.t_step, :, :, :, 5] if config.training.reference_gradients else None
        
        p_pred = uvw_pred[config.plot.t_step, :, :, :, 3] if (config.setup.include_pressure and not config.training.reference_gradients) else None 

    else:
        u_pred = uvw_pred[0, :, :, :, 0]
        v_pred = uvw_pred[0, :, :, :, 1]
        w_pred = uvw_pred[0, :, :, :, 2]
        #p_pred = uvw_pred[config.plot.t_step*config.ref_temporal_factor, :, :, :, 3] if config.setup.include_pressure else None
        px_pred = uvw_pred[0, :, :, :, 3] if config.training.reference_gradients else None
        py_pred = uvw_pred[0, :, :, :, 4] if config.training.reference_gradients else None
        pz_pred = uvw_pred[0, :, :, :, 5] if config.training.reference_gradients else None
        p_pred = uvw_pred[0, :, :, :, 3] if (config.setup.include_pressure and not config.training.reference_gradients) else None 

    if config.setup.include_time:
        t_step = config.plot.t_step
    else:
        t_step = 0
        u_lr = np.expand_dims(u_lr, axis=0)
        v_lr = np.expand_dims(v_lr, axis=0)
        w_lr = np.expand_dims(w_lr, axis=0)
        #p_lr = np.expand_dims(p_lr, axis=0) if config.setup.include_pressure else None
        #px_lr = np.expand_dims(px_lr, axis=0) if config.training.reference_gradients else None
        #py_lr = np.expand_dims(py_lr, axis=0) if config.training.reference_gradients else None
        #pz_lr = np.expand_dims(pz_lr, axis=0) if config.training.reference_gradients else None

        u_ref = np.expand_dims(u_ref, axis=0)
        v_ref = np.expand_dims(v_ref, axis=0)
        w_ref = np.expand_dims(w_ref, axis=0)
        #_ref = np.expand_dims(p_ref, axis=0) if config.setup.include_pressure else None
        px_ref = np.expand_dims(px_ref, axis=0) if config.training.reference_gradients else None
        py_ref = np.expand_dims(py_ref, axis=0) if config.training.reference_gradients else None
        pz_ref = np.expand_dims(pz_ref, axis=0) if config.training.reference_gradients else None
        p_ref = np.expand_dims(p_ref, axis=0) if config.setup.include_pressure and not config.training.reference_gradients else None

    # Create LR vs HR vs SR plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('LR u')
    plt.imshow(u_lr[int(t_step/2), :, :, config.plot.z_slice].T, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title('Reference u')
    plt.imshow(u_ref[t_step, :, :, config.plot.z_slice*config.ref_spatial_factor].T, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title('Predicted u')
    plt.imshow(u_pred[:, :, config.plot.z_slice*config.ref_spatial_factor].T, origin='lower', cmap='viridis')
    plt.colorbar()

    plt.savefig(os.path.join(directory, f"prediction_vs_reference_u.png"))
    plt.close()  

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('LR v')
    plt.imshow(v_lr[int(t_step/2), :, :, config.plot.z_slice].T, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title('Reference v')
    plt.imshow(v_ref[t_step, :, :, config.plot.z_slice*config.ref_spatial_factor].T, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title('Predicted v')
    plt.imshow(v_pred[:, :, config.plot.z_slice*config.ref_spatial_factor].T, origin='lower', cmap='viridis')
    plt.colorbar()

    plt.savefig(os.path.join(directory, f"prediction_vs_reference_v.png"))
    plt.close() 

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('LR w')
    plt.imshow(w_lr[int(t_step/2), :, :, config.plot.z_slice].T, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title('Reference w')
    plt.imshow(w_ref[t_step, :, :, config.plot.z_slice*config.ref_spatial_factor].T, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title('Predicted w')
    plt.imshow(w_pred[:, :, config.plot.z_slice*config.ref_spatial_factor].T, origin='lower', cmap='viridis')
    plt.colorbar()

    plt.savefig(os.path.join(directory, f"prediction_vs_reference_w.png"))
    plt.close()
    
    if config.setup.include_pressure and not config.training.reference_gradients:

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.title('Reference p')
        plt.imshow(p_ref[t_step, :, :, config.plot.z_slice*config.ref_spatial_factor].T, origin='lower', cmap='viridis')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title('Reference p')
        plt.imshow(p_ref[t_step, :, :, config.plot.z_slice*config.ref_spatial_factor], cmap='viridis'.T, origin='lower', vmin=14100)
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.title('Predicted p')
        plt.imshow(p_pred[:, :, config.plot.z_slice*config.ref_spatial_factor].T, origin='lower', cmap='viridis')
        plt.colorbar()

        plt.savefig(os.path.join(directory, f"prediction_vs_reference_p.png"))
        plt.close()
    
    if config.training.reference_gradients:

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Reference p')
        plt.imshow(px_ref[t_step, :, :, config.plot.z_slice*config.ref_spatial_factor].T, origin='lower', cmap='viridis')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title('Predicted p')
        plt.imshow(px_pred[:, :, config.plot.z_slice*config.ref_spatial_factor].T, origin='lower', cmap='viridis')
        plt.colorbar()
        plt.savefig(os.path.join(directory, f"prediction_vs_reference_px.png"))
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Reference p')
        plt.imshow(py_ref[t_step, :, :, config.plot.z_slice*config.ref_spatial_factor].T, origin='lower', cmap='viridis')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title('Predicted p')
        plt.imshow(py_pred[:, :, config.plot.z_slice*config.ref_spatial_factor].T, origin='lower', cmap='viridis')
        plt.colorbar()
        plt.savefig(os.path.join(directory, f"prediction_vs_reference_py.png"))
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Reference p')
        plt.imshow(pz_ref[t_step, :, :, config.plot.z_slice*config.ref_spatial_factor].T, origin='lower', cmap='viridis')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title('Predicted p')
        plt.imshow(pz_pred[:, :, config.plot.z_slice*config.ref_spatial_factor].T, origin='lower', cmap='viridis')
        plt.colorbar()
        plt.savefig(os.path.join(directory, f"prediction_vs_reference_pz.png"))
        plt.close()

    # with h5py.File(f"{config.log_dir}/pred.h5", 'w') as f:
    #     f.create_dataset('u', data=u_pred)
    #     f.create_dataset('v', data=v_pred)
    #     f.create_dataset('w', data=w_pred)

    return

def plot_predictions(config, model, device, it, u, mask, U_max):
    
    # Create directory
    directory = f'{config.log_dir}/plots/iter_{it}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Extract boundaries
    if config.plot.expand_mask:
        boundary_mask = compute_outer_boundary_mask(mask)
        mask = mask + boundary_mask

    if config.setup.include_time:
        t_len, x_len, y_len, z_len = u.shape
    else:
        x_len, y_len, z_len = u.shape
        t_len = 1

    t_normalized, x_normalized, y_normalized, z_normalized, standardization_factors = create_and_normalize_coords(config, t_len, x_len, y_len, z_len)

    # Upsample each coordinate
    t_ups = upsample_1d(t_normalized, config.plot.temporal_factor, config.plot.temp_upsampling_mode) if config.setup.include_time else []
    x_ups = upsample_1d(x_normalized, config.plot.spatial_factor, mode=config.plot.spat_upsampling_mode)
    y_ups = upsample_1d(y_normalized, config.plot.spatial_factor, mode=config.plot.spat_upsampling_mode)
    z_ups = upsample_1d(z_normalized, config.plot.spatial_factor, mode=config.plot.spat_upsampling_mode)
    
    if config.setup.include_time:
        grids = np.meshgrid(t_ups, x_ups, y_ups, z_ups, indexing='ij')
    else:
        grids = np.meshgrid(x_ups, y_ups, z_ups, indexing='ij')
    
    flat_coords = [grid.ravel() for grid in grids]
    xyz_plot_full = np.stack(flat_coords, axis=-1) 

    if config.plot.fluid_region:
        # Upsample mask
        mask_plot = zoom(mask, zoom=config.plot.spatial_factor, order=0, grid_mode=True, mode='nearest')
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

    if config.plot.fluid_region:
        uvw_pred_full = np.zeros(((len(xyz_plot_full), len(uvw_pred_plot[0])))) + config.plot.non_fluid_value
        uvw_pred_full[fluid_indices] = uvw_pred_plot
        uvw_pred_plot = uvw_pred_full

    if config.setup.include_time:
        uvw_pred_plot = uvw_pred_plot.reshape(len(t_ups), len(x_ups), len(y_ups), len(z_ups), len(uvw_pred_plot[0]))
    else:
        uvw_pred_plot = uvw_pred_plot.reshape(1, len(x_ups), len(y_ups), len(z_ups), len(uvw_pred_plot[0]))

    if config.setup.include_time:
        u_pred = uvw_pred_plot[config.plot.t_step_2, :, :, :, 0]
        v_pred = uvw_pred_plot[config.plot.t_step_2, :, :, :, 1]
        w_pred = uvw_pred_plot[config.plot.t_step_2, :, :, :, 2]
        p_pred = uvw_pred_plot[config.plot.t_step_2, :, :, :, 3] if config.setup.include_pressure else None
    else:
        u_pred = uvw_pred_plot[0, :, :, :, 0]
        v_pred = uvw_pred_plot[0, :, :, :, 1]
        w_pred = uvw_pred_plot[0, :, :, :, 2]
        p_pred = uvw_pred_plot[0, :, :, :, 3] if config.setup.include_pressure else None

    # Denormalize predictions
    if config.plot.denormalize:
        if config.vel_normalization == "characteristic":
            u_pred = u_pred*config.constants.U
            v_pred = v_pred*config.constants.U
            w_pred = w_pred*config.constants.U
            p_pred = p_pred*(config.constants.rho*(config.constants.U**2)) if config.setup.include_pressure else None
        elif config.vel_normalization == "max_velocity":
            u_pred = u_pred*U_max
            v_pred = v_pred*U_max
            w_pred = w_pred*U_max
            p_pred = p_pred*(config.constants.rho*(config.constants.U**2)) if config.setup.include_pressure else None
    
    z_slice = config.plot.z_slice

    # Plotting (example using matplotlib)
    plt.figure(figsize=(12, 6))


    plt.subplot(2, 2, 1)
    plt.title('Predicted u')
    plt.imshow(u_pred[:, :, z_slice].T, origin='lower', extent=[x_ups.min(), x_ups.max(), y_ups.min(), y_ups.max()])
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title('Predicted v')
    plt.imshow(v_pred[:, :, z_slice].T, origin='lower', extent=[x_ups.min(), x_ups.max(), y_ups.min(), y_ups.max()])
    plt.colorbar()
    
    plt.subplot(2, 2, 3)
    plt.title('Predicted w')
    plt.imshow(w_pred[:, :, z_slice].T, origin='lower', extent=[x_ups.min(), x_ups.max(), y_ups.min(), y_ups.max()])
    plt.colorbar()

    if config.setup.include_pressure and p_pred is not None:
        plt.subplot(2, 2, 4)
        plt.title('Predicted px')
        plt.imshow(p_pred[:, :, z_slice].T, origin='lower', extent=[x_ups.min(), x_ups.max(), y_ups.min(), y_ups.max()])
        plt.colorbar()

    plt.savefig(os.path.join(directory, f"predictions.png"))
    plt.close()

    return