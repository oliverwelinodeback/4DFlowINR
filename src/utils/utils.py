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
from utils.evaluation_utils import create_boundary_and_core_masks, calculate_relative_error, calculate_absolute_error, calculate_rmse, calculate_absolute_error_pressure, calculate_rmse_pressure
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

def evaluate_predictions(config, model, device, it, xyz_ref, u_ref, v_ref, w_ref, p_ref, mask_ref, mask_flat_ref, U_max):

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
            if config.setup.include_pressure:
                uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2)  # p
        elif config.vel_normalization == "max_velocity":
            uvw_pred[:, 0] *= U_max  # u
            uvw_pred[:, 1] *= U_max  # v
            uvw_pred[:, 2] *= U_max  # w
            if config.setup.include_pressure:
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


    D_pred = uvw_pred.shape[1]

    uvw_pred = uvw_pred.reshape(T, X, Y, Z, D_pred)

    u_pred = uvw_pred[:, :, :, :, 0]
    v_pred = uvw_pred[:, :, :, :, 1]
    w_pred = uvw_pred[:, :, :, :, 2]
    p_pred = uvw_pred[:, :, :, :, 3] if config.setup.include_pressure else None

    # Get metrics
    T = len(u_pred)
    nf_mask = 1.0 - mask_ref
    boundary_mask, core_mask = create_boundary_and_core_masks(mask_ref, 0.1, 'voxels')

    rel_err = np.zeros((T,3))
    abs_err = np.zeros((T,5))
    rmse = np.zeros((T,5))

    for t in range(T):
        rel_err[t,0] = (calculate_relative_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
        rel_err[t,1] = (calculate_relative_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
        rel_err[t,2] = (calculate_relative_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], core_mask))

        abs_err[t,0] = (calculate_absolute_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
        abs_err[t,1] = (calculate_absolute_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
        abs_err[t,2] = (calculate_absolute_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], core_mask))
        abs_err[t,3] = (calculate_absolute_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], nf_mask))
        #abs_err[t,4] = (calculate_absolute_error_pressure(p_pred[t], p_ref[t], mask_ref))

        rmse[t,0] = (calculate_rmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
        rmse[t,1] = (calculate_rmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
        rmse[t,2] = (calculate_rmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], core_mask))
        rmse[t,3] = (calculate_rmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], nf_mask))
        #rmse[t,4] = (calculate_rmse_pressure(p_pred[t], p_ref[t], mask_ref))
    
        # New metrics
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
    #print(f'Absolute error Pressure [Fluid] {abs_err_tot[4]:.4f}')

    rmse_tot = np.mean(rmse, axis=0)
    print(f'R.M.S.   error [Fluid] {rmse_tot[0]:.4f}')
    print(f'R.M.S.   error [Bound] {rmse_tot[1]:.4f}')
    print(f'R.M.S.   error [Core] {rmse_tot[2]:.4f}')
    print(f'R.M.S.   error [Non-F] {rmse_tot[3]:.4f}')
    #print(f'R.M.S.   error Pressure [Fluid] {rmse_tot[4]:.4f}')

    # Save metrics to csv
    metrics = {
        'Relative error [Fluid]': rel_err_tot[0],
        'Relative error [Bound]': rel_err_tot[1],
        'Relative error [Core]': rel_err_tot[2],

        'Absolute error [Fluid]': abs_err_tot[0],
        'Absolute error [Bound]': abs_err_tot[1],
        'Absolute error [Core]': abs_err_tot[2],
        'Absolute error [Non-F]': abs_err_tot[3],
        #'Absolute error Pressure [Fluid]': abs_err_tot[4],

        'R.M.S. error [Fluid]': rmse_tot[0],
        'R.M.S. error [Bound]': rmse_tot[1],
        'R.M.S. error [Core]': rmse_tot[2],
        'R.M.S. error [Non-F]': rmse_tot[3],
        #'R.M.S. error Pressure [Fluid]': rmse_tot[4],
    }

    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    metrics_filename = f"{directory}/metrics.csv"
    metrics_df.to_csv(metrics_filename, index=False)

    return 

def plot_predictions_vs_reference(config, model, device, it, xyz_ref, u_lr, v_lr, w_lr, p_lr, u_ref, v_ref, w_ref, p_ref, mask_ref, mask_flat_ref, U_max):

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
            if config.setup.include_pressure:
                uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2)  # p
        elif config.vel_normalization == "max_velocity":
            uvw_pred[:, 0] *= U_max  # u
            uvw_pred[:, 1] *= U_max  # v
            uvw_pred[:, 2] *= U_max  # w
            if config.setup.include_pressure:
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
        u_pred = uvw_pred[config.plot.t_step*config.ref_temporal_factor, :, :, :, 0]
        v_pred = uvw_pred[config.plot.t_step*config.ref_temporal_factor, :, :, :, 1]
        w_pred = uvw_pred[config.plot.t_step*config.ref_temporal_factor, :, :, :, 2]
        p_pred = uvw_pred[config.plot.t_step*config.ref_temporal_factor, :, :, :, 3] if config.setup.include_pressure else None
    else:
        u_pred = uvw_pred[0, :, :, :, 0]
        v_pred = uvw_pred[0, :, :, :, 1]
        w_pred = uvw_pred[0, :, :, :, 2]
        p_pred = uvw_pred[0, :, :, :, 3] if config.setup.include_pressure else None

    if config.setup.include_time:
        t_step = config.plot.t_step
    else:
        t_step = 0
        u_lr = np.expand_dims(u_lr, axis=0)
        v_lr = np.expand_dims(v_lr, axis=0)
        w_lr = np.expand_dims(w_lr, axis=0)
        p_lr = np.expand_dims(p_lr, axis=0) if config.setup.include_pressure else None

        u_ref = np.expand_dims(u_ref, axis=0)
        v_ref = np.expand_dims(v_ref, axis=0)
        w_ref = np.expand_dims(w_ref, axis=0)
        p_ref = np.expand_dims(p_ref, axis=0) if config.setup.include_pressure else None

    # Create LR vs HR vs SR plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('LR u')
    plt.imshow(u_lr[t_step, :, :, config.plot.z_slice], cmap='viridis')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title('Reference u')
    plt.imshow(u_ref[t_step*config.ref_temporal_factor, :, :, config.plot.z_slice*config.ref_spatial_factor], cmap='viridis')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title('Predicted u')
    plt.imshow(u_pred[:, :, config.plot.z_slice*config.ref_spatial_factor], cmap='viridis')
    plt.colorbar()

    plt.savefig(os.path.join(directory, f"prediction_vs_reference_u.png"))
    plt.close()  

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('LR v')
    plt.imshow(v_lr[t_step, :, :, config.plot.z_slice], cmap='viridis')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title('Reference v')
    plt.imshow(v_ref[t_step*config.ref_temporal_factor, :, :, config.plot.z_slice*config.ref_spatial_factor], cmap='viridis')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title('Predicted v')
    plt.imshow(v_pred[:, :, config.plot.z_slice*config.ref_spatial_factor], cmap='viridis')
    plt.colorbar()

    plt.savefig(os.path.join(directory, f"prediction_vs_reference_v.png"))
    plt.close() 

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('LR w')
    plt.imshow(w_lr[t_step, :, :, config.plot.z_slice], cmap='viridis')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title('Reference w')
    plt.imshow(w_ref[t_step*config.ref_temporal_factor, :, :, config.plot.z_slice*config.ref_spatial_factor], cmap='viridis')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title('Predicted w')
    plt.imshow(w_pred[:, :, config.plot.z_slice*config.ref_spatial_factor], cmap='viridis')
    plt.colorbar()

    plt.savefig(os.path.join(directory, f"prediction_vs_reference_w.png"))
    plt.close()

    if config.setup.include_pressure:

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.title('LR p')
        plt.imshow(p_lr[t_step, :, :, config.plot.z_slice], cmap='viridis')
        plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.title('Reference p')
        plt.imshow(p_ref[t_step*config.ref_temporal_factor, :, :, config.plot.z_slice*config.ref_spatial_factor], cmap='viridis')
        plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.title('Predicted p')
        plt.imshow(p_pred[:, :, config.plot.z_slice*config.ref_spatial_factor], cmap='viridis')
        plt.colorbar()

        plt.savefig(os.path.join(directory, f"prediction_vs_reference_p.png"))
        plt.close()
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
        u_pred = uvw_pred_plot[config.plot.t_step*config.plot.temporal_factor, :, :, :, 0]
        v_pred = uvw_pred_plot[config.plot.t_step*config.plot.temporal_factor, :, :, :, 1]
        w_pred = uvw_pred_plot[config.plot.t_step*config.plot.temporal_factor, :, :, :, 2]
        p_pred = uvw_pred_plot[config.plot.t_step*config.plot.temporal_factor, :, :, :, 3] if config.setup.include_pressure else None
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
    
    z_slice = config.plot.z_slice*config.plot.spatial_factor

    # Plotting (example using matplotlib)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Predicted w')
    plt.imshow(w_pred[:, :, z_slice], origin='lower', extent=[x_ups.min(), x_ups.max(), y_ups.min(), y_ups.max()])
    plt.colorbar()

    if config.setup.include_pressure and p_pred is not None:
        plt.subplot(1, 2, 2)
        plt.title('Predicted p')
        plt.imshow(p_pred[:, :, z_slice], origin='lower', extent=[x_ups.min(), x_ups.max(), y_ups.min(), y_ups.max()])
        plt.colorbar()

    plt.savefig(os.path.join(directory, f"predictions.png"))
    plt.close()

    return