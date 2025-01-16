# Imports
import os
import time
import numpy as np
import torch
import pandas as pd
from scipy.ndimage import zoom
from utils.prepare_data import create_and_normalize_coords, upsample_1d, extract_fluid_region, compute_outer_boundary_mask
from utils.evaluation_utils import (
    create_boundary_and_core_masks, calculate_relative_error, calculate_absolute_error, 
    calculate_rmse, calculate_absolute_error_pressure, calculate_rmse_pressure, linreg)
from utils.prepare_data import prepare_data, load_data, extract_fluid_region, load_ref_data, prepare_ref_data
from utils.utils import save_to_h5, h5_to_paraview
from utils.preprocessing_utils import compute_outer_boundary_mask
import SIREN
from configs.SIREN_x2 import get_config
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    print("Starting script")

    config = get_config()

    # Path to stored weights
    network_path = "../models/250115_Tests/SIREN_x2_20250115-1656/SIREN_x2_final.pth"
    results_directory = "../results/250115_Tests/SIREN_x2_20250115-1656"
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    # Load data
    u, v, w, p, mask = load_data(config)

    # Save noisy data truth to results directory
    #save_to_h5(f"{results_directory}/healthy-05mm3_LR_SNR5_x1.h5", "u", u*mask)
    #save_to_h5(f"{results_directory}/healthy-05mm3_LR_SNR5_x1.h5", "v", v*mask)
    #save_to_h5(f"{results_directory}/healthy-05mm3_LR_SNR5_x1.h5", "w", w*mask)
    #save_to_h5(f"{results_directory}/healthy-05mm3_LR_SNR5_x1.h5", "p", p*mask)
    save_to_h5(f"{results_directory}/healthy-05mm3_LR_dv_241211.h5", "u", u*mask)
    save_to_h5(f"{results_directory}/healthy-05mm3_LR_dv_241211.h5", "v", v*mask)
    save_to_h5(f"{results_directory}/healthy-05mm3_LR_dv_241211.h5", "w", w*mask)
    #save_to_h5(f"{results_directory}/healthy-05mm3_LR_dv_241211.h5", "p", p*mask)

    # Save to vtk file
    #h5_to_paraview(u, v, w, p, (config.resolution.dx, config.resolution.dy, config.resolution.dz), f"{results_directory}/healthy-05mm3_LR_SNR5_x1.vti")

    # Prepare data
    uvw_data, xyz_data, mask_flat, boundary_mask_flat, standardization_factors, U_max  = prepare_data(config, u, v, w, p, mask)

    # Load and prepare reference data
    if config.include_ref:
        u_ref, v_ref, w_ref, p_ref, mask_ref = load_ref_data(config)
        xyz_data_ref, mask_flat_ref, boundary_mask_flat_ref = prepare_ref_data(config, u, mask_ref)

        # Save noisy data truth to results directory
        save_to_h5(f"{results_directory}/healthy-05mm3.h5", "u", u_ref)
        save_to_h5(f"{results_directory}/healthy-05mm3.h5", "v", v_ref)
        save_to_h5(f"{results_directory}/healthy-05mm3.h5", "w", w_ref)
        #save_to_h5(f"{results_directory}/healthy-05mm3.h5", "p", p_ref)

    # Include fluid region data
    if config.setup.fluid_region:
        uvw_train, xyz_train = extract_fluid_region(uvw_data, xyz_data, mask_flat)
        if config.include_ref:
            xyz_ref = xyz_data_ref[mask_flat_ref==1]
    else:
        uvw_train, xyz_train = uvw_data, xyz_data
        if config.include_ref:
            xyz_ref = xyz_data_ref

    # Initialize network
    DEVICE = torch.device('cuda')
    model = SIREN.SIREN(
        in_dim=config.network.in_dim,
        out_dim=config.network.out_dim,
        depth=config.network.depth,
        hidden_features=config.network.hidden_features,
        first_omega_0=config.network.first_omega_0,
        hidden_omega_0=config.network.hidden_omega_0
    ).to(DEVICE)

    # Load trained model
    checkpoint = torch.load(network_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Predict and compare with reference data
    if config.predictions.predict_reference_data:

        # Create directory
        ref_directory = f'{results_directory}/ref_data'
        if not os.path.exists(ref_directory):
            os.makedirs(ref_directory)

        # Predict reference coordinates
        model.eval()
        with torch.no_grad():
            xyz_ref = torch.from_numpy(xyz_ref).float().to(DEVICE)
            uvw_pred = model(xyz_ref)  # shape (N_fluid, out_dim)

        # Detach
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

        u_pred = uvw_pred[:, :, :, :, 0]
        v_pred = uvw_pred[:, :, :, :, 1]
        w_pred = uvw_pred[:, :, :, :, 2]
        p_pred = uvw_pred[:, :, :, :, 3] if config.setup.include_pressure else None

        # Save ref predictions to results directory
        save_to_h5(f"{ref_directory}/healthy-05mm3_SR.h5", "u", u_pred)
        save_to_h5(f"{ref_directory}/healthy-05mm3_SR.h5", "v", v_pred)
        save_to_h5(f"{ref_directory}/healthy-05mm3_SR.h5", "w", w_pred)
        #save_to_h5(f"{ref_directory}/healthy-05mm3_SR.h5", "p", p_pred)

        # Get metrics
        peak_flow_idx = config.predictions.peak_flow_idx
        T = len(u_pred)
        nf_mask = 1.0 - mask_ref
        boundary_mask, core_mask = create_boundary_and_core_masks(mask_ref, 0.1, 'voxels')

        X,Y,Z = mask_ref.shape
        cov_a = np.sum(mask_ref)/(X*Y*Z)
        cov_b = np.sum(boundary_mask)/(X*Y*Z)
        cov_c = np.sum(core_mask)/(X*Y*Z)
        ratio_b = np.sum(boundary_mask)/np.sum(mask_ref)
        ratio_c = np.sum(core_mask)/np.sum(mask_ref)

        print(' ')
        print(f'Coverage: {100*cov_a:.3f} %')
        print(f'Boundary --- cov: {100*cov_b:.3f} %, ratio: {100*ratio_b:.3f} %')
        print(f'Core --- cov: {100*cov_c:.3f} %, ratio: {100*ratio_c:.3f} %')

        rel_err = np.zeros((T,3))
        abs_err = np.zeros((T,5))
        rmse = np.zeros((T,5))

        Ks = np.zeros((T,3,3))
        Ms = np.zeros((T,3,3))
        Rs = np.zeros((T,3,3))

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

            Ks[t][0][0], Ms[t][0][0], Rs[t][0][0] = linreg(u_pred[t], u_ref[t], mask_ref)
            Ks[t][1][0], Ms[t][1][0], Rs[t][1][0] = linreg(v_pred[t], v_ref[t], mask_ref)
            Ks[t][2][0], Ms[t][2][0], Rs[t][2][0] = linreg(w_pred[t], w_ref[t], mask_ref)

            Ks[t][0][1], Ms[t][0][1], Rs[t][0][1] = linreg(u_pred[t], u_ref[t], boundary_mask)
            Ks[t][1][1], Ms[t][1][1], Rs[t][1][1] = linreg(v_pred[t], v_ref[t], boundary_mask)
            Ks[t][2][1], Ms[t][2][1], Rs[t][2][1] = linreg(w_pred[t], w_ref[t], boundary_mask)

            Ks[t][0][2], Ms[t][0][2], Rs[t][0][2] = linreg(u_pred[t], u_ref[t], core_mask)
            Ks[t][1][2], Ms[t][1][2], Rs[t][1][2] = linreg(v_pred[t], v_ref[t], core_mask)
            Ks[t][2][2], Ms[t][2][2], Rs[t][2][2] = linreg(w_pred[t], w_ref[t], core_mask)
        
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
        print(f'Absolute error Pressure [Fluid] {abs_err_tot[4]:.4f}')

        rmse_tot = np.mean(rmse, axis=0)
        print(f'R.M.S.   error [Fluid] {rmse_tot[0]:.4f}')
        print(f'R.M.S.   error [Bound] {rmse_tot[1]:.4f}')
        print(f'R.M.S.   error [Core] {rmse_tot[2]:.4f}')
        print(f'R.M.S.   error [Non-F] {rmse_tot[3]:.4f}')
        print(f'R.M.S.   error Pressure [Fluid] {rmse_tot[4]:.4f}')

        print(' ')
        print(f'U [Fluid] k: {Ks[peak_flow_idx][0][0]:.4f} \t m: {Ms[peak_flow_idx][0][0]:.4f} \t r^2: {Rs[peak_flow_idx][0][0]:.4f}')
        print(f'  [Bound] k: {Ks[peak_flow_idx][0][1]:.4f} \t m: {Ms[peak_flow_idx][0][1]:.4f} \t r^2: {Rs[peak_flow_idx][0][1]:.4f}')
        print(f'  [Core] k: {Ks[peak_flow_idx][0][2]:.4f} \t m: {Ms[peak_flow_idx][0][2]:.4f} \t r^2: {Rs[peak_flow_idx][0][2]:.4f}')

        print(' ')
        print(f'V [Fluid] k: {Ks[peak_flow_idx][1][0]:.4f} \t m: {Ms[peak_flow_idx][1][0]:.4f} \t r^2: {Rs[peak_flow_idx][1][0]:.4f}')
        print(f'  [Bound] k: {Ks[peak_flow_idx][1][1]:.4f} \t m: {Ms[peak_flow_idx][1][1]:.4f} \t r^2: {Rs[peak_flow_idx][1][1]:.4f}')
        print(f'  [Core] k: {Ks[peak_flow_idx][1][2]:.4f} \t m: {Ms[peak_flow_idx][1][2]:.4f} \t r^2: {Rs[peak_flow_idx][1][2]:.4f}')

        print(' ')
        print(f'W [Fluid] k: {Ks[peak_flow_idx][2][0]:.4f} \t m: {Ms[peak_flow_idx][2][0]:.4f} \t r^2: {Rs[peak_flow_idx][2][0]:.4f}')
        print(f'  [Bound] k: {Ks[peak_flow_idx][2][1]:.4f} \t m: {Ms[peak_flow_idx][2][1]:.4f} \t r^2: {Rs[peak_flow_idx][2][1]:.4f}')
        print(f'  [Core] k: {Ks[peak_flow_idx][2][2]:.4f} \t m: {Ms[peak_flow_idx][2][2]:.4f} \t r^2: {Rs[peak_flow_idx][2][2]:.4f}')


        # Save metrics to csv
        metrics = {
            'Relative error [Fluid]': rel_err_tot[0],
            'Relative error [Bound]': rel_err_tot[1],
            'Relative error [Core]': rel_err_tot[2],

            'Absolute error [Fluid]': abs_err_tot[0],
            'Absolute error [Bound]': abs_err_tot[1],
            'Absolute error [Core]': abs_err_tot[2],
            'Absolute error [Non-F]': abs_err_tot[3],
            'Absolute error Pressure [Fluid]': abs_err_tot[4],

            'R.M.S. error [Fluid]': rmse_tot[0],
            'R.M.S. error [Bound]': rmse_tot[1],
            'R.M.S. error [Core]': rmse_tot[2],
            'R.M.S. error [Non-F]': rmse_tot[3],
            'R.M.S. error Pressure [Fluid]': rmse_tot[4],

            'PEAK FLOW INDEX:': peak_flow_idx,
            'Relative error [Fluid] Peak': rel_err[peak_flow_idx][0],
            'Relative error [Bound] Peak': rel_err[peak_flow_idx][1],
            'Relative error [Core] Peak': rel_err[peak_flow_idx][2],
            'Absolute error [Fluid] Peak': abs_err[peak_flow_idx][0],
            'Absolute error [Bound] Peak': abs_err[peak_flow_idx][1],
            'Absolute error [Core] Peak': abs_err[peak_flow_idx][2],
            'Absolute error [Non-F] Peak': abs_err[peak_flow_idx][3],
            'R.M.S. error [Fluid] Peak': rmse[peak_flow_idx][0],
            'R.M.S. error [Bound] Peak': rmse[peak_flow_idx][1],
            'R.M.S. error [Core] Peak': rmse[peak_flow_idx][2],
            'R.M.S. error [Non-F] Peak': rmse[peak_flow_idx][3],

            'U [Fluid] k': Ks[peak_flow_idx][0][0],
            'U [Bound] k': Ks[peak_flow_idx][0][1],
            'U [Core] k': Ks[peak_flow_idx][0][2],
            'U [Fluid] m': Ms[peak_flow_idx][0][0],
            'U [Bound] m': Ms[peak_flow_idx][0][1],
            'U [Core] m': Ms[peak_flow_idx][0][2],
            'U [Fluid] r^2': Rs[peak_flow_idx][0][0],
            'U [Bound] r^2': Rs[peak_flow_idx][0][1],
            'U [Core] r^2': Rs[peak_flow_idx][0][2],

            'V [Fluid] k': Ks[peak_flow_idx][1][0],
            'V [Bound] k': Ks[peak_flow_idx][1][1],
            'V [Core] k': Ks[peak_flow_idx][1][2],
            'V [Fluid] m': Ms[peak_flow_idx][1][0],
            'V [Bound] m': Ms[peak_flow_idx][1][1],
            'V [Core] m': Ms[peak_flow_idx][1][2],
            'V [Fluid] r^2': Rs[peak_flow_idx][1][0],
            'V [Bound] r^2': Rs[peak_flow_idx][1][1],
            'V [Core] r^2': Rs[peak_flow_idx][1][2],


            'W [Fluid] k': Ks[peak_flow_idx][2][0],
            'W [Bound] k': Ks[peak_flow_idx][2][1],
            'W [Core] k': Ks[peak_flow_idx][2][2],
            'W [Fluid] m': Ms[peak_flow_idx][2][0],
            'W [Bound] m': Ms[peak_flow_idx][2][1],
            'W [Core] m': Ms[peak_flow_idx][2][2],
            'W [Fluid] r^2': Rs[peak_flow_idx][2][0],
            'W [Bound] r^2': Rs[peak_flow_idx][2][1],
            'W [Core] r^2': Rs[peak_flow_idx][2][2],

        }

        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        metrics_filename = f"{ref_directory}/metrics.csv"
        metrics_df.to_csv(metrics_filename, index=False)

    # Predict super-resolved data
    if config.predictions.predict_SR_data:

        # Create directory
        SR_directory = f'{results_directory}/SR_data'
        if not os.path.exists(SR_directory):
            os.makedirs(SR_directory)

        # Extract boundaries
        if config.plot.expand_mask:
            boundary_mask = compute_outer_boundary_mask(mask)
            mask = mask + boundary_mask

        t_len, x_len, y_len, z_len = u.shape
        t_normalized, x_normalized, y_normalized, z_normalized, standardization_factors = create_and_normalize_coords(config, t_len, x_len, y_len, z_len)

        # Upsample each coordinate
        t_ups = upsample_1d(t_normalized, config.plot.temporal_factor,'extend') if config.setup.include_time else []
        x_ups = upsample_1d(x_normalized, config.plot.spatial_factor, mode='extend')
        y_ups = upsample_1d(y_normalized, config.plot.spatial_factor, mode='extend')
        z_ups = upsample_1d(z_normalized, config.plot.spatial_factor, mode='extend')
        
        if config.setup.include_time:
            grids = np.meshgrid(t_ups, x_ups, y_ups, z_ups, indexing='ij')
        else:
            grids = np.meshgrid(x_ups, y_ups, z_ups, indexing='ij')
        
        flat_coords = [grid.ravel() for grid in grids]
        xyz_ups_full = np.stack(flat_coords, axis=-1) 

        if config.predictions.fluid_region:
            # Upsample mask
            mask_ups = zoom(mask, zoom=config.plot.spatial_factor, order=0, grid_mode=True, mode='nearest')
            mask_ups_flat = np.tile(mask_ups.ravel(), len(t_ups)) if config.setup.include_time else mask_ups.ravel()
            fluid_indices = mask_ups_flat == 1

            xyz_ups = xyz_ups_full[fluid_indices]
        else:
            xyz_ups = xyz_ups_full  
        
        # Predict fluid data poinst grid
        model.eval()
        with torch.no_grad():
            xyz_ups = torch.from_numpy(xyz_ups).float().to(DEVICE)
            uvw_pred_ups = model(xyz_ups)  # shape (N_fluid, out_dim)

        # Detach
        uvw_pred_ups = uvw_pred_ups.cpu().numpy()

        if config.plot.fluid_region:
            uvw_pred_full = np.zeros(((len(xyz_ups_full), len(uvw_pred_ups[0])))) + config.plot.non_fluid_value
            uvw_pred_full[fluid_indices] = uvw_pred_ups
            uvw_pred_ups = uvw_pred_full

        uvw_pred_ups = uvw_pred_ups.reshape(len(t_ups), len(x_ups), len(y_ups), len(z_ups), len(uvw_pred_ups[0]))

        u_pred = uvw_pred[:, :, :, :, 0]
        v_pred = uvw_pred[:, :, :, :, 1]
        w_pred = uvw_pred[:, :, :, :, 2]
        p_pred = uvw_pred[:, :, :, :, 3] if config.setup.include_pressure else None

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

        # Save SR predictions
        save_to_h5(f"{SR_directory}/healthy-05mm3_SR.h5", "u", u_pred)
        save_to_h5(f"{SR_directory}/healthy-05mm3_SR.h5", "v", v_pred)
        save_to_h5(f"{SR_directory}/healthy-05mm3_SR.h5", "w", w_pred)
        #save_to_h5(f"{SR_directory}/healthy-05mm3_SR.h5", "p", p_pred)

    if config.predictions.compare_noisy_vs_ref:

        assert config.ref_spatial_factor == 1

        # Create directory
        ref_directory = f'{results_directory}/ref_data'
        if not os.path.exists(ref_directory):
            os.makedirs(ref_directory)

        # Get metrics
        peak_flow_idx = config.predictions.peak_flow_idx
        T = len(u)
        nf_mask = 1.0 - mask_ref
        boundary_mask, core_mask = create_boundary_and_core_masks(mask_ref, 0.1, 'voxels')

        X,Y,Z = mask_ref.shape
        cov_a = np.sum(mask_ref)/(X*Y*Z)
        cov_b = np.sum(boundary_mask)/(X*Y*Z)
        cov_c = np.sum(core_mask)/(X*Y*Z)
        ratio_b = np.sum(boundary_mask)/np.sum(mask_ref)
        ratio_c = np.sum(core_mask)/np.sum(mask_ref)

        print(' ')
        print(f'Coverage: {100*cov_a:.3f} %')
        print(f'Boundary --- cov: {100*cov_b:.3f} %, ratio: {100*ratio_b:.3f} %')
        print(f'Core --- cov: {100*cov_c:.3f} %, ratio: {100*ratio_c:.3f} %')

        rel_err = np.zeros((T,3))
        abs_err = np.zeros((T,5))
        rmse = np.zeros((T,5))

        Ks = np.zeros((T,3,3))
        Ms = np.zeros((T,3,3))
        Rs = np.zeros((T,3,3))

        for t in range(T):
            rel_err[t,0] = (calculate_relative_error(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
            rel_err[t,1] = (calculate_relative_error(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
            rel_err[t,2] = (calculate_relative_error(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], core_mask))

            abs_err[t,0] = (calculate_absolute_error(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
            abs_err[t,1] = (calculate_absolute_error(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
            abs_err[t,2] = (calculate_absolute_error(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], core_mask))
            abs_err[t,3] = (calculate_absolute_error(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], nf_mask))
            # abs_err[t,4] = (calculate_absolute_error_pressure(p[t], p_ref[t], mask_ref))

            rmse[t,0] = (calculate_rmse(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
            rmse[t,1] = (calculate_rmse(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
            rmse[t,2] = (calculate_rmse(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], core_mask))
            rmse[t,3] = (calculate_rmse(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], nf_mask))
            # rmse[t,4] = (calculate_rmse_pressure(p[t], p_ref[t], mask_ref))

            Ks[t][0][0], Ms[t][0][0], Rs[t][0][0] = linreg(u[t], u_ref[t], mask_ref)
            Ks[t][1][0], Ms[t][1][0], Rs[t][1][0] = linreg(v[t], v_ref[t], mask_ref)
            Ks[t][2][0], Ms[t][2][0], Rs[t][2][0] = linreg(w[t], w_ref[t], mask_ref)

            Ks[t][0][1], Ms[t][0][1], Rs[t][0][1] = linreg(u[t], u_ref[t], boundary_mask)
            Ks[t][1][1], Ms[t][1][1], Rs[t][1][1] = linreg(v[t], v_ref[t], boundary_mask)
            Ks[t][2][1], Ms[t][2][1], Rs[t][2][1] = linreg(w[t], w_ref[t], boundary_mask)

            Ks[t][0][2], Ms[t][0][2], Rs[t][0][2] = linreg(u[t], u_ref[t], core_mask)
            Ks[t][1][2], Ms[t][1][2], Rs[t][1][2] = linreg(v[t], v_ref[t], core_mask)
            Ks[t][2][2], Ms[t][2][2], Rs[t][2][2] = linreg(w[t], w_ref[t], core_mask)
        
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
        # print(f'Absolute error Pressure [Fluid] {abs_err_tot[4]:.4f}')

        rmse_tot = np.mean(rmse, axis=0)
        print(f'R.M.S.   error [Fluid] {rmse_tot[0]:.4f}')
        print(f'R.M.S.   error [Bound] {rmse_tot[1]:.4f}')
        print(f'R.M.S.   error [Core] {rmse_tot[2]:.4f}')
        print(f'R.M.S.   error [Non-F] {rmse_tot[3]:.4f}')
        # print(f'R.M.S.   error Pressure [Fluid] {rmse_tot[4]:.4f}')

        print(' ')
        print(f'U [Fluid] k: {Ks[peak_flow_idx][0][0]:.4f} \t m: {Ms[peak_flow_idx][0][0]:.4f} \t r^2: {Rs[peak_flow_idx][0][0]:.4f}')
        print(f'  [Bound] k: {Ks[peak_flow_idx][0][1]:.4f} \t m: {Ms[peak_flow_idx][0][1]:.4f} \t r^2: {Rs[peak_flow_idx][0][1]:.4f}')
        print(f'  [Core] k: {Ks[peak_flow_idx][0][2]:.4f} \t m: {Ms[peak_flow_idx][0][2]:.4f} \t r^2: {Rs[peak_flow_idx][0][2]:.4f}')

        print(' ')
        print(f'V [Fluid] k: {Ks[peak_flow_idx][1][0]:.4f} \t m: {Ms[peak_flow_idx][1][0]:.4f} \t r^2: {Rs[peak_flow_idx][1][0]:.4f}')
        print(f'  [Bound] k: {Ks[peak_flow_idx][1][1]:.4f} \t m: {Ms[peak_flow_idx][1][1]:.4f} \t r^2: {Rs[peak_flow_idx][1][1]:.4f}')
        print(f'  [Core] k: {Ks[peak_flow_idx][1][2]:.4f} \t m: {Ms[peak_flow_idx][1][2]:.4f} \t r^2: {Rs[peak_flow_idx][1][2]:.4f}')

        print(' ')
        print(f'W [Fluid] k: {Ks[peak_flow_idx][2][0]:.4f} \t m: {Ms[peak_flow_idx][2][0]:.4f} \t r^2: {Rs[peak_flow_idx][2][0]:.4f}')
        print(f'  [Bound] k: {Ks[peak_flow_idx][2][1]:.4f} \t m: {Ms[peak_flow_idx][2][1]:.4f} \t r^2: {Rs[peak_flow_idx][2][1]:.4f}')
        print(f'  [Core] k: {Ks[peak_flow_idx][2][2]:.4f} \t m: {Ms[peak_flow_idx][2][2]:.4f} \t r^2: {Rs[peak_flow_idx][2][2]:.4f}')


        # Save metrics to csv
        metrics = {
            'Relative error [Fluid]': rel_err_tot[0],
            'Relative error [Bound]': rel_err_tot[1],
            'Relative error [Core]': rel_err_tot[2],

            'Absolute error [Fluid]': abs_err_tot[0],
            'Absolute error [Bound]': abs_err_tot[1],
            'Absolute error [Core]': abs_err_tot[2],
            'Absolute error [Non-F]': abs_err_tot[3],
            # 'Absolute error Pressure [Fluid]': abs_err_tot[4],

            'R.M.S. error [Fluid]': rmse_tot[0],
            'R.M.S. error [Bound]': rmse_tot[1],
            'R.M.S. error [Core]': rmse_tot[2],
            'R.M.S. error [Non-F]': rmse_tot[3],
            # 'R.M.S. error Pressure [Fluid]': rmse_tot[4],

            'PEAK FLOW INDEX:': peak_flow_idx,
            'Relative error [Fluid] Peak': rel_err[peak_flow_idx][0],
            'Relative error [Bound] Peak': rel_err[peak_flow_idx][1],
            'Relative error [Core] Peak': rel_err[peak_flow_idx][2],
            'Absolute error [Fluid] Peak': abs_err[peak_flow_idx][0],
            'Absolute error [Bound] Peak': abs_err[peak_flow_idx][1],
            'Absolute error [Core] Peak': abs_err[peak_flow_idx][2],
            'Absolute error [Non-F] Peak': abs_err[peak_flow_idx][3],
            'R.M.S. error [Fluid] Peak': rmse[peak_flow_idx][0],
            'R.M.S. error [Bound] Peak': rmse[peak_flow_idx][1],
            'R.M.S. error [Core] Peak': rmse[peak_flow_idx][2],
            'R.M.S. error [Non-F] Peak': rmse[peak_flow_idx][3],

            'U [Fluid] k': Ks[peak_flow_idx][0][0],
            'U [Bound] k': Ks[peak_flow_idx][0][1],
            'U [Core] k': Ks[peak_flow_idx][0][2],
            'U [Fluid] m': Ms[peak_flow_idx][0][0],
            'U [Bound] m': Ms[peak_flow_idx][0][1],
            'U [Core] m': Ms[peak_flow_idx][0][2],
            'U [Fluid] r^2': Rs[peak_flow_idx][0][0],
            'U [Bound] r^2': Rs[peak_flow_idx][0][1],
            'U [Core] r^2': Rs[peak_flow_idx][0][2],

            'V [Fluid] k': Ks[peak_flow_idx][1][0],
            'V [Bound] k': Ks[peak_flow_idx][1][1],
            'V [Core] k': Ks[peak_flow_idx][1][2],
            'V [Fluid] m': Ms[peak_flow_idx][1][0],
            'V [Bound] m': Ms[peak_flow_idx][1][1],
            'V [Core] m': Ms[peak_flow_idx][1][2],
            'V [Fluid] r^2': Rs[peak_flow_idx][1][0],
            'V [Bound] r^2': Rs[peak_flow_idx][1][1],
            'V [Core] r^2': Rs[peak_flow_idx][1][2],

            'W [Fluid] k': Ks[peak_flow_idx][2][0],
            'W [Bound] k': Ks[peak_flow_idx][2][1],
            'W [Core] k': Ks[peak_flow_idx][2][2],
            'W [Fluid] m': Ms[peak_flow_idx][2][0],
            'W [Bound] m': Ms[peak_flow_idx][2][1],
            'W [Core] m': Ms[peak_flow_idx][2][2],
            'W [Fluid] r^2': Rs[peak_flow_idx][2][0],
            'W [Bound] r^2': Rs[peak_flow_idx][2][1],
            'W [Core] r^2': Rs[peak_flow_idx][2][2],
        }

        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        metrics_filename = f"{ref_directory}/metrics_noisyvsref.csv"
        metrics_df.to_csv(metrics_filename, index=False)