# Imports
import time
import torch
import wandb
from utils.loss_utils import compute_data_loss, compute_physics_loss, compute_boundary_loss, update_loss_weights
from utils.prepare_data import prepare_data, load_data, extract_fluid_region, sample_collocation_points, sample_boundary_points, load_ref_data, prepare_ref_data
from utils.utils import copy_source_code, save_checkpoint, sample_to_device, sample_from_gpu, sample_ref_from_gpu, sample_ref_to_device, plot_predictions, evaluate_predictions, plot_predictions_vs_reference, set_seed
import networks
#from configs.Config_1x_HV01_highSNR_momentum_PG import get_config
#from configs.Config_ICAD_1t_2x_healthy_lowSNR import get_config
from configs.Config_251016_sweep_WIRE_complex_abstract1 import get_config

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def train(config=None, run_name=None):

    print("Starting script")
    
    run = wandb.init(
            project="SRFlowNIR",
            name=run_name,
            config=config.to_dict()
        )
    config = run.config
    if config["sweep"]:

        # Sweep parameters:
        omega_0 = config["network"]["omega_0"]
        sigma_0 = config["network"]["sigma_0"]
        #fourier_mapping_size = sweep_config["network.fourier_mapping_size"]
        #fourier_scale = sweep_config["network.fourier_scale"]

        #t_len = config["template"]["t_len"]

        # Run names
        #run.name = f"SIREN_sweep_Omega{omega_0}"
        #run.name = f"GAUSS_sweep_Sigma{sigma_0}"+
        #run.name = f"WIRE_COMPLEX_sweep_Sigma{sigma_0}_Omega{omega_0}"
        #run.name = f"FFN_bias_sweep_fourier_mapping_size{fourier_mapping_size}_fourier_scale{fourier_scale}"

        run.name = f"WIRE_sweep_omega{omega_0}_sigma{sigma_0}"

        #run.log({"run_name": run.name})


        timestamp = datetime.now().strftime('%Y%m%d-%H%M')
        networks_folder = config["networks_folder"]
        new_log_dir = f"{networks_folder}/{run.name}_{timestamp}"
        config.update({"log_dir": new_log_dir}, allow_val_change=True)
        if sigma_0 == 0:
            config["network"].update({"complex": False}, allow_val_change=True)

    # Store source files
    # copy_source_code(config.log_dir, directory_to_backup= [".", "configs"])

    # Set random seed
    set_seed(config["random_seed"])

    # Load data
    u, v, w, p, px, py, pz, mask, config = load_data(config)

    # Prepare data
    uvw_data, xyz_data, mask_flat, boundary_mask_flat, standardization_factors, U_max  = prepare_data(config, u, v, w, p, px, py, pz, mask)

    config["U_max"] = U_max

    # Load and prepare reference data
    if config["include_ref"]:
        u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask_ref = load_ref_data(config)
        uvw_data_ref, xyz_data_ref, mask_flat_ref, boundary_mask_flat_ref = prepare_ref_data(config, u, u_ref, v_ref, w_ref, 
                                                                                             p_ref, px_ref, py_ref, pz_ref,
                                                                                             mask_ref, U_max)
    mask_flat = mask_flat.astype(np.uint8)
    mask_flat_ref = mask_flat_ref.astype(np.uint8)

    # Sample collocation points
    xyz_collocation = None
    if config["sample_collocation"]:
        xyz_collocation = sample_collocation_points(config, xyz_data, mask_flat)
    ### xyz_collocation = np.copy(xyz_train)

    # Sample boundary points
    xyz_boundary = None
    if config["sample_boundary"]:
        xyz_boundary = sample_boundary_points(config, xyz_data, boundary_mask_flat)
    
    # Expand mask
    if config["setup"]["expand_mask"]:
        mask_flat = mask_flat + boundary_mask_flat

    # Include fluid region data
    if config["setup"]["fluid_region"]:
        uvw_train, xyz_train = extract_fluid_region(uvw_data, xyz_data, mask_flat, print_fluid_points=True)
        if config["include_ref"]:
            uvw_ref, xyz_ref = extract_fluid_region(uvw_data_ref, xyz_data_ref, mask_flat_ref)
    else:
        uvw_train, xyz_train = uvw_data, xyz_data
        if config["include_ref"]:
            uvw_ref, xyz_ref = uvw_data_ref, xyz_data_ref

    # Initialize network
    DEVICE = torch.device('cuda')
    if config["network"]["arch"] == "SIREN":
        model = networks.SIREN(
            in_dim=config["network"]["in_dim"],
            out_dim=config["network"]["out_dim"],
            depth=config["network"]["depth"],
            hidden_features=config["network"]["hidden_features"],
            first_omega_0=config["network"]["omega_0"],
            hidden_omega_0=config["network"]["omega_0"]
        ).to(DEVICE)
    elif config["network"]["arch"] == "FF_SIREN":
        model = networks.FF_SIREN(
            in_dim=config["network"]["in_dim"],
            out_dim=config["network"]["out_dim"],
            depth=config["network"]["depth"],
            hidden_features=config["network"]["hidden_features"],
            first_omega_0=config["network"]["omega_0"],
            hidden_omega_0=config["network"]["omega_0"],
            fourier_mapping_size=config["network"]["fourier_mapping_size"],
            scale=config["network"]["fourier_scale"]
        ).to(DEVICE)
    elif config["network"]["arch"] == "FFN":
        model = networks.FFN(
            input_dim=config["network"]["in_dim"],
            output_dim=config["network"]["out_dim"],
            depth=config["network"]["depth"],
            hidden_dim=config["network"]["hidden_features"],
            fourier_mapping_size=config["network"]["fourier_mapping_size"],
            scale=config["network"]["fourier_scale"]
        ).to(DEVICE)
    elif config["network"]["arch"] == "WIRE":
        model = networks.WIRE(
            in_dim=config["network"]["in_dim"],
            out_dim=config["network"]["out_dim"],
            depth=config["network"]["depth"],
            hidden_features=config["network"]["hidden_features"],
            first_omega_0=config["network"]["omega_0"],
            hidden_omega_0=config["network"]["omega_0"],
            scale=config["network"]["sigma_0"],
            complex=config["network"]["complex"]
        ).to(DEVICE)
    else:
        raise ValueError("Unknown network.")
    
    
    xyz_train_gpu = torch.from_numpy(xyz_train).float().to(DEVICE)
    uvw_train_gpu = torch.from_numpy(uvw_train).float().to(DEVICE)
    mask_flat_gpu = torch.from_numpy(mask_flat).float().to(DEVICE).view(-1, 1)
    
    xyz_collocation_gpu = None
    if config["sample_collocation"]:
        xyz_collocation_gpu = torch.from_numpy(xyz_collocation).float().to(DEVICE)

    xyz_boundary_gpu = None
    if config["sample_boundary"]:
        xyz_boundary_gpu = torch.from_numpy(xyz_boundary).float().to(DEVICE)
    if config["include_ref"]:
        xyz_ref_gpu = torch.from_numpy(xyz_ref).float().to(DEVICE)
        uvw_ref_gpu = torch.from_numpy(uvw_ref).float().to(DEVICE)
        mask_flat_ref_gpu = torch.from_numpy(mask_flat_ref).float().to(DEVICE).view(-1, 1)

    # Initialize optimizers
    Adam_optimizer = torch.optim.Adam(params=model.parameters(), lr=config["training"]["lr"])
    if config["training"]["use_LBFGS"]:
        BFGS_optimizer = torch.optim.LBFGS(params=model.parameters(), lr=config["training"]["BFGS_lr"])

        def closure():

                # Zero out gradients
                BFGS_optimizer.zero_grad()

                # Recompute forward pass
                model.train()

                # Data loss
                data_loss, _, _, _, _ = compute_data_loss(config, model, xyz_data_batch, uvw_data_batch, mask_batch, standardization_factors)

                # PDE residuals (physics loss)
                physics_losses = compute_physics_loss(
                    config,
                    it,
                    model,
                    xyz_collocation_batch,
                    xyz_data_batch,  
                    standardization_factors
                )
                physics_loss = physics_losses["physics_loss"] if config["training"]["use_physics_loss"] else 0.0
                bound_loss = compute_boundary_loss(
                    config, model, xyz_boundary_batch
                ) if config["sample_boundary"] else 0.0

                # Weighted total loss depending on LBFGS logic
                if config["training"]["grad_weight_scheme"]:
                    # Suppose data_weight, physics_weight, bound_weight come from loss_weights
                    if config["sample_boundary"]:
                        data_weight, physics_weight, bound_weight = loss_weights
                        total_loss = (
                            data_weight * data_loss 
                            + physics_weight * config["training"]["physics_weight"] * physics_loss
                            + bound_weight * config["training"]["boundary_weight"] * bound_loss
                        )
                    else:
                        data_weight, physics_weight = loss_weights
                        total_loss = (
                            data_weight * data_loss 
                            + physics_weight * config["training"]["physics_weight"] * physics_loss
                        )
                else:
                    total_loss = data_loss + config["training"]["physics_weight"] * physics_loss + config["training"]["boundary_weight"] * bound_loss

                # Backprop
                total_loss.backward()
                return total_loss

    # Initialize loss weights
    loss_weights = None

    # Initialize csv logger    
    writer = SummaryWriter(log_dir=f"{config['log_dir']}/tensorboard")
    
    # Start training
    start_time = time.time()
    for it in range(config["training"]["iterations"]):

        # Time iteration
        it_start_time = time.time()

        # Train
        model.train()

        # Sample random points
        (
            xyz_data_batch, 
            uvw_data_batch, 
            mask_batch,
            xyz_collocation_batch, 
            xyz_boundary_batch
        ) = sample_from_gpu(
            config, 
            xyz_train_gpu,
            xyz_collocation_gpu,
            xyz_boundary_gpu,
            uvw_train_gpu,
            mask_flat_gpu
        )
        """ (
            xyz_data_batch, 
            uvw_data_batch, 
            mask_batch,
            xyz_collocation_batch, 
            xyz_boundary_batch
        ) = sample_to_device(config, xyz_train, xyz_collocation, xyz_boundary, uvw_train, mask_flat, DEVICE) """
        
        # Track gradients
        xyz_data_batch.requires_grad = True
        if config["training"]["use_physics_loss"]:
            xyz_collocation_batch.requires_grad = True

        # Update gradient-based loss weights
        if config["training"]["grad_weight_scheme"]:
            loss_weights = update_loss_weights(config, model, loss_weights, it, xyz_data_batch,
                                               uvw_data_batch, mask, xyz_collocation_batch,
                                               xyz_boundary_batch, standardization_factors)

        # Predict and calculate data loss
        data_loss, _, _, _, _ = compute_data_loss(config, model, xyz_data_batch, uvw_data_batch, mask_batch, standardization_factors)

        # Predict and calculate PDE residuals (physics loss)
        physics_losses = compute_physics_loss(
            config,
            it,
            model,
            xyz_collocation_batch,
            xyz_data_batch,
            standardization_factors
        )

        physics_loss = physics_losses["physics_loss"]
        momentum_loss = physics_losses["momentum_loss"]
        div_loss = physics_losses["div_loss"]
        physics_loss_data = physics_losses["physics_loss_data"]
        momentum_loss_data = physics_losses["momentum_loss_data"]
        div_loss_data = physics_losses["div_loss_data"]

        # Predict and calculate boundary points
        bound_loss = compute_boundary_loss(config, model, xyz_boundary_batch)
        
        # Total loss
        if config["training"]["grad_weight_scheme"]:
            if config["sample_boundary"]:
                data_weight, physics_weight, bound_weight = loss_weights[0], loss_weights[1], loss_weights[2]
                total_loss = data_weight*data_loss + physics_weight*config["training"]["physics_weight"]*physics_loss + bound_weight*config["training"]["boundary_weight"]*bound_loss
            else:
                data_weight, physics_weight = loss_weights[0], loss_weights[1]
                total_loss = data_weight*data_loss + physics_weight*config["training"]["physics_weight"]*physics_loss
        else:
            total_loss = data_loss + config["training"]["physics_weight"]*physics_loss + config["training"]["boundary_weight"]*bound_loss
            data_weight, physics_weight = None, None

        # Optimizer Step
        if config["training"]["use_LBFGS"]:
            if it < config["training"]["iterations_before_BFGS"]:
                # Update Adam optimizer
                Adam_optimizer.zero_grad()
                total_loss.backward()
                Adam_optimizer.step()
            else:
                # Update LBFGS optimizer
                BFGS_optimizer.step(closure)
        else:
            # Update Adam optimizer
            Adam_optimizer.zero_grad()
            total_loss.backward()
            Adam_optimizer.step()

            # Learning rate decay
            if (it + 1) % config["training"]["lr_decay_iter"] == 0:
                for param_group in Adam_optimizer.param_groups:
                    param_group['lr'] *= config["training"]["lr_decay_factor"]

        if config["include_ref_loss"]:
            # Sample random points and set to device
            # xyz_ref_batch, uvw_ref_batch, mask_ref_batch = sample_ref_to_device(config, xyz_ref, uvw_ref, mask_flat_ref, DEVICE)
            xyz_ref_batch, uvw_ref_batch, mask_ref_batch = sample_ref_from_gpu(
                config, 
                xyz_ref_gpu,
                uvw_ref_gpu,
                mask_flat_ref_gpu
            )
            if config["training"]["use_vector_potential"]:
                xyz_ref_batch.requires_grad = True
            ref_loss, _, mse_px, mse_py, mse_pz = compute_data_loss(config, model, xyz_ref_batch, uvw_ref_batch, mask_ref_batch, standardization_factors, denormalize=True, reference=True)
        else:
            ref_loss, mse_px, mse_py, mse_pz = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        # Logging
        metrics = {
            "Loss/Train": total_loss.item(),
            "Loss/Data": data_loss.item(),
            "Loss/Physics": physics_loss.item(),
            "Loss/Boundary": bound_loss.item(),
            "Loss/Momentum": momentum_loss.item(),
            "Loss/Divergence": div_loss.item(),
            "Loss/Physics_data": physics_loss_data.item(),
            "Loss/Momentum_data": momentum_loss_data.item(),
            "Loss/Divergence_data": div_loss_data.item(),
            "Loss/Pressure_x": mse_px.item() if mse_px is not None else 0.0,
            "Loss/Pressure_y": mse_py.item() if mse_px is not None else 0.0,
            "Loss/Pressure_z": mse_pz.item() if mse_px is not None else 0.0,
            "Loss/Ref": ref_loss.item(), 
            "Loss/data_weight": data_weight if data_weight is not None else 0.0,
            "Loss/physics_weight": physics_weight if physics_weight is not None else 0.0,
            "Loss/boundary_weight": bound_weight if config["sample_boundary"] else 0.0,
        }

        for key, value in metrics.items():
            writer.add_scalar(key, value, it)
        if (it + 1) % config["training"]["log_iter"] == 0:
            print(f"[Iteration {it+1}] total_loss={total_loss.item():.4f}, data_loss={data_loss.item():.4f}, ref_loss={ref_loss.item():.4f}, physics_loss={physics_loss.item():.4E}, it_time={round((time.time()-it_start_time)/config['training']['log_iter'], 5)} s total_time={round((time.time()-start_time)/60, 1)} min")

            wandb.log({
                "Loss/Train": total_loss.item(),
                "Loss/Data": data_loss.item(),
                "Loss/Physics": physics_loss.item(),
                "Loss/Boundary": bound_loss.item(),
                "Loss/Momentum": momentum_loss.item(),
                "Loss/Divergence": div_loss.item(),
                "Loss/Physics_data": physics_loss_data.item(),
                "Loss/Momentum_data": momentum_loss_data.item(),
                "Loss/Divergence_data": div_loss_data.item(),
                "Loss/Pressure_x": mse_px.item() if mse_px is not None else 0.0,
                "Loss/Pressure_y": mse_py.item() if mse_px is not None else 0.0,
                "Loss/Pressure_z": mse_pz.item() if mse_px is not None else 0.0,
                "Loss/Ref": ref_loss.item(),
                "Loss/data_weight": data_weight if data_weight is not None else 0.0,
                "Loss/physics_weight": physics_weight if physics_weight is not None else 0.0,
                "Loss/boundary_weight": bound_weight if config["sample_boundary"] else 0.0,
            },step=it+1)
            
        # Plot current model predictions
        if (it + 1) % config["plot"]["iter"] == 0:
            plot_predictions(config, model, DEVICE, it+1, u, mask, U_max)
       
        # Compare with reference data
        if config["include_ref"]:
            if (it + 1) % config["training"]["error_iter"] == 0 or it == 0:
                
                metrics_eval = evaluate_predictions(config, model, DEVICE, it+1, xyz_ref, u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask_ref, mask_flat_ref, U_max, standardization_factors)
        
                plot_predictions_vs_reference(config, model, DEVICE, it+1, xyz_ref, 
                                            u, v, w, p, u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask_ref, 
                                            mask_flat_ref, U_max, standardization_factors)
                
            # Log metrics to wandb
            log_dict = {
                "Relative Error [Fluid]": metrics_eval['Relative error [Fluid]'],
                "VNRMSE [Fluid]": metrics_eval['VNRMSE [Fluid]'],
                "Directional error [Fluid]": metrics_eval["Directional error [Fluid]"],
                "Divergence prediction [Fluid]": metrics_eval["Divergence prediction [Fluid]"],
                "Divergence reference [Fluid]": metrics_eval["Divergence reference [Fluid]"],
                "W k [Core]": metrics_eval['W k [Core]'],
                "W r^2 [Core]": metrics_eval['W r^2 [Core]'],
            }

            # Add pressure metrics only if reference gradients are used
            if config["training"]["reference_gradients"]:
                log_dict.update({
                    "Pressure Gradient Relative Error [Fluid]": metrics_eval['Relative error Pressure Gradient (%) [Fluid]'],
                    "Pressure gradient PX k [Core]": metrics_eval['PX k [Core]'],
                    "Pressure gradient PX r^2 [Core]": metrics_eval['PX r^2 [Core]'],
                    "Pressure gradient PY k [Core]": metrics_eval['PY k [Core]'],
                    "Pressure gradient PY r^2 [Core]": metrics_eval['PY r^2 [Core]'],
                    "Pressure gradient PZ k [Core]": metrics_eval['PZ k [Core]'],
                    "Pressure gradient PZ r^2 [Core]": metrics_eval['PZ r^2 [Core]'],
                })

            # Log to W&B
            wandb.log(log_dict, step=it + 1)
                    
        # Save model at checkpoint
        if (it + 1) % config["training"]["summary_iter"] == 0:
            save_checkpoint(model, it+1, config)

        # Save model at end of training
        if (it + 1) == config["training"]["iterations"]:
            save_checkpoint(model, it+1, config, final=True)

            metrics_eval = evaluate_predictions(config, model, DEVICE, it+1, xyz_ref, u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask_ref, mask_flat_ref, U_max, standardization_factors, save_pred=False)

            final_log_time = time.time() - start_time

            final_log_dict = {
                "FINAL Relative Error [Fluid]": metrics_eval['Relative error [Fluid]'],
                "FINAL VNRMSE [Fluid]": metrics_eval['VNRMSE [Fluid]'],
                "FINAL training time [min]": round(final_log_time/60, 2),
                "FINAL W k [Core]": metrics_eval['W k [Core]'],
                "FINAL W r^2 [Core]": metrics_eval['W r^2 [Core]'],     
            }
            if config["training"]["reference_gradients"]:
                final_log_dict.update({
                    "FINAL Pressure Gradient Relative Error [Fluid]": metrics_eval['Relative error Pressure Gradient (%) [Fluid]'],
                    "FINAL Pressure gradient PZ k [Core]": metrics_eval['PZ k [Core]'],
                    "FINAL Pressure gradient PZ r^2 [Core]": metrics_eval['PZ r^2 [Core]'],
                })

            # Log metrics to wandb
            wandb.log(final_log_dict)

    wandb.finish()

if __name__ == "__main__":

    config = get_config()
    run_name = f"{config.network_name}"
    train(config=config, run_name=run_name)
