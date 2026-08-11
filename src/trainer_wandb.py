# Imports
import time
import torch
import wandb
from utils.loss_utils import compute_data_loss, compute_physics_loss, compute_boundary_loss, update_loss_weights
from utils.prepare_data import prepare_data, load_data, extract_fluid_region, sample_collocation_points, sample_boundary_points, load_ref_data, prepare_ref_data, merge_timeframes
from utils.utils import copy_cource_code, save_checkpoint, sample_to_device, sample_ref_to_device, plot_predictions, evaluate_predictions, plot_predictions_vs_reference, set_seed, plot_3D
import networks

# from configs.FF_1t_AoModel_4dsim_V250_F import get_config
# from configs.FF_1t_AoModel_4dsim_V250_FathiNS import get_config
from configs.FF_1t_AoModel_4dsim_V250_VanillaPINN import get_config
# from configs.FF_1t_AoModel_4dsim_V250_SIREN import get_config
# from configs.FF_1t_AoModel_4dsim_V250_n6 import get_config

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils.evaluation_utils import (
    create_boundary_and_core_masks, 
    calculate_relative_error, 
    calculate_absolute_error, 
    calculate_rmse, 
    calculate_absolute_error_pressure, 
    calculate_rmse_pressure, 
    calculate_divergence,
    calculate_directional_error,
    calculate_vnrmse,
    )

from scipy.ndimage import binary_erosion, generate_binary_structure, binary_dilation

def train(config=None, run_name=None,use_sweep=False):

    print("Starting script")
    
    DEBUG = False

    if use_sweep:
        # Initialize wandb for this run
        wandb.init(
            project="SRFlowNIR", 
        )
        config = wandb.config
    elif DEBUG:
        pass
    else:
        # Initialize wandb for this run
        wandb.init(
            project="SRFlowNIR",
            name=run_name,
            config=config.to_dict()
        )
        
    if use_sweep:
        training_config = get_config()
        
        # Sweep overrides:
        # training_config.network.arch = config.network_arch
        # training_config.training.lr = config.learning_rate
        # training_config.network.depth = config.num_layers
        # training_config.network.hidden_features = config.hidden_dim
        # training_config.network.fourier_mapping_size = config.embed_dim
        training_config.network.fourier_scale = config.fourier_scale
        # training_config.training.physics_weight = config.physics_weight
        config = training_config

    # Store source files
    copy_cource_code(config.log_dir, directory_to_backup= [".", "configs"])

    # Set random seed
    set_seed(config.random_seed)

    # Load data
    u, v, w, p, mask, config = load_data(config)
    
    SEG_experiment = False
    if SEG_experiment:
        struct_3d = generate_binary_structure(3, 2)
        # mask = binary_erosion(mask, structure=struct_3d, iterations=1)
        mask = binary_dilation(mask, structure=struct_3d, iterations=1)

    # Prepare data
    uvw_data, xyz_data, mask_flat, boundary_mask_flat, standardization_factors, U_max  = prepare_data(config, u, v, w, p, mask)

    config.U_max = U_max

    # Load and prepare reference data
    if config.include_ref:
        u_ref, v_ref, w_ref, p_ref, mask_ref = load_ref_data(config)
        if SEG_experiment:
            struct_3d = generate_binary_structure(3, 2)
            # mask_ref = binary_erosion(mask_ref, structure=struct_3d, iterations=1)
            mask_ref = binary_dilation(mask_ref, structure=struct_3d, iterations=1)
        uvw_data_ref, xyz_data_ref, mask_flat_ref, boundary_mask_flat_ref = prepare_ref_data(config, u, u_ref, v_ref, w_ref, p_ref, mask_ref, U_max)

    # Include fluid region data
    if config.setup.fluid_region:
        uvw_train, xyz_train = extract_fluid_region(uvw_data, xyz_data, mask_flat)
        if config.include_ref:
            uvw_ref, xyz_ref = extract_fluid_region(uvw_data_ref, xyz_data_ref, mask_flat_ref)
    else:
        uvw_train, xyz_train = uvw_data, xyz_data
        if config.include_ref:
            uvw_ref, xyz_ref = uvw_data_ref, xyz_data_ref

    # Sample collocation points
    xyz_collocation = None
    if config.sample_collocation:
        xyz_collocation = sample_collocation_points(config, xyz_data, mask_flat)

    xyz_collocation = np.copy(xyz_train)

    # Sample boundary points
    xyz_boundary = None
    if config.sample_boundary:
        xyz_boundary = sample_boundary_points(config, xyz_data, boundary_mask_flat)


    # Initialize network
    DEVICE = torch.device('cuda')
    if config.network.arch == "SIREN":
        model = networks.SIREN(
            in_dim=config.network.in_dim,
            out_dim=config.network.out_dim,
            depth=config.network.depth,
            hidden_features=config.network.hidden_features,
            first_omega_0=config.network.first_omega_0,
            hidden_omega_0=config.network.hidden_omega_0
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
    elif config.network.arch == "FathiMLP":
        model = networks.FathiMLP(
            in_dim=config.network.in_dim,
            out_dim=config.network.out_dim,
            depth=config.network.depth,
            hidden_features=config.network.hidden_features,
            use_magnitude_output=config.training.use_magnitude_output,
        ).to(DEVICE)
    else:
        raise ValueError("Unknown network.")

    # Initialize optimizers
    Adam_optimizer = torch.optim.Adam(params=model.parameters(), lr=config.training.lr)
    if config.training.use_LBFGS:
        BFGS_optimizer = torch.optim.LBFGS(params=model.parameters(), lr=config.training.BFGS_lr)

        def closure():

                # Zero out gradients
                BFGS_optimizer.zero_grad()

                # Recompute forward pass
                model.train()

                # Data loss
                data_loss, _ = compute_data_loss(config, model, xyz_data_batch, uvw_data_batch, mask_batch)

                # PDE residuals (physics loss)
                physics_losses = compute_physics_loss(
                    config,
                    it,
                    model,
                    xyz_collocation_batch,
                    xyz_data_batch,  
                    standardization_factors
                )
                physics_loss = physics_losses["physics_loss"] if config.training.use_physics_loss else 0.0
                bound_loss = compute_boundary_loss(
                    config, model, xyz_boundary_batch
                ) if config.sample_boundary else 0.0

                # Weighted total loss depending on LBFGS logic
                if config.training.grad_weight_scheme:
                    # Suppose data_weight, physics_weight, bound_weight come from loss_weights
                    if config.sample_boundary:
                        data_weight, physics_weight, bound_weight = loss_weights
                        total_loss = (
                            data_weight * data_loss 
                            + physics_weight * config.training.physics_weight * physics_loss
                            + bound_weight * config.training.boundary_weight * bound_loss
                        )
                    else:
                        data_weight, physics_weight = loss_weights
                        total_loss = (
                            data_weight * data_loss 
                            + physics_weight * config.training.physics_weight * physics_loss
                        )
                else:
                    total_loss = data_loss + config.training.physics_weight * physics_loss + config.training.boundary_weight * bound_loss

                # Backprop
                total_loss.backward()
                return total_loss

    # Initialize loss weights
    loss_weights = None

    # Initialize csv logger    
    writer = SummaryWriter(log_dir=f"{config.log_dir}/tensorboard")
    

    vnrmse_noisy = calculate_vnrmse(u, v, w, u_ref, v_ref, w_ref, mask_ref)
    d_error_noisy = calculate_directional_error(u, v, w, u_ref, v_ref, w_ref, mask_ref)
    div_noisy = calculate_divergence([u, v, w],[config.resolution.dx,config.resolution.dy,config.resolution.dz], mask_ref)
    print(f"VNRMSE [Noisy]: {vnrmse_noisy:.6f}")
    print(f"Directional error [Noisy]: {d_error_noisy:.6f}")
    print(f"Divergence [Noisy]: {div_noisy:.6f}")

    # Start training
    start_time = time.time()
    for it in range(config.training.iterations):
        
        if it == 0:
            wandb.log({
                "VNRMSE [Noisy]": vnrmse_noisy,
                "Directional error [Noisy]": d_error_noisy,
                "Divergence [Noisy]": div_noisy,
            })
        # Time iteration
        it_start_time = time.time()

        # Train
        model.train()

        # Sample random points and set to device
        (
            xyz_data_batch, 
            uvw_data_batch, 
            mask_batch,
            xyz_collocation_batch, 
            xyz_boundary_batch
        ) = sample_to_device(config, xyz_train, xyz_collocation, xyz_boundary, uvw_train, mask_flat, DEVICE)
        
        # Track gradients
        xyz_data_batch.requires_grad = True
        if config.training.use_physics_loss:
            xyz_collocation_batch.requires_grad = True

        # Update gradient-based loss weights
        if config.training.grad_weight_scheme:
            loss_weights = update_loss_weights(config, model, loss_weights, it, xyz_data_batch,
                                               uvw_data_batch, mask, xyz_collocation_batch,
                                               xyz_boundary_batch, standardization_factors)

        # Predict and calculate data loss
        data_loss, _ = compute_data_loss(config, model, xyz_data_batch, uvw_data_batch, mask_batch)

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
        if config.training.grad_weight_scheme:
            if config.sample_boundary:
                data_weight, physics_weight, bound_weight = loss_weights[0], loss_weights[1], loss_weights[2]
                total_loss = data_weight*data_loss + physics_weight*config.training.physics_weight*physics_loss + bound_weight*config.training.boundary_weight*bound_loss
            else:
                data_weight, physics_weight = loss_weights[0], loss_weights[1]
                total_loss = data_weight*data_loss + physics_weight*config.training.physics_weight*physics_loss
        else:
            total_loss = data_loss + config.training.physics_weight*physics_loss + config.training.boundary_weight*bound_loss

        # Optimizer Step
        if config.training.use_LBFGS:
            if it < config.training.iterations_before_BFGS:
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
            if (it + 1) % config.training.lr_decay_iter == 0:
                for param_group in Adam_optimizer.param_groups:
                    param_group['lr'] *= config.training.lr_decay_factor

        


        
        if (it + 1) % config.training.log_iter == 0:
            
            if config.include_ref_loss:
                # Sample random points and set to device
                xyz_ref_batch, uvw_ref_batch, mask_ref_batch = sample_ref_to_device(config, xyz_ref, uvw_ref, mask_flat_ref, DEVICE)
            if config.training.use_vector_potential:
                xyz_ref_batch.requires_grad = True
                ref_loss, _ = compute_data_loss(config, model, xyz_ref_batch, uvw_ref_batch, mask_ref_batch)
            else:
                ref_loss, _ = compute_data_loss(config, model, xyz_ref_batch, uvw_ref_batch, mask_ref_batch)

            print(f"[Iteration {it+1}] total_loss={total_loss.item():.4f}, data_loss={data_loss.item():.4f}, ref_loss={ref_loss.item():.4f}, physics_loss={physics_loss.item():.4E}, it_time={round((time.time()-it_start_time)/config.training.log_iter, 5)} s total_time={round((time.time()-start_time)/60, 1)} min")
            
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
                "Loss/Ref": ref_loss.item(),
            }
            for key, value in metrics.items():
                writer.add_scalar(key, value, it)
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
                "Loss/Ref": ref_loss.item(),
                },step=it+1)
            
        # Plot current model predictions
        if (it + 1) % config.plot.iter == 0:
            plot_predictions(config, model, DEVICE, it+1, u, mask, U_max)
       
        # Compare with reference data
        if config.include_ref:
            if (it + 1) % config.training.error_iter == 0 or it == 0 or ((it + 1) == config.training.iterations):
                metrics_eval = evaluate_predictions(config, model, DEVICE, it+1, xyz_ref, u_ref, v_ref, w_ref, p_ref, mask_ref, mask_flat_ref, U_max)
                plot_predictions_vs_reference(config, model, DEVICE, it+1, xyz_ref, 
                                            u, v, w, p, u_ref, v_ref, w_ref, p_ref, mask_ref, 
                                            mask_flat_ref, U_max)
            # Log metrics to wandb
            wandb.log({
                "VNRMSE [Fluid]": metrics_eval['VNRMSE [Fluid]'],
                "Directional error [Fluid]" : metrics_eval["Directional error [Fluid]"],
                "Divergence prediction [Fluid]" : metrics_eval["Divergence prediction [Fluid]"],
                "Divergence reference [Fluid]" : metrics_eval["Divergence reference [Fluid]"]
            }, step=it+1)
                    
        # Save model at checkpoint
        if (it + 1) % config.training.summary_iter == 0:
            save_checkpoint(model, it+1, config)

        # Save model at end of training
        if (it + 1) == config.training.iterations:
            save_checkpoint(model, it+1, config, final=True)

    wandb.finish()

if __name__ == "__main__":
    import h5py

    sweep = False
    train_all_timeframes = False

    if sweep:

        # Def}]ine sweep configuration
        sweep_configuration = {
            'method': 'grid', #
            'metric': {'name': 'Loss/Ref', 'goal': 'minimize'},
            'parameters': {
                'num_layers': {
                'values': [3, 5, 8, 10, 15]
                },
            'hidden_dim': {
                'values': [50, 100, 150, 200, 300, 400]
                },
            'embed_dim': {
                'values': [64, 128, 256]
                },

            'fourier_scale': {
                'values': [0.05,0.1,1.0,3.0]
            }
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="SRFlowNIR")
        wandb.agent(sweep_id, function=lambda: train(use_sweep=True))
    
    elif train_all_timeframes and not sweep:
        config = get_config()
        with h5py.File(config.data_file, mode='r') as hf:
            time_frames = hf['u'].shape[0]
            print(f"Training for {time_frames} time frames")
        for t in range(time_frames):
            config = get_config()
            config.domain.t_start = t
            config.network_name = f"{config.network_name}_timeframe_{t}"
            config.log_dir = f"{config.log_dir}_timeframe_{t}"
            run_name = config.network_name
            train(config=config, run_name=run_name)

        results_path = "/".join(config.log_dir.split("/")[0:-1])
        out_file = f"{results_path}/4dsim.h5"
        merge_timeframes(results_path,out_file)
        print(f"Saved time resolved predictions to {out_file}")
    else:
        config = get_config()
        run_name = f"{config.network_name}"
        train(config=config, run_name=run_name)
