# Imports
import time
import torch
import wandb
from utils.loss_utils import compute_data_loss, compute_physics_loss, compute_boundary_loss, update_loss_weights
from utils.prepare_data import prepare_data, load_data, extract_fluid_region, sample_collocation_points, sample_boundary_points, load_ref_data, prepare_ref_data
from utils.utils import copy_cource_code, save_checkpoint, sample_to_device, sample_ref_to_device, plot_predictions, evaluate_predictions, plot_predictions_vs_reference, set_seed
import networks
from configs.Config_250923_KI_HV01_x2_tx2_momentum import get_config
#from configs.Config_250923_KI_HV01_x2_tx2_momentum import get_config

from torch.utils.tensorboard import SummaryWriter
import numpy as np

def train(config=None, run_name=None, use_sweep=False):

    print("Starting script")

    if use_sweep:
        # Initialize wandb for this run
        wandb.init(
            project="SRFlowNIR", 
        )
        config = wandb.config
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
        #training_config.training.epochs_before_PDE = config.training_epochs_before_PDE
        #training_config.training.grad_weight_scheme = config.training_grad_weight_scheme
        #training_config.coords_normalization = config.coords_normalization
        #training_config.global_normalization = config.global_normalization
        #training_config.network.omega_0 = config.network_omega_0
        #training_config.constants.U = config.constants_U
        #training_config.constants.L = config.constants_L
        #training_config.training.use_LBFGS = config.training_use_LBFGS
        #training_config.training.physics_weight = config.training_physics_weight
        config = training_config

    # Store source files
    copy_cource_code(config.log_dir, directory_to_backup= [".", "configs", "utils"])

    # Set random seed
    set_seed(config.random_seed)

    # Load data
    u, v, w, p, px, py, pz, mask, config = load_data(config)

    # Prepare data
    uvw_data, xyz_data, mask_flat, boundary_mask_flat, standardization_factors, U_max  = prepare_data(config, u, v, w, p, px, py, pz, mask)

    config.U_max = U_max

    # Load and prepare reference data
    if config.include_ref:
        u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask_ref = load_ref_data(config)
        uvw_data_ref, xyz_data_ref, mask_flat_ref, boundary_mask_flat_ref = prepare_ref_data(config, u, u_ref, v_ref, w_ref, 
                                                                                             p_ref, px_ref, py_ref, pz_ref,
                                                                                             mask_ref, U_max)
    mask_flat = mask_flat.astype(np.uint8)
    mask_flat_ref = mask_flat_ref.astype(np.uint8)

    # Sample collocation points
    xyz_collocation = None
    if config.sample_collocation:
        xyz_collocation = sample_collocation_points(config, xyz_data, mask_flat)
    ### xyz_collocation = np.copy(xyz_train)

    # Sample boundary points
    xyz_boundary = None
    if config.sample_boundary:
        xyz_boundary = sample_boundary_points(config, xyz_data, boundary_mask_flat)
    
    # Expand mask
    if config.setup.expand_mask:
        mask_flat = mask_flat + boundary_mask_flat

    # Include fluid region data
    if config.setup.fluid_region:
        uvw_train, xyz_train = extract_fluid_region(uvw_data, xyz_data, mask_flat, print_fluid_points=True)
        if config.include_ref:
            uvw_ref, xyz_ref = extract_fluid_region(uvw_data_ref, xyz_data_ref, mask_flat_ref)
    else:
        uvw_train, xyz_train = uvw_data, xyz_data
        if config.include_ref:
            uvw_ref, xyz_ref = uvw_data_ref, xyz_data_ref

    # Initialize network
    DEVICE = torch.device('cuda')
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
    
    # Start training
    start_time = time.time()
    for it in range(config.training.iterations):

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

        if config.include_ref_loss:
            # Sample random points and set to device
            xyz_ref_batch, uvw_ref_batch, mask_ref_batch = sample_ref_to_device(config, xyz_ref, uvw_ref, mask_flat_ref, DEVICE)
            if config.training.use_vector_potential:
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
            "Loss/data_weight": data_weight,
            "Loss/physics_weight": physics_weight,
            "Loss/boundary_weight": bound_weight if config.sample_boundary else 0.0,
        }
        for key, value in metrics.items():
            writer.add_scalar(key, value, it)
        if (it + 1) % config.training.log_iter == 0:
            print(f"[Iteration {it+1}] total_loss={total_loss.item():.4f}, data_loss={data_loss.item():.4f}, ref_loss={ref_loss.item():.4f}, physics_loss={physics_loss.item():.4E}, it_time={round((time.time()-it_start_time)/config.training.log_iter, 5)} s total_time={round((time.time()-start_time)/60, 1)} min")

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
                "Loss/data_weight": data_weight,
                "Loss/physics_weight": physics_weight,
                "Loss/boundary_weight": bound_weight if config.sample_boundary else 0.0,
                },step=it+1)
            
        # Plot current model predictions
        if (it + 1) % config.plot.iter == 0:
            plot_predictions(config, model, DEVICE, it+1, u, mask, U_max)
       
        # Compare with reference data
        if config.include_ref:
            if (it + 1) % config.training.error_iter == 0 or it == 0:
                
                metrics_eval = evaluate_predictions(config, model, DEVICE, it+1, xyz_ref, u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask_ref, mask_flat_ref, U_max, standardization_factors)
        
                plot_predictions_vs_reference(config, model, DEVICE, it+1, xyz_ref, 
                                            u, v, w, p, u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask_ref, 
                                            mask_flat_ref, U_max, standardization_factors)
                
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

    sweep = True

    if sweep:
        # Define sweep configuration
        sweep_configuration = {
            'method': 'random', #
            'metric': {'name': 'Loss/Ref', 'goal': 'minimize'},
            'parameters': { 

                # epochs_before_PDE: 0 or 100
                'training_epochs_before_PDE': {
                    'values': [0]
                },
                # grad_weight_scheme: True or False
                'training_grad_weight_scheme': {
                    'values': [True] 
                },
                # coords_normalization: "standardize" or "min_max"
                'coords_normalization': {
                    'values': ["standardize"]#, "min_max"]
                },
                # global_normalization: True or False
                'global_normalization': {
                    'values': [True]
                },
                # network.omega_0: 10, 30, or 50
                'network_omega_0': {
                    'values': [1, 10, 30, 70]
                },

                'constants_U': {
                    'values': [1.0, 2.0, 0.2]
                },

                'constants_L': {
                    'values': [0.005, 0.1, 0.0005]
                },
                # training.use_LBFGS: True or False
                'training_use_LBFGS': {
                    'values': [True]
                },
                'training_physics_weight': {
                    'values': [1.0]
                },
            }
        }

        # Nerea - Sweep (FFN/SIREN)
        # sweep_configuration = {
        #     'method': 'grid', #
        #     'metric': {'name': 'Loss/Ref', 'goal': 'minimize'},
        #     'parameters': { 
        #         'network_arch': {
        #             'value': 'SIREN'
        #         },
        #         ''
        #         'training_use_vector_potential' : { 
        #             'values': [True, False]
        #         },  
        #         'training_use_physics_loss'  : { 
        #             'values': [True, False]
        #         }, 
        #         'network_omega'   :   { 
        #             'values': [0.1, 1.0, 10.0, 30.0, 70.0, 200.0]
        #         }, 
        #     }
        # }
        # sweep_configuration = {
        #    'method': 'grid', #
        #    'metric': {'name': 'Loss/Ref', 'goal': 'minimize'},
        #    'parameters': { 
        #        'network_arch': {
        #            'value': 'FFN'
        #        },
        #        'training_use_vector_potential' : { 
        #            'values': [True, False]
        #        },  
        #        'training_use_physics_loss'  : { 
        #            'values': [True, False]
        #        }, 
        #        'network_fourier_scale'   :   { 
        #            'values': [0.1, 1.0, 3.0, 5.0, 7.0, 10.0, 50.0, 100.0]
        #        }, 
        #    }
        # }

        sweep_id = wandb.sweep(sweep=sweep_configuration, project="SRFlowNIR")
        wandb.agent(sweep_id, function=lambda: train(use_sweep=True),count=1)
    else:
        config = get_config()
        run_name = f"{config.network_name}"
        train(config=config, run_name=run_name)