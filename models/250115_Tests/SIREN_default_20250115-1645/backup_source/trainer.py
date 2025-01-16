# Imports
import time
import torch
from utils.loss_utils import compute_data_loss, compute_physics_loss, compute_boundary_loss, update_loss_weights
from utils.prepare_data import prepare_data, load_data, extract_fluid_region, sample_collocation_points, sample_boundary_points, load_ref_data, prepare_ref_data
from utils.utils import copy_cource_code, save_checkpoint, sample_to_device, plot_predictions, evaluate_predictions, plot_predictions_vs_reference, set_seed
import SIREN
from configs.SIREN_default import get_config
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    print("Starting script")

    config = get_config()

    # Store source files
    copy_cource_code(config.log_dir, directory_to_backup= [".", "configs"])

    # Set random seed
    set_seed(config.random_seed)

    # Load data
    u, v, w, p, mask = load_data(config)

    # Prepare data
    uvw_data, xyz_data, mask_flat, boundary_mask_flat, standardization_factors, U_max  = prepare_data(config, u, v, w, p, mask)

    # Load and prepare reference data
    if config.include_ref:
        u_ref, v_ref, w_ref, p_ref, mask_ref = load_ref_data(config)
        xyz_data_ref, mask_flat_ref, boundary_mask_flat_ref = prepare_ref_data(config, u, mask_ref)

    # Expand mask
    if config.setup.expand_mask:
        mask_flat = mask_flat + boundary_mask_flat
        if config.include_ref:
            mask_flat_ref = mask_flat_ref + boundary_mask_flat_ref

    # Include fluid region data
    if config.setup.fluid_region:
        uvw_train, xyz_train = extract_fluid_region(uvw_data, xyz_data, mask_flat)
        if config.include_ref:
            xyz_ref = xyz_data_ref[mask_flat_ref==1]
    else:
        uvw_train, xyz_train = uvw_data, xyz_data
        if config.include_ref:
            xyz_ref = xyz_data_ref

    # Sample collocation points
    xyz_collocation = None
    if config.sample_collocation:
        xyz_collocation = sample_collocation_points(config, xyz_data, mask_flat)

    # Sample boundary points
    xyz_boundary = None
    if config.sample_boundary:
        xyz_boundary = sample_boundary_points(config, xyz_data, boundary_mask_flat)

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
    
    # Start training
    start_time = time.time()
    for it in range(config.training.iterations):

        # Time epoch
        start_epoch = time.time()

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
        data_loss, uvw_pred_data = compute_data_loss(config, model, xyz_data_batch, uvw_data_batch, mask_batch)

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
        }
        for key, value in metrics.items():
            writer.add_scalar(key, value, it)
        if (it + 1) % config.training.log_iter == 0:
            print(f"[Iteration {it+1}] total_loss={total_loss.item():.4f}, data_loss={data_loss.item():.4f}, physics_loss={physics_loss.item():.4f}, time={round((time.time()-start_time)/60, 1)} min")

        # Plot current model predictions
        if (it + 1) % config.plot.iter == 0:
            plot_predictions(config, model, DEVICE, it+1, u, mask, U_max)
       
        # Compare with reference data
        if config.include_ref:
            if (it + 1) % config.training.error_iter == 0 or it == 0:
                evaluate_predictions(config, model, DEVICE, it+1, xyz_ref, u_ref, v_ref, w_ref, p_ref, mask_ref, mask_flat_ref, U_max)
                plot_predictions_vs_reference(config, model, DEVICE, it+1, xyz_ref, 
                                            u, v, w, p, u_ref, v_ref, w_ref, p_ref, mask_ref, 
                                            mask_flat_ref, U_max)
                
        # Save model at checkpoint
        if (it + 1) % config.training.summary_iter == 0:
            save_checkpoint(model, it+1, config)

        # Save model at end of training
        if (it + 1) == config.training.iterations:
            save_checkpoint(model, it+1, config, final=True) 
