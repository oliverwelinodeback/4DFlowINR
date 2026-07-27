# Imports
import time
import torch
import wandb
from utils.loss_utils import (
    compute_data_loss, 
    compute_physics_loss, 
    compute_boundary_loss, 
    update_loss_weights, 
    navier_stokes_loss
)
from utils.data_io import load_data, load_ref_data
from utils.prepare_data import (
    extract_fluid_region, 
    sample_collocation_points, 
    sample_boundary_points, 
    prepare_data, 
    prepare_ref_data
)
from utils.checkpoints import save_ckpt
from utils.sampling import sample_from_gpu, sample_ref_from_gpu
from utils.reproducibility import set_seed, copy_source_code
from utils.prediction_utils import (
    evaluate_predictions,
    plot_reference_comparison,
    predict_superresolved_grid,
    save_h5_predictions,
)
import networks
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from meta.train_meta import train_meta_learning

# Metrics reported to W&B during reference-data evaluation.
EVAL_METRIC_KEYS = (
    "Relative error [Fluid]",
    "VNRMSE [Fluid]",
    "Directional error [Fluid]",
    "Divergence prediction [Fluid]",
    "Divergence reference [Fluid]",
    "W K [Core]",
    "W R2 [Core]",
    "W k [Core] Peak",
    "W r^2 [Core] Peak",
    "W 2 k [Core]",
    "W 2 r^2 [Core]",
)

PRESSURE_EVAL_METRIC_KEYS = (
    "Relative error Pressure Gradient (%) [Fluid]",
    "PX K [Core]",
    "PX R2 [Core]",
    "PY K [Core]",
    "PY R2 [Core]",
    "PZ K [Core]",
    "PZ R2 [Core]",
    "PZ k [Core] Peak",
    "PZ r^2 [Core] Peak",
)

def build_eval_log(metrics_eval, include_pressure=False, prefix="Eval"):
    """Select evaluation metrics for W&B logging."""

    keys = list(EVAL_METRIC_KEYS)

    if include_pressure:
        keys.extend(PRESSURE_EVAL_METRIC_KEYS)

    missing_keys = [key for key in keys if key not in metrics_eval]
    if missing_keys:
        raise KeyError(
            "Missing evaluation metrics: "
            + ", ".join(missing_keys)
        )

    return {
        f"{prefix}/{key}": float(metrics_eval[key])
        for key in keys
    }


def init_wandb(config, run_name=None, use_sweep=False):
    """Initialize W&B with consistent experiment metadata."""

    if config.meta_learning.enabled:
        job_type = "meta-train"
    elif use_sweep:
        job_type = "sweep"
    else:
        job_type = "train"

    init_kwargs = {
        "project": config.wandb.project,
        "group": config.wandb.group,
        "job_type": job_type,
        "tags": list(config.wandb.tags),
    }

    if run_name is not None:
        init_kwargs["name"] = run_name

    # For non-sweep runs, store the complete configuration immediately
    # Sweep parameters are populated by the W&B agent instead
    if not use_sweep:
        init_kwargs["config"] = config.to_dict()

    return wandb.init(**init_kwargs)

def train(config=None, run_name=None, use_sweep=False):

    print("Starting script")

    if use_sweep:
        run = init_wandb(
            config,
            run_name=run_name,
            use_sweep=True,
        )
        sweep_config = wandb.config

        data_file = sweep_config.get("data_file", config.data_file)
        LR_ROUTING = {
            "../data/healthy/HV01_05mm3_20ms_LR_sv17_tSNR10_newMask.h5":        ("../data/healthy/HV01_05mm3_20ms.h5",        "HV01_sv17",    1.7, 12),
            "../data/healthy/HV03_05mm3_20ms_LR_sv13_tSNR10_newMask.h5":        ("../data/healthy/HV03_05mm3_20ms.h5",        "HV03_sv13",    1.3,  4),
            "../data/healthy/HV06_05mm3_20ms_LR_sv12_tSNR10_newMask.h5":        ("../data/healthy/HV06_05mm3_20ms.h5",        "HV06_sv12",    1.2,  2),
            "../data/stenosis_50/ICAD28_05mm3_20ms_LR_sv13_tSNR10_newMask.h5":  ("../data/stenosis_50/ICAD28_05mm3_20ms.h5",  "ICAD28_sv13",  1.3,  2),
            "../data/stenosis_50/ICAD48_05mm3_20ms_LR_sv13_tSNR10_newMask.h5":  ("../data/stenosis_50/ICAD48_05mm3_20ms.h5",  "ICAD48_sv13",  1.3, 14),
            "../data/stenosis_50/ICAD98_05mm3_20ms_LR_sv51_tSNR10_newMask.h5":  ("../data/stenosis_50/ICAD98_05mm3_20ms.h5",  "ICAD98_sv51",  5.1, 12),
            "../data/stenosis_70/ICAD17_05mm3_20ms_LR_sv41_tSNR10_newMask.h5":  ("../data/stenosis_70/ICAD17_05mm3_20ms.h5",  "ICAD17_sv41",  4.1,  8),
            "../data/stenosis_70/ICAD21_05mm3_20ms_LR_sv26_tSNR10_newMask.h5":  ("../data/stenosis_70/ICAD21_05mm3_20ms.h5",  "ICAD21_sv26",  2.6, 12),
            "../data/stenosis_70/ICAD146_05mm3_20ms_LR_sv17_tSNR10_newMask.h5": ("../data/stenosis_70/ICAD146_05mm3_20ms.h5", "ICAD146_sv17", 1.7,  8),

            # SA-PINN test cases - continue here...
            "../data/healthy/HV01_05mm3_20ms_HRLR_sv17_tSNR10.h5": ("../data/healthy/HV01_05mm3_20ms.h5", "HV01_sv17", 1.7, 12),

            "../data/healthy/HV01_05mm3_20ms_HRLR_sv17_tSNR2.h5": ("../data/healthy/HV01_05mm3_20ms.h5", "HV01_sv17", 1.7, 12),
        }

        if data_file in LR_ROUTING:
            ref_file, _, venc, peak_idx = LR_ROUTING[data_file]
            config.data_file = data_file
            config.data_file_ref = ref_file
            config.constants.venc = venc
            config.predictions.peak_flow_idx = peak_idx

        config.network.omega_0 = sweep_config.get(
            "network.omega_0",
            config.network.omega_0,
        )

        config.network.sigma_0 = sweep_config.get(
            "network.sigma_0",
            config.network.sigma_0,
        )

        config.load_meta_init = sweep_config.get(
            "load_meta_init",
            config.load_meta_init,
        )

        # Store the complete resolved configuration used by this run
        run.config.update(
            {"resolved_config": config.to_dict()},
            allow_val_change=True,
        )

        timestamp = datetime.now().strftime('%Y%m%d-%H%M')
        config.log_dir = f"{config.networks_folder}/{run.name}_{timestamp}"

    else:
        run = init_wandb(
            config,
            run_name=run_name,
            use_sweep=False,
        )

    if config.meta_learning.enabled:
        # Meta-learning with TrainingStep integration
        print("\n" + "="*60)
        print("Using META-LEARNING with TrainingStep Integration")
        print("="*60 + "\n")
        return train_meta_learning(config, run_name, use_sweep)

    # Store source files
    copy_source_code(config.log_dir, directory_to_backup= [".", "configs", "utils"])

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
    if config.include_ref:
        mask_flat_ref = mask_flat_ref.astype(np.uint8)

    # Sample collocation points
    xyz_collocation = None
    if config.sample_collocation:
        xyz_collocation = sample_collocation_points(config, xyz_data, mask_flat)

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
    model = networks.build_model(config).to(DEVICE)

    if config.load_meta_init:
        meta_checkpoint_path = config.meta_init_path
        print(f"\n{'='*60}")
        print(f"Loading meta-learned initialization from:")
        print(f"  {meta_checkpoint_path}")
        print(f"{'='*60}\n")
        
        checkpoint = torch.load(meta_checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("Meta-learned weights loaded successfully")

    c_weights = None
    if config.sample_collocation and config.training.self_adaptive:
        N_c = xyz_collocation.shape[0]
        c_weights = np.ones(N_c, dtype=np.float32) 


    xyz_train_gpu = torch.from_numpy(xyz_train).float().to(DEVICE)
    uvw_train_gpu = torch.from_numpy(uvw_train).float().to(DEVICE)
    mask_flat_gpu = torch.from_numpy(mask_flat).float().to(DEVICE).view(-1, 1)
    
    xyz_collocation_gpu = None
    if config.sample_collocation:
        xyz_collocation_gpu = torch.from_numpy(xyz_collocation).float().to(DEVICE)
    
    xyz_boundary_gpu = None
    if config.sample_boundary:
        xyz_boundary_gpu = torch.from_numpy(xyz_boundary).float().to(DEVICE)
    if config.include_ref:
        xyz_ref_gpu = torch.from_numpy(xyz_ref).float().to(DEVICE)
        uvw_ref_gpu = torch.from_numpy(uvw_ref).float().to(DEVICE)
        mask_flat_ref_gpu = torch.from_numpy(mask_flat_ref).float().to(DEVICE).view(-1, 1)

    # Initialize optimizers
    Adam_optimizer = torch.optim.Adam(params=model.parameters(), lr=config.training.lr)
    
    if config.decay_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            Adam_optimizer, T_max=8000, eta_min=1e-7
        )

    elif config.decay_type == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            Adam_optimizer, gamma=0.9991
        )

    elif config.decay_type == 'multi':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            Adam_optimizer, milestones=[2000, 5000, 7000], gamma=0.1
        )
    else:
        scheduler = LambdaLR(Adam_optimizer, lambda x: 1)

    if config.training.use_LBFGS:
        BFGS_optimizer = torch.optim.LBFGS(
            params=model.parameters(),
            lr=config.training.BFGS_lr,
            max_iter=config.training.BFGS_max_iter,
            history_size=config.training.BFGS_history_size,
            tolerance_grad=config.training.BFGS_tolerance_grad,
            tolerance_change=config.training.BFGS_tolerance_change,
            line_search_fn='strong_wolfe'
            )

        def closure():

                # Zero out gradients
                BFGS_optimizer.zero_grad()

                # Recompute forward pass
                model.train()

                # Data loss
                data_loss, _, _, _, _ = compute_data_loss(config, model, xyz_data_batch, uvw_data_batch, mask_batch, standardization_factors)

                # PDE residuals (physics loss)
                physics_losses = compute_physics_loss(
                    config, it, model,
                    xyz_collocation_batch, xyz_data_batch,
                    standardization_factors,
                    c_weights=c_batch_weights 
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

    # Initialize TensorBoard writer    
    writer = SummaryWriter(
        log_dir=f"{config.log_dir}/tensorboard"
    )
    start_it = 0

    # Start training
    start_time = time.time()
    for it in range(start_it, config.training.iterations):

        # Time iteration
        it_start_time = time.time()

        # Train
        model.train()

        # Sample random points
        if config.training.self_adaptive and config.training.adaptive_sampling:
            (
                xyz_data_batch, 
                uvw_data_batch, 
                mask_batch,
                xyz_collocation_batch, 
                xyz_boundary_batch,
                coll_indices
            ) = sample_from_gpu(config, xyz_train_gpu, xyz_collocation_gpu, xyz_boundary_gpu, uvw_train_gpu, mask_flat_gpu, c_weights)
        else:
            (
                xyz_data_batch, 
                uvw_data_batch, 
                mask_batch,
                xyz_collocation_batch, 
                xyz_boundary_batch,
                coll_indices
            ) = sample_from_gpu(config, xyz_train_gpu, xyz_collocation_gpu, xyz_boundary_gpu, uvw_train_gpu, mask_flat_gpu)

        # Build batch weights tensor if available
        c_batch_weights = None
        if (coll_indices is not None) and (c_weights is not None):
            c_batch_weights = torch.as_tensor(
                c_weights[coll_indices.detach().cpu().numpy()],
                device=DEVICE, dtype=torch.float32
            )

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
        if config.training.self_adaptive:
            physics_losses = compute_physics_loss(
                config,
                it,
                model,
                xyz_collocation_batch,
                xyz_data_batch,
                standardization_factors,
                c_weights=c_batch_weights
            )
        else:
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
                bound_weight = None
        else:
            total_loss = data_loss + config.training.physics_weight*physics_loss + config.training.boundary_weight*bound_loss
            data_weight, physics_weight, bound_weight = None, None, None

        # Optimizer Step
        if config.training.use_LBFGS:
            if it < config.training.iterations_before_BFGS:
                # Update Adam optimizer
                Adam_optimizer.zero_grad()
                total_loss.backward()
                Adam_optimizer.step()
            
                if (it + 1) % config.training.lr_decay_iter == 0:
                    for param_group in Adam_optimizer.param_groups:
                        param_group['lr'] *= config.training.lr_decay_factor
            else:
                # Update LBFGS optimizer
                BFGS_optimizer.step(closure)
        else:
            # Update Adam optimizer
            Adam_optimizer.zero_grad()
            total_loss.backward()
            Adam_optimizer.step()
            scheduler.step()

            # Learning rate decay
            if (it + 1) % config.training.lr_decay_iter == 0:
                for param_group in Adam_optimizer.param_groups:
                    param_group['lr'] *= config.training.lr_decay_factor

        if config.include_ref_loss:
            # Sample random points and set to device
            xyz_ref_batch, uvw_ref_batch, mask_ref_batch = sample_ref_from_gpu(config, xyz_ref_gpu, uvw_ref_gpu, mask_flat_ref_gpu)
            if config.training.use_vector_potential:
                xyz_ref_batch.requires_grad = True
            ref_loss, _, mse_px, mse_py, mse_pz = compute_data_loss(config, model, xyz_ref_batch, uvw_ref_batch, mask_ref_batch, standardization_factors, denormalize=True, reference=True)
        else:
            ref_loss, mse_px, mse_py, mse_pz = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        # Logging
        train_metrics = {
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
            "Loss/Pressure_y": mse_py.item() if mse_py is not None else 0.0,
            "Loss/Pressure_z": mse_pz.item() if mse_pz is not None else 0.0,
            "Loss/Ref": ref_loss.item(),
            "Loss/data_weight": data_weight if data_weight is not None else 0.0,
            "Loss/physics_weight": physics_weight if physics_weight is not None else 0.0,
            "Loss/boundary_weight": bound_weight if bound_weight is not None else 0.0,
        }

        for key, value in train_metrics.items():
            writer.add_scalar(key, value, it+1)

        if (it + 1) % config.training.log_iter == 0:
            print(f"[Iteration {it+1}] total_loss={total_loss.item():.4f}, "
            f"data_loss={data_loss.item():.4f}, ref_loss={ref_loss.item():.4f}, "
            f"physics_loss={physics_loss.item():.4E}, "
            f"it_time={round((time.time()-it_start_time)/config.training.log_iter, 5)}s, "
            f"total_time={round((time.time()-start_time)/60, 1)} min")

            wandb.log(train_metrics, step=it+1)
       
        # Save h5 predictions (lightweight, no metrics)
        if config.include_ref:
            if (it + 1) in getattr(config.training, 'save_h5_iters', []):
                save_h5_predictions(config, model, DEVICE, it+1, xyz_ref, u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask_ref, mask_flat_ref, U_max, standardization_factors)
                model.train()

        # Compare with reference data
        if config.include_ref:
            if (
                ((it + 1) % config.training.error_iter == 0 or it == 0)
                and (it + 1) != config.training.iterations
            ):
                metrics_eval = evaluate_predictions(config, model, DEVICE, it+1, xyz_ref, u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask_ref, mask_flat_ref, U_max, standardization_factors)
                if config.visualization.enabled:
                    plot_reference_comparison(config, model, DEVICE, it+1, xyz_ref,
                                                u, v, w, p, u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask_ref,
                                                mask_flat_ref, U_max, standardization_factors)

                # Log selected evaluation metrics to W&B.
                eval_log = build_eval_log(
                    metrics_eval,
                    include_pressure=config.training.reference_gradients,
                    prefix="Eval",
                )
                wandb.log(eval_log, step=it + 1)
                    
        # Save model at checkpoint
        if (it + 1) % config.training.summary_iter == 0 and (it + 1) != config.training.iterations:
            save_ckpt(
                f"{config.log_dir}/checkpoints/{config.network_name}_it{it+1:06d}.pth",
                model, Adam_optimizer, BFGS_optimizer if config.training.use_LBFGS else None,
                scheduler=scheduler, loss_weights=loss_weights, c_weights=c_weights, 
                standardization_factors=standardization_factors, U_max=U_max, 
                config_dict=config.to_dict(), iteration=it+1
            )

        # Save model at end of training
        if (it + 1) == config.training.iterations:
            save_ckpt(
                f"{config.log_dir}/checkpoints/{config.network_name}_it{it+1:06d}.pth",
                model, Adam_optimizer, BFGS_optimizer if config.training.use_LBFGS else None, 
                scheduler=scheduler, loss_weights=loss_weights, c_weights=c_weights, 
                standardization_factors=standardization_factors, U_max=U_max, 
                iteration=it+1
            )
            
            final_log_time = time.time() - start_time

            if config.include_ref:
                metrics_eval = evaluate_predictions(config, model, DEVICE, it+1, xyz_ref, 
                u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, 
                mask_ref, mask_flat_ref, U_max, standardization_factors, 
                save_pred=True
                )

                if config.visualization.enabled:
                    plot_reference_comparison(
                        config, model, DEVICE, it+1, xyz_ref,
                        u, v, w, p, u_ref, v_ref, w_ref,
                        p_ref, px_ref, py_ref, pz_ref,
                        mask_ref, mask_flat_ref, U_max,
                        standardization_factors,
                    )

                final_log_dict = build_eval_log(
                    metrics_eval,
                    include_pressure=config.training.reference_gradients,
                    prefix="Final",
                )

                final_log_dict["Final/training_time_min"] = round(
                    final_log_time / 60,
                    2,
                )
            
            else:
                final_log_dict = {
                    "Final/training_time_min": round(
                        final_log_time / 60,
                        2,
                    ),
                }
                predict_superresolved_grid(config, model, DEVICE, it+1, u, mask, U_max, save_pred=True)

            # Log metrics to wandb
            wandb.log(final_log_dict)

        # Self-adaptive collocation point weight update
        if config.sample_collocation and config.training.self_adaptive:
            step = it + 1
            do_refresh = (
                (step == config.training.K_initial) or
                (step > config.training.K_initial and ((step - config.training.K_initial) % config.training.K == 0))
            )

            if do_refresh:
                model.eval()  # eval mode for stable residuals

                N_c = xyz_collocation.shape[0]
                # choose a large subset for refresh (OOM-safe)
                subset_size = min(N_c, config.training.points_to_update)
                refresh_chunk = config.training.chunk_size

                # random subset of collocation indices (CPU numpy)
                subset_idx = np.random.choice(N_c, size=subset_size, replace=False)

                # compute per-point residuals on this subset, chunked
                resid_list = []
                with torch.enable_grad():
                    for s in range(0, subset_size, refresh_chunk):
                        e = min(s + refresh_chunk, subset_size)
                        idx_block = subset_idx[s:e]

                        X_block = torch.from_numpy(xyz_collocation[idx_block]).to(DEVICE).float()
                        X_block.requires_grad_(True)

                        uvw_block = model(X_block)
                        per_point, _, _ = navier_stokes_loss(
                            uvw_block, X_block, standardization_factors, config,
                            return_per_point=True, build_graph=True
                        )
                        # per_point: 1D tensor of per-point residual magnitudes
                        resid_list.append(per_point.detach().cpu().numpy())

                resid_subset = np.concatenate(resid_list)  # shape (subset_size,)

                # ---- map residuals -> attention weights (stable) ----
                tau = float(config.training.tau)
                eps = 1e-12

                # 1) Clean residuals
                res = np.asarray(resid_subset, dtype=np.float64)
                is_finite = np.isfinite(res)
                if not np.all(is_finite):
                    res = res[is_finite]
                    if res.size == 0:
                        # fallback: uniform weights for this refresh
                        w_new = np.ones_like(subset_idx, dtype=np.float64)
                        beta = float(config.training.beta)
                        c_weights[subset_idx] = (1.0 - beta) * c_weights[subset_idx] + beta * w_new
                        c_weights = np.maximum(c_weights, eps)
                        c_weights /= (c_weights.mean() + eps)
                        model.train()
                        continue

                # 2) Clip extreme tails to prevent exp overflow but keep ranking
                p99 = np.percentile(res, 99.5)
                res = np.minimum(res, p99)

                # 3) Stable softmax via centering
                r = res / max(tau, eps)
                r -= np.max(r)
                z = np.exp(r)
                den = z.mean()
                if not np.isfinite(den) or den <= 0:
                    w_new_core = np.ones_like(z, dtype=np.float64)
                else:
                    w_new_core = z / (den + eps)

                # 4) Clip & sanitize
                clip_vals = list(config.training.weight_clip)
                clip_min, clip_max = min(clip_vals), max(clip_vals)
                w_new_core = np.clip(w_new_core, clip_min, clip_max)
                w_new_core = np.nan_to_num(w_new_core, nan=1.0, posinf=clip_max, neginf=clip_min)

                # 5) Use the newly computed weights for the sampled subset
                w_new = w_new_core 

                # 6) EMA blend + renorm + floor
                beta = float(config.training.beta)
                c_weights[subset_idx] = (1.0 - beta) * c_weights[subset_idx] + beta * w_new

                c_weights = np.nan_to_num(c_weights, nan=1.0, posinf=clip_max, neginf=clip_min)
                c_weights = np.maximum(c_weights, eps)
                c_weights /= (c_weights.mean() + eps)

                # diagnostics (optional)
                a, b = c_weights.min(), c_weights.max()
                print(f"[Self-adaptive] Refreshed c_weights on {subset_size} / {N_c} "
                    f"min={a:.3g}, max={b:.3g})")
                
                # Log metrics to wandb
                log_dict = {
                    "c_weights/min":a,
                    "c_weights/max":b,
                    "c_weights/p50": float(np.percentile(c_weights, 50)),
                    "c_weights/p90": float(np.percentile(c_weights, 90)),
                }

                # Log to W&B
                wandb.log(log_dict, step=it + 1)                
                
                del X_block, uvw_block, per_point
                torch.cuda.empty_cache()

                model.train()

    writer.close()
    wandb.finish()
