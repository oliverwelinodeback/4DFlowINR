"""
train_meta_v2.py - Meta-Learning for 4D Flow MRI Super-Resolution

Supports multiple meta-learning algorithms:
- MAML (Model-Agnostic Meta-Learning) - Second-order, full gradient tracking
- FOMAML (First-Order MAML) - First-order approximation, memory efficient
- Reptile - First-order alternative, no higher library needed

Key features:
- Simple, direct data loss computation (compute_loss_from_pred)
- Optional physics loss (Navier-Stokes) in inner and outer loops
- SGD inner loop, Adam outer loop
- Case-specific venc handling for cosine loss

Config options:
- meta_method: 'MAML', 'FOMAML', or 'Reptile'
- reptile_epsilon: Interpolation factor for Reptile (default: 1.0)
"""

import torch
import torch.nn.functional as F
import numpy as np
import wandb
from pathlib import Path
import os
import higher
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

# Import existing utils
from utils.prepare_data import (
    load_data, prepare_data, extract_fluid_region,
    load_ref_data, prepare_ref_data, sample_collocation_points,
    sample_boundary_points
)
from utils.loss_utils import compute_data_loss, compute_physics_loss, navier_stokes_loss
import networks


# ==========================================
# Simple Loss Function (Original Working Version)
# ==========================================
def compute_loss_from_pred(config, pred, velocities, venc, U_max=None):
    """
    Simple data loss computation - matches original working MAML implementation.

    This function computes loss directly from predictions without calling the model,
    and uses the case-specific venc for proper denormalization in cosine loss.

    Args:
        config: Configuration object
        pred: Model predictions [N, 3] (u, v, w)
        velocities: Ground truth velocities [N, 3] (u, v, w)
        venc: Case-specific velocity encoding value
        U_max: Maximum velocity for denormalization (optional)

    Returns:
        Scalar loss tensor
    """
    if config.training.use_cosine:
        # Denormalize for cosine loss
        if config.vel_normalization == "characteristic":
            U = config.constants.U
            pred_denorm = pred * U
            vel_denorm = velocities * U
        else:
            # Use passed U_max or fall back to config
            U_max_val = U_max if U_max is not None else getattr(config, 'U_max', 1.0)
            pred_denorm = pred * U_max_val
            vel_denorm = velocities * U_max_val

        # Use case-specific venc for Kv
        Kv = torch.pi / venc

        u_pred, v_pred, w_pred = pred_denorm[:, 0], pred_denorm[:, 1], pred_denorm[:, 2]
        u, v, w = vel_denorm[:, 0], vel_denorm[:, 1], vel_denorm[:, 2]

        cosine_u = 1 - torch.cos(Kv * (u_pred - u))
        cosine_v = 1 - torch.cos(Kv * (v_pred - v))
        cosine_w = 1 - torch.cos(Kv * (w_pred - w))

        loss = torch.mean(
            config.training.u_weight * cosine_u +
            config.training.v_weight * cosine_v +
            config.training.w_weight * cosine_w
        )
    else:
        # MSE loss (no denormalization needed)
        mse_u = (pred[:, 0] - velocities[:, 0]) ** 2
        mse_v = (pred[:, 1] - velocities[:, 1]) ** 2
        mse_w = (pred[:, 2] - velocities[:, 2]) ** 2

        loss = torch.mean(
            config.training.u_weight * mse_u +
            config.training.v_weight * mse_v +
            config.training.w_weight * mse_w
        )
    return loss


# ==========================================
# 1. Data Container
# ==========================================
@dataclass
class CaseData:
    """Container for patient case data."""
    case_name: str
    venc: float
    std_factors: List[float]
    U_max: float

    # LR Data - Used for MAML (support + query)
    coords_LR: torch.Tensor
    vel_LR: torch.Tensor
    n_LR: int
    mask_LR: torch.Tensor  # Mask for LR data

    # HR Data - ONLY for monitoring (not in meta-loss)
    coords_HR: Optional[torch.Tensor] = None
    vel_HR: Optional[torch.Tensor] = None
    n_HR: Optional[int] = None

    # Pre-sampled collocation points for physics (GPU tensors, ready to use)
    coll_points_pool: Optional[torch.Tensor] = None
    n_coll_pool: Optional[int] = None

    # Pre-sampled boundary points (GPU tensors)
    boundary_points: Optional[torch.Tensor] = None
    n_boundary: Optional[int] = None


# ==========================================
# 2. Helper: Load Multiple Cases (LR + HR Pair)
# ==========================================
def load_all_cases(file_list, config, device):
    """
    Loads pairs of Low-Res (for meta-learning) and High-Res (for monitoring).
    """
    loaded_cases = []
    print(f"\nProcessing {len(file_list)} cases (LR + Ref Pair)...")

    pattern = re.compile(r"_LR_.*_newMask")

    for case_path in file_list:
        case_name = Path(case_path).stem
        print(f"  Loading: {case_name}")

        original_data_file = config.data_file
        original_venc = config.constants.venc

        config.data_file = case_path
        config.constants.venc = config.meta_learning.case_venc.get(
            case_name, config.constants.venc
        )

        try:
            # --- Load Low-Resolution (LR) Data ---
            u, v, w, p, px, py, pz, mask, _ = load_data(config)

            uvw_data, xyz_data, mask_flat, boundary_mask_flat, std_factors, U_max = prepare_data(
                config, u, v, w, p, px, py, pz, mask
            )

            uvw_LR, xyz_LR = extract_fluid_region(
                uvw_data, xyz_data, mask_flat, print_fluid_points=False
            )

            # Create mask tensor for LR points (all ones in fluid region)
            mask_LR = torch.ones(len(xyz_LR), 1, device=device)

            # --- Load High-Resolution (Reference) Data for MONITORING ---
            ref_path = pattern.sub("", case_path)
            coords_HR, vel_HR, n_HR = None, None, None

            if os.path.exists(ref_path):
                config.data_file_ref = ref_path

                u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask_ref = load_ref_data(config)

                uvw_data_ref, xyz_data_ref, mask_flat_ref, _ = prepare_ref_data(
                    config, u, u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref,
                    mask_ref, U_max
                )

                uvw_HR, xyz_HR = extract_fluid_region(
                    uvw_data_ref, xyz_data_ref, mask_flat_ref
                )

                coords_HR = torch.from_numpy(xyz_HR).float().to(device)
                vel_HR = torch.from_numpy(uvw_HR).float().to(device)
                n_HR = len(xyz_HR)

                print(f"    -> LR: {len(xyz_LR):,} pts | HR: {n_HR:,} pts (monitoring only)")
            else:
                print(f"    -> LR: {len(xyz_LR):,} pts | HR: Not found (will skip HR monitoring)")

            # --- Pre-sample collocation points ---
            coll_points_pool = None
            n_coll_pool = None

            # Use physics loss settings from config
            use_physics = getattr(config.meta_learning, 'use_physics_loss', True)
            if use_physics:
                coll_points_inner = getattr(config.meta_learning, 'coll_points_inner', 5000)
                pool_size = coll_points_inner * 20  # Large pool for random sampling

                original_coll_points = config.collocation_points
                config.collocation_points = pool_size

                coll_pool_np = sample_collocation_points(config, xyz_data, mask_flat)
                coll_points_pool = torch.from_numpy(coll_pool_np).float().to(device)
                n_coll_pool = len(coll_points_pool)

                config.collocation_points = original_coll_points
                print(f"    -> Pre-sampled {n_coll_pool:,} collocation points on GPU")

            # --- Pre-sample boundary points ---
            boundary_points = None
            n_boundary = None

            use_boundary = getattr(config.meta_learning, 'use_boundary_loss', config.sample_boundary)
            if use_boundary:
                boundary_points_inner = getattr(config.meta_learning, 'boundary_points_inner', 2000)
                boundary_pool_size = boundary_points_inner * 10

                boundary_pool_np = sample_boundary_points(config, xyz_data, boundary_mask_flat)
                if boundary_pool_np is not None and len(boundary_pool_np) > 0:
                    boundary_points = torch.from_numpy(boundary_pool_np).float().to(device)
                    n_boundary = len(boundary_points)
                    print(f"    -> Pre-sampled {n_boundary:,} boundary points on GPU")

            # --- Store Data ---
            case_data = CaseData(
                case_name=case_name,
                venc=config.constants.venc,
                std_factors=std_factors,
                U_max=U_max,

                coords_LR=torch.from_numpy(xyz_LR).float().to(device),
                vel_LR=torch.from_numpy(uvw_LR).float().to(device),
                n_LR=len(xyz_LR),
                mask_LR=mask_LR,

                coords_HR=coords_HR,
                vel_HR=vel_HR,
                n_HR=n_HR,

                coll_points_pool=coll_points_pool,
                n_coll_pool=n_coll_pool,

                boundary_points=boundary_points,
                n_boundary=n_boundary,
            )
            loaded_cases.append(case_data)

        except Exception as e:
            print(f"    [ERROR] Failed to load {case_name}: {e}")
            import traceback
            traceback.print_exc()

        config.data_file = original_data_file
        config.constants.venc = original_venc

    return loaded_cases


# ==========================================
# 3. MAML Step (Original Working Version)
# ==========================================
def maml_step_v2(model, meta_optimizer, all_cases_data, case_indices, config, device, current_iter=0, meta_method='FOMAML'):
    """
    MAML step with simple, direct loss computation.

    Inner loop: Data loss + optional physics loss
    Outer loop: Evaluate on query set

    Uses SGD for inner loop (required by `higher` library).
    Supports FOMAML (first-order) and full MAML (second-order).
    Uses case-specific venc for proper cosine loss denormalization.

    Memory-efficient physics mode (use_physics_outer_only=True):
    - Inner loop: Data loss ONLY (no physics, saves memory)
    - Outer loop: Data loss + Physics loss (guides meta-objective)
    This learns a physics-aware initialization without the memory cost
    of computing physics gradients through the inner loop.

    Curriculum learning (physics_curriculum_start/end):
    - Before curriculum_start: Pure data-driven (like working MAML)
    - Between start and end: Physics weight ramps up linearly
    - After curriculum_end: Full physics_weight applied
    """
    meta_optimizer.zero_grad()
    query_losses = []
    support_losses = []
    physics_losses_inner = []
    physics_losses_outer = []
    data_losses_outer = []  # NEW: Track data loss separately
    inner_step_improvements = []
    per_task_query_losses = []

    # FOMAML uses first-order gradients (no second-order graph tracking)
    use_first_order = (meta_method == 'FOMAML')

    # Inner optimizer (SGD - required by higher)
    inner_opt = torch.optim.SGD(model.parameters(), lr=config.meta_learning.inner_lr)

    # Get inner loop settings (once, outside the case loop)
    inner_points = getattr(config.meta_learning, 'inner_points', 5000)
    use_physics_inner = getattr(config.meta_learning, 'use_physics_loss', False)
    coll_points_inner = getattr(config.meta_learning, 'coll_points_inner', 3000)
    physics_weight_base = getattr(config.meta_learning, 'physics_weight', 1.0)

    # ==========================================
    # CURRICULUM LEARNING FOR PHYSICS
    # ==========================================
    # Gradually introduce physics to avoid conflicting gradients early in training
    physics_curriculum_start = getattr(config.meta_learning, 'physics_curriculum_start', 0)
    physics_curriculum_end = getattr(config.meta_learning, 'physics_curriculum_end', 0)

    # Compute effective physics weight based on curriculum
    if physics_curriculum_end > physics_curriculum_start and current_iter < physics_curriculum_end:
        if current_iter < physics_curriculum_start:
            # Phase 1: Pure data-driven (no physics)
            effective_physics_weight = 0.0
        else:
            # Phase 2: Ramp up physics weight linearly
            progress = (current_iter - physics_curriculum_start) / (physics_curriculum_end - physics_curriculum_start)
            progress = min(1.0, max(0.0, progress))
            effective_physics_weight = physics_weight_base * progress
    else:
        # No curriculum or past curriculum end: use full physics weight
        effective_physics_weight = physics_weight_base

    # Memory-efficient mode: Physics in outer loop only
    # When True: Inner loop = data only, Outer loop = data + physics
    # This saves significant memory while still learning physics-aware initialization
    use_physics_outer_only = getattr(config.meta_learning, 'use_physics_outer_only', False)

    # If physics_outer_only is enabled, disable physics in inner loop
    if use_physics_outer_only:
        use_physics_inner_actual = False
        use_physics_outer = True
    else:
        use_physics_inner_actual = use_physics_inner
        use_physics_outer = use_physics_inner  # Same as before

    for case_idx in case_indices:
        case = all_cases_data[case_idx]

        # --- A. Split LR Data into Support/Query ---
        n_total = case.n_LR
        support_fraction = getattr(config.meta_learning, 'support_fraction', 0.5)
        n_support = int(n_total * support_fraction)

        perm = torch.randperm(n_total, device=device)
        idx_support = perm[:n_support]
        idx_query = perm[n_support:]

        support_coords = case.coords_LR[idx_support]
        support_vel = case.vel_LR[idx_support]

        query_coords = case.coords_LR[idx_query]
        query_vel = case.vel_LR[idx_query]

        # --- B. Higher Inner Loop ---

        with higher.innerloop_ctx(
            model,
            inner_opt,
            copy_initial_weights=False,
            track_higher_grads=not use_first_order
        ) as (fmodel, diffopt):

            # Track loss before first inner step
            with torch.no_grad():
                pred_init = fmodel(support_coords)
                loss_before_adapt = compute_loss_from_pred(
                    config, pred_init, support_vel, case.venc, case.U_max
                ).item()

            # Inner Loop: Simple data loss (original working version)
            # Pre-sample collocation points ONCE before inner loop (original behavior)
            coll_support = None
            if use_physics_inner_actual and case.coll_points_pool is not None:
                n_coll = case.coll_points_pool.shape[0]
                if coll_points_inner < n_coll:
                    coll_idx = torch.randperm(n_coll, device=device)[:coll_points_inner]
                    coll_support = case.coll_points_pool[coll_idx]
                else:
                    coll_support = case.coll_points_pool

            for step in range(config.meta_learning.inner_steps):
                # Use ALL support points every step (original behavior - no subsampling)
                pred = fmodel(support_coords)

                # Data loss (simple, direct computation)
                data_loss_support = compute_loss_from_pred(
                    config, pred, support_vel, case.venc, case.U_max
                )

                # Physics loss (optional) - uses pre-sampled coll_support
                # NOTE: Disabled when use_physics_outer_only=True (memory-efficient mode)
                physics_loss_support = torch.tensor(0.0, device=device)
                if use_physics_inner_actual and coll_support is not None:
                    coll_support.requires_grad = True
                    pred_coll = fmodel(coll_support)

                    # Compute Navier-Stokes residual
                    # navier_stokes_loss returns (momentum_loss, div_loss) when return_per_point=False
                    momentum_loss, div_loss = navier_stokes_loss(
                        pred_coll, coll_support, case.std_factors, config,
                        return_per_point=False, build_graph=True
                    )
                    physics_loss_support = momentum_loss + div_loss

                    if step == config.meta_learning.inner_steps - 1:  # Last step
                        physics_losses_inner.append(physics_loss_support.item())

                # Total inner loss
                if use_physics_inner_actual and coll_support is not None:
                    total_loss_support = data_loss_support + physics_weight * physics_loss_support
                else:
                    total_loss_support = data_loss_support

                # Inner update (SGD step)
                diffopt.step(total_loss_support)

            # Record final support loss
            with torch.no_grad():
                pred_final = fmodel(support_coords)
                final_support_loss = compute_loss_from_pred(
                    config, pred_final, support_vel, case.venc, case.U_max
                ).item()
                support_losses.append(final_support_loss)
                inner_improvement = loss_before_adapt - final_support_loss
                inner_step_improvements.append(inner_improvement)

            # --- C. Outer Loop: Evaluate on Query Set ---
            # Use simple data loss for query (outer) evaluation
            pred_query = fmodel(query_coords)
            query_data_loss = compute_loss_from_pred(
                config, pred_query, query_vel, case.venc, case.U_max
            )

            # Physics loss on query (for outer loop)
            # NOTE: When use_physics_outer_only=True, physics is computed HERE only
            # This makes the meta-objective physics-aware without inner loop memory cost
            query_physics_loss = torch.tensor(0.0, device=device)
            if use_physics_outer and case.coll_points_pool is not None:
                # Sample collocation points for outer loop physics
                n_coll = case.coll_points_pool.shape[0]
                coll_points_outer = getattr(config.meta_learning, 'coll_points_outer', coll_points_inner)
                if coll_points_outer < n_coll:
                    coll_idx = torch.randperm(n_coll, device=device)[:coll_points_outer]
                    coll_query = case.coll_points_pool[coll_idx]
                else:
                    coll_query = case.coll_points_pool

                coll_query.requires_grad = True
                pred_coll_query = fmodel(coll_query)

                # navier_stokes_loss returns (momentum_loss, div_loss) when return_per_point=False
                momentum_loss_query, div_loss_query = navier_stokes_loss(
                    pred_coll_query, coll_query, case.std_factors, config,
                    return_per_point=False, build_graph=True
                )
                query_physics_loss = momentum_loss_query + div_loss_query

            # Total query loss (for meta-gradient)
            # Uses effective_physics_weight from curriculum learning
            total_query_loss = query_data_loss + effective_physics_weight * query_physics_loss
            query_losses.append(total_query_loss)

            # Track metrics (separate data and physics for analysis)
            data_losses_outer.append(query_data_loss.item())
            physics_losses_outer.append(query_physics_loss.item())
            per_task_query_losses.append((case.case_name, total_query_loss.item()))

    # --- D. Meta Update ---
    if not query_losses:
        return 0.0, {}

    meta_loss = torch.stack(query_losses).mean()
    meta_loss.backward()

    # Compute gradient norm
    total_grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.data.norm(2).item() ** 2
    meta_grad_norm = total_grad_norm ** 0.5

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    meta_optimizer.step()

    # Find hardest task
    hardest_task_name, hardest_task_loss = max(per_task_query_losses, key=lambda x: x[1]) if per_task_query_losses else ("N/A", 0.0)

    metrics = {
        'support_loss': np.mean(support_losses) if support_losses else 0.0,
        'query_loss': meta_loss.item(),
        'query_data_loss': np.mean(data_losses_outer) if data_losses_outer else 0.0,  # NEW
        'query_physics_loss': np.mean(physics_losses_outer) if physics_losses_outer else 0.0,  # NEW (raw, unweighted)
        'meta_grad_norm': meta_grad_norm,
        'inner_step_improvement': np.mean(inner_step_improvements) if inner_step_improvements else 0.0,
        'hardest_task_name': hardest_task_name,
        'hardest_task_loss': hardest_task_loss,
        'physics_loss_inner': np.mean(physics_losses_inner) if physics_losses_inner else 0.0,
        'physics_loss_outer': np.mean(physics_losses_outer) if physics_losses_outer else 0.0,
        'effective_physics_weight': effective_physics_weight,  # NEW: Track curriculum progress
    }

    return meta_loss.item(), metrics


# ==========================================
# 3b. Reptile Step (First-Order Alternative)
# ==========================================
def reptile_step(model, meta_optimizer, all_cases_data, case_indices, config, device):
    """
    Reptile meta-learning step.

    Unlike MAML, Reptile doesn't differentiate through the inner loop.
    Instead, it moves the initialization toward the average of adapted parameters.

    Algorithm:
    1. For each task: save init params, run SGD, get adapted params
    2. Compute average direction: (adapted - init) across tasks
    3. Update init params in that direction

    Benefits over MAML:
    - No second-order gradients (much faster, less memory)
    - No need for `higher` library
    - More stable training

    Reference: Nichol et al., "On First-Order Meta-Learning Algorithms" (2018)
    """
    support_losses = []
    query_losses_values = []
    physics_losses_inner = []
    inner_step_improvements = []
    per_task_query_losses = []

    # Get settings
    use_physics_inner = getattr(config.meta_learning, 'use_physics_loss', False)
    coll_points_inner = getattr(config.meta_learning, 'coll_points_inner', 3000)
    physics_weight = getattr(config.meta_learning, 'physics_weight', 1.0)
    reptile_epsilon = getattr(config.meta_learning, 'reptile_epsilon', 1.0)  # Interpolation factor

    # Store initial parameters
    init_params = {name: param.clone() for name, param in model.named_parameters()}

    # Accumulate parameter differences across tasks
    param_diffs = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    for case_idx in case_indices:
        case = all_cases_data[case_idx]

        # --- A. Split LR Data into Support/Query ---
        n_total = case.n_LR
        support_fraction = getattr(config.meta_learning, 'support_fraction', 0.5)
        n_support = int(n_total * support_fraction)

        perm = torch.randperm(n_total, device=device)
        idx_support = perm[:n_support]
        idx_query = perm[n_support:]

        support_coords = case.coords_LR[idx_support]
        support_vel = case.vel_LR[idx_support]

        query_coords = case.coords_LR[idx_query]
        query_vel = case.vel_LR[idx_query]

        # --- B. Reset to initial parameters ---
        for name, param in model.named_parameters():
            param.data.copy_(init_params[name])

        # --- C. Inner Loop (Standard SGD, no higher library) ---
        inner_opt = torch.optim.SGD(model.parameters(), lr=config.meta_learning.inner_lr)

        # Pre-sample collocation points ONCE
        coll_support = None
        if use_physics_inner and case.coll_points_pool is not None:
            n_coll = case.coll_points_pool.shape[0]
            if coll_points_inner < n_coll:
                coll_idx = torch.randperm(n_coll, device=device)[:coll_points_inner]
                coll_support = case.coll_points_pool[coll_idx]
            else:
                coll_support = case.coll_points_pool

        # Track loss before adaptation
        with torch.no_grad():
            pred_init = model(support_coords)
            loss_before_adapt = compute_loss_from_pred(
                config, pred_init, support_vel, case.venc, case.U_max
            ).item()

        # Inner loop adaptation
        for step in range(config.meta_learning.inner_steps):
            inner_opt.zero_grad()

            # Forward pass on ALL support points
            pred = model(support_coords)

            # Data loss
            data_loss = compute_loss_from_pred(
                config, pred, support_vel, case.venc, case.U_max
            )

            # Physics loss (optional)
            physics_loss = torch.tensor(0.0, device=device)
            if use_physics_inner and coll_support is not None:
                coll_support.requires_grad = True
                pred_coll = model(coll_support)

                # navier_stokes_loss returns (momentum_loss, div_loss) when return_per_point=False
                momentum_loss, div_loss = navier_stokes_loss(
                    pred_coll, coll_support, case.std_factors, config,
                    return_per_point=False, build_graph=True
                )
                physics_loss = momentum_loss + div_loss

                if step == config.meta_learning.inner_steps - 1:
                    physics_losses_inner.append(physics_loss.item())

            # Total loss
            if use_physics_inner and coll_support is not None:
                total_loss = data_loss + physics_weight * physics_loss
            else:
                total_loss = data_loss

            # Backward and step
            total_loss.backward()
            inner_opt.step()

        # --- D. Record adapted parameters difference ---
        for name, param in model.named_parameters():
            param_diffs[name] += (param.data - init_params[name])

        # --- E. Evaluate on query set (for logging only) ---
        with torch.no_grad():
            # Final support loss
            pred_final = model(support_coords)
            final_support_loss = compute_loss_from_pred(
                config, pred_final, support_vel, case.venc, case.U_max
            ).item()
            support_losses.append(final_support_loss)
            inner_step_improvements.append(loss_before_adapt - final_support_loss)

            # Query loss
            pred_query = model(query_coords)
            query_loss = compute_loss_from_pred(
                config, pred_query, query_vel, case.venc, case.U_max
            ).item()
            query_losses_values.append(query_loss)
            per_task_query_losses.append((case.case_name, query_loss))

    # --- F. Reptile Update: Move init params toward adapted params ---
    n_tasks = len(case_indices)
    if n_tasks > 0:
        # Average the parameter differences
        for name in param_diffs:
            param_diffs[name] /= n_tasks

        # Update model parameters: θ = θ + ε * (θ' - θ)
        # Where θ' is average adapted params, θ is init params
        # This is equivalent to: θ = θ + ε * avg_diff
        meta_optimizer.zero_grad()

        # Set gradients manually (negative of diff because optimizer subtracts)
        for name, param in model.named_parameters():
            param.data.copy_(init_params[name])  # Reset to init
            # We want: param = param + lr * diff
            # Optimizer does: param = param - lr * grad
            # So: grad = -diff / outer_lr, then optimizer gives us param + diff
            # But simpler: just manually update
            param.data.add_(param_diffs[name], alpha=reptile_epsilon * config.meta_learning.outer_lr)

    # Find hardest task
    hardest_task_name, hardest_task_loss = max(per_task_query_losses, key=lambda x: x[1]) if per_task_query_losses else ("N/A", 0.0)

    metrics = {
        'support_loss': np.mean(support_losses) if support_losses else 0.0,
        'query_loss': np.mean(query_losses_values) if query_losses_values else 0.0,
        'meta_grad_norm': 0.0,  # Not applicable for Reptile
        'inner_step_improvement': np.mean(inner_step_improvements) if inner_step_improvements else 0.0,
        'hardest_task_name': hardest_task_name,
        'hardest_task_loss': hardest_task_loss,
        'use_first_order': True,  # Reptile is first-order
        'physics_loss_inner': np.mean(physics_losses_inner) if physics_losses_inner else 0.0,
        'physics_loss_outer': 0.0,  # Not computed in Reptile
    }

    avg_query_loss = np.mean(query_losses_values) if query_losses_values else 0.0
    return avg_query_loss, metrics


# ==========================================
# 4. Validation (matches training loss computation)
# ==========================================
def validate_meta_v2(model, val_cases_data, config, device):
    """
    Validation using simple loss function (matches training).
    Uses case-specific venc for proper cosine loss denormalization.
    """
    model.eval()
    pre_losses_LR = []
    post_losses_LR = []
    hr_losses = []
    per_case_losses = []

    inner_opt = torch.optim.SGD(model.parameters(), lr=config.meta_learning.inner_lr)
    inner_points = getattr(config.meta_learning, 'inner_points', 5000)
    use_physics_inner = getattr(config.meta_learning, 'use_physics_loss', False)
    coll_points_inner = getattr(config.meta_learning, 'coll_points_inner', 3000)
    physics_weight = getattr(config.meta_learning, 'physics_weight', 1.0)

    for case in val_cases_data:
        n_total = case.n_LR
        support_fraction = getattr(config.meta_learning, 'support_fraction', 0.5)
        n_support = int(n_total * support_fraction)

        perm = torch.randperm(n_total, device=device)
        support_coords = case.coords_LR[perm[:n_support]]
        support_vel = case.vel_LR[perm[:n_support]]
        query_coords = case.coords_LR[perm[n_support:]]
        query_vel = case.vel_LR[perm[n_support:]]

        # Pre-adaptation (evaluate on query set before adaptation)
        with torch.no_grad():
            pred_pre = model(query_coords)
            pre_loss = compute_loss_from_pred(
                config, pred_pre, query_vel, case.venc, case.U_max
            ).item()
            pre_losses_LR.append(pre_loss)

        # Adaptation
        with torch.enable_grad():
            with higher.innerloop_ctx(
                model, inner_opt,
                copy_initial_weights=False,
                track_higher_grads=False
            ) as (fmodel, diffopt):

                # Pre-sample collocation points ONCE before inner loop (original behavior)
                coll_support = None
                if use_physics_inner and case.coll_points_pool is not None:
                    n_coll = case.coll_points_pool.shape[0]
                    if coll_points_inner < n_coll:
                        coll_idx = torch.randperm(n_coll, device=device)[:coll_points_inner]
                        coll_support = case.coll_points_pool[coll_idx]
                    else:
                        coll_support = case.coll_points_pool

                for _ in range(config.meta_learning.inner_steps):
                    # Use ALL support points every step (original behavior)
                    pred = fmodel(support_coords)
                    data_loss = compute_loss_from_pred(
                        config, pred, support_vel, case.venc, case.U_max
                    )

                    # Physics loss (optional) - uses pre-sampled coll_support
                    physics_loss = torch.tensor(0.0, device=device)
                    if use_physics_inner and coll_support is not None:
                        coll_support.requires_grad = True
                        pred_coll = fmodel(coll_support)
                        momentum_loss, div_loss = navier_stokes_loss(
                            pred_coll, coll_support, case.std_factors, config,
                            return_per_point=False, build_graph=True
                        )
                        physics_loss = momentum_loss + div_loss

                    # Total loss
                    if use_physics_inner and coll_support is not None:
                        total_loss = data_loss + physics_weight * physics_loss
                    else:
                        total_loss = data_loss

                    diffopt.step(total_loss)

                # Post-adaptation (evaluate on query set after adaptation)
                with torch.no_grad():
                    pred_post = fmodel(query_coords)
                    post_loss = compute_loss_from_pred(
                        config, pred_post, query_vel, case.venc, case.U_max
                    ).item()
                    post_losses_LR.append(post_loss)
                    per_case_losses.append((case.case_name, post_loss))

                    # HR monitoring
                    if case.coords_HR is not None:
                        n_hr_sample = min(5000, case.n_HR)
                        idx_hr = torch.randperm(case.n_HR, device=device)[:n_hr_sample]
                        hr_x = case.coords_HR[idx_hr]
                        hr_y = case.vel_HR[idx_hr]

                        pred_hr = fmodel(hr_x)
                        hr_mse = F.mse_loss(pred_hr[:, :3], hr_y[:, :3])
                        hr_losses.append(hr_mse.item())

    model.train()

    hardest_task_name, hardest_task_loss = max(per_case_losses, key=lambda x: x[1]) if per_case_losses else ("N/A", 0.0)

    return {
        'pre_LR': np.mean(pre_losses_LR),
        'post_LR': np.mean(post_losses_LR),
        'improvement_LR': np.mean(pre_losses_LR) - np.mean(post_losses_LR),
        'hr_quality': np.mean(hr_losses) if hr_losses else None,
        'hardest_task_name': hardest_task_name,
        'hardest_task_loss': hardest_task_loss,
    }


# ==========================================
# 5. Main Training Loop
# ==========================================
def train_meta_learning_v2(config, run_name, use_sweep=False):
    """
    Main meta-learning training function.
    Supports MAML, FOMAML, and Reptile via config.meta_learning.meta_method.
    """
    device = torch.device('cuda')
    print("\n" + "="*60)
    print("Meta-Learning with TrainingStep Integration")
    print("="*60)

    os.makedirs(config.log_dir, exist_ok=True)

    # Determine meta-learning method: 'MAML', 'FOMAML', or 'Reptile'
    meta_method = config.meta_learning.meta_method.upper()

    if meta_method == 'REPTILE':
        method_str = "Reptile (First-Order, No Higher)"
    elif meta_method == 'FOMAML':
        method_str = "FOMAML (First-Order)"
    else:  # MAML
        method_str = "MAML (Second-Order)"
        meta_method = 'MAML'  # Normalize

    # Check for memory-efficient physics mode
    use_physics_outer_only = getattr(config.meta_learning, 'use_physics_outer_only', False)
    use_physics_loss = getattr(config.meta_learning, 'use_physics_loss', False)

    print(f"\n[Configuration]")
    print(f"  Method: {method_str}")
    print(f"  Inner LR: {config.meta_learning.inner_lr}")
    print(f"  Inner Steps: {config.meta_learning.inner_steps}")
    print(f"  Outer LR: {config.meta_learning.outer_lr}")
    print(f"  Meta Batch Size: {config.meta_learning.meta_batch_size}")
    print(f"  Use Physics Loss: {use_physics_loss}")
    if use_physics_outer_only:
        print(f"  Physics Mode: OUTER LOOP ONLY (memory-efficient)")
        print(f"    -> Inner loop: Data loss only")
        print(f"    -> Outer loop: Data + Physics loss")
    elif use_physics_loss:
        print(f"  Physics Mode: Inner + Outer loops")
    print(f"  Use Boundary Loss: {getattr(config.meta_learning, 'use_boundary_loss', config.sample_boundary)}")
    if meta_method == 'REPTILE':
        reptile_epsilon = getattr(config.meta_learning, 'reptile_epsilon', 1.0)
        print(f"  Reptile Epsilon: {reptile_epsilon}")

    # Load data
    train_files = config.meta_learning.train_cases
    val_files = config.meta_learning.val_cases

    print("\n--- Loading Training Data ---")
    train_cases_data = load_all_cases(train_files, config, device)

    print("\n--- Loading Validation Data ---")
    val_cases_data = load_all_cases(val_files, config, device)

    if not train_cases_data:
        raise ValueError("No training data loaded!")

    print(f"\nLoaded {len(train_cases_data)} training, {len(val_cases_data)} validation cases")

    # Create model
    model = networks.WIRE(
        in_dim=config.network.in_dim,
        out_dim=config.network.out_dim,
        depth=config.network.depth,
        hidden_features=config.network.hidden_features,
        first_omega_0=config.network.omega_0,
        hidden_omega_0=config.network.omega_0,
        scale=config.network.sigma_0,
        complex=config.network.complex
    ).to(device)

    print(f"\n[Model] WIRE - {sum(p.numel() for p in model.parameters()):,} parameters")

    # Meta optimizer (Adam)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=config.meta_learning.outer_lr)

    # Scheduler
    scheduler = None
    if getattr(config.meta_learning, 'use_scheduler', False):
        scheduler_gamma = getattr(config.meta_learning, 'scheduler_gamma', 0.9991)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(meta_optimizer, gamma=scheduler_gamma)

    # Training loop
    print("\n--- Starting Meta-Learning ---")
    print(f"  Max Iterations: {config.meta_learning.max_iters}")

    best_val_post_LR = np.inf

    for meta_iter in range(config.meta_learning.max_iters):
        n_cases = min(config.meta_learning.meta_batch_size, len(train_cases_data))
        case_indices = np.random.choice(len(train_cases_data), n_cases, replace=False)

        # Choose step function based on meta-learning method
        if meta_method == 'REPTILE':
            meta_loss, train_metrics = reptile_step(
                model, meta_optimizer, train_cases_data, case_indices, config, device
            )
        else:
            # MAML or FOMAML
            meta_loss, train_metrics = maml_step_v2(
                model, meta_optimizer, train_cases_data, case_indices, config, device,
                current_iter=meta_iter, meta_method=meta_method
            )

        if scheduler is not None:
            scheduler.step()

        # Logging
        if (meta_iter + 1) % 10 == 0:
            eff_phys_w = train_metrics.get('effective_physics_weight', 0.0)
            log_str = f"[Iter {meta_iter+1:04d}] Loss: {meta_loss:.5f}"
            log_str += f" | Support: {train_metrics['support_loss']:.5f}"
            log_str += f" | Query: {train_metrics['query_loss']:.5f}"
            if train_metrics['physics_loss_outer'] > 0:
                log_str += f" | Phys: {train_metrics['physics_loss_outer']:.5f} (w={eff_phys_w:.3f})"
            print(log_str)

            wandb.log({
                'Train/Meta_Loss': meta_loss,
                'Train/Support_Loss': train_metrics['support_loss'],
                'Train/Query_Loss': train_metrics['query_loss'],
                'Train/Query_Data_Loss': train_metrics.get('query_data_loss', 0.0),  # NEW
                'Train/Query_Physics_Loss': train_metrics.get('query_physics_loss', 0.0),  # NEW (raw)
                'Train/Physics_Loss_Inner': train_metrics['physics_loss_inner'],
                'Train/Physics_Loss_Outer': train_metrics['physics_loss_outer'],
                'Train/Effective_Physics_Weight': eff_phys_w,  # NEW: Curriculum progress
                'Train/Learning_Rate': meta_optimizer.param_groups[0]['lr'],
                'Stability/Meta_Grad_Norm': train_metrics['meta_grad_norm'],
                'Stability/Inner_Improvement': train_metrics['inner_step_improvement'],
                'Config/Meta_Method': meta_method,
            }, step=meta_iter + 1)

        # Validation
        if (meta_iter + 1) % 100 == 0:
            val_metrics = validate_meta_v2(model, val_cases_data, config, device)

            print(f"   [VAL] Pre: {val_metrics['pre_LR']:.4f} -> Post: {val_metrics['post_LR']:.4f} (Δ={val_metrics['improvement_LR']:.4f})")

            wandb.log({
                'Val/Pre_LR': val_metrics['pre_LR'],
                'Val/Post_LR': val_metrics['post_LR'],
                'Val/Improvement_LR': val_metrics['improvement_LR'],
            }, step=meta_iter + 1)

            if val_metrics['hr_quality'] is not None:
                wandb.log({'Val/HR_Quality': val_metrics['hr_quality']}, step=meta_iter + 1)

            if val_metrics['post_LR'] < best_val_post_LR:
                best_val_post_LR = val_metrics['post_LR']
                torch.save({
                    'iteration': meta_iter + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': meta_optimizer.state_dict(),
                    'val_post_LR': best_val_post_LR,
                    'val_metrics': val_metrics,
                    'config': config.to_dict()
                }, f"{config.log_dir}/meta_best.pth")
                print(f"   [NEW BEST] Post_LR: {best_val_post_LR:.4f}")

        # Checkpoints
        if (meta_iter + 1) % 500 == 0:
            torch.save({
                'iteration': meta_iter + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': meta_optimizer.state_dict(),
                'config': config.to_dict()
            }, f"{config.log_dir}/meta_ckpt_iter{meta_iter+1}.pth")

    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
    }, f"{config.log_dir}/meta_learned_init_FINAL.pth")
    print(f"\nTraining complete! Final model saved.")

    wandb.finish()
