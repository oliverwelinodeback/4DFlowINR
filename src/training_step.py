"""
training_step.py - Unified Training Step Abstraction for PINN and Meta-Learning

This module provides a TrainingStep class that encapsulates one training iteration,
enabling code sharing between:
1. Standard PINN training (fine-tuning)
2. Meta-learning inner loop (FOMAML)
3. Physical pre-conditioning (Phase 0)

The same loss computation and sampling logic is used in all cases, ensuring
meta-learned initializations are directly compatible with fine-tuning.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Any, Dict

from utils.loss_utils import (
    compute_data_loss,
    compute_physics_loss,
    compute_boundary_loss,
    navier_stokes_loss
)


@dataclass
class TrainingStepConfig:
    """Configuration for a single training step.

    This mirrors the relevant parts of the main config but allows
    independent configuration for different contexts (inner loop, outer loop, etc.)
    """
    # Data loss settings
    use_data_loss: bool = True
    use_mse: bool = False
    use_cosine: bool = True
    u_weight: float = 1.0
    v_weight: float = 1.0
    w_weight: float = 1.0

    # Physics loss settings
    use_physics_loss: bool = True
    physics_weight: float = 1.0
    use_navier_stokes: bool = True
    use_divergence: bool = True

    # Boundary loss settings
    use_boundary_loss: bool = False
    boundary_weight: float = 1.0

    # Batch sizes
    data_points_per_batch: Optional[int] = 20000
    coll_points_per_batch: Optional[int] = 20000
    boundary_points_per_batch: Optional[int] = 10000

    # Advanced features
    use_adaptive_weighting: bool = False  # Per-point collocation weights
    use_gradient_balancing: bool = False  # Gradient-based loss weight balancing

    # Context flags
    is_inner_loop: bool = False  # True for meta-learning inner loop (disables expensive features)


@dataclass
class BatchData:
    """Container for a batch of training data."""
    # Data points (always present)
    xyz_data: Optional[torch.Tensor] = None
    uvw_data: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None

    # Collocation points (optional)
    xyz_collocation: Optional[torch.Tensor] = None
    coll_indices: Optional[torch.Tensor] = None
    c_weights: Optional[torch.Tensor] = None

    # Boundary points (optional)
    xyz_boundary: Optional[torch.Tensor] = None


@dataclass
class StepResult:
    """Result of a training step containing all loss components."""
    total_loss: torch.Tensor
    data_loss: torch.Tensor
    physics_loss: torch.Tensor
    boundary_loss: torch.Tensor

    # Physics breakdown
    momentum_loss: torch.Tensor
    div_loss: torch.Tensor

    # Additional info
    loss_weights: Optional[Tuple[float, ...]] = None
    physics_loss_data: Optional[torch.Tensor] = None
    momentum_loss_data: Optional[torch.Tensor] = None
    div_loss_data: Optional[torch.Tensor] = None


class TrainingStep:
    """
    Encapsulates one training iteration for PINN.

    Can operate in multiple modes:
    1. Standard mode: Full PINN features (physics, boundary, adaptive weights)
    2. Meta-inner mode: For differentiable inner loop (disables per-point tracking)
    3. Physics-only mode: For pre-conditioning (no data loss)

    The same step logic is used by both standard training and meta-learning,
    ensuring feature parity and code reuse.
    """

    def __init__(
        self,
        step_config: TrainingStepConfig,
        full_config: Any,  # ml_collections.ConfigDict
        standardization_factors: Tuple,
        device: torch.device
    ):
        """
        Initialize TrainingStep.

        Args:
            step_config: TrainingStepConfig with step-specific settings
            full_config: Full ml_collections.ConfigDict for accessing all settings
            standardization_factors: Tuple of normalization factors from data preparation
            device: torch.device (cuda/cpu)
        """
        self.step_config = step_config
        self.full_config = full_config
        self.std_factors = standardization_factors
        self.device = device

        # State for gradient-based loss weight balancing
        self._loss_weights = None
        self._iteration = 0

    def sample_batch(
        self,
        xyz_train: Optional[torch.Tensor] = None,
        uvw_train: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        xyz_coll: Optional[torch.Tensor] = None,
        xyz_bound: Optional[torch.Tensor] = None,
        c_weights: Optional[np.ndarray] = None
    ) -> BatchData:
        """
        Sample a batch from GPU tensors.

        This method extracts and unifies sampling logic from sample_from_gpu()
        to allow both standard training and meta-learning to use the same logic.

        Args:
            xyz_train: Coordinates tensor [N, 4] (x, y, z, t)
            uvw_train: Velocities tensor [N, 3 or 4] (u, v, w, [p])
            mask: Mask tensor [N, 1]
            xyz_coll: Collocation points tensor [M, 4]
            xyz_bound: Boundary points tensor [B, 4]
            c_weights: Per-point collocation weights (numpy array)

        Returns:
            BatchData containing sampled tensors
        """
        cfg = self.step_config

        # --- Sample Data Points ---
        xyz_data_batch = None
        uvw_data_batch = None
        mask_batch = None

        if xyz_train is not None and cfg.use_data_loss:
            n_data = xyz_train.shape[0]
            n_sample = cfg.data_points_per_batch or n_data

            if n_sample < n_data:
                indices = torch.randint(0, n_data, (n_sample,), device=self.device)
                xyz_data_batch = xyz_train[indices]
                uvw_data_batch = uvw_train[indices] if uvw_train is not None else None
                mask_batch = mask[indices] if mask is not None else None
            else:
                xyz_data_batch = xyz_train
                uvw_data_batch = uvw_train
                mask_batch = mask

        # --- Sample Collocation Points ---
        xyz_coll_batch = None
        coll_indices = None
        c_batch_weights = None

        if xyz_coll is not None and cfg.use_physics_loss:
            n_coll = xyz_coll.shape[0]
            n_sample = cfg.coll_points_per_batch or n_coll

            if n_sample < n_coll:
                # Weighted sampling (only if adaptive weighting enabled and not in inner loop)
                if c_weights is not None and cfg.use_adaptive_weighting and not cfg.is_inner_loop:
                    # Convert to tensor for sampling
                    c_weights_tensor = torch.from_numpy(c_weights).float().to(self.device)
                    p = c_weights_tensor / (c_weights_tensor.sum() + 1e-12)
                    coll_indices = torch.multinomial(p, n_sample, replacement=False)
                else:
                    coll_indices = torch.randperm(n_coll, device=self.device)[:n_sample]

                xyz_coll_batch = xyz_coll[coll_indices]

                # Get weights for sampled points
                if c_weights is not None and not cfg.is_inner_loop:
                    c_batch_weights = torch.from_numpy(
                        c_weights[coll_indices.cpu().numpy()]
                    ).float().to(self.device)
            else:
                xyz_coll_batch = xyz_coll
                coll_indices = torch.arange(n_coll, device=self.device)
                if c_weights is not None and not cfg.is_inner_loop:
                    c_batch_weights = torch.from_numpy(c_weights).float().to(self.device)

        # --- Sample Boundary Points ---
        xyz_bound_batch = None

        if xyz_bound is not None and cfg.use_boundary_loss:
            n_bound = xyz_bound.shape[0]
            n_sample = cfg.boundary_points_per_batch or n_bound

            if n_sample < n_bound:
                bound_indices = torch.randint(0, n_bound, (n_sample,), device=self.device)
                xyz_bound_batch = xyz_bound[bound_indices]
            else:
                xyz_bound_batch = xyz_bound

        return BatchData(
            xyz_data=xyz_data_batch,
            uvw_data=uvw_data_batch,
            mask=mask_batch,
            xyz_collocation=xyz_coll_batch,
            coll_indices=coll_indices,
            c_weights=c_batch_weights,
            xyz_boundary=xyz_bound_batch
        )

    def compute_grad_weights_detached(
        self,
        model: nn.Module,
        batch: BatchData
    ) -> Tuple[float, ...]:
        """
        Compute gradient-based loss weights WITHOUT tracking higher-order gradients.

        This allows gradient balancing in FOMAML without memory overhead.
        Works the same way in both meta-learning and fine-tuning.

        The weights are computed based on the inverse of gradient norms,
        following the GradNorm / MGDA approach for multi-task learning.

        Args:
            model: The PINN model (or functional model from `higher`)
            batch: BatchData containing the current batch

        Returns:
            Tuple of (data_weight, physics_weight, [boundary_weight])
        """
        cfg = self.step_config
        alpha = getattr(self.full_config.training, 'alpha', 0.9)

        # Compute individual losses (no grad tracking for the weights themselves)
        losses = []

        # Data loss
        if cfg.use_data_loss and batch.xyz_data is not None:
            with torch.enable_grad():
                xyz = batch.xyz_data.clone().requires_grad_(True)
                pred = model(xyz)
                data_loss, _, _, _, _ = compute_data_loss(
                    self.full_config, model, xyz, batch.uvw_data,
                    batch.mask, self.std_factors
                )
                losses.append(('data', data_loss))

        # Physics loss
        if cfg.use_physics_loss and batch.xyz_collocation is not None:
            with torch.enable_grad():
                xyz_coll = batch.xyz_collocation.clone().requires_grad_(True)
                pred_coll = model(xyz_coll)
                momentum_loss, div_loss = navier_stokes_loss(
                    pred_coll, xyz_coll, self.std_factors, self.full_config,
                    return_per_point=False, build_graph=True
                )
                physics_loss = momentum_loss + div_loss
                losses.append(('physics', physics_loss))

        # Boundary loss
        if cfg.use_boundary_loss and batch.xyz_boundary is not None:
            with torch.enable_grad():
                bound_loss = compute_boundary_loss(
                    self.full_config, model, batch.xyz_boundary
                )
                losses.append(('boundary', bound_loss))

        if len(losses) <= 1:
            # No balancing needed
            return (1.0,) * len(losses) if losses else (1.0,)

        # Compute gradient norms for each loss
        grad_norms = []
        for name, loss in losses:
            model.zero_grad()
            loss.backward(retain_graph=True)

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norms.append(total_norm ** 0.5)

        model.zero_grad()  # Clean up

        # Compute weights (inverse of gradient norms, normalized)
        grad_norms = np.array(grad_norms) + 1e-12
        inv_norms = 1.0 / grad_norms
        weights = inv_norms / inv_norms.sum()

        # EMA update if we have previous weights
        if self._loss_weights is not None:
            weights = alpha * np.array(self._loss_weights) + (1 - alpha) * weights

        self._loss_weights = tuple(weights.tolist())
        return self._loss_weights

    def forward(
        self,
        model: nn.Module,
        batch: BatchData,
        compute_grad_weights: bool = False
    ) -> StepResult:
        """
        Compute losses for one training step.

        Args:
            model: The PINN model (or functional model from `higher`)
            batch: BatchData containing sampled points
            compute_grad_weights: Whether to update gradient-based loss weights

        Returns:
            StepResult with all loss components
        """
        cfg = self.step_config
        device = self.device

        # Initialize losses
        data_loss = torch.tensor(0.0, device=device)
        physics_loss = torch.tensor(0.0, device=device)
        boundary_loss = torch.tensor(0.0, device=device)
        momentum_loss = torch.tensor(0.0, device=device)
        div_loss = torch.tensor(0.0, device=device)
        physics_loss_data = torch.tensor(0.0, device=device)
        momentum_loss_data = torch.tensor(0.0, device=device)
        div_loss_data = torch.tensor(0.0, device=device)

        # Check if gradients are enabled (we might be inside torch.no_grad())
        grad_enabled = torch.is_grad_enabled()

        # --- Enable gradients for physics if needed ---
        # Only set requires_grad if we're not in a no_grad context
        if cfg.use_physics_loss and grad_enabled:
            if batch.xyz_collocation is not None:
                batch.xyz_collocation.requires_grad_(True)
            if batch.xyz_data is not None:
                batch.xyz_data.requires_grad_(True)

        # --- Data Loss ---
        if cfg.use_data_loss and batch.xyz_data is not None:
            data_loss, _, _, _, _ = compute_data_loss(
                self.full_config,
                model,
                batch.xyz_data,
                batch.uvw_data,
                batch.mask,
                self.std_factors
            )

        # --- Physics Loss ---
        # Skip physics loss if we're in a no_grad context (requires autograd for PDE residuals)
        if cfg.use_physics_loss and batch.xyz_collocation is not None and grad_enabled:
            physics_dict = compute_physics_loss(
                self.full_config,
                self._iteration,
                model,
                batch.xyz_collocation,
                batch.xyz_data if batch.xyz_data is not None else batch.xyz_collocation,
                self.std_factors,
                c_weights=batch.c_weights if not cfg.is_inner_loop else None
            )
            physics_loss = physics_dict["physics_loss"]
            momentum_loss = physics_dict["momentum_loss"]
            div_loss = physics_dict["div_loss"]
            physics_loss_data = physics_dict.get("physics_loss_data", torch.tensor(0.0, device=device))
            momentum_loss_data = physics_dict.get("momentum_loss_data", torch.tensor(0.0, device=device))
            div_loss_data = physics_dict.get("div_loss_data", torch.tensor(0.0, device=device))

        # --- Boundary Loss ---
        if cfg.use_boundary_loss and batch.xyz_boundary is not None:
            boundary_loss = compute_boundary_loss(
                self.full_config,
                model,
                batch.xyz_boundary
            )

        # --- Compute Gradient-Based Weights (Detached) ---
        if compute_grad_weights and cfg.use_gradient_balancing and not cfg.is_inner_loop:
            self._loss_weights = self.compute_grad_weights_detached(model, batch)

        # --- Compute Total Loss ---
        if self._loss_weights is not None and not cfg.is_inner_loop:
            if cfg.use_boundary_loss and len(self._loss_weights) >= 3:
                dw, pw, bw = self._loss_weights[:3]
                total_loss = (
                    dw * data_loss +
                    pw * cfg.physics_weight * physics_loss +
                    bw * cfg.boundary_weight * boundary_loss
                )
            elif len(self._loss_weights) >= 2:
                dw, pw = self._loss_weights[:2]
                total_loss = dw * data_loss + pw * cfg.physics_weight * physics_loss
            else:
                total_loss = data_loss + cfg.physics_weight * physics_loss
        else:
            total_loss = (
                data_loss +
                cfg.physics_weight * physics_loss +
                cfg.boundary_weight * boundary_loss
            )

        self._iteration += 1

        return StepResult(
            total_loss=total_loss,
            data_loss=data_loss,
            physics_loss=physics_loss,
            boundary_loss=boundary_loss,
            momentum_loss=momentum_loss,
            div_loss=div_loss,
            loss_weights=self._loss_weights,
            physics_loss_data=physics_loss_data,
            momentum_loss_data=momentum_loss_data,
            div_loss_data=div_loss_data
        )

    def reset_iteration(self, iteration: int = 0):
        """Reset iteration counter (used when resuming)."""
        self._iteration = iteration

    def set_loss_weights(self, weights: Optional[Tuple[float, ...]]):
        """Set loss weights (used when loading checkpoint)."""
        self._loss_weights = weights

    def get_loss_weights(self) -> Optional[Tuple[float, ...]]:
        """Get current loss weights."""
        return self._loss_weights


def physics_preconditioning(
    model: nn.Module,
    cases_data: list,
    full_config: Any,
    device: torch.device,
    n_iterations: int = 100,
    n_cases: int = 5,
    lr: float = 1e-3
) -> nn.Module:
    """
    Phase 0: Pre-condition model with physics before meta-learning.

    Moves model parameters from random initialization into a state where
    Navier-Stokes residuals are already low. This ensures MAML doesn't
    waste iterations learning basic physical relationships.

    Based on NTK analysis (Yüce et al.): Without warm-start, random initialization
    creates "noisy harmonics". With warm-start, the NTK eigenfunctions are "primed"
    with smooth, PDE-consistent shapes.

    Args:
        model: The PINN model to pre-condition
        cases_data: List of CaseData objects with collocation/boundary points
        full_config: Full ml_collections.ConfigDict
        device: torch.device
        n_iterations: Number of pre-conditioning iterations
        n_cases: Number of cases to sample for pre-conditioning
        lr: Learning rate for pre-conditioning

    Returns:
        Pre-conditioned model (same object, modified in-place)
    """
    import random

    print(f"\n{'='*60}")
    print("Phase 0: Physical Pre-Conditioning")
    print(f"  Iterations: {n_iterations}")
    print(f"  Cases: {min(n_cases, len(cases_data))}")
    print(f"  Learning Rate: {lr}")
    print(f"{'='*60}\n")

    # Step 1: Select sample cases
    sample_cases = random.sample(cases_data, min(n_cases, len(cases_data)))

    # Step 2: Create physics-only TrainingStepConfig
    physics_step_config = TrainingStepConfig(
        use_data_loss=False,  # No data fitting in warm-start
        use_physics_loss=True,
        physics_weight=1.0,
        use_navier_stokes=full_config.training.use_navier_stokes,
        use_divergence=full_config.training.use_divergence,
        use_boundary_loss=True,
        boundary_weight=1.0,
        coll_points_per_batch=getattr(full_config.meta_learning, 'coll_points_inner', 5000),
        boundary_points_per_batch=getattr(full_config.meta_learning, 'boundary_points_inner', 2000),
        use_adaptive_weighting=False,
        use_gradient_balancing=False,
        is_inner_loop=False  # Not in higher context, can use all features
    )

    # Step 3: Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    # Track initial and final physics residuals
    initial_residuals = []
    final_residuals = []

    # Step 4: Run pre-conditioning iterations
    for it in range(n_iterations):
        total_physics_loss = 0.0
        n_batches = 0

        for case in sample_cases:
            # Create TrainingStep for this case
            physics_step = TrainingStep(
                physics_step_config,
                full_config,
                case.std_factors,
                device
            )

            # Get collocation and boundary points
            xyz_coll = getattr(case, 'coll_points_pool', None)
            xyz_bound = getattr(case, 'boundary_points', None)

            if xyz_coll is None:
                continue

            # Sample batch
            batch = physics_step.sample_batch(
                xyz_coll=xyz_coll,
                xyz_bound=xyz_bound
            )

            # Forward pass
            result = physics_step.forward(model, batch)

            # Backward pass
            optimizer.zero_grad()
            result.total_loss.backward()
            optimizer.step()

            total_physics_loss += result.physics_loss.item()
            n_batches += 1

            # Track initial residuals
            if it == 0:
                initial_residuals.append(result.physics_loss.item())
            elif it == n_iterations - 1:
                final_residuals.append(result.physics_loss.item())

        # Log progress
        if (it + 1) % 10 == 0 or it == 0:
            avg_loss = total_physics_loss / max(n_batches, 1)
            print(f"  [Pre-cond Iter {it+1:3d}/{n_iterations}] Physics Loss: {avg_loss:.6f}")

    # Report improvement
    if initial_residuals and final_residuals:
        init_avg = np.mean(initial_residuals)
        final_avg = np.mean(final_residuals)
        reduction = (1 - final_avg / (init_avg + 1e-12)) * 100
        print(f"\n  Pre-conditioning complete!")
        print(f"  Initial NS residual: {init_avg:.6f}")
        print(f"  Final NS residual:   {final_avg:.6f}")
        print(f"  Reduction: {reduction:.1f}%")
        print(f"{'='*60}\n")

    return model


def create_step_config_from_full_config(
    full_config: Any,
    is_inner_loop: bool = False,
    override_physics: Optional[bool] = None,
    override_boundary: Optional[bool] = None,
    override_data: Optional[bool] = None
) -> TrainingStepConfig:
    """
    Create TrainingStepConfig from full ml_collections config.

    Args:
        full_config: Full ml_collections.ConfigDict
        is_inner_loop: Whether this is for meta-learning inner loop
        override_physics: Override use_physics_loss setting
        override_boundary: Override use_boundary_loss setting
        override_data: Override use_data_loss setting

    Returns:
        TrainingStepConfig
    """
    use_physics = override_physics if override_physics is not None else full_config.training.use_physics_loss
    use_boundary = override_boundary if override_boundary is not None else full_config.sample_boundary
    use_data = override_data if override_data is not None else True

    return TrainingStepConfig(
        use_data_loss=use_data,
        use_mse=full_config.training.use_mse,
        use_cosine=full_config.training.use_cosine,
        u_weight=full_config.training.u_weight,
        v_weight=full_config.training.v_weight,
        w_weight=full_config.training.w_weight,
        use_physics_loss=use_physics,
        physics_weight=full_config.training.physics_weight,
        use_navier_stokes=full_config.training.use_navier_stokes,
        use_divergence=full_config.training.use_divergence,
        use_boundary_loss=use_boundary,
        boundary_weight=full_config.training.boundary_weight,
        data_points_per_batch=full_config.training.data_points_per_batch,
        coll_points_per_batch=full_config.training.coll_points_per_batch,
        boundary_points_per_batch=full_config.training.boundary_points_per_batch,
        use_adaptive_weighting=full_config.training.self_adaptive,
        use_gradient_balancing=full_config.training.grad_weight_scheme,
        is_inner_loop=is_inner_loop
    )
