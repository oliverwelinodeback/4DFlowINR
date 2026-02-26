"""
Meta-learning algorithms for WIRE 4D Flow MRI Super-Resolution.

Implements:
- MAML (Model-Agnostic Meta-Learning) with full second-order gradients
- FOMAML (First-Order MAML) - faster approximation
- REPTILE - simple and memory-efficient alternative

Reference: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation"
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Dict, List, Optional, Tuple, Callable
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np

from .functional_model import (
    FunctionalWIRE,
    sgd_step,
    interpolate_params,
    average_params,
    compute_param_distance
)
from .meta_dataset import TaskBatch, MetaFlowDataset


@dataclass
class MetaLearnerConfig:
    """Configuration for meta-learning."""
    method: str = 'FOMAML'          # 'MAML', 'FOMAML', or 'REPTILE'
    inner_lr: float = 0.01          # Learning rate for inner loop
    outer_lr: float = 1e-4          # Learning rate for meta-update
    inner_steps: int = 2            # Gradient steps per task
    meta_batch_size: int = 3        # Tasks per meta-batch
    reptile_epsilon: float = 1.0    # Interpolation factor for REPTILE


class MetaLearner:
    """
    Meta-learning trainer for WIRE networks.

    Supports MAML, FOMAML, and REPTILE algorithms for learning good
    initialization parameters across multiple patient cases.

    Args:
        model: FunctionalWIRE model
        config: MetaLearnerConfig with algorithm settings
        loss_fn: Loss function (coords, velocities, model, params) -> loss
        device: Torch device
    """

    def __init__(
        self,
        model: FunctionalWIRE,
        config: MetaLearnerConfig,
        loss_fn: Callable,
        device: torch.device = torch.device('cuda')
    ):
        self.model = model
        self.config = config
        self.loss_fn = loss_fn
        self.device = device

        # Meta-optimizer (for outer loop)
        self.meta_optimizer = Adam(model.parameters(), lr=config.outer_lr)

        # Track statistics
        self.stats = {
            'inner_losses': [],
            'outer_losses': [],
            'param_distances': []
        }

    def inner_loop(
        self,
        task: TaskBatch,
        params: Dict[str, torch.Tensor],
        n_steps: Optional[int] = None,
        create_graph: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], List[float]]:
        """
        Perform inner loop adaptation on a single task.

        Args:
            task: TaskBatch with coords and velocities
            params: Starting parameters
            n_steps: Number of gradient steps (default: config.inner_steps)
            create_graph: If True, track gradients for second-order MAML

        Returns:
            Tuple of (adapted_params, list of losses)
        """
        if n_steps is None:
            n_steps = self.config.inner_steps

        losses = []
        current_params = params

        for step in range(n_steps):
            # Forward pass with current params, using per-case venc
            loss = self.loss_fn(
                task.coords,
                task.velocities,
                self.model,
                current_params,
                venc=task.venc
            )
            losses.append(loss.item())

            # Compute gradients w.r.t. current params
            grads = torch.autograd.grad(
                loss,
                current_params.values(),
                create_graph=create_graph,
                allow_unused=True
            )

            # Create gradient dictionary
            grad_dict = OrderedDict()
            for (name, _), grad in zip(current_params.items(), grads):
                if grad is not None:
                    grad_dict[name] = grad
                else:
                    # Handle unused parameters (set gradient to zero)
                    grad_dict[name] = torch.zeros_like(current_params[name])

            # SGD step
            current_params = sgd_step(current_params, grad_dict, self.config.inner_lr)

        return current_params, losses

    def maml_step(
        self,
        tasks: List[TaskBatch],
        support_points: int,
        query_points: int
    ) -> Tuple[float, Dict]:
        """
        Perform one MAML meta-update step.

        For each task:
        1. Adapt parameters on support set (inner loop)
        2. Evaluate adapted params on query set
        3. Backprop through the entire process

        Args:
            tasks: List of TaskBatch for this meta-batch
            support_points: Points for inner loop adaptation
            query_points: Points for meta-gradient computation

        Returns:
            Tuple of (meta_loss, info_dict)
        """
        self.meta_optimizer.zero_grad()

        meta_loss = 0.0
        inner_losses_all = []
        query_losses = []

        # Get current meta-parameters
        meta_params = self.model.get_params()

        for task in tasks:
            # Split task data into support/query sets
            n_total = task.coords.shape[0]
            perm = torch.randperm(n_total)

            support_idx = perm[:support_points]
            query_idx = perm[support_points:support_points + query_points]

            support_task = TaskBatch(
                coords=task.coords[support_idx],
                velocities=task.velocities[support_idx],
                case_id=task.case_id,
                case_idx=task.case_idx,
                venc=task.venc
            )

            query_task = TaskBatch(
                coords=task.coords[query_idx],
                velocities=task.velocities[query_idx],
                case_id=task.case_id,
                case_idx=task.case_idx,
                venc=task.venc
            )

            # Clone params to enable gradient tracking
            task_params = OrderedDict(
                (name, param.clone()) for name, param in meta_params.items()
            )

            # Inner loop with graph creation for second-order gradients
            create_graph = (self.config.method == 'MAML')
            adapted_params, inner_losses = self.inner_loop(
                support_task,
                task_params,
                create_graph=create_graph
            )
            inner_losses_all.extend(inner_losses)

            # Query loss with adapted parameters, using per-case venc
            query_loss = self.loss_fn(
                query_task.coords,
                query_task.velocities,
                self.model,
                adapted_params,
                venc=query_task.venc
            )
            query_losses.append(query_loss.item())
            meta_loss = meta_loss + query_loss

        # Average meta-loss
        meta_loss = meta_loss / len(tasks)

        # Backprop through everything
        meta_loss.backward()

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

        # Meta-update
        self.meta_optimizer.step()

        info = {
            'meta_loss': meta_loss.item(),
            'mean_inner_loss': np.mean(inner_losses_all),
            'mean_query_loss': np.mean(query_losses)
        }

        return meta_loss.item(), info

    def fomaml_step(
        self,
        tasks: List[TaskBatch],
        support_points: int,
        query_points: int
    ) -> Tuple[float, Dict]:
        """
        First-Order MAML step (no second-order gradients).

        Same as MAML but doesn't backprop through the inner loop,
        making it much faster and more memory efficient.
        """
        self.meta_optimizer.zero_grad()

        meta_loss = 0.0
        inner_losses_all = []
        query_losses = []

        meta_params = self.model.get_params()

        for task in tasks:
            n_total = task.coords.shape[0]
            perm = torch.randperm(n_total)

            support_idx = perm[:support_points]
            query_idx = perm[support_points:support_points + query_points]

            support_task = TaskBatch(
                coords=task.coords[support_idx],
                velocities=task.velocities[support_idx],
                case_id=task.case_id,
                case_idx=task.case_idx,
                venc=task.venc
            )

            query_task = TaskBatch(
                coords=task.coords[query_idx],
                velocities=task.velocities[query_idx],
                case_id=task.case_id,
                case_idx=task.case_idx,
                venc=task.venc
            )

            # Clone params (no gradient tracking for inner loop)
            task_params = OrderedDict(
                (name, param.clone().detach().requires_grad_(True))
                for name, param in meta_params.items()
            )

            # Inner loop without graph creation
            adapted_params, inner_losses = self.inner_loop(
                support_task,
                task_params,
                create_graph=False
            )
            inner_losses_all.extend(inner_losses)

            # For FOMAML: compute query gradient w.r.t. adapted params,
            # then apply to meta params (using per-case venc)
            query_loss = self.loss_fn(
                query_task.coords,
                query_task.velocities,
                self.model,
                adapted_params,
                venc=query_task.venc
            )
            query_losses.append(query_loss.item())

            # Compute gradients w.r.t. adapted params
            adapted_grads = torch.autograd.grad(
                query_loss,
                adapted_params.values(),
                allow_unused=True
            )

            # Apply gradients to meta parameters
            for (name, meta_param), grad in zip(meta_params.items(), adapted_grads):
                if grad is not None:
                    if meta_param.grad is None:
                        meta_param.grad = grad.clone()
                    else:
                        meta_param.grad = meta_param.grad + grad

            meta_loss = meta_loss + query_loss.item()

        # Average gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = param.grad / len(tasks)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

        # Meta-update
        self.meta_optimizer.step()

        info = {
            'meta_loss': meta_loss / len(tasks),
            'mean_inner_loss': np.mean(inner_losses_all),
            'mean_query_loss': np.mean(query_losses)
        }

        return meta_loss / len(tasks), info

    def reptile_step(
        self,
        tasks: List[TaskBatch],
        points_per_task: int
    ) -> Tuple[float, Dict]:
        """
        REPTILE meta-learning step.

        For each task, perform multiple SGD steps, then interpolate
        meta-parameters towards the average of adapted parameters.

        Args:
            tasks: List of TaskBatch
            points_per_task: Points to use per task

        Returns:
            Tuple of (mean_loss, info_dict)
        """
        initial_params = self.model.clone_params()
        adapted_params_list = []
        all_losses = []

        for task in tasks:
            # Clone initial params
            task_params = OrderedDict(
                (name, param.clone().detach().requires_grad_(True))
                for name, param in initial_params.items()
            )

            # Sample points for this task
            n_total = task.coords.shape[0]
            if points_per_task < n_total:
                idx = torch.randperm(n_total)[:points_per_task]
                task_subset = TaskBatch(
                    coords=task.coords[idx],
                    velocities=task.velocities[idx],
                    case_id=task.case_id,
                    case_idx=task.case_idx,
                    venc=task.venc
                )
            else:
                task_subset = task

            # Inner loop
            adapted_params, losses = self.inner_loop(
                task_subset,
                task_params,
                create_graph=False
            )
            adapted_params_list.append(adapted_params)
            all_losses.extend(losses)

        # Average adapted parameters
        avg_adapted = average_params(adapted_params_list)

        # Compute distance moved
        param_dist = compute_param_distance(initial_params, avg_adapted)

        # REPTILE update: interpolate towards average
        epsilon = self.config.reptile_epsilon * self.config.outer_lr
        new_params = interpolate_params(initial_params, avg_adapted, epsilon)

        # Apply new parameters
        self.model.set_params(new_params)

        info = {
            'meta_loss': np.mean(all_losses),
            'mean_inner_loss': np.mean(all_losses),
            'param_distance': param_dist
        }

        return np.mean(all_losses), info

    def meta_step(
        self,
        dataset: MetaFlowDataset,
        points_per_task: int
    ) -> Tuple[float, Dict]:
        """
        Perform one meta-learning step using the configured method.

        Args:
            dataset: MetaFlowDataset to sample tasks from
            points_per_task: Points to sample per task

        Returns:
            Tuple of (loss, info_dict)
        """
        # Sample tasks
        tasks = dataset.sample_meta_batch(
            n_tasks=self.config.meta_batch_size,
            n_points_per_task=points_per_task * 2  # Support + query
        )

        if self.config.method == 'MAML':
            return self.maml_step(tasks, points_per_task, points_per_task)
        elif self.config.method == 'FOMAML':
            return self.fomaml_step(tasks, points_per_task, points_per_task)
        elif self.config.method == 'REPTILE':
            # For REPTILE, use all points for inner loop
            tasks = dataset.sample_meta_batch(
                n_tasks=self.config.meta_batch_size,
                n_points_per_task=points_per_task
            )
            return self.reptile_step(tasks, points_per_task)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    def evaluate(
        self,
        dataset: MetaFlowDataset,
        n_inner_steps: int = 5,
        points_per_task: int = 5000
    ) -> Dict:
        """
        Evaluate meta-learned initialization on validation tasks.

        For each task:
        1. Start from meta-learned init
        2. Adapt with n_inner_steps
        3. Compute final loss

        Returns:
            Dictionary with per-task and aggregate metrics
        """
        self.model.eval()
        results = {
            'pre_adapt_losses': [],
            'post_adapt_losses': [],
            'case_ids': []
        }

        with torch.no_grad():
            meta_params = self.model.get_params()

        for task_idx in range(len(dataset)):
            task = dataset.sample_task_batch(task_idx, points_per_task)

            # Pre-adaptation loss (using per-case venc)
            with torch.no_grad():
                pre_loss = self.loss_fn(
                    task.coords,
                    task.velocities,
                    self.model,
                    meta_params,
                    venc=task.venc
                ).item()

            # Clone for adaptation
            task_params = OrderedDict(
                (name, param.clone().detach().requires_grad_(True))
                for name, param in meta_params.items()
            )

            # Adapt
            adapted_params, _ = self.inner_loop(
                task,
                task_params,
                n_steps=n_inner_steps,
                create_graph=False
            )

            # Post-adaptation loss (using per-case venc)
            with torch.no_grad():
                post_loss = self.loss_fn(
                    task.coords,
                    task.velocities,
                    self.model,
                    adapted_params,
                    venc=task.venc
                ).item()

            results['pre_adapt_losses'].append(pre_loss)
            results['post_adapt_losses'].append(post_loss)
            results['case_ids'].append(task.case_id)

        # Aggregate metrics
        results['mean_pre_loss'] = np.mean(results['pre_adapt_losses'])
        results['mean_post_loss'] = np.mean(results['post_adapt_losses'])
        results['improvement'] = results['mean_pre_loss'] - results['mean_post_loss']

        self.model.train()
        return results

    def save_checkpoint(self, path: str, iteration: int, extra_info: Optional[Dict] = None):
        """Save meta-learned checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'config': {
                'method': self.config.method,
                'inner_lr': self.config.inner_lr,
                'outer_lr': self.config.outer_lr,
                'inner_steps': self.config.inner_steps,
                'meta_batch_size': self.config.meta_batch_size
            },
            'model_config': {
                'in_dim': self.model.in_dim,
                'out_dim': self.model.out_dim,
                'depth': self.model.depth,
                'hidden_features': self.model.hidden_features,
                'omega_0': self.model.omega_0,
                'sigma_0': self.model.sigma_0,
                'use_complex': self.model.use_complex
            }
        }
        if extra_info:
            checkpoint['extra_info'] = extra_info

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint. Returns iteration number."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['iteration']


def create_cosine_loss(config) -> Callable:
    """
    Create the cosine loss function for meta-learning.

    This wraps the existing cosine_loss to work with the functional interface.
    Now supports per-case venc values for proper phase unwrapping.
    """
    def loss_fn(
        coords: torch.Tensor,
        velocities: torch.Tensor,
        model: FunctionalWIRE,
        params: Dict[str, torch.Tensor],
        venc: float = None
    ) -> torch.Tensor:
        # Forward pass with given params
        pred = model.functional_forward(coords, params)

        # Denormalize for cosine loss computation
        if config.vel_normalization == "characteristic":
            U = config.constants.U
            pred_denorm = pred * U
            vel_denorm = velocities * U
        else:
            U_max = getattr(config, 'U_max', 1.0)
            pred_denorm = pred * U_max
            vel_denorm = velocities * U_max

        # Use per-case venc if provided, otherwise use config default
        actual_venc = venc if venc is not None else config.constants.venc
        Kv = torch.pi / actual_venc

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

        return loss

    return loss_fn


def create_mse_loss(config) -> Callable:
    """Create MSE loss function for meta-learning."""
    def loss_fn(
        coords: torch.Tensor,
        velocities: torch.Tensor,
        model: FunctionalWIRE,
        params: Dict[str, torch.Tensor],
        venc: float = None  # Added for API consistency, not used in MSE
    ) -> torch.Tensor:
        pred = model.functional_forward(coords, params)

        mse_u = (pred[:, 0] - velocities[:, 0]) ** 2
        mse_v = (pred[:, 1] - velocities[:, 1]) ** 2
        mse_w = (pred[:, 2] - velocities[:, 2]) ** 2

        loss = torch.mean(
            config.training.u_weight * mse_u +
            config.training.v_weight * mse_v +
            config.training.w_weight * mse_w
        )

        return loss

    return loss_fn
