"""
Multi-case dataset for meta-learning on 4D Flow MRI data.

This module provides:
- MetaFlowDataset: Loads and manages multiple patient cases
- TaskBatch: Container for task-specific data batches
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing_utils import standardize, compute_outer_boundary_mask


@dataclass
class TaskBatch:
    """Container for a single task's data batch."""
    coords: torch.Tensor         # (N, 4) normalized coordinates [t, x, y, z]
    velocities: torch.Tensor     # (N, 3) normalized velocities [u, v, w]
    case_id: str                 # Identifier for this case
    case_idx: int                # Index in the dataset
    venc: float = 1.2            # Per-case venc for cosine loss
    mask: Optional[torch.Tensor] = None  # Optional fluid mask


@dataclass
class CaseMetadata:
    """Per-case metadata for parameters that vary between patients."""
    venc: float                  # Velocity encoding from MRI acquisition
    peak_flow_idx: int           # Timestep of peak flow (for evaluation)
    U_max: float                 # Maximum velocity in the case


@dataclass
class TaskData:
    """Full data for a single patient case (stored in memory)."""
    coords: np.ndarray           # (M, 4) all fluid coordinates
    velocities: np.ndarray       # (M, 3) all fluid velocities
    case_id: str                 # Identifier
    case_path: str               # Original file path
    n_points: int                # Number of fluid points
    standardization_factors: List[float]  # For denormalization
    metadata: Optional[CaseMetadata] = None  # Per-case parameters


class MetaFlowDataset(Dataset):
    """
    Dataset for meta-learning across multiple 4D Flow MRI patient cases.

    Each patient case is treated as a separate "task" in the meta-learning framework.
    The dataset pre-loads and normalizes all cases using a common template for
    consistent coordinate ranges across patients.

    Args:
        case_paths: List of paths to .h5 files
        config: Configuration object with normalization settings
        device: torch device for tensors
        preload: If True, load all data into memory at initialization
    """

    def __init__(
        self,
        case_paths: List[str],
        config,
        device: torch.device = torch.device('cuda'),
        preload: bool = True
    ):
        self.case_paths = case_paths
        self.config = config
        self.device = device
        self.preload = preload

        # Compute template normalization factors once
        self.template_factors = self._compute_template_factors()

        # Storage for loaded data
        self.tasks: List[TaskData] = []

        if preload:
            self._preload_all_cases()

    def _compute_template_factors(self) -> Dict[str, float]:
        """Compute normalization factors from template dimensions."""
        cfg = self.config

        # Extract template dimensions
        dx, dy, dz, dt = (
            cfg.template.dx, cfg.template.dy,
            cfg.template.dz, cfg.template.dt
        )
        x_len, y_len, z_len, t_len = (
            cfg.template.x_len, cfg.template.y_len,
            cfg.template.z_len, cfg.template.t_len
        )

        # Create template coordinate arrays
        t = np.linspace(dt, t_len * dt, t_len)
        x = np.linspace(dx, x_len * dx, x_len)
        y = np.linspace(dy, y_len * dy, y_len)
        z = np.linspace(dz, z_len * dz, z_len)

        # Apply characteristic normalization if enabled
        if cfg.coords_characteristic:
            L, T = cfg.constants.L, cfg.constants.T
            t = t / T
            x = x / L
            y = y / L
            z = z / L

        factors = {}

        if cfg.coords_normalization == "standardize":
            # Compute per-axis statistics
            _, factors['mean_x'], factors['std_x'] = standardize(x)
            _, factors['mean_y'], factors['std_y'] = standardize(y)
            _, factors['mean_z'], factors['std_z'] = standardize(z)
            if cfg.setup.include_time:
                _, factors['mean_t'], factors['std_t'] = standardize(t)

            # Global normalization uses largest spatial dimension
            if cfg.global_normalization:
                ranges = [np.ptp(arr) for arr in (x, y, z)]
                idx_largest = np.argmax(ranges)
                if idx_largest == 0:
                    factors['global_mean'] = factors['mean_x']
                    factors['global_std'] = factors['std_x']
                elif idx_largest == 1:
                    factors['global_mean'] = factors['mean_y']
                    factors['global_std'] = factors['std_y']
                else:
                    factors['global_mean'] = factors['mean_z']
                    factors['global_std'] = factors['std_z']

        return factors

    def _load_single_case(self, case_path: str) -> TaskData:
        """Load and preprocess a single patient case."""
        cfg = self.config
        case_id = Path(case_path).stem

        with h5py.File(case_path, mode='r') as hf:
            # Load velocity components
            u = np.asarray(hf['u'])
            v = np.asarray(hf['v'])
            w = np.asarray(hf['w'])

            # Load mask
            mask = np.asarray(hf['mask'])
            if len(mask.shape) == 4:
                mask = mask[0]

        # Get dimensions
        if cfg.setup.include_time:
            t_len, x_len, y_len, z_len = u.shape
        else:
            x_len, y_len, z_len = u.shape
            t_len = 1

        # Create and normalize coordinates
        coords, std_factors = self._create_normalized_coords(
            t_len, x_len, y_len, z_len
        )

        # Normalize velocities
        U_max = max(u.max(), v.max(), w.max())
        if cfg.vel_normalization == "characteristic":
            U = cfg.constants.U
            u_norm = u / U
            v_norm = v / U
            w_norm = w / U
        elif cfg.vel_normalization == "max_velocity":
            u_norm = u / U_max
            v_norm = v / U_max
            w_norm = w / U_max

        # Flatten data
        u_flat = u_norm.ravel()
        v_flat = v_norm.ravel()
        w_flat = w_norm.ravel()
        velocities = np.stack([u_flat, v_flat, w_flat], axis=1)

        # Create mask for fluid region
        if cfg.setup.include_time:
            mask_flat = np.tile(mask.ravel(), t_len)
        else:
            mask_flat = mask.ravel()

        # Extract fluid region only
        fluid_mask = mask_flat == 1
        coords_fluid = coords[fluid_mask]
        velocities_fluid = velocities[fluid_mask]

        # Load per-case metadata from config
        case_params = getattr(cfg.meta, 'case_params', {})
        if case_id in case_params:
            params = case_params[case_id]
            venc = params.get('venc', cfg.constants.venc)
            peak_flow_idx = params.get('peak_flow_idx', getattr(cfg.meta, 'default_peak_flow_idx', 10))
        else:
            # Use default values
            venc = cfg.constants.venc
            peak_flow_idx = getattr(cfg.meta, 'default_peak_flow_idx', 10)

        metadata = CaseMetadata(
            venc=venc,
            peak_flow_idx=peak_flow_idx,
            U_max=U_max
        )

        return TaskData(
            coords=coords_fluid,
            velocities=velocities_fluid,
            case_id=case_id,
            case_path=case_path,
            n_points=len(coords_fluid),
            standardization_factors=std_factors,
            metadata=metadata
        )

    def _create_normalized_coords(
        self,
        t_len: int,
        x_len: int,
        y_len: int,
        z_len: int
    ) -> Tuple[np.ndarray, List[float]]:
        """Create normalized coordinate grid for given dimensions."""
        cfg = self.config

        # Create coordinate arrays
        dx, dy, dz = cfg.resolution.dx, cfg.resolution.dy, cfg.resolution.dz
        dt = cfg.resolution.dt

        t = np.linspace(dt, t_len * dt, t_len)
        x = np.linspace(dx, x_len * dx, x_len)
        y = np.linspace(dy, y_len * dy, y_len)
        z = np.linspace(dz, z_len * dz, z_len)

        # Characteristic normalization
        if cfg.coords_characteristic:
            L, T = cfg.constants.L, cfg.constants.T
            t = t / T
            x = x / L
            y = y / L
            z = z / L

        # Apply template-based standardization
        tf = self.template_factors
        if cfg.global_normalization:
            global_mean = tf['global_mean']
            global_std = tf['global_std']
            x_norm = (x - global_mean) / global_std
            y_norm = (y - global_mean) / global_std
            z_norm = (z - global_mean) / global_std
            std_factors = [
                tf.get('mean_t', 0), tf.get('std_t', 1),
                global_mean, global_std,
                global_mean, global_std,
                global_mean, global_std
            ]
        else:
            x_norm = (x - tf['mean_x']) / tf['std_x']
            y_norm = (y - tf['mean_y']) / tf['std_y']
            z_norm = (z - tf['mean_z']) / tf['std_z']
            std_factors = [
                tf.get('mean_t', 0), tf.get('std_t', 1),
                tf['mean_x'], tf['std_x'],
                tf['mean_y'], tf['std_y'],
                tf['mean_z'], tf['std_z']
            ]

        if cfg.setup.include_time:
            t_norm = (t - tf['mean_t']) / tf['std_t']
            grids = np.meshgrid(t_norm, x_norm, y_norm, z_norm, indexing='ij')
        else:
            grids = np.meshgrid(x_norm, y_norm, z_norm, indexing='ij')

        flat_coords = [grid.ravel() for grid in grids]
        coords = np.stack(flat_coords, axis=1).astype(np.float32)

        return coords, std_factors

    def _preload_all_cases(self):
        """Load all cases into memory."""
        print(f"Preloading {len(self.case_paths)} cases...")
        for i, case_path in enumerate(self.case_paths):
            print(f"  Loading case {i+1}/{len(self.case_paths)}: {Path(case_path).name}")
            task_data = self._load_single_case(case_path)
            self.tasks.append(task_data)
            print(f"    -> {task_data.n_points:,} fluid points")
        print("Preloading complete.")

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> TaskData:
        if not self.preload:
            return self._load_single_case(self.case_paths[idx])
        return self.tasks[idx]

    def sample_task_batch(
        self,
        task_idx: int,
        n_points: int,
        to_device: bool = True
    ) -> TaskBatch:
        """
        Sample a random batch of points from a specific task.

        Args:
            task_idx: Index of the task/case
            n_points: Number of points to sample
            to_device: If True, move tensors to self.device

        Returns:
            TaskBatch with sampled coordinates and velocities
        """
        task = self.tasks[task_idx]

        # Random sampling with replacement if needed
        if n_points >= task.n_points:
            indices = np.arange(task.n_points)
        else:
            indices = np.random.choice(task.n_points, size=n_points, replace=False)

        coords = torch.from_numpy(task.coords[indices]).float()
        velocities = torch.from_numpy(task.velocities[indices]).float()

        if to_device:
            coords = coords.to(self.device)
            velocities = velocities.to(self.device)

        # Get venc from metadata if available
        venc = task.metadata.venc if task.metadata else self.config.constants.venc

        return TaskBatch(
            coords=coords,
            velocities=velocities,
            case_id=task.case_id,
            case_idx=task_idx,
            venc=venc
        )

    def sample_meta_batch(
        self,
        n_tasks: int,
        n_points_per_task: int
    ) -> List[TaskBatch]:
        """
        Sample a batch of tasks for meta-learning.

        Args:
            n_tasks: Number of tasks to sample
            n_points_per_task: Points per task

        Returns:
            List of TaskBatch objects
        """
        # Sample task indices without replacement
        n_available = len(self.tasks)
        if n_tasks > n_available:
            task_indices = np.random.choice(n_available, size=n_tasks, replace=True)
        else:
            task_indices = np.random.choice(n_available, size=n_tasks, replace=False)

        batches = []
        for task_idx in task_indices:
            batch = self.sample_task_batch(task_idx, n_points_per_task)
            batches.append(batch)

        return batches

    def get_full_task(self, task_idx: int, to_device: bool = True) -> TaskBatch:
        """Get all points from a task (for evaluation)."""
        task = self.tasks[task_idx]

        coords = torch.from_numpy(task.coords).float()
        velocities = torch.from_numpy(task.velocities).float()

        if to_device:
            coords = coords.to(self.device)
            velocities = velocities.to(self.device)

        # Get venc from metadata if available
        venc = task.metadata.venc if task.metadata else self.config.constants.venc

        return TaskBatch(
            coords=coords,
            velocities=velocities,
            case_id=task.case_id,
            case_idx=task_idx,
            venc=venc
        )


class MetaFlowDatasetLazy(MetaFlowDataset):
    """
    Memory-efficient version that loads data on-demand.

    Use this when the combined dataset is too large to fit in memory.
    """

    def __init__(self, case_paths: List[str], config, device: torch.device):
        super().__init__(case_paths, config, device, preload=False)

        # Only store metadata
        self._case_metadata = []
        for path in case_paths:
            with h5py.File(path, 'r') as hf:
                u_shape = hf['u'].shape
            self._case_metadata.append({
                'path': path,
                'shape': u_shape
            })

    def sample_task_batch(
        self,
        task_idx: int,
        n_points: int,
        to_device: bool = True
    ) -> TaskBatch:
        """Load and sample from a case on-demand."""
        task = self._load_single_case(self.case_paths[task_idx])

        indices = np.random.choice(task.n_points, size=min(n_points, task.n_points), replace=False)

        coords = torch.from_numpy(task.coords[indices]).float()
        velocities = torch.from_numpy(task.velocities[indices]).float()

        if to_device:
            coords = coords.to(self.device)
            velocities = velocities.to(self.device)

        return TaskBatch(
            coords=coords,
            velocities=velocities,
            case_id=task.case_id,
            case_idx=task_idx
        )
