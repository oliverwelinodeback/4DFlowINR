import torch
import numpy as np
import random


def sample_to_device(config, xyz_train, xyz_collocation, xyz_boundary, uvw_train, mask_flat, device):
    
    # Data / Fluid Points
    if config.training.data_points_per_batch is not None:
        data_indices = np.random.choice(len(xyz_train), size=config.training.data_points_per_batch, replace=False)
        xyz_data_batch = xyz_train[data_indices]
        uvw_data_batch = uvw_train[data_indices]
        mask_batch = mask_flat[data_indices]
    else:
        xyz_data_batch = xyz_train
        uvw_data_batch = uvw_train
        mask_batch = mask_flat

    xyz_data_batch = torch.from_numpy(xyz_data_batch).float().to(device)
    uvw_data_batch = torch.from_numpy(uvw_data_batch).float().to(device)
    mask_batch = torch.from_numpy(mask_batch).float().to(device)
    mask_batch = mask_batch.view(-1, 1)

    # Collocation Points
    xyz_collocation_batch = None
    if config.sample_collocation:
        if config.training.coll_points_per_batch is not None:
            coll_indices = np.random.choice(len(xyz_collocation), size=config.training.coll_points_per_batch, replace=False)
            xyz_collocation_batch = xyz_collocation[coll_indices]
        else:
            xyz_collocation_batch = xyz_collocation

        xyz_collocation_batch = torch.from_numpy(xyz_collocation_batch).float().to(device)

    # Boundary Points
    xyz_boundary_batch = None
    if config.sample_boundary:
        if config.training.boundary_points_per_batch is not None:
            boundary_indices = np.random.choice(len(xyz_boundary), size=config.training.boundary_points_per_batch, replace=False)
            xyz_boundary_batch = xyz_boundary[boundary_indices]
        else:
            xyz_boundary_batch = xyz_boundary

        xyz_boundary_batch = torch.from_numpy(xyz_boundary_batch).float().to(device)

    return xyz_data_batch, uvw_data_batch, mask_batch, xyz_collocation_batch, xyz_boundary_batch

def sample_from_gpu(config, xyz_train_gpu, xyz_collocation_gpu, xyz_boundary_gpu, uvw_train_gpu, mask_flat_gpu, c_weights=None):
    
    # Get the device from one of the tensors
    device = xyz_train_gpu.device

    # Data / Fluid Points
    if config.training.data_points_per_batch is not None:
        # Use torch.randint on the GPU. This is near-instant.
        data_indices = torch.randint(
            low=0, 
            high=xyz_train_gpu.shape[0], 
            size=(config.training.data_points_per_batch,), 
            device=device
        )
        # Index the GPU tensor directly. This is a fast VRAM-to-VRAM copy.
        xyz_data_batch = xyz_train_gpu[data_indices]
        uvw_data_batch = uvw_train_gpu[data_indices]
        mask_batch = mask_flat_gpu[data_indices]
    else:
        xyz_data_batch = xyz_train_gpu
        uvw_data_batch = uvw_train_gpu
        mask_batch = mask_flat_gpu

    # Collocation Points
    xyz_collocation_batch = None
    coll_indices = None
    if config.sample_collocation:
        if config.training.coll_points_per_batch is not None:
            
            n = config.training.coll_points_per_batch
            
            if c_weights is not None:
                # weighted sampling without replacement
                if isinstance(c_weights, torch.Tensor):
                    p = c_weights.to(device=xyz_collocation_gpu.device, dtype=torch.float32)
                else:
                    p = torch.as_tensor(c_weights, device=xyz_collocation_gpu.device, dtype=torch.float32)
                p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                p.clamp_(min=1e-12)
                p /= p.sum() + 1e-12
                coll_indices = torch.multinomial(p, n, replacement=False)
            else:
                # uniform sampling without replacement
                coll_indices = torch.randperm(xyz_collocation_gpu.shape[0], device=device)[:n]

            xyz_collocation_batch = xyz_collocation_gpu[coll_indices]
        else:
            xyz_collocation_batch = xyz_collocation_gpu
    else:
        xyz_collocation_batch = None

    # Boundary Points
    xyz_boundary_batch = None
    if config["sample_boundary"]:
        if config.training.boundary_points_per_batch is not None:
            boundary_indices = torch.randint(
                low=0, 
                high=xyz_boundary_gpu.shape[0], 
                size=(config.training.boundary_points_per_batch,), 
                device=device
            )
            xyz_boundary_batch = xyz_boundary_gpu[boundary_indices]
        else:
            xyz_boundary_batch = xyz_boundary_gpu

    return xyz_data_batch, uvw_data_batch, mask_batch, xyz_collocation_batch, xyz_boundary_batch, coll_indices

def sample_ref_to_device(config, xyz_train, uvw_train, mask_flat, device):
    
    # Data / Fluid Points
    if config.training.data_points_per_batch is not None:
        data_indices = np.random.choice(len(xyz_train), size=config.training.data_points_per_batch, replace=False)
        xyz_data_batch = xyz_train[data_indices]
        uvw_data_batch = uvw_train[data_indices]
        mask_batch = mask_flat[data_indices]
    else:
        xyz_data_batch = xyz_train
        uvw_data_batch = uvw_train
        mask_batch = mask_flat

    xyz_data_batch = torch.from_numpy(xyz_data_batch).float().to(device)
    uvw_data_batch = torch.from_numpy(uvw_data_batch).float().to(device)
    mask_batch = torch.from_numpy(mask_batch).float().to(device)
    mask_batch = mask_batch.view(-1, 1)

    return xyz_data_batch, uvw_data_batch, mask_batch

def sample_ref_from_gpu(config, xyz_ref_gpu, uvw_ref_gpu, mask_flat_ref_gpu):
    
    # Get the device from one of the input tensors
    device = xyz_ref_gpu.device

    # Data / Fluid Points
    if config.training.data_points_per_batch is not None:
        # Use torch.randint on the GPU to get random indices
        data_indices = torch.randint(
            low=0, 
            high=xyz_ref_gpu.shape[0], 
            size=(config.training.data_points_per_batch,), 
            device=device
        )
        
        # Index the GPU tensors directly (fast VRAM-to-VRAM copy)
        xyz_data_batch = xyz_ref_gpu[data_indices]
        uvw_data_batch = uvw_ref_gpu[data_indices]
        mask_batch = mask_flat_ref_gpu[data_indices]
    else:
        # Use the full tensors if batch size is not specified
        xyz_data_batch = xyz_ref_gpu
        uvw_data_batch = uvw_ref_gpu
        mask_batch = mask_flat_ref_gpu

    
    return xyz_data_batch, uvw_data_batch, mask_batch