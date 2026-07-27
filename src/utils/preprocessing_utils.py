import numpy as np
import h5py

def standardize(data, mean=None, std=None):
    
    if mean is None:
        mean = np.mean(data)
        std = np.std(data)
        normalized_data = (data - mean) / std
    else:
        normalized_data = (data - mean) / std
    return normalized_data, mean, std

def min_max_normalize(data, min_val=None, max_val=None):
    """Normalize the data to a [0, 1] range using min-max scaling."""

    if min_val is None:
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
    else:
        normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data, min_val, max_val

def compute_boundary_mask(mask):
    t, x, y, z = mask.shape
    boundary_mask = np.zeros_like(mask, dtype=int)

    for time_step in range(t):
        current_mask = mask[time_step]
        
        # Check for boundary in each direction
        for i in range(1, x-1):
            for j in range(1, y-1):
                for k in range(1, z-1):
                    if current_mask[i, j, k] == 1:
                        if (current_mask[i+1, j, k] == 0 or current_mask[i-1, j, k] == 0 or
                            current_mask[i, j+1, k] == 0 or current_mask[i, j-1, k] == 0 or
                            current_mask[i, j, k+1] == 0 or current_mask[i, j, k-1] == 0):
                            boundary_mask[time_step, i, j, k] = 1
    return boundary_mask

def compute_outer_boundary_mask(mask):
    x, y, z = mask.shape
    outer_boundary_mask = np.zeros_like(mask, dtype=int)

    current_mask = mask
    
    # Check for outer boundary in each direction
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if current_mask[i, j, k] == 0:
                    if ((i > 0 and current_mask[i-1, j, k] == 1) or
                        (i < x-1 and current_mask[i+1, j, k] == 1) or
                        (j > 0 and current_mask[i, j-1, k] == 1) or
                        (j < y-1 and current_mask[i, j+1, k] == 1) or
                        (k > 0 and current_mask[i, j, k-1] == 1) or
                        (k < z-1 and current_mask[i, j, k+1] == 1)):
                        outer_boundary_mask[i, j, k] = 1

    return outer_boundary_mask
