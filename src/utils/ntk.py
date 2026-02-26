from torch.func import functional_call, vmap, jacrev
import torch
import scipy
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- 1. SAMPLING HELPER ---
def get_structured_batch(data, n_points, device='cpu'):
    """
    Selects a subset of data concentrated at a SINGLE time step (Middle of the range).
    Ensures high-quality spatial density for NTK visualization (Option B).
    """
    print(f"Sampling structured batch from {len(data)} points...")
    
    # Ensure CPU for easy numpy operations
    if isinstance(data, torch.Tensor):
        coords = data.detach().cpu().numpy()
    else:
        coords = data
        
    # 1. Find Unique Times
    # Rounding is important to group float times correctly
    times = np.round(coords[:, 0], decimals=5) 
    unique_times = np.sort(np.unique(times))
    
    # 2. Select Middle Time
    mid_idx = len(unique_times) // 2
    target_time = unique_times[mid_idx]
    print(f" -> Selected specific time t = {target_time} (Index {mid_idx}/{len(unique_times)})")
    
    # 3. Filter Data to ONLY this time
    mask = (times == target_time)
    spatial_slice = data[mask] 
    
    n_available = len(spatial_slice)
    print(f" -> Found {n_available} spatial points at this time step.")
    
    # 4. Subsample Spatially if needed
    if n_available > n_points:
        print(f" -> Subsampling {n_points} points randomly from this slice.")
        perm = torch.randperm(n_available) if isinstance(data, torch.Tensor) else np.random.permutation(n_available)
        subset = spatial_slice[perm[:n_points]]
    else:
        print(f" -> Using all {n_available} points.")
        subset = spatial_slice
        
    if isinstance(subset, np.ndarray):
        subset = torch.from_numpy(subset).float().to(device)
    else:
        subset = subset.to(device)
        
    return subset


# --- 2. NTK COMPUTATION LOGIC ---
def get_ntk_fn(model):
    """
    Creates a function to compute the NTK for a WIRE model.
    Handles Complex parameters and Per-Component outputs.
    """
    # Pre-process parameters: Detect complex weights and convert to Real views
    params_real = {}
    complex_keys = set() 

    # print("Pre-processing parameters for NTK...")
    for name, param in model.named_parameters():
        p_detached = param.detach()
        if p_detached.is_complex():
            params_real[name] = torch.view_as_real(p_detached)
            complex_keys.add(name)
        else:
            params_real[name] = p_detached
            
    # Wrapper to reconstruct complex params inside the function
    def fnet_single_wrapper(p_real_dict, x):
        p_complex_dict = {}
        for name, p in p_real_dict.items():
            if name in complex_keys:
                p_complex_dict[name] = torch.view_as_complex(p)
            else:
                p_complex_dict[name] = p
        
        # Forward pass
        output = functional_call(model, p_complex_dict, (x.unsqueeze(0),)).squeeze(0)
        
        # Ensure output is real for Jacobian (split if complex)
        if output.is_complex():
            return torch.view_as_real(output).flatten()
        return output

    def flatten_jacobian(jac_dict):
        jac_tensors = []
        for val in jac_dict.values():
            jac_tensors.append(val.flatten(start_dim=2))
        return torch.cat(jac_tensors, dim=2)

    def ntk_fn(x1, x2):
        # Compute Jacobians
        # Shape: [batch, output_dim, params]
        jac1_dict = vmap(jacrev(fnet_single_wrapper), (None, 0))(params_real, x1)
        jac2_dict = vmap(jacrev(fnet_single_wrapper), (None, 0))(params_real, x2)
        
        # Flatten parameters
        jac1 = flatten_jacobian(jac1_dict)
        jac2 = flatten_jacobian(jac2_dict)
        
        # Permute to [output_dim, batch, params] for batch matrix multiplication
        jac1 = jac1.permute(1, 0, 2)
        jac2 = jac2.permute(1, 0, 2)
        
        # Result shape: [output_dim, batch1, batch2]
        # This returns the full NTK for each component separately (u, v, w...)
        return torch.bmm(jac1, jac2.transpose(1, 2))
    
    return ntk_fn


def ntk_eigendecomposition(model, data, k=200, batch_size=64, component_idx=None):
    """
    Computes NTK eigenvalues/vectors.
    Args:
        component_idx (int): Index of output to analyze (0=u, 1=v, 2=w). 
                             If None, sums all components.
    """
    device = next(model.parameters()).device
    n_data = len(data)
    
    # Check output info
    dummy_out = model(data[:1].to(device))
    n_outputs = dummy_out.shape[-1]
    
    msg = f"Output Component {component_idx}" if component_idx is not None else "SUM of all components"
    print(f"Computing NTK for {n_data} points. Mode: {msg}")

    if k >= n_data - 1:
        k = n_data - 2
        if k < 1: k = 1

    ntk_fn = get_ntk_fn(model)
    ntk_matrix = torch.zeros((n_data, n_data), device=device)
    
    for i in tqdm(range(0, n_data, batch_size), desc="NTK Rows"):
        x1_batch = data[i:i+batch_size]
        for j in range(0, n_data, batch_size):
            x2_batch = data[j:j+batch_size]
            
            # shape: [output_dim, batch_i, batch_j]
            kernel_block_all = ntk_fn(x1_batch, x2_batch)
            
            if component_idx is not None:
                if component_idx >= kernel_block_all.shape[0]:
                    raise ValueError(f"component_idx {component_idx} out of range for model output dim {kernel_block_all.shape[0]}")
                kernel_block = kernel_block_all[component_idx]
            else:
                kernel_block = kernel_block_all.sum(dim=0)

            ntk_matrix[i:i+batch_size, j:j+batch_size] = kernel_block
    
    print("Computing eigenvalues (scipy.sparse.linalg.eigsh)...")
    ntk_np = ntk_matrix.cpu().numpy()
    
    try:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(ntk_np, k=k, which='LM')
    except Exception as e:
        print(f"Eigendecomposition failed: {e}. Retrying with fewer k...")
        k_fallback = min(k // 2, n_data - 2)
        if k_fallback < 1: k_fallback = 1
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(ntk_np, k=k_fallback, which='LM')

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = torch.from_numpy(eigvals[idx].copy())
    eigvecs = torch.from_numpy(eigvecs[:, idx].copy())
    
    return eigvals, eigvecs, ntk_matrix


# --- 3. VISUALIZATION LOGIC ---
def visualize_ntk_results(eigvals, eigvecs, coords, outdir, keyword, 
                          plot_eigvec_indices=range(10),
                          z_slice_relative_positions=[0.5]): 
    """
    Visualizes NTK results.
    - Assumes 'coords' is a Structured Batch (single time slice).
    - Supports multiple Z-slice depths.
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    if isinstance(eigvals, torch.Tensor): eigvals_np = eigvals.cpu().numpy()
    else: eigvals_np = eigvals
        
    if isinstance(eigvecs, torch.Tensor): eigvecs = eigvecs.cpu().numpy()
    if isinstance(coords, torch.Tensor): coords = coords.cpu().numpy()

    # 1. Plot Eigenvalues
    plt.figure(figsize=(8, 5))
    plt.plot(eigvals_np)
    plt.title(f"{keyword} NTK Eigenvalues")
    plt.yscale("log") 
    sns.despine()
    plt.savefig(os.path.join(outdir, f"{keyword}_eigvals.png"), bbox_inches="tight")
    plt.close()

    # 2. Check Dimensions
    dim = coords.shape[1]
    
    if dim == 4:
        # Check if T is actually constant
        times = coords[:, 0]
        target_t = np.mean(times)
        if np.max(times) - np.min(times) > 1e-5:
            print("WARNING: Data is not a single time slice! Visualization may be jagged.")
        
        # Drop Time dimension -> (X, Y, Z)
        spatial_coords = coords[:, 1:4] 
        base_info_text = f"t={target_t:.2f}"
        dim = 3
    else:
        spatial_coords = coords
        base_info_text = ""

    # 3. Setup Grid
    grid_res = 100
    min_c = spatial_coords.min(axis=0)
    max_c = spatial_coords.max(axis=0)
    
    if dim == 3:
        grid_x, grid_y, grid_z = np.mgrid[
            min_c[0]:max_c[0]:grid_res*1j,
            min_c[1]:max_c[1]:grid_res*1j,
            min_c[2]:max_c[2]:grid_res*1j
        ]
        grid_tuple = (grid_x, grid_y, grid_z)
    elif dim == 2:
        grid_x, grid_y = np.mgrid[
            min_c[0]:max_c[0]:grid_res*1j,
            min_c[1]:max_c[1]:grid_res*1j
        ]
        grid_tuple = (grid_x, grid_y)
    else:
        print("Skipping viz: Dimension not supported.")
        return

    # 4. Interpolate and Plot
    print(f"Interpolating onto {grid_res}x{grid_res} grid...")
    
    for i in plot_eigvec_indices:
        if i >= eigvecs.shape[1]: break
        v_i = eigvecs[:, i]
        
        try:
            # Linear Interpolation
            grid_data = scipy.interpolate.griddata(
                spatial_coords, v_i, grid_tuple, method='linear'
            )
            # Fill borders with 0 (Background)
            grid_data[np.isnan(grid_data)] = 0.0

            # --- Loop over requested Z slices ---
            current_z_slices = z_slice_relative_positions if dim == 3 else [None]

            for z_pos in current_z_slices:
                plt.figure(figsize=(6, 5))
                slice_suffix = ""
                slice_title_text = base_info_text

                if dim == 3:
                    z_idx = int(z_pos * (grid_res - 1))
                    z_idx = np.clip(z_idx, 0, grid_res - 1)
                    z_real = min_c[2] + z_pos * (max_c[2] - min_c[2])
                    
                    xy_slice = grid_data[:, :, z_idx]
                    
                    # Plot with Transpose for correct X/Y orientation
                    plt.imshow(xy_slice.T, origin='lower', cmap='viridis',
                               extent=[min_c[0], max_c[0], min_c[1], max_c[1]])
                    
                    slice_suffix = f"_z{z_pos:.2f}"
                    slice_title_text += f", z={z_real:.2f}"
                    
                elif dim == 2:
                    plt.imshow(grid_data.T, origin='lower', cmap='viridis',
                               extent=[min_c[0], max_c[0], min_c[1], max_c[1]])

                plt.xlabel("X")
                plt.ylabel("Y")
                plt.colorbar(label="Eigenvector Value")
                plt.title(f"Eigenvector {i}\n({slice_title_text})")
                
                filename = f"{keyword}_eigvec_{i}{slice_suffix}.png"
                plt.savefig(os.path.join(outdir, filename), bbox_inches="tight")
                plt.close()

        except Exception as e:
            print(f"Could not interpolate eigenvector {i}: {e}")