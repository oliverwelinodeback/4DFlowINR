
import torch

from torch.func import functional_call, vmap, jacrev

import scipy

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm



def get_ntk_fn(model):

    """

    Creates a function to compute the NTK for a given model.

    Handles multi-output models by summing over the output dimensions.

    """

    params = {k: v.detach() for k, v in model.named_parameters()}

   

    def fnet_single(p, x):

        return functional_call(model, p, (x.unsqueeze(0),)).squeeze(0)



    def flatten_jacobian(jac_dict):

        """Flattens and concatenates all parameter jacobians."""

        jac_tensors = [v.flatten(2) for v in jac_dict.values()]

        return torch.cat(jac_tensors, dim=2)



    def ntk_fn(x1, x2):

        """Computes the empirical NTK for two batches of data."""

        # Compute Jacobians for both batches.

        jac1_dict = vmap(jacrev(fnet_single), (None, 0))(params, x1)

        jac2_dict = vmap(jacrev(fnet_single), (None, 0))(params, x2)

       

        # Flatten and concatenate all parameter gradients.

        # Shape: [batch_size, output_dim, total_params]

        jac1_flat = flatten_jacobian(jac1_dict)

        jac2_flat = flatten_jacobian(jac2_dict)



        # For multi-output networks (e.g., u,v,w), we sum over the output dimension.

        # This is a standard approach to get a single kernel matrix.

        # Shape becomes: [batch_size, total_params]

        jac1 = jac1_flat.sum(dim=1)

        jac2 = jac2_flat.sum(dim=1)

       

        # Compute the kernel: J(x1) @ J(x2).T

        # Result shape: [batch1_size, batch2_size]

        return jac1 @ jac2.T

   

    return ntk_fn



def ntk_eigendecomposition(model, data, k=200, batch_size=64):

    """

    Computes the NTK matrix and its top k eigenvalues and eigenvectors.

    Uses batching to manage memory during NTK matrix construction.

    """

    device = next(model.parameters()).device

    n_data = len(data)

   

    # Ensure k is valid for eigsh

    if k >= n_data - 1:

        print(f"Warning: k={k} is too large for {n_data} data points. Setting k to {n_data - 2}.")

        k = n_data - 2

    if k < 1:

        raise ValueError("Number of eigenvalues 'k' must be at least 1.")



    print(f"Computing NTK matrix for {n_data} points (batch size: {batch_size})...")

    ntk_fn = get_ntk_fn(model)

   

    # Compute the full NTK matrix in batches to avoid OOM on the GPU

    ntk_matrix = torch.zeros((n_data, n_data), device=device)

   

    for i in tqdm(range(0, n_data, batch_size), desc="NTK Matrix Rows"):

        x1_batch = data[i:i+batch_size]

        for j in range(0, n_data, batch_size):

            x2_batch = data[j:j+batch_size]

           

            kernel_block = ntk_fn(x1_batch, x2_batch)

            ntk_matrix[i:i+batch_size, j:j+batch_size] = kernel_block

   

    print("NTK matrix computed. Moving to CPU for eigendecomposition.")

    ntk_np = ntk_matrix.cpu().numpy()

   

    print(f"Computing top {k} eigenvalues/eigenvectors using scipy.linalg.eigsh...")

    try:

        # 'LM' means largest magnitude eigenvalues. For a positive semi-definite

        # matrix like the NTK, this corresponds to the largest eigenvalues.

        eigvals, eigvecs = scipy.sparse.linalg.eigsh(ntk_np, k=k, which='LM')

    except Exception as e:

        print(f"Eigendecomposition failed with k={k}: {e}")

        # Fallback to a smaller k if it fails

        k_fallback = min(k // 2, n_data - 2)

        if k_fallback < 1: raise ValueError("Eigendecomposition failed even with fallback.")

        print(f"Retrying with k={k_fallback}...")

        eigvals, eigvecs = scipy.sparse.linalg.eigsh(ntk_np, k=k_fallback, which='LM')



    # Sort eigenvalues and corresponding eigenvectors in descending order

    idx = np.argsort(eigvals)[::-1]

    eigvals = torch.from_numpy(eigvals[idx].copy())

    eigvecs = torch.from_numpy(eigvecs[:, idx].copy())

   

    return eigvals, eigvecs, ntk_matrix



def visualize_ntk_results(eigvals, eigvecs, coords, outdir, keyword, plot_eigvec_indices=range(10), fmt="svg"):

    """

    Visualizes NTK eigenvalues and eigenvectors for 3D spatial data.

    - Saves plots of raw and normalized eigenvalues.

    - For eigenvectors, interpolates values onto a regular grid and saves 2D slices.

    """

    if not os.path.exists(outdir):

        os.makedirs(outdir)

       

    eigvals_np = eigvals.cpu().numpy()



    # Plot and save eigenvalues

    plt.figure(figsize=(8, 5))

    plt.plot(eigvals_np)

    plt.title(f"{keyword.replace('_', ' ').title()} NTK Eigenvalues")

    plt.xlabel("Index")

    plt.ylabel("Eigenvalue")

    sns.despine()

    plt.savefig(os.path.join(outdir, f"{keyword}_eigvals.{fmt}"), format=fmt, bbox_inches="tight")

    plt.close()



    # Plot and save normalized eigenvalues

    plt.figure(figsize=(8, 5))

    plt.plot(eigvals_np / eigvals_np[0])

    plt.title(f"{keyword.replace('_', ' ').title()} Normalized NTK Eigenvalues")

    plt.xlabel("Index")

    plt.ylabel("Normalized Eigenvalue")

    sns.despine()

    plt.savefig(os.path.join(outdir, f"{keyword}_eigvals_normalized.{fmt}"), format=fmt, bbox_inches="tight")

    plt.close()



    # --- Visualize Eigenvectors by Interpolating onto a Grid ---

    print("Visualizing eigenvectors by plotting 2D slices...")



    # Create a regular grid to interpolate onto. Adjust resolution as needed.

    grid_res = 100

    min_coords = coords.min(axis=0)

    max_coords = coords.max(axis=0)

   

    grid_x, grid_y, grid_z = np.mgrid[

        min_coords[0]:max_coords[0]:grid_res*1j,

        min_coords[1]:max_coords[1]:grid_res*1j,

        min_coords[2]:max_coords[2]:grid_res*1j

    ]



    for i in plot_eigvec_indices:

        v_i = eigvecs[:, i].cpu().numpy()

       

        print(f"  Interpolating eigenvector {i}...")

        # Interpolate the scattered eigenvector data onto the regular grid

        interpolated_grid = scipy.interpolate.griddata(

            coords, v_i, (grid_x, grid_y, grid_z), method='linear'

        )

       

        # Handle points outside the convex hull of data, which will be NaN

        interpolated_grid[np.isnan(interpolated_grid)] = 0

       

        # --- Plot XY slice at the middle Z-depth ---

        z_slice_idx = interpolated_grid.shape[2] // 2

        xy_slice = interpolated_grid[:, :, z_slice_idx]



        plt.figure(figsize=(7, 6))

        plt.imshow(xy_slice.T, origin='lower', cmap='viridis', extent=[min_coords[0], max_coords[0], min_coords[1], max_coords[1]])

        plt.title(f"Eigenvector {i} (XY Slice)")

        plt.xlabel("X coordinate")

        plt.ylabel("Y coordinate")

        plt.colorbar(label="Value")

        plt.axis('on')

        plt.savefig(os.path.join(outdir, f"{keyword}_eigvec_{i}_xy_slice.{fmt}"), format=fmt, bbox_inches="tight")

        plt.close()