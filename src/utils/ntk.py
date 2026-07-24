import torch
from torch.func import functional_call, vmap, jacrev
import scipy
import numpy as np
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
