"""
ntk_analysis.py - Simplified NTK Energy Concentration for 4D Flow MRI

Neural Tangent Kernel (NTK) analysis for understanding meta-learned initialization.
Implements energy concentration metric from "A Structured Dictionary Perspective on INRs".

For 4D data (t, x, y, z), we use a heavily subsampled grid to make computation tractable.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import time


def compute_ntk_gram_matrix_chunked(
    model: torch.nn.Module,
    coords: torch.Tensor,
    chunk_size: int = 64,  # Kept for API compatibility, not used
    max_coords: int = 1024
) -> torch.Tensor:
    """
    Compute empirical NTK Gram matrix: Θ(r1, r2) = <∇_θ f(r1), ∇_θ f(r2)>

    Computes per-coordinate Jacobians and builds the Gram matrix.

    Args:
        model: The WIRE network
        coords: (N, 4) coordinates [t, x, y, z]
        chunk_size: (Unused, kept for API compatibility)
        max_coords: Maximum coordinates to use (subsample if N > max_coords)

    Returns:
        ntk_matrix: (M, M) Gram matrix, where M = min(N, max_coords)
    """
    # Ensure model is in train mode and gradients are enabled
    was_training = model.training
    model.train()

    # Ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad_(True)

    device = coords.device
    N = coords.shape[0]

    # Subsample if too large
    if N > max_coords:
        print(f"  Subsampling coords: {N} -> {max_coords}")
        idx = torch.randperm(N, device=device)[:max_coords]
        coords = coords[idx]
        N = max_coords

    # Compute Jacobians for all coordinates
    print(f"  Computing NTK for {N} points...")
    start_time = time.time()

    # Get number of parameters
    n_params = sum(p.numel() for p in model.parameters())
    out_dim = 3  # velocity components (u, v, w)

    # Pre-allocate jacobian storage: (N, out_dim * n_params)
    jacobians = torch.zeros(N, out_dim * n_params, device=device)

    # Use torch.enable_grad() to ensure gradients are computed even if called from no_grad context
    with torch.enable_grad():
        for i in range(N):
            coord_single = coords[i:i+1].clone().detach().requires_grad_(True)  # (1, 4)

            # Forward pass for single coordinate
            output = model(coord_single)  # (1, 3)

            # For each output dimension, compute gradient w.r.t. all parameters
            jac_row = []
            for out_idx in range(out_dim):
                model.zero_grad()
                output[0, out_idx].backward(retain_graph=(out_idx < out_dim - 1))

                # Collect gradients from all parameters
                grads = []
                for p in model.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.flatten().clone())
                    else:
                        grads.append(torch.zeros(p.numel(), device=device))

                grad_flat = torch.cat(grads)  # (n_params,)
                jac_row.append(grad_flat)

            # Concatenate all output dimensions: (out_dim * n_params,)
            jacobians[i] = torch.cat(jac_row)

            if (i + 1) % 100 == 0:
                print(f"    Processed {i+1}/{N} coordinates...")

    print(f"  Computing Gram matrix...")

    # Compute NTK Gram matrix: (N, N)
    # Θ(i, j) = <J_i, J_j> where J is the flattened Jacobian
    ntk_matrix = jacobians @ jacobians.t()

    elapsed = time.time() - start_time
    print(f"  NTK computation time: {elapsed:.2f}s")

    # Restore original training mode
    if not was_training:
        model.eval()

    return ntk_matrix


def compute_energy_concentration_curve(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    signals: np.ndarray,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute energy concentration curve E(λ) for plotting (like Figure 4 in the paper).

    Args:
        eigvals: (K,) eigenvalues sorted descending
        eigvecs: (K, N) eigenvectors (rows are eigenvectors)
        signals: (M, N) M signals, each of length N
        n_points: Number of points on the curve

    Returns:
        thresholds: (n_points,) normalized eigenvalue thresholds (λ/λ_0)
        energies: (n_points,) energy concentration at each threshold
    """
    λ_max = eigvals[0]
    λ_min = max(eigvals[-1], 1e-10)  # Avoid log(0)
    normalized_eigvals = eigvals / λ_max

    # Create log-spaced thresholds from 1 (λ_max) down to λ_min/λ_max
    min_ratio = λ_min / λ_max
    thresholds = np.logspace(0, np.log10(min_ratio), n_points)

    # Pre-compute all projections for efficiency
    # signals: (M, N), eigvecs: (K, N) -> projections: (M, K)
    projections = signals @ eigvecs.T  # (M, K)

    # Signal norms
    signal_norms_sq = np.sum(signals ** 2, axis=1)  # (M,)
    signal_norms_sq = np.maximum(signal_norms_sq, 1e-12)  # Avoid division by zero

    energies = []
    for thresh in thresholds:
        # Find eigenvalues above threshold
        mask = normalized_eigvals >= thresh
        if mask.sum() == 0:
            energies.append(0.0)
            continue

        # Sum squared projections for selected eigenvectors
        proj_selected = projections[:, mask]  # (M, n_selected)
        energy_per_signal = np.sum(proj_selected ** 2, axis=1) / signal_norms_sq  # (M,)
        energies.append(np.mean(energy_per_signal))

    return thresholds, np.array(energies)


def compute_energy_concentration(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    signals: np.ndarray,
    threshold: float = 0.1
) -> Tuple[float, Dict]:
    """
    Compute energy concentration of signals on NTK eigenfunctions.

    From paper Eq. 11:
    E(λ) = (1/N) * Σ_n Σ_{λ_i/λ_0 >= λ} |<φ_i, g_n>|² / ||g_n||²

    Measures how much signal energy is concentrated on large eigenvalues.
    Higher = better/faster learning (signal aligns with high-NTK modes).

    Args:
        eigvals: (K,) eigenvalues sorted descending
        eigvecs: (K, N) eigenvectors (rows are eigenvectors)
        signals: (M, N) M signals, each of length N
        threshold: Eigenvalue threshold (λ_i/λ_max >= threshold)

    Returns:
        energy: Scalar energy concentration at threshold
        metrics: Dict with additional statistics
    """
    λ_max = eigvals[0]
    normalized_eigvals = eigvals / λ_max

    # Find eigenvalues above threshold
    mask = normalized_eigvals >= threshold
    n_selected = mask.sum()

    if n_selected == 0:
        return 0.0, {'n_eigvals_selected': 0}

    # Select eigenvectors corresponding to large eigenvalues
    eigvecs_selected = eigvecs[mask, :]  # (n_selected, N)

    # For each signal, compute projection onto selected eigenvectors
    energies = []
    for signal in signals:
        # Normalize signal
        signal_norm_sq = np.dot(signal, signal)
        if signal_norm_sq < 1e-12:
            energies.append(0.0)
            continue

        # Project onto eigenvectors: <φ_i, g>
        projections = eigvecs_selected @ signal  # (n_selected,)

        # Sum of squared projections: Σ |<φ_i, g>|²
        energy_captured = np.sum(projections ** 2)

        # Normalize by signal energy
        energy_fraction = energy_captured / signal_norm_sq
        energies.append(energy_fraction)

    # Average over all signals
    mean_energy = np.mean(energies)

    metrics = {
        'n_eigvals_selected': n_selected,
        'energy_per_signal': energies,
        'energy_std': np.std(energies),
        'threshold': threshold
    }

    return mean_energy, metrics


def analyze_ntk_energy(
    model: torch.nn.Module,
    val_cases: list,
    config,
    device: torch.device,
    max_coords: int = 512
) -> Dict:
    """
    Perform full NTK energy concentration analysis.

    Steps:
    1. Sample coordinates from a subset of validation cases
    2. Compute NTK Gram matrix
    3. Eigendecompose
    4. Project validation velocity fields onto eigenvectors
    5. Compute energy concentration at different thresholds

    Args:
        model: WIRE network
        val_cases: List of CaseData objects
        config: Configuration
        device: torch device
        max_coords: Max coordinates for NTK computation (memory limit)

    Returns:
        results: Dict with energy concentration metrics
    """
    print("\n[NTK Energy Concentration Analysis]")

    model.eval()

    # 1. Sample a grid of coordinates from validation cases
    # Use first validation case for NTK basis
    if not val_cases:
        print("  No validation cases available")
        return {}

    case = val_cases[0]
    print(f"  Using case: {case.case_name}")

    # Sample coordinates uniformly from LR data
    coords = case.coords_LR  # (N, 4)
    N = coords.shape[0]

    if N > max_coords:
        sample_idx = torch.randperm(N, device=device)[:max_coords]
        coords_sample = coords[sample_idx]
    else:
        sample_idx = None
        coords_sample = coords

    # 2. Compute NTK Gram matrix
    # Note: compute_ntk_gram_matrix_chunked handles gradient enabling internally
    ntk_matrix = compute_ntk_gram_matrix_chunked(
        model, coords_sample, chunk_size=64, max_coords=max_coords
    )

    # Move to CPU/numpy for eigendecomposition
    ntk_np = ntk_matrix.cpu().numpy()

    # 3. Eigendecompose
    print(f"  Eigendecomposing {ntk_np.shape[0]}x{ntk_np.shape[0]} matrix...")
    start_time = time.time()

    # Symmetrize (numerical stability)
    ntk_np = (ntk_np + ntk_np.T) / 2

    eigvals, eigvecs = np.linalg.eigh(ntk_np)

    # Sort descending (use .copy() to avoid negative stride issues)
    sort_idx = np.argsort(eigvals)[::-1].copy()
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx].T  # (K, N) - rows are eigenvectors

    elapsed = time.time() - start_time
    print(f"  Eigendecomposition time: {elapsed:.2f}s")
    print(f"  Top 5 eigenvalues: {eigvals[:5]}")
    print(f"  Eigenvalue range: [{eigvals.min():.2e}, {eigvals.max():.2e}]")

    # 4. Project validation velocity fields onto eigenvectors
    # Use velocity from sampled coordinates
    vel_sample = case.vel_LR
    if sample_idx is not None:
        vel_sample = vel_sample[sample_idx]

    vel_np = vel_sample.cpu().numpy()  # (M, 3)

    # Flatten to treat as signals: each velocity component is a signal
    signals = vel_np.T  # (3, M) - 3 signals

    # 5. Compute energy concentration at different thresholds
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
    energy_results = {}

    for thresh in thresholds:
        energy, metrics = compute_energy_concentration(
            eigvals, eigvecs, signals, threshold=thresh
        )
        energy_results[f'energy_{int(thresh*100):02d}pct'] = energy
        print(f"  Energy @ λ/λ_max >= {thresh:.2f}: {energy:.4f} (using {metrics['n_eigvals_selected']} eigenvectors)")

    # Summary metrics
    results = {
        'eigenvalue_ratio': eigvals[10] / eigvals[0] if len(eigvals) > 10 else 0.0,
        'eigenvalue_decay': eigvals[50] / eigvals[0] if len(eigvals) > 50 else 0.0,
        **energy_results
    }

    model.train()
    return results


def analyze_ntk_with_curve(
    model: torch.nn.Module,
    val_cases: list,
    config,
    device: torch.device,
    max_coords: int = 512,
    n_curve_points: int = 100
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Perform NTK analysis and return the full energy concentration curve.

    Averages energy concentration across ALL validation cases.

    Args:
        model: WIRE network
        val_cases: List of CaseData objects
        config: Configuration
        device: torch device
        max_coords: Max coordinates for NTK computation
        n_curve_points: Number of points for the energy curve

    Returns:
        results: Dict with energy concentration metrics (averaged across cases)
        thresholds: (n_curve_points,) normalized eigenvalue thresholds
        energies: (n_curve_points,) energy concentration values (averaged across cases)
    """
    print("\n[NTK Energy Concentration Analysis with Curve]")
    print(f"  Averaging across {len(val_cases)} validation cases")

    model.eval()

    if not val_cases:
        print("  No validation cases available")
        return {}, np.array([]), np.array([])

    # Store results for each case
    all_energies = []
    all_thresholds = None
    all_eigval_ratios = []
    all_eigval_decays = []
    all_energy_metrics = {thresh: [] for thresh in [0.01, 0.05, 0.1, 0.2, 0.5]}

    for case_idx, case in enumerate(val_cases):
        print(f"\n  [{case_idx + 1}/{len(val_cases)}] Processing case: {case.case_name}")

        # Sample coordinates
        coords = case.coords_LR
        N = coords.shape[0]

        if N > max_coords:
            sample_idx = torch.randperm(N, device=device)[:max_coords]
            coords_sample = coords[sample_idx]
        else:
            sample_idx = None
            coords_sample = coords

        # Compute NTK Gram matrix
        ntk_matrix = compute_ntk_gram_matrix_chunked(
            model, coords_sample, chunk_size=64, max_coords=max_coords
        )

        ntk_np = ntk_matrix.cpu().numpy()

        # Eigendecompose
        print(f"    Eigendecomposing {ntk_np.shape[0]}x{ntk_np.shape[0]} matrix...")
        ntk_np = (ntk_np + ntk_np.T) / 2
        eigvals, eigvecs = np.linalg.eigh(ntk_np)

        sort_idx = np.argsort(eigvals)[::-1].copy()
        eigvals = eigvals[sort_idx]
        eigvecs = eigvecs[:, sort_idx].T

        print(f"    Top 5 eigenvalues: {eigvals[:5]}")
        print(f"    Eigenvalue range: [{eigvals.min():.2e}, {eigvals.max():.2e}]")

        # Store eigenvalue metrics
        all_eigval_ratios.append(eigvals[10] / eigvals[0] if len(eigvals) > 10 else 0.0)
        all_eigval_decays.append(eigvals[50] / eigvals[0] if len(eigvals) > 50 else 0.0)

        # Get velocity signals
        vel_sample = case.vel_LR
        if sample_idx is not None:
            vel_sample = vel_sample[sample_idx]
        vel_np = vel_sample.cpu().numpy()
        signals = vel_np.T  # (3, M)

        # Compute full energy concentration curve
        print(f"    Computing energy concentration curve ({n_curve_points} points)...")
        thresholds, energies = compute_energy_concentration_curve(
            eigvals, eigvecs, signals, n_points=n_curve_points
        )

        all_energies.append(energies)
        if all_thresholds is None:
            all_thresholds = thresholds

        # Energy at specific thresholds
        for thresh in [0.01, 0.05, 0.1, 0.2, 0.5]:
            energy, _ = compute_energy_concentration(eigvals, eigvecs, signals, threshold=thresh)
            all_energy_metrics[thresh].append(energy)
            print(f"    Energy @ {thresh:.0%}: {energy:.4f}")

    # Average across all cases
    print(f"\n  [Averaging results across {len(val_cases)} cases]")

    avg_energies = np.mean(all_energies, axis=0)
    std_energies = np.std(all_energies, axis=0)

    # Build averaged results
    results = {
        'eigenvalue_ratio': np.mean(all_eigval_ratios),
        'eigenvalue_decay': np.mean(all_eigval_decays),
        'eigenvalue_ratio_std': np.std(all_eigval_ratios),
        'eigenvalue_decay_std': np.std(all_eigval_decays),
        'n_cases': len(val_cases),
    }

    # Averaged energy at specific thresholds
    for thresh in [0.01, 0.05, 0.1, 0.2, 0.5]:
        key = f'energy_{int(thresh*100):02d}pct'
        results[key] = np.mean(all_energy_metrics[thresh])
        results[f'{key}_std'] = np.std(all_energy_metrics[thresh])
        print(f"  Avg Energy @ {thresh:.0%}: {results[key]:.4f} ± {results[f'{key}_std']:.4f}")

    model.train()
    return results, all_thresholds, avg_energies


def compare_ntk_random_vs_metalearned(
    model_meta: torch.nn.Module,
    model_random: torch.nn.Module,
    val_cases: list,
    config,
    device: torch.device
) -> Dict:
    """
    Compare NTK energy concentration between meta-learned and random initialization.

    This shows if meta-learning has shaped the NTK to better align with the data distribution.

    Args:
        model_meta: Meta-learned WIRE
        model_random: Randomly initialized WIRE
        val_cases: Validation cases
        config: Config
        device: Device

    Returns:
        comparison: Dict with metrics for both models
    """
    print("\n[Comparing NTK: Meta-learned vs Random Init]")

    results_meta = analyze_ntk_energy(model_meta, val_cases, config, device, max_coords=512)
    print("\n  Random initialization...")
    results_random = analyze_ntk_energy(model_random, val_cases, config, device, max_coords=512)

    comparison = {
        'meta': results_meta,
        'random': results_random,
        'improvement': {
            k: results_meta.get(k, 0) - results_random.get(k, 0)
            for k in results_meta.keys()
        }
    }

    print("\n[Comparison Summary]")
    for k in results_meta.keys():
        meta_val = results_meta.get(k, 0)
        rand_val = results_random.get(k, 0)
        improve = meta_val - rand_val
        print(f"  {k}: Meta={meta_val:.4f}, Random={rand_val:.4f}, Δ={improve:+.4f}")

    return comparison


def compare_ntk_with_curves(
    model_meta: torch.nn.Module,
    model_random: torch.nn.Module,
    val_cases: list,
    config,
    device: torch.device,
    max_coords: int = 512,
    n_curve_points: int = 100
) -> Dict:
    """
    Compare NTK energy concentration and return curves for plotting.

    Args:
        model_meta: Meta-learned WIRE
        model_random: Randomly initialized WIRE
        val_cases: Validation cases
        config: Config
        device: Device
        max_coords: Max coordinates for NTK
        n_curve_points: Points for the curve

    Returns:
        comparison: Dict with metrics and curves for both models
    """
    print("\n[Comparing NTK with Energy Curves: Meta-learned vs Random Init]")

    print("\n  Meta-learned initialization...")
    results_meta, thresh_meta, energy_meta = analyze_ntk_with_curve(
        model_meta, val_cases, config, device, max_coords, n_curve_points
    )

    print("\n  Random initialization...")
    results_random, thresh_random, energy_random = analyze_ntk_with_curve(
        model_random, val_cases, config, device, max_coords, n_curve_points
    )

    comparison = {
        'meta': results_meta,
        'random': results_random,
        'improvement': {
            k: results_meta.get(k, 0) - results_random.get(k, 0)
            for k in results_meta.keys()
        },
        'curves': {
            'meta': {'thresholds': thresh_meta, 'energies': energy_meta},
            'random': {'thresholds': thresh_random, 'energies': energy_random}
        }
    }

    print("\n[Comparison Summary]")
    for k in results_meta.keys():
        meta_val = results_meta.get(k, 0)
        rand_val = results_random.get(k, 0)
        improve = meta_val - rand_val
        print(f"  {k}: Meta={meta_val:.4f}, Random={rand_val:.4f}, Δ={improve:+.4f}")

    return comparison
