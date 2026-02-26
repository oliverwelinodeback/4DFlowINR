"""
NTK Energy Concentration Analysis for different model configurations.

Usage:
    cd SRFlow/src
    python run_ntk_analysis.py --model_name "WIRE-Meta" --use_checkpoint
    python run_ntk_analysis.py --model_name "WIRE-Random"  # No checkpoint = random init

Network parameters (omega_0, sigma_0, hidden_features, depth) are loaded from config file.
Logs raw curve data to W&B for later plotting/comparison.
"""

import argparse
import torch
import ml_collections
import numpy as np

import wandb
import networks
from meta.ntk_analysis import analyze_ntk_with_curve
from meta.train_meta_v2 import load_all_cases
from configs.tunings_251106.Config_MetaLearning_MAML_DataDriven import get_config


# ============ CONFIGURATION PARAMETERS (edit these) ============

# Path to meta-learned checkpoint (Data-Driven MAML, out_dim=3, omega_0=20, sigma_0=20)
META_CHECKPOINT = "../models/MetaLearning_BestHyperPArameters/260202_MAML_DataDriven_20260203-1034/meta_best.pth"

# Validation cases to analyze (use cases NOT in meta-learning training)
VAL_CASE_PATHS = [
    "../data/healthy/HV06_05mm3_20ms_LR_sv12_tSNR10_newMask.h5",
    "../data/stenosis_50/ICAD98_05mm3_20ms_LR_sv51_tSNR10_newMask.h5",
    "../data/stenosis_70/ICAD146_05mm3_20ms_LR_sv17_tSNR10_newMask.h5"
]

# VENC values for each case (needed for data loading)
CASE_VENC = {
    "HV06_05mm3_20ms_LR_sv12_tSNR10_newMask": 1.2,
    "ICAD98_05mm3_20ms_LR_sv51_tSNR10_newMask": 5.1,
    "ICAD146_05mm3_20ms_LR_sv17_tSNR10_newMask": 1.7
}

# ============ END CONFIGURATION ============


def parse_args():
    parser = argparse.ArgumentParser(description="NTK Energy Concentration Analysis")
    parser.add_argument("--omega_0", type=int, default=None,
                        help="Override omega_0 from config")
    parser.add_argument("--sigma_0", type=int, default=None,
                        help="Override sigma_0 from config")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name for this model (used in W&B run name and for identification)")
    parser.add_argument("--use_checkpoint", action="store_true",
                        help="Load weights from META_CHECKPOINT. If not set, uses random initialization.")
    parser.add_argument("--max_coords", type=int, default=512,
                        help="Max coordinates for NTK computation (default: 512)")
    parser.add_argument("--n_curve_points", type=int, default=100,
                        help="Number of points for energy curve (default: 100)")
    parser.add_argument("--wandb_project", type=str, default="SRFlow-NTK-Analysis",
                        help="W&B project name")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration from config file (network parameters come from here)
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load overrides from command line
    if args.omega_0 is not None:
        config.network.omega_0 = args.omega_0
    if args.sigma_0 is not None:
        config.network.sigma_0 = args.sigma_0

    # Network parameters from config file
    print(f"\n[Model Configuration from Config File]")
    print(f"  Model name: {args.model_name}")
    print(f"  omega_0: {config.network.omega_0}")
    print(f"  sigma_0: {config.network.sigma_0}")
    print(f"  depth: {config.network.depth}")
    print(f"  hidden_features: {config.network.hidden_features}")
    print(f"  in_dim: {config.network.in_dim}")
    print(f"  out_dim: {config.network.out_dim}")
    print(f"  complex: {config.network.complex}")
    print(f"  Checkpoint: {META_CHECKPOINT if args.use_checkpoint else 'Random initialization'}")

    # Setup meta_learning config for data loading
    if not hasattr(config, 'meta_learning'):
        config.meta_learning = ml_collections.ConfigDict()
    config.meta_learning.case_venc = CASE_VENC

    # ============ Load Validation Cases ============
    print("\n[Loading Validation Cases]")
    val_cases = load_all_cases(VAL_CASE_PATHS, config, device)
    print(f"Loaded {len(val_cases)} validation cases")

    # ============ Create Model (using config parameters) ============
    def create_model():
        return networks.WIRE(
            in_dim=config.network.in_dim,
            out_dim=config.network.out_dim,
            depth=config.network.depth,
            hidden_features=config.network.hidden_features,
            first_omega_0=config.network.omega_0,
            hidden_omega_0=config.network.omega_0,
            scale=config.network.sigma_0,
            complex=config.network.complex
        ).to(device)

    model = create_model()

    # Load checkpoint if requested
    if args.use_checkpoint:
        try:
            checkpoint = torch.load(META_CHECKPOINT, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded weights from: {META_CHECKPOINT}")
        except FileNotFoundError:
            print(f"WARNING: Could not find {META_CHECKPOINT}")
            print("Using random initialization instead...")
    else:
        print("Using random initialization")

    # ============ Run NTK Analysis ============
    print("\n" + "="*60)
    print(f"NTK Energy Concentration Analysis: {args.model_name}")
    print("="*60)

    results, thresholds, energies = analyze_ntk_with_curve(
        model=model,
        val_cases=val_cases,
        config=config,
        device=device,
        max_coords=args.max_coords,
        n_curve_points=args.n_curve_points
    )

    print("\n[Results]")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    # ============ Log to W&B ============
    print("\n" + "="*60)
    print("Logging results to W&B...")
    print("="*60)

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=f"NTK_{args.model_name}",
        config={
            "model_name": args.model_name,
            "omega_0": config.network.omega_0,
            "sigma_0": config.network.sigma_0,
            "depth": config.network.depth,
            "hidden_features": config.network.hidden_features,
            "in_dim": config.network.in_dim,
            "out_dim": config.network.out_dim,
            "complex": config.network.complex,
            "max_coords": args.max_coords,
            "n_curve_points": args.n_curve_points,
            "checkpoint": META_CHECKPOINT if args.use_checkpoint else "random",
        }
    )

    # Log scalar metrics (averaged across all validation cases)
    wandb.log({
        "n_cases": results.get('n_cases', 1),
        "eigenvalue_ratio": results.get('eigenvalue_ratio', 0),
        "eigenvalue_decay": results.get('eigenvalue_decay', 0),
        "eigenvalue_ratio_std": results.get('eigenvalue_ratio_std', 0),
        "eigenvalue_decay_std": results.get('eigenvalue_decay_std', 0),
        "energy_01pct": results.get('energy_01pct', 0),
        "energy_05pct": results.get('energy_05pct', 0),
        "energy_10pct": results.get('energy_10pct', 0),
        "energy_20pct": results.get('energy_20pct', 0),
        "energy_50pct": results.get('energy_50pct', 0),
        "energy_01pct_std": results.get('energy_01pct_std', 0),
        "energy_05pct_std": results.get('energy_05pct_std', 0),
        "energy_10pct_std": results.get('energy_10pct_std', 0),
        "energy_20pct_std": results.get('energy_20pct_std', 0),
        "energy_50pct_std": results.get('energy_50pct_std', 0),
    })

    # Log raw curve data as a table (for later plotting)
    curve_data = []
    for i in range(len(thresholds)):
        curve_data.append([
            thresholds[i],                    # Raw threshold (λ/λ₀)
            energies[i],                      # Raw energy (0-1)
            energies[i] * 100,                # Energy as percentage
            np.log10(thresholds[i]),          # Log10 of threshold (for plotting)
        ])

    curve_table = wandb.Table(
        data=curve_data,
        columns=["threshold", "energy", "energy_pct", "log10_threshold"]
    )
    wandb.log({"energy_curve": curve_table})

    # Also log as W&B artifact for easy download
    artifact = wandb.Artifact(
        name=f"ntk_curve_{args.model_name.replace(' ', '_')}",
        type="ntk_data"
    )

    # Save curve data as numpy arrays
    np.savez(
        f"/tmp/ntk_curve_{args.model_name.replace(' ', '_')}.npz",
        thresholds=thresholds,
        energies=energies,
        model_name=args.model_name,
        omega_0=config.network.omega_0,
        sigma_0=config.network.sigma_0
    )
    artifact.add_file(f"/tmp/ntk_curve_{args.model_name.replace(' ', '_')}.npz")
    wandb.log_artifact(artifact)

    wandb.finish()
    print("W&B logging complete!")
    print(f"\nTo compare multiple models, run this script with different --model_name and --checkpoint")
    print(f"Then use W&B to download the curve data and create comparison plots.")


if __name__ == "__main__":
    main()
