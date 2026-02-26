"""
Meta-training script for WIRE 4D Flow MRI Super-Resolution.

This script:
1. Loads multiple patient cases as meta-learning tasks
2. Trains a WIRE network using MAML, FOMAML, or REPTILE
3. Saves meta-learned initialization for later fine-tuning
4. Logs training progress to W&B (optional) and TensorBoard

Usage:
    python meta_train.py --method FOMAML --inner_steps 2 --meta_batch_size 3

    # With W&B logging
    python meta_train.py --wandb --wandb_project SRFlow-Meta

    # Resume from checkpoint
    python meta_train.py --resume path/to/checkpoint.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from configs.meta_config import get_meta_config
from meta.meta_dataset import MetaFlowDataset
from meta.meta_learner import (
    MetaLearner,
    MetaLearnerConfig,
    create_cosine_loss,
    create_mse_loss
)
from meta.functional_model import FunctionalWIRE
from utils.utils import set_seed, copy_cource_code


def parse_args():
    parser = argparse.ArgumentParser(description='Meta-learning for WIRE 4D Flow MRI SR')

    # Meta-learning algorithm
    parser.add_argument('--method', type=str, default='FOMAML',
                        choices=['MAML', 'FOMAML', 'REPTILE'],
                        help='Meta-learning algorithm')

    # Inner loop
    parser.add_argument('--inner_lr', type=float, default=0.01,
                        help='Inner loop learning rate')
    parser.add_argument('--inner_steps', type=int, default=2,
                        help='Number of inner loop steps')
    parser.add_argument('--inner_points', type=int, default=5000,
                        help='Points per inner step')

    # Outer loop
    parser.add_argument('--outer_lr', type=float, default=1e-4,
                        help='Outer loop (meta) learning rate')
    parser.add_argument('--meta_batch_size', type=int, default=3,
                        help='Number of tasks per meta-batch')
    parser.add_argument('--max_iters', type=int, default=5000,
                        help='Total meta-training iterations')

    # Network
    parser.add_argument('--depth', type=int, default=6,
                        help='Network depth')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Hidden layer width')
    parser.add_argument('--omega_0', type=float, default=30.0,
                        help='WIRE omega parameter')
    parser.add_argument('--sigma_0', type=float, default=30.0,
                        help='WIRE sigma parameter')

    # Checkpointing
    parser.add_argument('--save_every', type=int, default=500,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--val_every', type=int, default=100,
                        help='Validate every N iterations')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: auto-generated)')

    # W&B logging
    parser.add_argument('--wandb', action='store_true',
                        help='Enable W&B logging')
    parser.add_argument('--wandb_project', type=str, default='SRFlow-Meta',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity/team name')

    # Other
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    config = get_meta_config()

    # Override config with command line args
    config.meta.method = args.method
    config.meta.inner_lr = args.inner_lr
    config.meta.inner_steps = args.inner_steps
    config.meta.inner_points = args.inner_points
    config.meta.outer_lr = args.outer_lr
    config.meta.meta_batch_size = args.meta_batch_size
    config.meta.max_iters = args.max_iters

    config.network.depth = args.depth
    config.network.hidden_features = args.hidden
    config.network.omega_0 = args.omega_0
    config.network.sigma_0 = args.sigma_0

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        output_dir = Path(f"../models/meta_{args.method}_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir = str(output_dir)
    print(f"Output directory: {output_dir}")

    # Copy source code for reproducibility
    copy_cource_code(str(output_dir), directory_to_backup=[".", "configs", "meta"])

    # Initialize W&B
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"meta_{args.method}_{datetime.now().strftime('%m%d_%H%M')}",
            config={
                'method': args.method,
                'inner_lr': args.inner_lr,
                'outer_lr': args.outer_lr,
                'inner_steps': args.inner_steps,
                'meta_batch_size': args.meta_batch_size,
                'max_iters': args.max_iters,
                'depth': args.depth,
                'hidden': args.hidden,
                'omega_0': args.omega_0,
                'sigma_0': args.sigma_0,
            }
        )

    # TensorBoard writer
    tb_writer = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))

    # ==================== Load Data ====================
    print("\n" + "=" * 50)
    print("Loading training data...")
    print("=" * 50)

    train_dataset = MetaFlowDataset(
        case_paths=config.meta.train_cases,
        config=config,
        device=device,
        preload=True
    )

    print(f"\nLoaded {len(train_dataset)} training cases")

    # Load validation dataset if available
    val_dataset = None
    if config.meta.val_cases:
        print("\nLoading validation data...")
        val_dataset = MetaFlowDataset(
            case_paths=config.meta.val_cases,
            config=config,
            device=device,
            preload=True
        )
        print(f"Loaded {len(val_dataset)} validation cases")

    # ==================== Create Model ====================
    print("\n" + "=" * 50)
    print("Creating model...")
    print("=" * 50)

    model = FunctionalWIRE(
        in_dim=config.network.in_dim,
        out_dim=config.network.out_dim,
        depth=config.network.depth,
        hidden_features=config.network.hidden_features,
        omega_0=config.network.omega_0,
        sigma_0=config.network.sigma_0,
        use_complex=config.network.complex
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Architecture: WIRE (depth={config.network.depth}, hidden={config.network.hidden_features})")
    print(f"omega_0={config.network.omega_0}, sigma_0={config.network.sigma_0}")

    # ==================== Create Meta-Learner ====================
    meta_config = MetaLearnerConfig(
        method=config.meta.method,
        inner_lr=config.meta.inner_lr,
        outer_lr=config.meta.outer_lr,
        inner_steps=config.meta.inner_steps,
        meta_batch_size=config.meta.meta_batch_size
    )

    # Create loss function
    if config.training.use_cosine:
        loss_fn = create_cosine_loss(config)
        print("Using cosine loss")
    else:
        loss_fn = create_mse_loss(config)
        print("Using MSE loss")

    meta_learner = MetaLearner(
        model=model,
        config=meta_config,
        loss_fn=loss_fn,
        device=device
    )

    # ==================== Resume from Checkpoint ====================
    start_iter = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_iter = meta_learner.load_checkpoint(args.resume)
        print(f"Resumed at iteration {start_iter}")

    # ==================== Training Loop ====================
    print("\n" + "=" * 50)
    print(f"Starting meta-training with {config.meta.method}")
    print(f"Inner LR: {config.meta.inner_lr}, Outer LR: {config.meta.outer_lr}")
    print(f"Inner steps: {config.meta.inner_steps}, Meta-batch size: {config.meta.meta_batch_size}")
    print("=" * 50 + "\n")

    start_time = time.time()
    best_val_loss = float('inf')

    for iteration in range(start_iter, config.meta.max_iters):
        iter_start = time.time()

        # Meta-learning step
        loss, info = meta_learner.meta_step(
            dataset=train_dataset,
            points_per_task=config.meta.inner_points
        )

        iter_time = time.time() - iter_start

        # Logging
        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[Iter {iteration+1}/{config.meta.max_iters}] "
                  f"loss={info['meta_loss']:.4f} "
                  f"inner_loss={info['mean_inner_loss']:.4f} "
                  f"iter_time={iter_time:.2f}s "
                  f"elapsed={elapsed/60:.1f}min")

        # TensorBoard logging
        tb_writer.add_scalar('Loss/meta', info['meta_loss'], iteration)
        tb_writer.add_scalar('Loss/inner', info['mean_inner_loss'], iteration)
        if 'mean_query_loss' in info:
            tb_writer.add_scalar('Loss/query', info['mean_query_loss'], iteration)
        if 'param_distance' in info:
            tb_writer.add_scalar('Params/distance', info['param_distance'], iteration)

        # W&B logging
        if args.wandb:
            wandb.log({
                'iteration': iteration,
                'loss/meta': info['meta_loss'],
                'loss/inner': info['mean_inner_loss'],
                'loss/query': info.get('mean_query_loss', 0),
                'time/iter': iter_time
            })

        # Validation
        if val_dataset and (iteration + 1) % args.val_every == 0:
            print("\n--- Validation ---")
            val_results = meta_learner.evaluate(
                val_dataset,
                n_inner_steps=config.meta.val_inner_steps,
                points_per_task=config.meta.inner_points
            )

            print(f"Pre-adapt loss:  {val_results['mean_pre_loss']:.4f}")
            print(f"Post-adapt loss: {val_results['mean_post_loss']:.4f}")
            print(f"Improvement:     {val_results['improvement']:.4f}")

            # Per-case results
            for case_id, pre, post in zip(
                val_results['case_ids'],
                val_results['pre_adapt_losses'],
                val_results['post_adapt_losses']
            ):
                print(f"  {case_id}: {pre:.4f} -> {post:.4f}")

            tb_writer.add_scalar('Val/pre_adapt_loss', val_results['mean_pre_loss'], iteration)
            tb_writer.add_scalar('Val/post_adapt_loss', val_results['mean_post_loss'], iteration)
            tb_writer.add_scalar('Val/improvement', val_results['improvement'], iteration)

            if args.wandb:
                wandb.log({
                    'val/pre_adapt_loss': val_results['mean_pre_loss'],
                    'val/post_adapt_loss': val_results['mean_post_loss'],
                    'val/improvement': val_results['improvement'],
                    'iteration': iteration
                })

            # Save best model
            if val_results['mean_post_loss'] < best_val_loss:
                best_val_loss = val_results['mean_post_loss']
                best_path = output_dir / 'best_model.pt'
                meta_learner.save_checkpoint(
                    str(best_path),
                    iteration,
                    {'val_loss': best_val_loss}
                )
                print(f"Saved best model with val_loss={best_val_loss:.4f}")

            print("------------------\n")

        # Save checkpoint
        if (iteration + 1) % args.save_every == 0:
            ckpt_path = output_dir / f'checkpoint_{iteration+1}.pt'
            meta_learner.save_checkpoint(str(ckpt_path), iteration + 1)
            print(f"Saved checkpoint: {ckpt_path}")

    # ==================== Final Save ====================
    final_path = output_dir / 'final_model.pt'
    meta_learner.save_checkpoint(str(final_path), config.meta.max_iters)
    print(f"\nTraining complete! Final model saved to: {final_path}")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time/3600:.2f} hours")

    # Final evaluation
    if val_dataset:
        print("\n" + "=" * 50)
        print("Final Evaluation")
        print("=" * 50)
        final_results = meta_learner.evaluate(
            val_dataset,
            n_inner_steps=10,  # More steps for final eval
            points_per_task=10000
        )
        print(f"Pre-adapt loss:  {final_results['mean_pre_loss']:.4f}")
        print(f"Post-adapt loss: {final_results['mean_post_loss']:.4f}")
        print(f"Improvement:     {final_results['improvement']:.4f}")

    tb_writer.close()

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
