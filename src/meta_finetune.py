"""
Fine-tuning script for meta-learned WIRE models.

This script takes a meta-learned checkpoint and fine-tunes it on a
specific patient case. The meta-learned initialization should allow
faster convergence compared to random initialization.

Usage:
    # Fine-tune on a new case
    python meta_finetune.py \
        --checkpoint path/to/meta_checkpoint.pt \
        --data_file ../data/stenosis_70/ICAD146_05mm3_20ms_LR_sv17_tSNR10_newMask.h5 \
        --iterations 2000

    # Compare with random init baseline
    python meta_finetune.py \
        --data_file ../data/stenosis_70/ICAD146_05mm3_20ms_LR_sv17_tSNR10_newMask.h5 \
        --iterations 8000 \
        --random_init
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

sys.path.insert(0, str(Path(__file__).parent))

from configs.meta_config import get_meta_config
from meta.meta_dataset import MetaFlowDataset
from meta.functional_model import FunctionalWIRE
from meta.meta_learner import create_cosine_loss, create_mse_loss
from utils.utils import set_seed, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune meta-learned WIRE model')

    # Required
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to .h5 file for fine-tuning')

    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to meta-learned checkpoint')
    parser.add_argument('--random_init', action='store_true',
                        help='Use random initialization (baseline)')

    # Training
    parser.add_argument('--iterations', type=int, default=2000,
                        help='Number of fine-tuning iterations')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=20000,
                        help='Points per batch')

    # Network (only used if random_init)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--omega_0', type=float, default=30.0)
    parser.add_argument('--sigma_0', type=float, default=30.0)

    # Output
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='SRFlow-Finetune')

    # Other
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=500)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    config = get_meta_config()
    config.data_file = args.data_file

    # Output directory
    case_name = Path(args.data_file).stem
    init_type = 'random' if args.random_init else 'meta'
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"../models/finetune_{init_type}_{case_name}_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # TensorBoard
    tb_writer = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))

    # W&B
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"finetune_{init_type}_{case_name}",
            config=vars(args)
        )

    # ==================== Load Model ====================
    if args.checkpoint and not args.random_init:
        print(f"\nLoading meta-learned checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Extract model config from checkpoint
        model_cfg = checkpoint['model_config']
        model = FunctionalWIRE(
            in_dim=model_cfg['in_dim'],
            out_dim=model_cfg['out_dim'],
            depth=model_cfg['depth'],
            hidden_features=model_cfg['hidden_features'],
            omega_0=model_cfg['omega_0'],
            sigma_0=model_cfg['sigma_0'],
            use_complex=model_cfg['use_complex']
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded meta-learned initialization")

        # Update config with model settings
        config.network.depth = model_cfg['depth']
        config.network.hidden_features = model_cfg['hidden_features']
        config.network.omega_0 = model_cfg['omega_0']
        config.network.sigma_0 = model_cfg['sigma_0']

    else:
        print("\nUsing random initialization")
        model = FunctionalWIRE(
            in_dim=config.network.in_dim,
            out_dim=config.network.out_dim,
            depth=args.depth,
            hidden_features=args.hidden,
            omega_0=args.omega_0,
            sigma_0=args.sigma_0,
            use_complex=config.network.complex
        ).to(device)

        config.network.depth = args.depth
        config.network.hidden_features = args.hidden
        config.network.omega_0 = args.omega_0
        config.network.sigma_0 = args.sigma_0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ==================== Load Data ====================
    print(f"\nLoading data: {args.data_file}")
    dataset = MetaFlowDataset(
        case_paths=[args.data_file],
        config=config,
        device=device,
        preload=True
    )
    print(f"Loaded {dataset.tasks[0].n_points:,} fluid points")

    # ==================== Setup Training ====================
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create loss function
    if config.training.use_cosine:
        loss_fn = create_cosine_loss(config)
    else:
        loss_fn = create_mse_loss(config)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iterations, eta_min=args.lr * 0.01
    )

    # ==================== Training Loop ====================
    print(f"\nStarting fine-tuning for {args.iterations} iterations")
    print(f"Learning rate: {args.lr}")
    print("=" * 50)

    start_time = time.time()
    losses = []

    for iteration in range(args.iterations):
        iter_start = time.time()

        model.train()
        optimizer.zero_grad()

        # Sample batch
        batch = dataset.sample_task_batch(0, args.batch_size)

        # Forward pass
        params = model.get_params()
        loss = loss_fn(batch.coords, batch.velocities, model, params)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        iter_time = time.time() - iter_start

        # Logging
        if (iteration + 1) % args.log_every == 0:
            avg_loss = np.mean(losses[-args.log_every:])
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            print(f"[Iter {iteration+1}/{args.iterations}] "
                  f"loss={avg_loss:.4f} "
                  f"lr={current_lr:.2e} "
                  f"time={elapsed/60:.1f}min")

            tb_writer.add_scalar('Loss/train', avg_loss, iteration)
            tb_writer.add_scalar('LR', current_lr, iteration)

            if args.wandb:
                wandb.log({
                    'iteration': iteration,
                    'loss': avg_loss,
                    'lr': current_lr
                })

        # Save checkpoint
        if (iteration + 1) % args.save_every == 0:
            ckpt_path = output_dir / f'checkpoint_{iteration+1}.pt'
            torch.save({
                'iteration': iteration + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'config': {
                    'depth': config.network.depth,
                    'hidden_features': config.network.hidden_features,
                    'omega_0': config.network.omega_0,
                    'sigma_0': config.network.sigma_0,
                }
            }, ckpt_path)

    # ==================== Final Save ====================
    final_path = output_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'in_dim': config.network.in_dim,
            'out_dim': config.network.out_dim,
            'depth': config.network.depth,
            'hidden_features': config.network.hidden_features,
            'omega_0': config.network.omega_0,
            'sigma_0': config.network.sigma_0,
            'use_complex': config.network.complex
        },
        'init_type': init_type,
        'data_file': args.data_file,
        'final_loss': losses[-1]
    }, final_path)

    print(f"\nTraining complete!")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Final model saved to: {final_path}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")

    # Save loss curve
    np.save(output_dir / 'losses.npy', np.array(losses))

    tb_writer.close()
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
