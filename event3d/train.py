"""Training script for Event-to-Voxel (E2V) 3D reconstruction.

Reproduces the Dense Voxel baseline (Chen 2023) and Xu2025 improvements.
"""
import os
import sys
import time
import random
import argparse
import datetime
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import E2VModel
from datasets import get_dataloader
from utils.loss import FocalLoss, DiceLoss, CombinedLoss
from utils.metrics import compute_iou, compute_fscore, compute_all_metrics
from utils.threshold import search_optimal_threshold


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train E2V model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--data_root", type=str,
                        default="D:/cugdocuments/科研/datasets/synthevox3d/SynthEVox3D-Tiny")
    parser.add_argument("--split_csv", type=str,
                        default="D:/cugdocuments/科研/datasets/synthevox3d/SynthEVox3D-Tiny/data_split/SynthEVox3D-Tiny_data_split.csv")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def build_model(config, device):
    """Build E2V model from config."""
    model_cfg = config['model']
    model = E2VModel(
        in_channels=model_cfg.get('encoder_in_channels', 1),
        use_eca=model_cfg.get('use_eca', False),
        voxel_size=model_cfg.get('voxel_size', 32),
    )
    model = model.to(device)
    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}")
    return model


def build_loss(config):
    """Build loss function from config."""
    loss_cfg = config['training']
    loss_name = loss_cfg.get('loss', 'focal')

    if loss_name == 'focal':
        return FocalLoss(
            alpha=loss_cfg.get('focal_alpha', 0.25),
            gamma=loss_cfg.get('focal_gamma', 2.0),
        )
    elif loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'dice':
        return DiceLoss()
    elif loss_name == 'combined':
        return CombinedLoss(
            focal_weight=loss_cfg.get('focal_weight', 1.0),
            dice_weight=loss_cfg.get('dice_weight', 0.5),
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


def build_optimizer(model, config):
    """Build optimizer from config."""
    train_cfg = config['training']
    lr = train_cfg.get('learning_rate', 1e-5)
    wd = train_cfg.get('weight_decay', 0.0)
    opt_name = train_cfg.get('optimizer', 'adamw')

    if opt_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(optimizer, config):
    """Build learning rate scheduler from config."""
    train_cfg = config['training']
    sch_name = train_cfg.get('scheduler', 'none')

    if sch_name == 'none':
        return None
    elif sch_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg.get('num_epochs', 100))
    elif sch_name == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_cfg.get('scheduler_step', 50),
            gamma=train_cfg.get('scheduler_gamma', 0.1))
    else:
        raise ValueError(f"Unknown scheduler: {sch_name}")


def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    epoch, writer, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = len(dataloader)
    print_interval = config['logging'].get('print_interval', 10)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        event_volume = batch['event_volume'].to(device)
        voxel_target = batch['voxel'].to(device)

        # Forward pass
        pred = model(event_volume)

        # Loss
        loss = criterion(torch.sigmoid(pred), voxel_target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute IoU for monitoring
        with torch.no_grad():
            iou = compute_iou(torch.sigmoid(pred), voxel_target, threshold=0.2)
            total_iou += iou

        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        avg_iou = total_iou / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'iou': f'{avg_iou:.4f}'})

        # TensorBoard logging
        if writer and batch_idx % print_interval == 0:
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/iou', iou, global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

    return total_loss / num_batches, total_iou / num_batches


@torch.no_grad()
def validate(model, dataloader, criterion, device, config):
    """Run validation."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_fscore = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc="[Val]")
    for batch in pbar:
        event_volume = batch['event_volume'].to(device)
        voxel_target = batch['voxel'].to(device)

        pred = model(event_volume)
        loss = criterion(torch.sigmoid(pred), voxel_target)

        total_loss += loss.item()
        iou = compute_iou(torch.sigmoid(pred), voxel_target, threshold=0.2)
        fscore = compute_fscore(torch.sigmoid(pred), voxel_target, threshold=0.2)
        total_iou += iou
        total_fscore += fscore

        avg_loss = total_loss / num_batches
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'iou': f'{iou:.4f}'})

    return (total_loss / num_batches,
            total_iou / num_batches,
            total_fscore / num_batches)


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override with CLI args
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs

    # Set seed
    set_seed(config['training'].get('seed', 42))

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup directories
    os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)

    # Data paths
    data_root = os.path.join(args.data_root, 'event_3d_scan_tiny')
    split_csv = args.split_csv

    # Build dataloaders
    data_cfg = config['data']
    train_cfg = config['training']
    batch_size = train_cfg.get('batch_size', 5)
    num_workers = args.num_workers

    train_loader = get_dataloader(
        root_dir=data_root, split_csv=split_csv, split="train",
        batch_size=batch_size,
        representation=data_cfg.get('event_representation', 'event_frame_pos'),
        image_size=data_cfg.get('image_size', 256),
        num_frames=data_cfg.get('num_frames', 100),
        augmentation=True,
        num_workers=num_workers,
        shuffle=True,
    )

    val_loader = get_dataloader(
        root_dir=data_root, split_csv=split_csv, split="val",
        batch_size=batch_size,
        representation=data_cfg.get('event_representation', 'event_frame_pos'),
        image_size=data_cfg.get('image_size', 256),
        num_frames=data_cfg.get('num_frames', 100),
        augmentation=False,
        num_workers=num_workers,
        shuffle=False,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Build model, loss, optimizer
    model = build_model(config, device)
    criterion = build_loss(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # Resume from checkpoint
    start_epoch = 0
    best_iou = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_iou = ckpt.get('best_iou', 0.0)
        print(f"Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}")

    # TensorBoard
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(config['logging']['log_dir'], time_str))
    writer.add_text('config', yaml.dump(config))

    # Training loop
    num_epochs = train_cfg.get('num_epochs', 100)
    eval_interval = config['logging'].get('eval_interval', 1)
    save_interval = config['checkpoint'].get('save_interval', 10)
    save_dir = config['checkpoint']['save_dir']

    print(f"\nTraining {num_epochs} epochs, batch_size={batch_size}")
    print(f"Save dir: {save_dir}")
    print("=" * 60)

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, writer, config)

        # Validation
        if (epoch + 1) % eval_interval == 0:
            val_loss, val_iou, val_fscore = validate(
                model, val_loader, criterion, device, config)

            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/iou', val_iou, epoch)
            writer.add_scalar('val/fscore', val_fscore, epoch)

            is_best = val_iou > best_iou
            if is_best:
                best_iou = val_iou

            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} IoU: {train_iou:.4f} | "
                  f"Val Loss: {val_loss:.4f} IoU: {val_iou:.4f} FScore: {val_fscore:.4f} | "
                  f"{'*BEST*' if is_best else ''}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} IoU: {train_iou:.4f}")

        # Save checkpoint
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
            ckpt_path = os.path.join(save_dir, f"e2v_epoch{epoch+1:03d}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'config': config,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Save best model
        if best_iou == val_iou:
            best_path = os.path.join(save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'config': config,
            }, best_path)

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"  Epoch time: {epoch_time:.1f}s")

    print("=" * 60)
    print(f"Training complete. Best val IoU: {best_iou:.4f}")

    # Search optimal threshold
    if config['evaluation'].get('search_threshold', True):
        print("\nSearching optimal binarization threshold...")
        best_t, best_m, results = search_optimal_threshold(
            model, val_loader, device,
            threshold_range=tuple(config['evaluation'].get('threshold_range', [0.10, 0.50])),
            step=config['evaluation'].get('threshold_step', 0.01),
        )
        print(f"Optimal threshold: {best_t:.2f}, Best IoU: {best_m:.4f}")

        # Print top-5 thresholds
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top-5 thresholds:")
        for t, m in sorted_results:
            print(f"  t={t:.2f}: IoU={m:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
