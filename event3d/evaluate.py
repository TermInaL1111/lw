"""Evaluation script for E2V model.

Evaluates a trained model on the test set and reports:
- mIoU (mean Intersection over Union)
- F-Score
- Per-class IoU
"""
import os
import sys
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import E2VModel
from datasets import get_dataloader
from utils.metrics import compute_iou, compute_fscore, compute_all_metrics, compute_per_class_iou
from utils.threshold import search_optimal_threshold, apply_threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate E2V model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--data_root", type=str,
                        default="D:/cugdocuments/科研/datasets/synthevox3d/SynthEVox3D-Tiny")
    parser.add_argument("--split_csv", type=str,
                        default="D:/cugdocuments/科研/datasets/synthevox3d/SynthEVox3D-Tiny/data_split/SynthEVox3D-Tiny_data_split.csv")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Binarization threshold (auto-search if not set)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save evaluation results")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, device, threshold=0.2):
    """Evaluate model on a dataset.

    Returns:
        metrics dict, all_predictions, all_targets, all_class_ids
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_class_ids = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        event_volume = batch['event_volume'].to(device)
        voxel_target = batch['voxel']
        class_ids = batch['class_id']

        pred = model(event_volume)
        pred_probs = torch.sigmoid(pred)

        all_preds.append(pred_probs.cpu())
        all_targets.append(voxel_target)
        all_class_ids.append(class_ids)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_class_ids = torch.cat(all_class_ids, dim=0)

    metrics = compute_all_metrics(all_preds, all_targets, threshold=threshold)
    per_class = compute_per_class_iou(all_preds, all_targets, all_class_ids,
                                       num_classes=13, threshold=threshold)

    return metrics, per_class, all_preds, all_targets, all_class_ids


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model_cfg = config['model']
    model = E2VModel(
        in_channels=model_cfg.get('encoder_in_channels', 1),
        use_eca=model_cfg.get('use_eca', False),
        voxel_size=model_cfg.get('voxel_size', 32),
    )
    model = model.to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    print(f"Checkpoint best IoU: {ckpt.get('best_iou', 'N/A')}")

    # Data
    data_root = os.path.join(args.data_root, 'event_3d_scan_tiny')
    split_csv = args.split_csv

    loader = get_dataloader(
        root_dir=data_root, split_csv=split_csv, split=args.split,
        batch_size=args.batch_size,
        representation=config['data'].get('event_representation', 'event_frame_pos'),
        image_size=config['data'].get('image_size', 256),
        num_frames=config['data'].get('num_frames', 100),
        augmentation=False,
        num_workers=4,
        shuffle=False,
    )

    print(f"Evaluating on {args.split} set ({len(loader.dataset)} samples)")

    # Search threshold if not provided
    threshold = args.threshold
    if threshold is None:
        print("Searching optimal threshold on evaluation set...")
        threshold, best_iou_at_t, _ = search_optimal_threshold(
            model, loader, device,
            threshold_range=tuple(config['evaluation'].get('threshold_range', [0.10, 0.50])),
            step=config['evaluation'].get('threshold_step', 0.01),
        )
        print(f"Optimal threshold: {threshold:.2f} (IoU: {best_iou_at_t:.4f})")

    # Evaluate
    metrics, per_class, all_preds, all_targets, all_class_ids = evaluate(
        model, loader, device, threshold=threshold)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Samples: {len(loader.dataset)}")
    print(f"\nOverall Metrics:")
    print(f"  mIoU:    {metrics['iou']:.4f}")
    print(f"  F-Score: {metrics['fscore']:.4f}")
    print(f"\nPer-Class IoU:")
    class_names = [
        'Airplane', 'Bench', 'Cabinet', 'Car', 'Chair', 'Displayer',
        'Lamp', 'Speaker', 'Rifle', 'Sofa', 'Table', 'Telephone', 'Watercraft'
    ]
    for c in range(13):
        if c in per_class:
            print(f"  {class_names[c]:12s}: {per_class[c]:.4f}")
        else:
            print(f"  {class_names[c]:12s}: N/A")
    print(f"  {'Mean':12s}: {np.mean(list(per_class.values())):.4f}")

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        # Save numerical results
        results_path = os.path.join(args.output_dir, f"results_{args.split}.txt")
        with open(results_path, 'w') as f:
            f.write(f"Split: {args.split}\n")
            f.write(f"Threshold: {threshold:.2f}\n")
            f.write(f"mIoU: {metrics['iou']:.4f}\n")
            f.write(f"F-Score: {metrics['fscore']:.4f}\n")
            f.write("\nPer-Class IoU:\n")
            for c in range(13):
                if c in per_class:
                    f.write(f"  {class_names[c]:12s}: {per_class[c]:.4f}\n")
        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
