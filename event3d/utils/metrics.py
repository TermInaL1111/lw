import torch
import numpy as np


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5,
                eps: float = 1e-7) -> float:
    """Compute Intersection over Union for binary voxel grids.

    Args:
        pred: (N, C, D, H, W) or (N, D, H, W) predicted probabilities
        target: same shape, binary ground truth
        threshold: binarization threshold for predictions
    Returns:
        mean IoU across batch
    """
    pred_binary = (pred > threshold).float()
    target = target.float()

    intersection = (pred_binary * target).sum(dim=tuple(range(1, pred.dim())))
    union = (pred_binary + target).clamp(0, 1).sum(dim=tuple(range(1, pred.dim())))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def compute_fscore(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5,
                   fscore_threshold: float = 0.2, eps: float = 1e-7) -> float:
    """Compute F-Score for binary voxel grids.

    In event-based 3D reconstruction, the F-Score threshold is often set
    to 20% (vs traditional 1%) due to event data sparsity.

    Args:
        pred: predicted probabilities
        target: binary ground truth
        threshold: binarization threshold
        fscore_threshold: minimum IoU to count as "correct"
    Returns:
        mean F-Score across batch
    """
    pred_binary = (pred > threshold).float()
    target = target.float()

    n = pred.shape[0]
    f_scores = []
    for i in range(n):
        intersection = (pred_binary[i] * target[i]).sum()
        precision = intersection / (pred_binary[i].sum() + eps)
        recall = intersection / (target[i].sum() + eps)
        f_score = 2 * precision * recall / (precision + recall + eps)
        f_scores.append(f_score.item())

    return float(np.mean(f_scores))


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5,
                        fscore_threshold: float = 0.2) -> dict:
    """Compute all evaluation metrics."""
    return {
        'iou': compute_iou(pred, target, threshold),
        'fscore': compute_fscore(pred, target, threshold, fscore_threshold),
    }


def compute_per_class_iou(pred: torch.Tensor, target: torch.Tensor,
                           class_ids: torch.Tensor, num_classes: int = 13,
                           threshold: float = 0.5, eps: float = 1e-7) -> dict:
    """Compute IoU per class.

    Args:
        pred: (N, 1, D, H, W) predictions
        target: (N, 1, D, H, W) ground truth
        class_ids: (N,) class indices
    Returns:
        dict mapping class_id -> mean IoU
    """
    pred_binary = (pred > threshold).float()
    target = target.float()
    per_class = {}
    for c in range(num_classes):
        mask = (class_ids == c)
        if mask.sum() == 0:
            continue
        pred_c = pred_binary[mask]
        target_c = target[mask]
        intersection = (pred_c * target_c).sum(dim=tuple(range(1, pred_c.dim())))
        union = (pred_c + target_c).clamp(0, 1).sum(dim=tuple(range(1, pred_c.dim())))
        iou_c = (intersection + eps) / (union + eps)
        per_class[c] = iou_c.mean().item()
    return per_class
