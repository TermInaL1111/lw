import torch
import numpy as np
from tqdm import tqdm
from .metrics import compute_iou


@torch.no_grad()
def search_optimal_threshold(model, dataloader, device,
                              threshold_range=(0.10, 0.50), step=0.01,
                              metric_fn=None):
    """Search for the optimal binarization threshold on validation set.

    Args:
        model: E2EModel instance
        dataloader: validation dataloader
        device: torch device
        threshold_range: (min, max) search range
        step: search step size
        metric_fn: metric function (defaults to compute_iou)
    Returns:
        best_threshold (float), best_metric (float)
    """
    if metric_fn is None:
        metric_fn = compute_iou

    model.eval()
    all_preds = []
    all_targets = []

    for batch in tqdm(dataloader, desc="Collecting predictions"):
        event_volume = batch['event_volume'].to(device)
        targets = batch['voxel'].to(device)

        pred = model(event_volume)
        all_preds.append(pred.cpu())
        all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    thresholds = np.arange(threshold_range[0], threshold_range[1] + step / 2, step)
    best_threshold = threshold_range[0]
    best_metric = 0.0
    results = {}

    for t in thresholds:
        metric = metric_fn(all_preds, all_targets, threshold=t)
        results[float(t)] = metric
        if metric > best_metric:
            best_metric = metric
            best_threshold = float(t)

    return best_threshold, best_metric, results


def apply_threshold(pred: torch.Tensor, threshold: float) -> torch.Tensor:
    """Binarize occupancy probabilities."""
    return (pred > threshold).float()
