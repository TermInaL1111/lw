import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for binary classification with class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (N, *) sigmoid probabilities in [0, 1]
            target: (N, *) binary labels in {0, 1}
        """
        pred = pred.clamp(min=1e-7, max=1.0 - 1e-7)
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation / occupancy prediction."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (N, *) sigmoid probabilities
            target: (N, *) binary labels
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """Combined Focal + Dice loss."""
    def __init__(self, focal_weight: float = 1.0, dice_weight: float = 0.5,
                 alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.focal_weight * self.focal(pred, target) + \
               self.dice_weight * self.dice(pred, target)
