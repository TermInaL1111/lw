"""Event representation module for 3D ResNet input.

Converts raw event streams into 3D volumes (C, D, H, W) for processing
by 3D convolutional networks.

E2V Architecture (from DenseVoxel paper):
  Event stream → Event Frame (Pos) → stack into 3D volume (1, 100, 256, 256)
  → 3D ResNet-152 encoder

Xu2025 improvement:
  Event stream → Sobel Event Frame (Pos) → stack into 3D volume (1, 100, 256, 256)
  → 3D ResNet-152 + ECA encoder

SAE-Temporal improvement (our work):
  Event stream → [Event Frame (Pos) + SAE time surface] → (2, 100, 256, 256)
  → Preserves μs-level temporal information that binary frames discard
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def build_event_volume(events, num_frames=100, image_size=256,
                        representation="event_frame_pos",
                        sae_decay_tau=None, sae_blur_sigma=0.5):
    """Convert event stream to 3D volume for 3D CNN processing.

    Args:
        events: (N, 4) array [x, y, t, p] float32
        num_frames: number of time bins (D dimension)
        image_size: (H, W) spatial size
        representation: one of:
            "event_frame_pos"        — (1, D, H, W) binary positive events
            "event_frame"            — (1, D, H, W) signed event count
            "sobel_event_frame_pos"  — (1, D, H, W) Sobel edge-enhanced
            "event_frame_sae"        — (2, D, H, W) EventFrame + SAE time surface
            "sobel_event_frame_sae"  — (2, D, H, W) Sobel EventFrame + SAE time surface
        sae_decay_tau: SAE exponential decay constant (None = no decay)
        sae_blur_sigma: SAE Gaussian blur sigma (default 0.5 for noise reduction)

    Returns:
        tensor (C, num_frames, H, W) — 3D volume
    """
    if representation == "event_frame_pos":
        frames = event_frame_pos(events, num_frames, image_size)
        frames = frames.permute(1, 0, 2, 3)  # (N,1,H,W)→(1,N,H,W)
    elif representation == "event_frame":
        frames = event_frame(events, num_frames, image_size)
        frames = frames.permute(1, 0, 2, 3)
    elif representation == "sobel_event_frame_pos":
        base = event_frame_pos(events, num_frames, image_size)
        frames = sobel_event_frame(base)
        frames = frames.permute(1, 0, 2, 3)
    elif representation == "event_frame_sae":
        ev_frames = event_frame_pos(events, num_frames, image_size)
        ev_frames = ev_frames.permute(1, 0, 2, 3)  # (1, D, H, W)
        sae = build_sae_volume(events, num_frames, image_size,
                               decay_tau=sae_decay_tau, blur_sigma=sae_blur_sigma)
        frames = torch.cat([ev_frames, sae], dim=0)   # (2, D, H, W)
    elif representation == "sobel_event_frame_sae":
        base = event_frame_pos(events, num_frames, image_size)
        ev_frames = sobel_event_frame(base)
        ev_frames = ev_frames.permute(1, 0, 2, 3)
        sae = build_sae_volume(events, num_frames, image_size,
                               decay_tau=sae_decay_tau, blur_sigma=sae_blur_sigma)
        frames = torch.cat([ev_frames, sae], dim=0)   # (2, D, H, W)
    else:
        raise ValueError(f"Unknown representation: {representation}")

    return frames


def event_frame_pos(events, num_frames=100, image_size=256):
    """Event Frame (Pos) — binary indicator of positive events per pixel per frame.

    For each time window: if pixel has >=1 positive event → 1, else 0.

    Args:
        events: (N, 4) tensor [x, y, t, p]
        num_frames: number of time windows
        image_size: output frame size
    Returns:
        (num_frames, 1, H, W) tensor
    """
    H = W = image_size
    frames = torch.zeros(num_frames, 1, H, W)

    if len(events) == 0:
        return frames

    x = events[:, 0].long().clamp(0, W - 1)
    y = events[:, 1].long().clamp(0, H - 1)
    t = events[:, 2]
    p = events[:, 3]

    t_min, t_max = t.min(), t.max()
    if t_max > t_min:
        t_norm = (t - t_min) / (t_max - t_min)
    else:
        t_norm = torch.zeros_like(t)

    frame_idx = (t_norm * num_frames).long().clamp(0, num_frames - 1)
    pos_mask = p > 0

    if pos_mask.any():
        flat_idx = frame_idx[pos_mask] * H * W + y[pos_mask] * W + x[pos_mask]
        frames.view(-1).scatter_(0, flat_idx, 1.0)

    return frames


def event_frame(events, num_frames=100, image_size=256):
    """Event Frame (Pos+Neg) — signed event count per pixel per frame.

    Positive events → +1, negative events → -1.
    """
    H = W = image_size
    frames = torch.zeros(num_frames, 1, H, W)

    if len(events) == 0:
        return frames

    x = events[:, 0].long().clamp(0, W - 1)
    y = events[:, 1].long().clamp(0, H - 1)
    t = events[:, 2]
    p = events[:, 3]

    t_min, t_max = t.min(), t.max()
    if t_max > t_min:
        t_norm = (t - t_min) / (t_max - t_min)
    else:
        t_norm = torch.zeros_like(t)

    frame_idx = (t_norm * num_frames).long().clamp(0, num_frames - 1)

    flat_idx = frame_idx * H * W + y * W + x
    frames.view(-1).scatter_add_(0, flat_idx, p)

    return frames


def sobel_event_frame(event_frames):
    """Apply Sobel edge detection to event frames (Xu2025).

    For each frame, compute gradient magnitude using 3x3 Sobel operators.

    Args:
        event_frames: (N, 1, H, W) Event Frame (Pos) tensor
    Returns:
        (N, 1, H, W) Sobel Event Frame (Pos), normalized to [0, 1]
    """
    N, C, H, W = event_frames.shape

    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                           device=event_frames.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                           device=event_frames.device).view(1, 1, 3, 3)

    frames_flat = event_frames.view(N * C, 1, H, W)
    frames_pad = F.pad(frames_flat, (1, 1, 1, 1), mode='replicate')

    gx = F.conv2d(frames_pad, sobel_x)
    gy = F.conv2d(frames_pad, sobel_y)
    grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    grad_mag = grad_mag.view(N, C, H, W)

    # Per-frame normalization to [0, 1]
    grad_flat = grad_mag.view(N, -1)
    g_min = grad_flat.min(dim=1, keepdim=True)[0]
    g_max = grad_flat.max(dim=1, keepdim=True)[0]
    g_range = g_max - g_min
    g_range[g_range < 1e-8] = 1.0
    grad_mag = (grad_flat - g_min) / g_range
    grad_mag = grad_mag.view(N, C, H, W)

    return grad_mag


# ==============================================================================
# SAE (Surface of Active Events) — Temporal Event Representation
# ==============================================================================
# SAE preserves per-pixel latest-event timestamps, retaining the μs-level
# temporal information that binary EventFrame discards.
#
# Key references:
#   - Benosman et al., "Event-based visual flow", IEEE TNNLS 2014
#   - Mueggler et al., "Fast Event-based Corner Detection", BMVC 2017
#   - Gallego et al., "Event-based Vision: A Survey", IEEE TPAMI 2022
# ==============================================================================


def build_sae(events, image_size=256, decay_tau=None, blur_sigma=None):
    """Build Surface of Active Events (SAE) for a batch of events.

    SAE[i,j] = normalized timestamp of the most recent event at pixel (i,j).
    Zero where no event occurred.

    Args:
        events: (N, 4) tensor [x, y, t, p]
        image_size: spatial resolution
        decay_tau: exponential decay constant (None = no decay, raw timestamp)
        blur_sigma: Gaussian blur sigma for noise suppression (None = no blur)

    Returns:
        (1, H, W) tensor — SAE time surface, values in [0, 1]
    """
    H = W = image_size

    if len(events) == 0:
        return torch.zeros(1, H, W)

    x = events[:, 0].long().clamp(0, W - 1)
    y = events[:, 1].long().clamp(0, H - 1)
    t = events[:, 2]

    t_min, t_max = t.min(), t.max()
    t_range = t_max - t_min
    if t_range < 1e-8:
        t_norm = torch.zeros_like(t)
    else:
        t_norm = (t - t_min) / t_range

    # Build SAE: keep the most recent (largest t_norm) at each pixel
    sae = torch.zeros(H, W)
    flat_idx = y * W + x
    # Sort by t_norm, then scatter — later (higher t_norm) overwrites earlier
    sort_idx = t_norm.argsort()
    sae.view(-1).scatter_(0, flat_idx[sort_idx], t_norm[sort_idx])
    sae = sae.view(1, H, W)

    # Optional: exponential decay
    if decay_tau is not None and decay_tau > 0:
        sae = 1.0 - torch.exp(-sae / decay_tau)

    # Optional: Gaussian blur for noise suppression
    if blur_sigma is not None and blur_sigma > 0:
        kernel_size = int(2 * blur_sigma * 3 + 1) | 1  # odd
        sae = _gaussian_blur(sae.unsqueeze(0), kernel_size, blur_sigma).squeeze(0)

    # Normalize to [0, 1]
    s_min, s_max = sae.min(), sae.max()
    if s_max > s_min:
        sae = (sae - s_min) / (s_max - s_min)

    return sae


def build_sae_volume(events, num_frames=100, image_size=256,
                     decay_tau=None, blur_sigma=None):
    """Build SAE volume across time windows.

    Each time bin gets its own SAE computed from events in that window.
    This preserves temporal structure while adding per-pixel timing info.

    Args:
        events: (N, 4) tensor [x, y, t, p]
        num_frames: number of time bins
        image_size: spatial resolution
        decay_tau: exponential decay constant
        blur_sigma: Gaussian blur sigma

    Returns:
        (1, num_frames, H, W) tensor — SAE volume
    """
    H = W = image_size

    if len(events) == 0:
        return torch.zeros(1, num_frames, H, W)

    x = events[:, 0].long().clamp(0, W - 1)
    y = events[:, 1].long().clamp(0, H - 1)
    t = events[:, 2]
    p = events[:, 3]

    t_min, t_max = t.min(), t.max()
    if t_max > t_min:
        t_norm = (t - t_min) / (t_max - t_min)
    else:
        t_norm = torch.zeros_like(t)

    frame_idx = (t_norm * num_frames).long().clamp(0, num_frames - 1)

    sae_volume = torch.zeros(1, num_frames, H, W)

    for f in range(num_frames):
        mask = frame_idx == f
        if mask.any():
            ev_f = torch.stack([x[mask], y[mask], t[mask], p[mask]], dim=1)
            sae_volume[0, f] = build_sae(ev_f, image_size=image_size,
                                         decay_tau=decay_tau,
                                         blur_sigma=blur_sigma).squeeze(0)

    return sae_volume


def sae_gradient(sae_volume, blur_sigma=1.0):
    """Compute spatial gradient of SAE volume for edge/motion detection.

    The SAE gradient reveals event edges and their motion direction.
    High gradient magnitude = strong edges, gradient direction = motion orientation.

    Args:
        sae_volume: (B, D, H, W) or (1, D, H, W) SAE volume
        blur_sigma: pre-blur sigma for noise reduction

    Returns:
        (B, 2, D, H, W) tensor with [grad_x, grad_y] channels
    """
    if sae_volume.dim() == 4:
        sae_volume = sae_volume.unsqueeze(0)
    B, C, D, H, W = sae_volume.shape

    # Merge B and D for batch processing
    sae_flat = sae_volume.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

    if blur_sigma > 0:
        ks = int(2 * blur_sigma * 3 + 1) | 1
        sae_flat = _gaussian_blur(sae_flat, ks, blur_sigma)

    # Sobel kernels
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                           device=sae_volume.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                           device=sae_volume.device).view(1, 1, 3, 3)

    padded = F.pad(sae_flat, (1, 1, 1, 1), mode='replicate')
    gx = F.conv2d(padded, sobel_x)
    gy = F.conv2d(padded, sobel_y)

    gx = gx.view(B, D, 1, H, W).permute(0, 2, 1, 3, 4)
    gy = gy.view(B, D, 1, H, W).permute(0, 2, 1, 3, 4)

    return torch.cat([gx, gy], dim=1)


def _gaussian_blur(x, kernel_size, sigma):
    """Apply 2D Gaussian blur."""
    ax = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - (kernel_size - 1) / 2
    gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
    gauss = gauss / gauss.sum()
    kernel = gauss[:, None] * gauss[None, :]
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(F.pad(x, (kernel_size//2,)*4, mode='replicate'),
                     kernel, groups=x.shape[1])
