"""E2V Model: End-to-End Event-to-Voxel 3D Reconstruction.

Architecture (from Dense Voxel paper, Chen et al. ICVR 2023):

  Event Stream → Event Frame stacking → 3D Volume (B, 1, D, H, W)
    → 3D ResNet-152 Encoder (all 2D convs → 3D convs)
    → Mid-Section blocks (multi-scale processing)
    → 3D U-Net Decoder (TConv3D + skip connections)
    → Sigmoid → (B, 1, 32, 32, 32) voxel occupancy

With optional Xu2025 improvements:
  - ECA attention in encoder Bottleneck blocks
  - Sobel Event Frame event representation
"""
import torch
import torch.nn as nn

from .encoder import ResNet152_3D_Encoder
from .decoder import Decoder3D_UNet


class E2VModel(nn.Module):
    """End-to-End Event-to-Voxel model.

    Args:
        in_channels: input channels (1 for Event Frame, 3 for MCEF)
        use_eca: add ECA modules to encoder (Xu2025)
        voxel_size: output resolution (32 for 32³)
    """
    def __init__(self, in_channels=1, use_eca=False, voxel_size=32, dropout=0.0):
        super().__init__()
        self.encoder = ResNet152_3D_Encoder(in_channels=in_channels, use_eca=use_eca)
        self.decoder = Decoder3D_UNet(out_channels=1, final_size=(32, 32, 32),
                                      dropout=dropout)
        self.voxel_size = voxel_size
        self.in_channels = in_channels

    def forward(self, event_volume):
        """Forward pass.

        Args:
            event_volume: (B, C, D, H, W)
                B = batch size
                C = in_channels (1 for Event Frame)
                D = num_frames (100 time windows)
                H, W = 256 (frame size)
        Returns:
            (B, 1, 32, 32, 32) occupancy logits (before sigmoid)
        """
        enc_features = self.encoder(event_volume)
        output = self.decoder(enc_features)
        return output

    @torch.no_grad()
    def predict(self, event_volume, threshold=0.5):
        """Predict binary voxel occupancy.

        Args:
            event_volume: (B, C, D, H, W)
            threshold: binarization threshold
        Returns:
            (B, 1, 32, 32, 32) binary voxel grid
        """
        self.eval()
        logits = self.forward(event_volume)
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
