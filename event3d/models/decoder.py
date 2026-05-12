"""3D U-Net Decoder for event-based voxel reconstruction.

Matches the E2V architecture diagram:
  - Mid-section blocks process encoder skip features
  - TConv3D upsampling (S2) at each stage
  - Append (concat) skip connections from encoder
  - Conv3D fusion after each concatenation
  - Final Sigmoid → (1, 32, 32, 32)

Decoder path:
  bottleneck (2048ch, smallest spatial)
    → mid_section → 512ch
    → TConv3D → 8³ → concat(256ch skip) → Conv3D
    → TConv3D → 16³ → concat(128ch skip) → Conv3D
    → TConv3D → 32³ → concat(64ch skip) → Conv3D
    → 1×1×1 Conv3D → Sigmoid → (1, 32, 32, 32)
"""
import torch
import torch.nn as nn


class MidSection(nn.Module):
    """Processes encoder features into skip connections for the decoder.

    The architecture diagram shows blue blocks that process each encoder
    stage output before feeding into the decoder via skip connections.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderStage(nn.Module):
    """One decoder stage: TConv3D upsampling → concat skip → Conv3D fusion."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.tconv = nn.ConvTranspose3d(in_channels, out_channels,
                                         kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_up = nn.BatchNorm3d(out_channels)
        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.tconv(x)
        x = self.bn_up(x)
        x = self.relu(x)
        # Match spatial size to skip (odd encoder dimensions cause mismatch)
        if x.shape[2:] != skip.shape[2:]:
            x = torch.nn.functional.interpolate(
                x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)  # Append (concat)
        x = self.fusion(x)
        return x


class Decoder3D_UNet(nn.Module):
    """3D U-Net decoder for E2V model.

    Takes multi-scale encoder features and produces 32³ voxel occupancy.

    Mid-section processing:
      enc4 (2048ch, smallest) → mid4 → 512ch bottleneck
      enc3 (1024ch) → mid3 → 256ch skip
      enc2 (512ch) → mid2 → 128ch skip
      enc1 (256ch) → mid1 → 64ch skip

    Decoder stages:
      bottleneck → stage3 (TConv+skip256) → stage2 (TConv+skip128) → stage1 (TConv+skip64)
    """
    def __init__(self, enc_channels=(256, 512, 1024, 2048),
                 mid_channels=(64, 128, 256, 512),
                 decoder_channels=(256, 128, 64, 32),
                 out_channels=1):
        super().__init__()
        # Mid-sections: process encoder features for skip connections
        self.mid1 = MidSection(enc_channels[0], mid_channels[0])   # 256→64
        self.mid2 = MidSection(enc_channels[1], mid_channels[1])   # 512→128
        self.mid3 = MidSection(enc_channels[2], mid_channels[2])   # 1024→256
        self.mid4 = MidSection(enc_channels[3], mid_channels[3])   # 2048→512

        # Decoder stages: upsample + concat skip + fusion
        # bottleneck (512) → TConv + skip(256) → 256 → TConv + skip(128) → 128 → TConv + skip(64) → 64
        self.stage3 = DecoderStage(
            mid_channels[3], mid_channels[2], decoder_channels[0])  # 512+256→256
        self.stage2 = DecoderStage(
            decoder_channels[0], mid_channels[1], decoder_channels[1])  # 256+128→128
        self.stage1 = DecoderStage(
            decoder_channels[1], mid_channels[0], decoder_channels[2])  # 128+64→64

        # Final output
        self.output = nn.Sequential(
            nn.Conv3d(decoder_channels[2], decoder_channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(decoder_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv3d(decoder_channels[3], out_channels, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, enc_features):
        """Forward pass.

        Args:
            enc_features: dict with 'enc1'..'enc4' keys from 3D ResNet encoder

        Returns:
            (B, out_channels, D, H, W) output logits
        """
        # Process encoder features through mid-sections
        skip1 = self.mid1(enc_features['enc1'])   # (B, 64, ...)
        skip2 = self.mid2(enc_features['enc2'])   # (B, 128, ...)
        skip3 = self.mid3(enc_features['enc3'])   # (B, 256, ...)
        bottleneck = self.mid4(enc_features['enc4'])  # (B, 512, ...)

        # Decoder with skip connections
        x = self.stage3(bottleneck, skip3)  # 512→256
        x = self.stage2(x, skip2)           # 256→128
        x = self.stage1(x, skip1)           # 128→64

        x = self.output(x)
        # Ensure exact 32³ output (upsampling from 25→32 depth, 64→32 spatial)
        if x.shape[2:] != (32, 32, 32):
            x = torch.nn.functional.interpolate(
                x, size=(32, 32, 32), mode='trilinear', align_corners=False)
        return x
