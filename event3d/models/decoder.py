"""3D U-Net Decoder for event-based voxel reconstruction.

Matches the E2V architecture diagram (DenseVoxel, Chen et al. ICVR 2023):
  - Mid-section blocks process encoder skip features (single Conv3D per block)
  - TConv3D upsampling at each stage
  - Append (concat) skip connections from encoder
  - Conv3D fusion after each concatenation (2-layer: 3x3x3 → 3x3x3)
  - AdaptiveAvgPool3d → (1, 32, 32, 32)

Decoder path:
  bottleneck (2048ch, smallest spatial)
    → mid4 (1x1x1 conv → 512ch)
    → TConv3D → concat(skip 256ch) → Conv3D fusion
    → TConv3D → concat(skip 128ch) → Conv3D fusion
    → TConv3D → concat(skip 64ch) → Conv3D fusion
    → output → AdaptiveAvgPool3d → (1, 32, 32, 32)

Total decoder params: ~32M, model total ~149M (matching paper: 149,155,905).
"""
import torch
import torch.nn as nn


class MidSection(nn.Module):
    """Processes one encoder feature level into a skip connection.

    Architecture diagram shows one Conv3D block per encoder output.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderStage(nn.Module):
    """One decoder stage: TConv3D upsampling → concat skip → Conv3D fusion.

    Standard U-Net pattern with 2-layer fusion after concatenation.
    """
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
        x = torch.cat([x, skip], dim=1)
        x = self.fusion(x)
        return x


class Decoder3D_UNet(nn.Module):
    """3D U-Net decoder for E2V model.

    Takes multi-scale encoder features and produces 32³ voxel occupancy.

    Encoder outputs (from ResNet152_3D_Encoder):
      enc1: (B, 256, ...)
      enc2: (B, 512, ...)
      enc3: (B, 1024, ...)
      enc4: (B, 2048, ...) — bottleneck, smallest spatial

    Mid-section processing (single Conv3D):
      enc4 → mid4 (1x1x1 conv, 512ch) — bottleneck
      enc3 → mid3 (3x3x3 conv, 256ch) — skip
      enc2 → mid2 (3x3x3 conv, 128ch) — skip
      enc1 → mid1 (3x3x3 conv, 64ch) — skip

    Decoder stages (TConv3D + 2-layer fusion):
      bottleneck → stage3 (+ skip3) → stage2 (+ skip2) → stage1 (+ skip1)
    """
    def __init__(self,
                 enc_channels=(256, 512, 1024, 2048),
                 mid_channels=(64, 128, 264, 512),
                 decoder_channels=(296, 128, 64, 32),
                 out_channels=1,
                 final_size=(32, 32, 32)):
        super().__init__()

        # Mid-sections: reduce encoder channels for skip connections
        self.mid1 = MidSection(enc_channels[0], mid_channels[0], kernel_size=3)
        self.mid2 = MidSection(enc_channels[1], mid_channels[1], kernel_size=3)
        self.mid3 = MidSection(enc_channels[2], mid_channels[2], kernel_size=3)
        self.mid4 = MidSection(enc_channels[3], mid_channels[3], kernel_size=1)

        # Decoder stages
        self.stage3 = DecoderStage(mid_channels[3], mid_channels[2],
                                   decoder_channels[0])
        self.stage2 = DecoderStage(decoder_channels[0], mid_channels[1],
                                   decoder_channels[1])
        self.stage1 = DecoderStage(decoder_channels[1], mid_channels[0],
                                   decoder_channels[2])

        # Final output
        self.output = nn.Sequential(
            nn.Conv3d(decoder_channels[2], decoder_channels[3],
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(decoder_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv3d(decoder_channels[3], out_channels, kernel_size=1),
        )

        # Adaptive pooling to ensure exact 32³ output
        self.pool = nn.AdaptiveAvgPool3d(final_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, enc_features):
        # Process encoder features through mid-sections
        skip1 = self.mid1(enc_features['enc1'])
        skip2 = self.mid2(enc_features['enc2'])
        skip3 = self.mid3(enc_features['enc3'])
        bottleneck = self.mid4(enc_features['enc4'])

        # Decoder with skip connections
        x = self.stage3(bottleneck, skip3)
        x = self.stage2(x, skip2)
        x = self.stage1(x, skip1)

        x = self.output(x)
        x = self.pool(x)
        return x
