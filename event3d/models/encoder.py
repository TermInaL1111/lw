"""3D ResNet-152 Encoder for event-based 3D reconstruction.

Based on the E2V architecture (Chen et al., ICVR 2023):
"All 2D convolutions replaced by 3D convolutions."

Architecture:
  Input: (B, 1, D, H, W) — stacked event frames as 3D volume
  conv1: Conv3d(1→64, k=7, s=2, p=3) → BN → ReLU
  maxpool: MaxPool3d(k=2, s=2, p=0)
  layer1: 3× Bottleneck3D (64→256), stride=1
  layer2: 8× Bottleneck3D (256→512), stride=2
  layer3: 36× Bottleneck3D (512→1024), stride=2
  layer4: 3× Bottleneck3D (1024→2048), stride=2

Each Bottleneck uses the standard 3D variant:
  1×1×1 conv → 3×3×3 conv → 1×1×1 conv + shortcut

Returns multi-scale features for U-Net skip connections.
"""
import math
import torch
import torch.nn as nn


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck3D(nn.Module):
    """3D Bottleneck: 1×1×1 → 3×3×3 → 1×1×1 + shortcut."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        width = planes
        self.conv1 = conv1x1x1(inplanes, width)
        self.bn1 = nn.BatchNorm3d(width)
        self.conv2 = conv3x3x3(width, width, stride)
        self.bn2 = nn.BatchNorm3d(width)
        self.conv3 = conv1x1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck3D_ECA(nn.Module):
    """3D Bottleneck with Efficient Channel Attention (Xu2025 improvement)."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        width = planes
        self.conv1 = conv1x1x1(inplanes, width)
        self.bn1 = nn.BatchNorm3d(width)
        self.conv2 = conv3x3x3(width, width, stride)
        self.bn2 = nn.BatchNorm3d(width)
        self.conv3 = conv1x1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        # ECA: adaptive 1D conv over channels
        channels = planes * self.expansion
        t = int(abs((math.log2(channels) + 1) / 2))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.eca_sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        # ECA channel attention
        b, c, d, h, w = out.shape
        y = self.avg_pool(out)                    # (B, C, 1, 1, 1)
        y = y.view(b, 1, c)                        # (B, 1, C) for Conv1d
        y = self.eca_conv(y)                       # (B, 1, C)
        y = self.eca_sigmoid(y)                    # (B, 1, C)
        y = y.view(b, c, 1, 1, 1)                 # (B, C, 1, 1, 1)
        out = out * y
        return out


class ResNet152_3D_Encoder(nn.Module):
    """3D ResNet-152 encoder for event frame volumes.

    Args:
        in_channels: input channels (1 for Event Frame)
        use_eca: use ECA attention (Xu2025 improvement)

    Outputs:
        Dictionary with encoder features at multiple scales for U-Net decoder.
    """
    def __init__(self, in_channels=1, use_eca=False):
        super().__init__()
        block = Bottleneck3D_ECA if use_eca else Bottleneck3D
        layers = [3, 8, 36, 3]
        self.inplanes = 64
        self.use_eca = use_eca

        # Initial: Conv3D(k=7,s=2,p=3)
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        # MaxPool3D(k=2,s=2,p=0) — as per the architecture diagram
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # ResNet stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, C, D, H, W) — stacked event frames as 3D volume

        Returns:
            dict with:
              'enc0': early feature (after conv1)
              'enc1': layer1 output
              'enc2': layer2 output
              'enc3': layer3 output
              'enc4': layer4 output (bottleneck)
        """
        # Initial conv
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)     # (B, 64, D/2, H/2, W/2)

        x = self.maxpool(x0)   # (B, 64, D/4, H/4, W/4)

        # ResNet stages
        x1 = self.layer1(x)    # (B, 256, D/4, H/4, W/4)
        x2 = self.layer2(x1)   # (B, 512, D/8, H/8, W/8)
        x3 = self.layer3(x2)   # (B, 1024, D/16, H/16, W/16)
        x4 = self.layer4(x3)   # (B, 2048, D/32, H/32, W/32)

        return {
            'enc0': x0,   # 64 channels
            'enc1': x1,   # 256 channels
            'enc2': x2,   # 512 channels
            'enc3': x3,   # 1024 channels
            'enc4': x4,   # 2048 channels (bottleneck)
        }
