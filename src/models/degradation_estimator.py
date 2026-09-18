"""
degradation_estimator.py
========================
Lightweight single-shot degradation parameter estimator.

Input:  NoisyLR [B, 1, H, W]  (normalized, H=W=128 for training but flexible)
Output: z [B, z_dim]           compact degradation embedding

Architecture:
  Conv(1->C) -> BN -> LReLU
  Conv(C->C, stride=2) -> BN -> LReLU    # H/2 x W/2
  Conv(C->C, stride=2) -> BN -> LReLU    # H/4 x W/4
  AdaptiveAvgPool(1x1)                    # [B, C, 1, 1]
  Flatten                                 # [B, C]
  Linear(C -> z_dim)                      # [B, z_dim]

Constraints (enforced by design):
  - Fully convolutional up to the pool: spatially resolution-flexible
  - No hard-coded H/W inside model logic
  - Differentiable and deterministic at inference
  - Does NOT see GT at any point
"""

import torch
import torch.nn as nn


class DegradationEstimator(nn.Module):
    """
    Lightweight Degradation Estimator.

    Args:
        in_channels (int): Input channels (1 for grayscale NoisyLR).
        base_channels (int): Internal channel width. Default 8.
        z_dim (int): Output embedding dimension. Default 4.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 8, z_dim: int = 4):
        super().__init__()
        C = base_channels
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            # Layer 1: full resolution feature extraction
            nn.Conv2d(in_channels, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: stride-2 downsampling
            nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: stride-2 downsampling again
            nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Global spatial pooling: removes spatial dependency
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Map to degradation embedding
        self.head = nn.Linear(C, z_dim)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: [B, 1, H, W]  (normalized NoisyLR — only sees degraded input, never GT)
        Returns z: [B, z_dim]
        """
        feat = self.encoder(y)        # [B, C, H/4, W/4]
        feat = self.pool(feat)        # [B, C, 1, 1]
        feat = feat.flatten(start_dim=1)  # [B, C]
        z = self.head(feat)           # [B, z_dim]
        return z


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    est = DegradationEstimator(in_channels=1, base_channels=8, z_dim=4)
    x = torch.randn(2, 1, 128, 128)
    z = est(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(z.shape)}")
    print(f"Params: {count_parameters(est):,}")
