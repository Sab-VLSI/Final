"""
unrolled_k3_film_hidden.py (E.D.I.T.H.-P2)
==========================================
Hidden-state variant of UnrolledK3FiLM. ONE structural change, nothing else.

WHY THIS EXISTS
---------------
Measured on the trained checkpoints, the original design is capacity-saturated:

  residual correlation between models (1.0 = identical function)
    W32_50ep vs W48_50ep    0.9849
    W32_50ep vs W16_stageA  0.9846      (11,296 vs 39,712 params)
    W32_50ep vs W32_augret  0.9858      (different data, ~1.8x steps)
    mean pairwise           0.9837
  ensembling all four        +0.0226 dB over the best single model

Models spanning an 8x parameter range converge to the same function, so extra
width, extra data, and ensembling all return ~nothing.

The bottleneck is the inter-iteration interface. In the original prior:

    feat = stem(x)                  #      1 channel -> C channels
    ... residual blocks + FiLM ...
    out  = x + tail(feat)           #      C channels -> 1 channel
    return sigmoid(out)

every iteration builds a C-channel representation and then destroys it. All
information passed to the next iteration is squeezed through a single-channel
image, so widening C widens something that is immediately collapsed. The
per-iteration profile shows the same story:

    x0 (bicubic init)   20.3469 dB
    after iteration 1   22.6088 dB   +2.2619   <- 77% of the work
    after iteration 2   22.7500 dB   +0.1411
    after iteration 3   23.3024 dB   +0.5525

Iterations 2 and 3 cannot build on iteration 1's features because those features
no longer exist.

THE CHANGE
----------
The prior now also consumes and returns a C-channel hidden state, so the feature
representation persists across the K unrolled iterations. Everything else is held
identical to UnrolledK3FiLM on purpose -- same K, same weight sharing, same FiLM
conditioning on z, same data-consistency step, same bicubic init, same sigmoid
tail. This is a one-variable experiment.

Cost at C=32: stem grows from Conv2d(1, C) to Conv2d(1 + C, C), i.e.
39,712 -> 48,928 trainable parameters (+23%). Still under 50k.
"""

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .forward_model import SurrogateForwardModel
from .degradation_estimator import DegradationEstimator
from .unrolled_k3_film import FiLMBlock


class SharedLearnedPriorFiLMHidden(nn.Module):
    """
    FiLM-conditioned residual prior that carries a hidden feature state.

    Same block structure as SharedLearnedPriorFiLM (stem: Conv-BN-LReLU;
    res: Conv-BN-LReLU-Conv-BN; tail: LReLU-Conv) with two differences:

    1. The stem accepts the previous iteration's features alongside the current
       image estimate, and the features are returned for the next iteration.

    2. PER-ITERATION BatchNorm. Convolution weights stay shared across all K
       iterations -- that is the architectural property being preserved -- but
       each iteration owns its own normalization statistics.

       Why (2) is required, learned the expensive way: with a single shared BN,
       iteration 1 sees cat([x, zeros]) while iterations 2..K see
       cat([x, real_features]). Those distributions are nothing alike, one set of
       running statistics fits neither, and the model trains fine while evaluating
       terribly. Measured at epoch 16 of the first attempt:

           BatchNorm eval mode  (running stats): 14.5823 dB
           BatchNorm train mode (batch stats)  : 20.6358 dB
           gap                                   +6.0534 dB

       Val PSNR sat at ~15 dB through epoch 15 while train loss fell normally --
       entirely an artifact of this mismatch, not a property of the hidden state.
       The original prior never hits this because its input is always a 1-channel
       image with consistent statistics across iterations.

       Cost at C=32, K=3: 5 BN layers x 32 channels x 2 params x 3 iterations
       = 960, against 320 when shared. +640 parameters.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 16, z_dim: int = 4, K: int = 3):
        super().__init__()
        b = base_channels
        self.base_channels = b
        self.K = K

        # Shared convolutions (weight-tied across iterations, as in the original).
        self.stem_conv = nn.Conv2d(in_channels + b, b, kernel_size=3, padding=1, bias=False)
        self.res1_conv1 = nn.Conv2d(b, b, kernel_size=3, padding=1, bias=False)
        self.res1_conv2 = nn.Conv2d(b, b, kernel_size=3, padding=1, bias=False)
        self.res2_conv1 = nn.Conv2d(b, b, kernel_size=3, padding=1, bias=False)
        self.res2_conv2 = nn.Conv2d(b, b, kernel_size=3, padding=1, bias=False)
        self.tail_conv = nn.Conv2d(b, in_channels, kernel_size=3, padding=1)

        # Per-iteration normalization: NOT shared.
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                "stem": nn.BatchNorm2d(b),
                "r1a": nn.BatchNorm2d(b),
                "r1b": nn.BatchNorm2d(b),
                "r2a": nn.BatchNorm2d(b),
                "r2b": nn.BatchNorm2d(b),
            })
            for _ in range(K)
        ])

        self.film1 = FiLMBlock(b, z_dim)
        self.film2 = FiLMBlock(b, z_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, z: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, 1, H, W] current image estimate
        h: [B, C, H, W] hidden state from the previous iteration (zeros on the first)
        z: [B, z_dim]   degradation embedding
        k: int          iteration index, selects the normalization set
        Returns (x_next, h_next).
        """
        bn = self.norms[k]

        feat = self.act(bn["stem"](self.stem_conv(torch.cat([x, h], dim=1))))

        r = self.act(bn["r1a"](self.res1_conv1(feat)))
        r = bn["r1b"](self.res1_conv2(r))
        feat = feat + self.film1(r, z)

        r = self.act(bn["r2a"](self.res2_conv1(feat)))
        r = bn["r2b"](self.res2_conv2(r))
        feat = feat + self.film2(r, z)

        out = x + self.tail_conv(self.act(feat))
        return torch.sigmoid(out), feat


class UnrolledK3FiLMHidden(nn.Module):
    """
    E.D.I.T.H.-P2 unrolled network with a persistent feature state across iterations.

    Drop-in replacement for UnrolledK3FiLM: identical constructor arguments and
    identical forward signature (y_norm, y_raw) -> [B, 1, 2H, 2W] in physical [0, 1].
    Normalization stays EXTERNAL; the model owns no normalization constants.
    """

    def __init__(
        self,
        K: int = 3,
        base_channels: int = 16,
        z_dim: int = 4,
        estimator_base_channels: int = 8,
        init_alpha: float = 0.01,
        learned_init: bool = False,
    ):
        super().__init__()
        self.K = K
        self.z_dim = z_dim
        self.base_channels = base_channels
        self.learned_init = learned_init

        self.estimator = DegradationEstimator(
            in_channels=1,
            base_channels=estimator_base_channels,
            z_dim=z_dim,
        )
        self.forward_model = SurrogateForwardModel(kernel_size=5, sigma=1.0, downsample_factor=2)
        self.shared_prior = SharedLearnedPriorFiLMHidden(
            in_channels=1,
            base_channels=base_channels,
            z_dim=z_dim,
            K=K,
        )

        # Learned de-speckling initializer, OFF by default.
        #
        # Bicubic upsampling of a noisy LR image smears speckle into structured
        # mid-frequency blobs the prior then cannot tell from real SEM texture.
        # Measured oracle bound on 120 val images, changing ONLY whether speckle is
        # present in x0 (same operator, same weights):
        #
        #     x0 alone   noisy bicubic 20.3658 dB -> clean bicubic 25.2618 (+4.90)
        #     full model noisy x0      22.2168 dB -> clean x0      24.4905 (+2.27)
        #
        # so a perfect de-speckling init is worth up to +2.27 dB -- the largest
        # headroom measured on this project (loss weights move ~0.2 dB, prior
        # architecture changes ~0).
        #
        # RESIDUAL and zero-initialised: the last conv starts at zero, so at step 0
        # x0 is EXACTLY the old bicubic estimate and this is a strict superset of the
        # previous behaviour rather than a replacement. Replacing bicubic outright was
        # measured to start 10.8 dB lower (8.92 dB, values in [-0.705, 1.204]) and to
        # feed an out-of-domain x0 into the data-consistency step, which compares A x
        # against y in physical intensity.
        #
        # OFF BY DEFAULT so every existing checkpoint (sweep_noise4_best.pt and the
        # whole sweep family, plus the packaged submission) still loads with
        # strict=True. Only a config that opts in pays the +740 parameters.
        if learned_init:
            self.init_net = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1, padding_mode="replicate"),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(16, 4, kernel_size=3, padding=1, padding_mode="replicate"),
            )
            self.pixel_shuffle = nn.PixelShuffle(2)
            nn.init.zeros_(self.init_net[-1].weight)
            nn.init.zeros_(self.init_net[-1].bias)

        init_raw = float(torch.log(torch.exp(torch.tensor(init_alpha)) - 1.0))
        self.raw_alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))
            for _ in range(K)
        ])

    def get_alphas(self) -> List[torch.Tensor]:
        return [F.softplus(raw_a) for raw_a in self.raw_alphas]

    def forward(
        self, y_norm: torch.Tensor, y_raw: torch.Tensor, return_all_stages: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        H, W = y_norm.shape[-2], y_norm.shape[-1]

        z = self.estimator(y_norm)

        x0 = F.interpolate(y_raw, size=(2 * H, 2 * W), mode="bicubic", align_corners=False)
        if self.learned_init:
            # Zero-init residual: identically bicubic at step 0, a learned
            # de-speckling correction thereafter. The unrolled loop below is untouched.
            x0 = x0 + self.pixel_shuffle(self.init_net(y_raw))
        x0 = torch.clamp(x0, 0.0, 1.0)

        stages = [x0]
        alphas = self.get_alphas()
        x_cur = x0
        # Zero state on the first iteration, so iteration 1 sees exactly what the
        # original prior saw. Any improvement therefore comes from iterations 2..K
        # actually being able to build on earlier features.
        h = torch.zeros(
            x0.shape[0], self.base_channels, x0.shape[-2], x0.shape[-1],
            device=x0.device, dtype=x0.dtype,
        )

        for k in range(self.K):
            g_k = self.forward_model.compute_data_fidelity_gradient(x_cur, y_raw)
            x_dc = x_cur - alphas[k] * g_k
            x_next, h = self.shared_prior(x_dc, h, z, k)
            stages.append(x_next)
            x_cur = x_next

        if return_all_stages:
            return stages
        return stages[-1]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from .unrolled_k3_film import UnrolledK3FiLM
    for w in (16, 32, 48):
        a = count_parameters(UnrolledK3FiLM(K=3, base_channels=w))
        b = count_parameters(UnrolledK3FiLMHidden(K=3, base_channels=w))
        print(f"W={w:2d}  original {a:7,}  hidden {b:7,}  (+{b - a:,}, +{100 * (b - a) / a:.0f}%)")

    m = UnrolledK3FiLMHidden(K=3, base_channels=32)
    for hw in (128, 256):
        out = m(torch.randn(2, 1, hw, hw), torch.rand(2, 1, hw, hw))
        print(f"{hw} -> {tuple(out.shape)}")
