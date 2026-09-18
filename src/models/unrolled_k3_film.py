"""
unrolled_k3_film.py
===================
FiLM-conditioned Deep Unrolled K=3 Iterative Restoration Network.

Architecture:
  y (NoisyLR, normalized) -> DegradationEstimator -> z [B, z_dim]
  x0 = bicubic_upsample(unnorm(y))

  For k = 1..K:
    x_dc = x - alpha_k * grad(fidelity(x, y))
    x_next = SharedLearnedPriorFiLM(x_dc, z)

  Output: x_K

Key design constraints:
  - Does NOT modify unrolled_k3.py (frozen checkpoint untouched)
  - Weight-shares the prior across all K iterations (same as frozen model)
  - FiLM modulation applied at 2 locations within the prior's feature branch
  - FiLM generators initialized to identity (gamma=1, beta=0)
  - Estimator initialized randomly, trained jointly via restoration objective
  - Fully convolutional, no hard-coded spatial dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .forward_model import SurrogateForwardModel
from .degradation_estimator import DegradationEstimator


# ── FiLM Block ────────────────────────────────────────────────────────────────

class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation block.
    Applies an affine transform to feature maps conditioned on z.

    y_modulated = gamma(z) * y + beta(z)

    Args:
        in_channels (int): Number of feature channels to modulate.
        z_dim (int): Dimensionality of conditioning vector z.
    """

    def __init__(self, in_channels: int, z_dim: int):
        super().__init__()
        self.gamma_gen = nn.Linear(z_dim, in_channels)
        self.beta_gen  = nn.Linear(z_dim, in_channels)

        # Near-identity initialization:
        #   bias  -> gamma=1, beta=0   (starts as no-op)
        #   weight -> small non-zero   (ensures d(output)/dz != 0 so gradients flow to estimator)
        # Using zeros_ for weight would zero out the gradient path from loss to estimator.
        nn.init.normal_(self.gamma_gen.weight, std=0.01)
        nn.init.ones_(self.gamma_gen.bias)
        nn.init.normal_(self.beta_gen.weight, std=0.01)
        nn.init.zeros_(self.beta_gen.bias)

    def forward(self, feat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        feat: [B, C, H, W]
        z:    [B, z_dim]
        Returns modulated features: [B, C, H, W]
        """
        gamma = self.gamma_gen(z).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta  = self.beta_gen(z).unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1]
        return gamma * feat + beta


# ── FiLM-Conditioned Prior ────────────────────────────────────────────────────

class SharedLearnedPriorFiLM(nn.Module):
    """
    FiLM-Conditioned Residual Convolutional Prior.
    Structurally identical to SharedLearnedPrior in unrolled_k3.py,
    with FiLM modulation inserted after each residual block.

    Input:  x [B, 1, 2H, 2W], z [B, z_dim]
    Output: x_updated [B, 1, 2H, 2W]

    Prior blocks (parallel to frozen model, enables weight transfer):
      stem: Conv -> BN -> LReLU
      res1: Conv -> BN -> LReLU -> Conv -> BN  (+ skip)
      film1: FiLMBlock applied to res1 output
      res2: Conv -> BN -> LReLU -> Conv -> BN  (+ skip)
      film2: FiLMBlock applied to res2 output
      tail: LReLU -> Conv
      output: sigmoid(x + tail(feat))
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 16, z_dim: int = 4):
        super().__init__()
        b = base_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(b, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(b, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
        )
        self.film1 = FiLMBlock(b, z_dim)

        self.res2 = nn.Sequential(
            nn.Conv2d(b, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(b, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
        )
        self.film2 = FiLMBlock(b, z_dim)

        self.tail = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(b, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = feat + self.film1(self.res1(feat), z)   # FiLM after res1
        feat = feat + self.film2(self.res2(feat), z)   # FiLM after res2
        out  = x + self.tail(feat)                     # residual addition
        return torch.sigmoid(out)


# ── Full Estimator + FiLM Unrolled Model ────────────────────────────────────

class UnrolledK3FiLM(nn.Module):
    """
    Deep Unrolled K=3 Restoration Network WITH Degradation Estimator + FiLM Conditioning.

    Architecture:
      y -> Estimator -> z
      x0 = bicubic(unnorm(y))
      For k=0..K-1:
        x_dc = x - alpha_k * grad(||A(x) - y_raw||^2)
        x_{k+1} = SharedPriorFiLM(x_dc, z)
      Output: x_K

    The estimator is trained jointly via the restoration objective.
    No GT information is provided to the estimator.

    Args:
        K (int): Number of unrolling iterations. Default 3.
        base_channels (int): Prior feature channels. Default 16.
        z_dim (int): Degradation embedding size. Default 4.
        estimator_base_channels (int): Estimator internal width. Default 8.
        init_alpha (float): Initial step-size value. Default 0.01.
        norm_mean (float): Dataset normalization mean.
        norm_std (float): Dataset normalization std.
    """

    def __init__(
        self,
        K: int = 3,
        base_channels: int = 16,
        z_dim: int = 4,
        estimator_base_channels: int = 8,
        init_alpha: float = 0.01,
        # Phase-2 train-split constants, matching config/normalization.json and the
        # packaged capacity_K3_W32_50ep checkpoint. run.py passes these explicitly;
        # the defaults exist only so a caller that forgets cannot silently fall back
        # to the Phase-1 values (0.432523 / 0.285066) and emit plausible-looking but
        # wrong output. This model un-normalizes internally, so a wrong constant here
        # corrupts every pixel without raising.
        norm_mean: float = 0.44862022165035664,
        norm_std: float = 0.23189431650723427,
    ):
        super().__init__()
        self.K        = K
        self.z_dim    = z_dim
        self.norm_mean = norm_mean
        self.norm_std  = norm_std

        # Degradation estimator (sees only NoisyLR)
        self.estimator = DegradationEstimator(
            in_channels=1,
            base_channels=estimator_base_channels,
            z_dim=z_dim,
        )

        # Surrogate forward operator (no trainable parameters)
        self.forward_model = SurrogateForwardModel(kernel_size=5, sigma=1.0, downsample_factor=2)

        # Shared FiLM-conditioned prior (weight-shared across K iterations)
        self.shared_prior = SharedLearnedPriorFiLM(
            in_channels=1,
            base_channels=base_channels,
            z_dim=z_dim,
        )

        # Learnable step sizes (one per iteration, positive via softplus)
        init_raw = float(torch.log(torch.exp(torch.tensor(init_alpha)) - 1.0))
        self.raw_alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))
            for _ in range(K)
        ])

    def get_alphas(self):
        """Returns positive alpha step sizes."""
        return [F.softplus(raw_a) for raw_a in self.raw_alphas]

    def forward(self, y: torch.Tensor, return_all_stages: bool = False):
        """
        y: [B, 1, H, W]  normalized NoisyLR
        Returns x_K [B, 1, 2H, 2W], or [x0, x1, ..., xK] if return_all_stages=True
        """
        H, W = y.shape[-2], y.shape[-1]

        # Step 1: Estimate degradation embedding from NoisyLR only
        z = self.estimator(y)   # [B, z_dim]

        # Step 2: Unnormalize and bicubic upsample to physical space
        y_raw = y * self.norm_std + self.norm_mean
        x0 = F.interpolate(y_raw, size=(2 * H, 2 * W), mode="bicubic", align_corners=False)
        x0 = torch.clamp(x0, 0.0, 1.0)

        stages  = [x0]
        alphas  = self.get_alphas()
        x_cur   = x0

        for k in range(self.K):
            # Data-consistency gradient step
            g_k  = self.forward_model.compute_data_fidelity_gradient(x_cur, y_raw)
            x_dc = x_cur - alphas[k] * g_k

            # FiLM-conditioned prior update
            x_next = self.shared_prior(x_dc, z)
            stages.append(x_next)
            x_cur = x_next

        if return_all_stages:
            return stages   # [x0, x1, x2, x3]
        return stages[-1]   # x3


# ── Weight Transfer Utility ──────────────────────────────────────────────────

def transfer_prior_weights(
    film_model: UnrolledK3FiLM,
    frozen_ckpt_path: str,
    verbose: bool = True,
) -> dict:
    """
    Transfer SharedLearnedPrior weights from frozen DeepUnrolledK3 checkpoint
    into the SharedLearnedPriorFiLM of film_model.

    Transferable layers (identical tensor shapes):
        shared_prior.stem.*
        shared_prior.res1.*
        shared_prior.res2.*
        shared_prior.tail.*

    NOT transferable (new in FiLM model):
        shared_prior.film1.*   (new FiLM generators)
        shared_prior.film2.*   (new FiLM generators)
        estimator.*            (new estimator)

    Returns: dict with transfer summary.
    """
    ckpt = torch.load(frozen_ckpt_path, map_location="cpu", weights_only=False)
    frozen_sd = ckpt["model_state_dict"]

    film_sd = film_model.state_dict()
    transferred = []
    skipped_new = []
    skipped_shape = []

    for name, param in frozen_sd.items():
        if name.startswith("shared_prior."):
            suffix = name[len("shared_prior."):]
            if suffix.startswith("film1.") or suffix.startswith("film2."):
                # These don't exist in the frozen model — skip
                continue
        
        # Check if the key exists in the new model
        if name in film_sd:
            if film_sd[name].shape == param.shape:
                film_sd[name] = param.clone()
                transferred.append(name)
            else:
                skipped_shape.append((name, film_sd[name].shape, param.shape))
        else:
            skipped_new.append(name)

    film_model.load_state_dict(film_sd)

    if verbose:
        print(f"  Transferred : {len(transferred)} tensors from frozen checkpoint")
        print(f"  Skipped (new in FiLM model, not in frozen): {len(skipped_new)}")
        print(f"  Skipped (shape mismatch): {len(skipped_shape)}")
        if skipped_shape:
            for n, s1, s2 in skipped_shape:
                print(f"    {n}: film={s1} vs frozen={s2}")

    return {
        "transferred": transferred,
        "skipped_new_in_frozen": skipped_new,
        "skipped_shape_mismatch": [(n, str(s1), str(s2)) for n, s1, s2 in skipped_shape],
    }


def count_parameters(model: nn.Module) -> dict:
    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    estimator = sum(p.numel() for p in model.estimator.parameters())
    prior     = sum(p.numel() for p in model.shared_prior.parameters())
    film      = sum(p.numel() for p in model.shared_prior.film1.parameters()) + \
                sum(p.numel() for p in model.shared_prior.film2.parameters())
    alphas    = sum(p.numel() for p in model.raw_alphas)
    return {
        "total": total,
        "trainable": trainable,
        "estimator": estimator,
        "prior_with_film": prior,
        "film_blocks_only": film,
        "alphas": alphas,
    }


if __name__ == "__main__":
    model = UnrolledK3FiLM(K=3, base_channels=16, z_dim=4, estimator_base_channels=8)
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    stages = model(x, return_all_stages=True)
    pc = count_parameters(model)
    print(f"Input:        {tuple(x.shape)}")
    print(f"Output:       {tuple(out.shape)}")
    print(f"Stages:       {[tuple(s.shape) for s in stages]}")
    print(f"Params total: {pc['total']:,}")
    print(f"  Estimator : {pc['estimator']:,}")
    print(f"  Prior+FiLM: {pc['prior_with_film']:,}")
    print(f"    FiLM only: {pc['film_blocks_only']:,}")
    print(f"  Alphas     : {pc['alphas']:,}")
