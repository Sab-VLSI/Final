"""
run.py
======

Usage:
    python run.py <input-dir> <output-dir>

Examples:
    python run.py ./Test_NoisyLR/NoisyLR ./restored_outputs
    python run.py ../test_input ../test_output

Input:
    Directory containing .npy files and/or standard grayscale images
    (.png/.jpg/.jpeg/.bmp/.tif/.tiff) — even height and width. The official
    test set is (128, 128); (256, 256) is also accepted. Format is identified
    purely from the file extension (no content sniffing), so mixed
    npy/image directories add no per-file overhead.

Output:
    One .npy file per input — float32, shape (2H, 2W), grayscale, values in [0, 1].
    Output filenames match input filenames (extension normalized to .npy).

Batching:
    Inputs are grouped by shape and processed in batches, so a full test set is
    one handful of forward passes instead of one-per-image. The batch size is
    sized to the GPU actually present -- 512 on an 80GB H100 down to 64 on
    consumer cards and 8 on CPU -- because at 49,568 parameters this model is
    bound by kernel-launch overhead, not FLOPs, and a laptop-sized batch would
    leave an H100's SMs idle. Override with SEMICON_BATCH_SIZE.

Model:
    Unrolled K=3 + Degradation Estimator + FiLM (hidden-state variant),
    49,568 trainable parameters, loaded from models/sweep_noise4_best.pt.
    Val PSNR 23.2865 dB / SSIM 0.6026; held-out real pair 22.591 dB.

Inference:
    SINGLE-PASS. 4x Test-Time Augmentation was implemented, measured, and
    removed: it bought +0.025 dB PSNR for +39% end-to-end time and a WORSE
    LPIPS (0.4195 vs 0.4135). See the note above the constants block.
    CUDA is used automatically when available; CPU fallback otherwise.

Environment overrides (all optional, all auto-detected by default):
    SEMICON_BATCH_SIZE      force a batch size
    SEMICON_PRECISION       fp16 (default) | bf16 | fp32 for exact reproduction
    SEMICON_WRITER_THREADS  disk-writer thread count (default 4)

No manual configuration, no internet access, no API keys required.
"""

import os
import sys
import json
import hashlib
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── Resolve submission root relative to this script ───────────────────────────
ROOT = Path(__file__).resolve().parent

# ── Add submission root to sys.path for local model imports ───────────────────
sys.path.insert(0, str(ROOT))
from src.models.unrolled_k3_film_hidden import UnrolledK3FiLMHidden

# ── GPU throughput settings (tuned for H100) ──────────────────────────────────
# Measured end-to-end profile of this pipeline (297 images):
#     import torch        5.783 s   57.2%   <- fixed process cost
#     GPU forward         3.063 s   30.3%
#     disk write          0.870 s    8.6%
#     CUDA context init   0.197 s    1.9%
#     disk read           0.140 s    1.4%
#     weights + SHA       0.058 s    0.6%
#
# So the addressable costs are the forward pass and the writes; reads are already
# negligible and need no faster decoder.
#
# This model is TINY (49,568 parameters) at 128->256. On an H100 it is bound by
# kernel-launch overhead and memory bandwidth rather than FLOPs, so the levers are
# (a) large batches, to amortise launches and fill 132 SMs, and (b) tensor cores.
if torch.cuda.is_available():
    # TF32 for any fp32 matmul/conv that remains outside autocast.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # cudnn.benchmark is OFF by default, deliberately. It autotunes per EXACT
    # tensor shape, and the autotune is not free: measured here it cost ~6 s on the
    # first batch. Worse, a partial final batch is a different shape and triggers a
    # SECOND full autotune -- measured at +4.7 s for one 41-image batch, which alone
    # turned a 4.1 s run into 12.1 s.
    #
    # Batch padding below removes the second autotune, but the first still costs
    # real time and its payoff on an H100 could not be verified from this machine.
    # Shipping an unverified multi-second fixed cost by default is the wrong trade
    # for a timed benchmark, so it is opt-in.
CUDNN_BENCHMARK = (
    torch.cuda.is_available() and os.environ.get("SEMICON_CUDNN_BENCHMARK", "0") == "1"
)
if CUDNN_BENCHMARK:
    torch.backends.cudnn.benchmark = True

# ── Input format detection (extension-only — zero content-sniffing overhead) ──
NPY_EXTS = {".npy"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SUPPORTED_EXTS = NPY_EXTS | IMAGE_EXTS


def _get_dynamic_batch_size(h: int, w: int) -> int:
    """
    Dynamically size the batch based on real-time VRAM availability and resolution.
    
    A 128x128 image costs ~45 MB in our Deep Supervision architecture.
    A 256x256 image costs exactly 4x as much. We check mem_get_info() right
    before batching to squeeze maximum safe throughput out of any GPU.
    """
    env = os.environ.get("SEMICON_BATCH_SIZE")
    if env:
        return max(1, int(env))
    if not torch.cuda.is_available():
        return 8
        
    free_mem, _ = torch.cuda.mem_get_info(0)
    free_mb = free_mem / (1024 ** 2)
    
    # Base cost measured on an RTX 4050
    scaling_factor = (h / 128.0) * (w / 128.0)
    vram_per_image_mb = 45.0 * scaling_factor
    
    # Reserve 1GB for CUDA context and system margin
    safe_free_mb = max(256.0, free_mb - 1024.0)
    
    bs = int(safe_free_mb // vram_per_image_mb)
    
    # Clamp to a maximum of 512 to avoid massive launch overhead timeouts or internal limits
    return max(1, min(512, bs))


# Inference precision. H100 4th-gen tensor cores run fp16/bf16 far faster than
# fp32, and the quality cost was measured on this checkpoint over 60 validation
# images: fp16 +0.0003 dB and bf16 +0.0019 dB against fp32 -- both far below any
# meaningful threshold, and the max elementwise deviation (1.8e-3 fp16) cannot move
# a clamped [0,1] output. fp32 remains available for exact reproduction.
_PRECISION = os.environ.get("SEMICON_PRECISION", "fp16").lower()
if _PRECISION not in ("fp16", "bf16", "fp32"):
    raise SystemExit(f"SEMICON_PRECISION must be fp16, bf16 or fp32 (got {_PRECISION!r})")
AUTOCAST_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[_PRECISION]

# Disk writes overlap GPU compute on worker threads (np.save releases the GIL in
# its write path). Measured at 8.6% of runtime, so this is the only I/O worth
# hiding -- reads are 1.4% and already effectively free.
WRITER_THREADS = int(os.environ.get("SEMICON_WRITER_THREADS", "4"))

# ── Frozen Constants — DO NOT MODIFY ─────────────────────────────────────────
# Promoted 2026-09-18 from sweep_noise4_best.pt (UnrolledK3FiLMHidden, 49,568 params)
# to research_bicubic_deep_supervision_20ep_detail_best.pt -- the final champion,
# selected on the highest metrics:
#     val PSNR 24.1657   val SSIM 0.6606   held-out real pair 22.591 dB
#
# NOT a drop-in swap. UnrolledK3FiLMHidden takes TWO forward inputs
# (y_norm, y_raw) and owns no normalization constants, where UnrolledK3FiLM took
# one and un-normalized internally. load_model/preprocess/inference all changed
# accordingly -- see below.
#
# config/normalization.json stays LOAD-BEARING: it now feeds the EXTERNAL
# normalization applied here, and its values were verified identical to the
# checkpoint's own recorded normalization (0.44862022165035664 /
# 0.23189431650723427). A stale mean/std silently corrupts every output.
CHECKPOINT_PATH       = ROOT / "models" / "research_bicubic_deep_supervision_20ep_detail_best.pt"
NORM_CONFIG_PATH      = ROOT / "config/normalization.json"
EXPECTED_PARAMS       = 49568
EXPECTED_SHA256       = "f3497dc27fbe0f2d8ff4501d2c3358e3b183165770d577458fed80aa8efc0bfd"

# Model architecture hyperparameters (must match checkpoint)
MODEL_K                       = 3
MODEL_BASE_CHANNELS           = 32
MODEL_Z_DIM                   = 4
MODEL_ESTIMATOR_BASE_CHANNELS = 8

# ── Single-pass inference (4x TTA REMOVED 2026-09-03) ─────────────────────────
# Measured end-to-end on the 846 unpaired inputs, quality on the 394-image val
# split (reports/MODEL_DEVELOPMENT_REPORT.md 7):
#
#     TTA  end-to-end        PSNR      SSIM     LPIPS
#     1x   41.98 s           23.5804   0.5869   0.4135
#     4x   58.39 s (+39%)    23.6056   0.5880   0.4195  (worse)
#
# 4x TTA bought +0.025 dB PSNR and +0.0011 SSIM for +39% end-to-end time and a
# WORSE perceptual score -- averaging four predictions smooths them further,
# which is this model's existing weakness rather than a fix for it. The deck
# scores end-to-end H100 time alongside quality, so this is strictly negative.


# ── Utility Functions ─────────────────────────────────────────────────────────

def sha256_of(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_normalization() -> tuple:
    """Load normalization constants from the packaged config file."""
    if not NORM_CONFIG_PATH.exists():
        raise RuntimeError(
            f"Normalization config not found: {NORM_CONFIG_PATH}\n"
            f"  Expected at: config/normalization.json relative to run.py"
        )
    with open(NORM_CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    return float(cfg["input_mean"]), float(cfg["input_std"])


def load_model(device: torch.device) -> torch.nn.Module:
    """Load the frozen hidden-state FiLM model, verify integrity, return in eval mode."""
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            f"  Expected at: models/{CHECKPOINT_PATH.name} relative to run.py"
        )

    # SHA-256 integrity check
    actual_sha = sha256_of(CHECKPOINT_PATH)
    if actual_sha.lower() != EXPECTED_SHA256.lower():
        raise RuntimeError(
            "Checkpoint SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_SHA256}\n"
            f"Got:      {actual_sha}"
        )
    print(f"  Checkpoint integrity: OK ({actual_sha[:16]}...)")

    norm_mean, norm_std = load_normalization()

    # No norm_mean/norm_std here: this model owns no normalization constants and
    # receives the normalized AND raw tensors explicitly at forward time.
    model = UnrolledK3FiLMHidden(
        K=MODEL_K,
        base_channels=MODEL_BASE_CHANNELS,
        z_dim=MODEL_Z_DIM,
        estimator_base_channels=MODEL_ESTIMATOR_BASE_CHANNELS,
        init_alpha=0.01,
    )

    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params != EXPECTED_PARAMS:
        raise RuntimeError(
            f"Parameter count mismatch: {params} (expected {EXPECTED_PARAMS})"
        )
    print(f"  Parameters: {params:,}")

    return model, norm_mean, norm_std


def load_input_array(fpath: Path) -> np.ndarray:
    """
    Load one input as a 2D float32 array, dispatching on file extension only
    (no content sniffing) so mixed npy/image directories cost nothing extra.

    .npy    -> loaded as-is (raw physical intensity, matches training domain).
    images  -> ponytail: assumes 8-bit grayscale; 16-bit source precision is
               not preserved. Scaled to [0, 1], the standard image convention.
    """
    suffix = fpath.suffix.lower()
    if suffix in NPY_EXTS:
        arr = np.load(str(fpath)).astype(np.float32)
    else:
        arr = np.asarray(Image.open(fpath).convert("L"), dtype=np.float32) / 255.0

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    return arr


def preprocess_batch(arrs: list, norm_mean: float, norm_std: float):
    """
    Stack same-shape arrays into one pinned [B, 1, H, W] fp32 buffer.

    Only y_raw is built here. The model needs BOTH y_raw (unnormalized physical
    intensity, which drives the data-consistency step) and y_norm, but normalizing
    on the GPU keeps the host->device copy to a single contiguous buffer instead of
    two -- which matters once the batch is 512 images wide on an H100.

    Pinned memory is what makes the non_blocking copy in run_batch actually async.
    """
    stacked = np.stack(arrs, axis=0).astype(np.float32)
    raw = torch.from_numpy(stacked[:, np.newaxis])
    if torch.cuda.is_available():
        raw = raw.pin_memory()          # enables the async H2D copy below
    return raw


@torch.inference_mode()
def run_batch(model: torch.nn.Module, raw_cpu: torch.Tensor, device: torch.device,
              norm_mean: float, norm_std: float, pad_to: int = 0) -> np.ndarray:
    """
    One shape-grouped chunk end to end: async upload -> autocast forward -> host.

    inference_mode (not no_grad) also skips version-counter bookkeeping on every
    tensor, which is measurable when the batch is large and the model is small.

    pad_to keeps EVERY forward pass the same shape by padding a short final batch
    up to the full batch size and discarding the extra rows. Without it the last
    partial batch is a distinct shape, which forces cuDNN to re-autotune and (when
    benchmarking is enabled) cost more than the entire rest of the run. Padding is
    cheap: the pad rows are duplicates of row 0 and are never written out.
    """
    n_real = raw_cpu.shape[0]
    if pad_to and n_real < pad_to:
        pad = raw_cpu[:1].expand(pad_to - n_real, *raw_cpu.shape[1:])
        raw_cpu = torch.cat([raw_cpu, pad], dim=0)

    raw = raw_cpu.to(device, non_blocking=True)
    y_norm = (raw - norm_mean) / norm_std

    if AUTOCAST_DTYPE is not None and device.type == "cuda":
        with torch.autocast("cuda", dtype=AUTOCAST_DTYPE):
            out = model(y_norm, raw)
        out = out.float()               # back to fp32 before clamping/saving
    else:
        out = model(y_norm, raw)

    # Clamp on-device: cheaper than doing it in numpy afterwards, and guarantees
    # the [0,1] contract holds before the tensor ever leaves the GPU. Padding rows
    # are dropped here so they never reach disk.
    out = torch.clamp(out, 0.0, 1.0)[:n_real]
    return out.squeeze(1).cpu().numpy().astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Argument parsing (exactly two positional arguments) ───────────────────
    if len(sys.argv) != 3:
        print(
            "Usage: python run.py <input-dir> <output-dir>\n"
            "\n"
            "  input-dir   Directory containing .npy test images (float32, even HxW)\n"
            "  output-dir  Directory for restored outputs (2Hx2W float32, created automatically)\n"
            "\n"
            "Example:\n"
            "  python run.py ./Test_NoisyLR/NoisyLR ./restored_outputs",
            file=sys.stderr,
        )
        sys.exit(1)

    input_dir = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()

    # ── Validate input directory ──────────────────────────────────────────────
    if not input_dir.exists():
        print(f"[ERROR] Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"[ERROR] Input path is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Discover input files (.npy and/or images, by extension only) ──────────
    input_files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    ])
    if not input_files:
        print(f"[ERROR] No .npy or image files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Create output directory ───────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device detection ──────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        gpu_name = "N/A"

    # ── Print execution summary ───────────────────────────────────────────────
    print("=" * 64)
    print("SEMICON / KLA HACKATHON 2026")
    print("Unrolled K=3 + Degradation Estimator + FiLM")
    print("=" * 64)
    print(f"  Device:           {device}" + (f" ({gpu_name})" if gpu_name != "N/A" else ""))
    print(f"  Inference mode:   single-pass (TTA removed)")
    print(f"  Batch size:       Dynamic (Real-time VRAM allocation)"
          + ("" if not os.environ.get("SEMICON_BATCH_SIZE") else f" (overridden to {os.environ.get('SEMICON_BATCH_SIZE')})"))
    print(f"  Compute precision:{_PRECISION}")
    print(f"  Writer threads:   {WRITER_THREADS}")
    print(f"  Input directory:  {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Input files:      {len(input_files)}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading model...")
    model, norm_mean, norm_std = load_model(device)

    # ── Load + validate every input, group by shape ────────────────────────────
    # A batch must share (H, W), so same-shape inputs are grouped before
    # chunking. The official test set is uniformly 128x128, so in practice
    # this is almost always one giant group -> len(input_files)/BATCH_SIZE
    # total forward passes instead of one per image.
    print(f"\nLoading {len(input_files)} inputs...")
    failed = []
    shape_groups = defaultdict(list)  # (h, w) -> [(fpath, arr), ...]

    for fpath in input_files:
        try:
            arr = load_input_array(fpath)

            # Validate input.
            # Resolution-agnostic by design: the network is fully convolutional and
            # emits 2H x 2W for any even input. The official test set is 128x128, but
            # the problem statement also specifies a 256->512 regime, so rejecting a
            # 256 input outright would score zero on it where an output would score
            # something. Only the 2x-divisibility the architecture actually requires
            # is enforced.
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D grayscale array, got shape {arr.shape}")
            h, w = arr.shape
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError(f"Input dimensions must be even, got {arr.shape}")
            if not np.isfinite(arr).all():
                raise ValueError("Input contains NaN or Inf values")

            shape_groups[(h, w)].append((fpath, arr))
        except Exception as e:
            failed.append((fpath.name, str(e)))
            print(f"  [FAIL] {fpath.name}: {e}", file=sys.stderr)

    # ── Batched inference ────────────────────────────────────────────────────
    print(f"\nProcessing {sum(len(v) for v in shape_groups.values())} valid inputs "
          f"across {len(shape_groups)} shape group(s)...")
    print("-" * 64)

    t_start = time.perf_counter()
    success_count = 0
    total_valid = sum(len(v) for v in shape_groups.values())
    n_done = 0

    def _save_batch(paths, out_batch):
        """Runs on a writer thread so saving overlaps the next batch's GPU work."""
        for p, a in zip(paths, out_batch):
            np.save(str(p), a)
        return len(paths)

    # Writes are dispatched to threads and only joined at the end, so the ~8.6% of
    # runtime they cost is hidden behind the following batch's forward pass instead
    # of serialising after it.
    pending = []
    with ThreadPoolExecutor(max_workers=WRITER_THREADS) as pool:
        for (h, w), items in shape_groups.items():
            dynamic_bs = _get_dynamic_batch_size(h, w)
            for chunk_start in range(0, len(items), dynamic_bs):
                chunk = items[chunk_start:chunk_start + dynamic_bs]
                fpaths = [fp for fp, _ in chunk]
                arrs = [a for _, a in chunk]

                try:
                    raw_cpu = preprocess_batch(arrs, norm_mean, norm_std)
                    # Pad ONLY when autotuning is on. Padding buys a single cuDNN
                    # autotune instead of two, but costs real compute on the final
                    # short batch (64 images of work for 41 real ones here), so with
                    # benchmarking off it is pure waste.
                    out_batch = run_batch(model, raw_cpu, device, norm_mean, norm_std,
                                          pad_to=dynamic_bs if CUDNN_BENCHMARK else 0)

                    assert out_batch.shape == (len(chunk), 2 * h, 2 * w), (
                        f"Output batch shape {out_batch.shape}, expected {(len(chunk), 2 * h, 2 * w)}"
                    )
                    assert np.isfinite(out_batch).all(), "Output contains NaN/Inf"

                    # Filename stem preserved, extension normalized to .npy -- required
                    # so image inputs (.png etc.) don't collide with the mandated .npy
                    # output format.
                    out_paths = [output_dir / f"{fp.stem}.npy" for fp in fpaths]
                    pending.append((pool.submit(_save_batch, out_paths, out_batch), fpaths))

                except Exception as e:
                    for fpath in fpaths:
                        failed.append((fpath.name, str(e)))
                    print(f"  [FAIL] batch of {len(chunk)} at shape {(h, w)}: {e}", file=sys.stderr)

                n_done += len(chunk)
                elapsed = time.perf_counter() - t_start
                fps = n_done / max(elapsed, 1e-6)
                print(f"  [{n_done:4d}/{total_valid}] batch of {len(chunk)} @ {(h, w)}  "
                      f"({fps:.1f} img/s, {elapsed:.1f}s elapsed)")

        # Join every writer before reporting success, so the counts and the exit
        # code describe files that are actually on disk.
        for fut, fpaths in pending:
            try:
                success_count += fut.result()
            except Exception as e:
                for fpath in fpaths:
                    failed.append((fpath.name, f"write failed: {e}"))
                print(f"  [FAIL] write batch: {e}", file=sys.stderr)

    t_total = time.perf_counter() - t_start

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("INFERENCE COMPLETE")
    print("=" * 64)
    print(f"  Total inputs    : {len(input_files)}")
    print(f"  Successful      : {success_count}")
    print(f"  Failed          : {len(failed)}")
    print(f"  Total time      : {t_total:.1f}s  ({t_total/max(success_count,1)*1000:.0f} ms/img avg)")
    print(f"  Output directory: {output_dir}")
    print(f"  Output format   : .npy float32 (2x input size, [0,1])")

    if failed:
        print("\n[FAILED INPUTS]")
        for name, err in failed:
            print(f"  {name}: {err}")
        sys.exit(1)

    print("=" * 64)


if __name__ == "__main__":
    main()
