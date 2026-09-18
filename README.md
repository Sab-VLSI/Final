<div align="center">

# SEM Image Restoration via Physics-Inspired Deep Unrolling

### Reconstructing 256×256 semiconductor imagery from noisy 128×128 observations
### with **49,568 parameters** — at **8 ms/image**

**SEMICON / KLA Hackathon 2026 — Phase 2 Submission**

`Deep Algorithm Unrolling` · `FiLM Conditioning` · `Learned Degradation Embedding` · `Noise-Aware Loss`

</div>

---

```
        ╔═══════════════════════════════════════════════════════════════╗
        ║   49,568 parameters          24.1657 dB PSNR                  ║
        ║   3 unrolled stages          0.6606 SSIM                      ║
        ║   8 ms / image               22.591 dB on a truly unseen pair ║
        ╚═══════════════════════════════════════════════════════════════╝
```

| | |
|---|---|
| **Entry point** | `python run.py <input-dir> <output-dir>` |
| **Architecture** | Unrolled K=3 + Degradation Estimator + FiLM, with persistent hidden state |
| **Trainable parameters** | **49,568** — three orders of magnitude below transformer baselines |
| **Validation PSNR / SSIM** | **24.1657 dB** / **0.6606** (Phase 2 validation split) |
| **Held-out real pair** | **22.591 dB** — a genuine acquisition never seen in training |
| **Throughput** | 1,197 images in **10.2 s** end-to-end (RTX 4050 Laptop) |
| **Dependencies** | `torch`, `numpy`, `pillow` — nothing else |
| **Offline** | 100% — no internet, API keys, downloads, or user interaction |


---

## 🔬 The Problem

Recover a clean high-resolution image `x` from a degraded low-resolution observation `y`,
formed by compound degradation applied in **arbitrary order** — Gaussian blur, downsampling,
multiplicative speckle, and additive sensor noise:

```
y = D(x) = S( B(x) · n_speckle ) + n_gaussian
```

Two properties of the real data — both **measured, not assumed** — shaped every design choice:

<table>
<tr><th>Finding</th><th>Measurement</th><th>Consequence for the design</th></tr>
<tr>
<td><b>Observations exceed [0,1]</b></td>
<td>Intensities reach <b>1.45</b> via constructive speckle interference</td>
<td>The pipeline <b>never clamps its input</b> — that overshoot is physical signal, and the
data-consistency step needs it. Only the final output is clamped.</td>
</tr>
<tr>
<td><b>Noise is signal-dependent</b></td>
<td><code>Var(n) = σ_a² + L²·σ_s²</code>, σ_a=0.038, σ_s=0.197, <b>R²=0.987</b>.
Speckle is ~15% of variance in dark regions but <b>~75% in bright ones</b></td>
<td>Directly motivated the <b>noise-aware loss</b> (§5) — a uniform pixel loss penalises
smoothing noise and smoothing texture identically, which is wrong in opposite directions.</td>
</tr>
</table>

---

## 🧠 Architecture

```
   y (NoisyLR, 128×128)
       │
       ├─────────────────► Degradation Estimator ──────► z ∈ R⁴
       │                      (1,308 params)         (per-image FiLM conditioning)
       │                                                       │
       ▼                                                       │
   x₀ = bicubic(y)                                             │
       │                                                       │
       ▼                                                       ▼
  ╔════════════════════════════════════════════════════════════════════╗
  ║   UNROLLED STAGE  k = 1 … 3        ── conv weights TIED across k ──║
  ║                                                                    ║
  ║     ┌── DATA CONSISTENCY (analytical, no autograd) ──┐             ║
  ║     │      xₖ  ←  xₖ − αₖ · Aᵀ( A xₖ − y )           │             ║
  ║     └───────────────────────┬──────────────────────── ┘             ║
  ║                             ▼                                      ║
  ║     ┌── LEARNED PRIOR (FiLM-modulated residual CNN) ──┐            ║
  ║     │      xₖ₊₁, hₖ₊₁  ←  Prior( xₖ, hₖ, z, k )       │            ║
  ║     └──────────────────────────────────────────────────┘            ║
  ║              ↳ persistent C-channel hidden state h                 ║
  ╚════════════════════════════════════════════════════════════════════╝
       │
       ▼
   x̂ (restored, 256×256, clamped to [0,1])
```

| Component | Params | Role |
|---|---:|---|
| Degradation Estimator | 1,308 | Compresses `y` into a 4-D embedding `z` describing *how* it was degraded |
| Shared Learned Prior | 48,257 | FiLM-modulated residual CNN — **weight-tied across all 3 stages** |
| Step sizes `α₁…α₃` | 3 | Learned per-stage data-consistency step (softplus-parameterised, always positive) |
| **Total** | **49,568** | |

---

## 💡 What Is Novel Here

### 1. Physics is in the architecture, not just the loss

The operator `A` (Gaussian blur + 2× decimation) and its adjoint `Aᵀ` are implemented
**analytically** — the data-fidelity gradient `Aᵀ(Ax − y)` is computed in closed form with no
autograd graph. Every stage is a genuine optimization step on a real energy functional, so
intermediate states `x₁, x₂, x₃` remain physically meaningful images rather than opaque
activations. This is what keeps 49.5K parameters sufficient.

### 2. Weight tying makes it an *unrolled optimizer*, not a deep stack

All convolutions are shared across the three stages. Only the BatchNorm layers are per-stage
— a necessity discovered empirically, not a design flourish:

> Stage 1 sees a **zero** hidden state; stages 2–3 see **real features**. Those distributions
> are nothing alike. With shared normalization the model trained normally but evaluated
> catastrophically — a **6.05 dB** train/eval gap. Per-stage statistics cost 640 parameters
> and closed it entirely.

### 3. Per-image conditioning through a learned degradation embedding

A lightweight estimator infers a 4-D code `z` from the observation alone (it never sees
ground truth) and modulates the prior's features via FiLM. One compact network therefore
adapts across varying noise and blur regimes instead of committing to an average.

### 4. A loss derived from measured noise physics

The noise-aware term reweights reconstruction error by **local input-noise energy**, so the
model is penalised *less* for smoothing genuine noise and *more* for smoothing genuine
texture. Its constants come from measurement — a global p99 edge scale of 2.2355 and noise
scale of 0.0236 computed over the dataset, not hand-tuned. The weight `λ = 4.0` was selected
by a dedicated sweep, and it measurably reversed the monotonic degradation of the flattest,
hardest samples that every prior sharpening configuration had made worse.

### 5. A persistent hidden state across unrolled iterations

Standard unrolling collapses each stage's representation back to a single-channel image,
destroying it. Here a C-channel state is carried forward, so stages 2–3 build on stage 1's
features. Diagnostics showed the original design spent **77% of its total gain in iteration
1** because later iterations were starting blind.

---

## 🎯 Engineering Discipline: Measured, Not Guessed

Every shipped choice was validated against a held-out split, and several plausible ideas were
**rejected on evidence**. This is the part of the work that does not show up in a single
metric:

| Decision | Evidence | Outcome |
|---|---|---|
| **4× TTA removed** | +0.025 dB PSNR for **+39% time** and a *worse* LPIPS (0.4195 vs 0.4135) | Rejected — strictly negative under a rubric that scores time |
| **Training precision** | Paired controls, only the AMP dtype differing: fp16 **23.2875** vs bf16 **21.6919** | fp16 retained — bf16 cost **1.60 dB** and was slower |
| **Loss weights** | Dedicated sweeps over gradient weight, SSIM weight, texture λ and noise-aware λ | Recipe in §6 selected on measured PSNR **and** SSIM |
| **Synthetic augmentation** | Three calibrated parametric datasets built and evaluated | All rejected — held-out performance fell every time |
| **Inference precision** | fp16 vs fp32 across 60 validation images: **+0.0003 dB** | fp16 default — free speed, no measurable quality cost |

Integrity is enforced in code, not documentation: the entry point verifies the checkpoint's
**SHA-256** and its **exact parameter count** before running, and fails hard on mismatch.

---

## 📊 Results

Measured on the **Phase 2 validation split**, plus a genuine unseen acquisition
pair reserved specifically to test generalization.

| Metric | Value |
|---|---|
| **PSNR** | **24.1657 dB** |
| **SSIM** | **0.6606** |
| **LPIPS** | 0.2872 |
| **Held-out real pair** | **22.591 dB** |

**Against bicubic on five representative samples** — the model wins on **all five**, by
**+1.40 dB** on average and by **+2.84 dB** on the hardest near-blank case:

| Sample | Model PSNR | Bicubic PSNR | Δ |
|---|---|---|---|
| 000845 | 22.13 | 21.29 | **+0.83** |
| 000847 | 20.67 | 19.22 | **+1.46** |
| 000848 *(held-out, near-blank)* | 21.78 | 18.94 | **+2.84** |
| 000849 | 21.97 | 21.05 | **+0.92** |
| 000850 | 20.54 | 19.60 | **+0.93** |

### Training recipe

| Setting | Value |
|---|---|
| Loss | `L1 + 0.4·(1 − SSIM) + 0.2·gradient + 4.0·noise-aware` |
| Optimizer | AdamW — lr 1e-3 → 1e-6 cosine, weight decay 1e-4 |
| Epochs / seed | 30 / 42 |
| Gradient clipping | 1.0 (global norm) |
| Mixed precision | fp16 + GradScaler |

---

## ⚙️ Performance Engineering

The pipeline was **profiled**, then optimized where the profile pointed — not by intuition:

| Stage | Time | Share |
|---|---|---|
| `import torch` | 5.783 s | **57.2%** — fixed process cost |
| GPU forward | 3.063 s | 30.3% |
| Disk write | 0.870 s | 8.6% |
| CUDA context init | 0.197 s | 1.9% |
| Disk read | 0.140 s | 1.4% |
| Weight load + SHA-256 | 0.058 s | 0.6% |

At 49,568 parameters the model is bound by **kernel-launch overhead and memory bandwidth**,
not FLOPs — which dictates the optimizations:

- **GPU-aware batch sizing** — Dynamic real-time VRAM allocation. Checks exactly how much free VRAM is available before every shape group batch to squeeze maximum safe throughput, easily scaling past a 1,800-image batch on an 80 GB H100 while avoiding CUDA out-of-memory errors on 6 GB laptop cards.
- **Shape-grouped batching** — a full test set becomes a handful of forward passes.
- **`cudnn.benchmark`** — constant shape per group, so autotuning is paid once and reused.
- **Half-precision inference** on 4th-gen tensor cores, **TF32** for anything outside autocast.
- **Threaded disk writes** overlapping GPU compute (`np.save` releases the GIL).
- **Extension-based format detection** — no content sniffing, zero per-file overhead.

### Optional overrides — all auto-detected, none required

| Variable | Default | Purpose |
|---|---|---|
| `SEMICON_BATCH_SIZE` | auto by VRAM | Force a batch size |
| `SEMICON_PRECISION` | `fp16` | `fp16` · `bf16` · `fp32` (exact reproduction) |
| `SEMICON_WRITER_THREADS` | `4` | Disk-writer thread count |

---

## ✅ Submission Compliance

| Requirement | Status |
|---|---|
| Reads all `.npy` files from the input directory | ✅ (plus `.png/.jpg/.jpeg/.bmp/.tif/.tiff`) |
| Creates the output directory if absent | ✅ `mkdir(parents=True, exist_ok=True)` |
| One restored `.npy` per input file | ✅ verified 297/297 |
| Output filename matches its input | ✅ same stem, `.npy` extension |
| Grayscale `(H, W)` or `(H, W, 1)` | ✅ `(2H, 2W)` 2-D float32 |
| Values in `[0,1]`, no NaN/Inf | ✅ measured range `[0.002467, 0.997814]`, all finite |
| Correct target resolution | ✅ exactly 2× input, asserted per file |
| Model weights and supporting files included | ✅ `models/`, `config/`, `src/` |
| `requirements.txt` with version details | ✅ fully pinned |
| `README.md` with setup and execution | ✅ this document |
| Runs on H100 with no internet / keys / downloads / interaction | ✅ fully offline |
| Entry script named `run.py` | ✅ |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
python run.py <input-dir> <output-dir>
```

**Example**

```bash
python run.py ./test_input ./restored_outputs
```

No source-code modification, parameter tuning, notebook, or extra CLI argument is required.
CUDA is detected automatically, with a clean CPU fallback.

> **Install note.** The pinned `torch==2.5.1+cu121` is a CUDA build hosted on the PyTorch
> index rather than plain PyPI, hence the `--extra-index-url` flag. For a CPU-only or
> different-CUDA harness, drop the `+cu121` suffix from the torch pin — no code changes needed.

---

## 📥 Input & Output Specification

<table>
<tr><th width="50%">Input</th><th width="50%">Output</th></tr>
<tr valign="top">
<td>

| Property | Value |
|---|---|
| Formats | `.npy` **(primary)** + common images |
| Shape | 2-D `(H, W)`; `(1,H,W)` / `(H,W,1)` squeezed |
| Dtype | float32 |
| Constraint | H, W must be **even** |
| Range | Unconstrained — **>1.0 preserved** |

</td>
<td>

| Property | Value |
|---|---|
| Format | NumPy `.npy` |
| Shape | `(2H, 2W)` grayscale |
| Dtype | float32 |
| Range | `[0.0, 1.0]`, guaranteed finite |
| Name | Same stem as input |

</td>
</tr>
</table>

```
test_input/000042.npy    →   test_output/000042.npy
test_input/sample_a.png  →   test_output/sample_a.npy
```

---

## 📦 Package Contents

```
submission/
├── run.py                  ← official entry point
├── requirements.txt        ← pinned runtime dependencies
├── README.md               ← this document
├── models/
│   └── sweep_noise4_best.pt        49,568 params · SHA-256 verified at load
├── config/
│   └── normalization.json          training mean/std · LOAD-BEARING
└── src/
    └── models/
        ├── unrolled_k3_film_hidden.py   ← shipped architecture
        ├── unrolled_k3_film.py          ← FiLM block + base variant
        ├── forward_model.py             ← analytical operator A, Aᵀ
        └── degradation_estimator.py     ← z-embedding network
```

`config/normalization.json` is **load-bearing, not decorative**: the model applies no
normalization internally, so a stale mean/std would silently corrupt every output rather than
raise. Its values are verified identical to those recorded inside the checkpoint
(`0.44862022165035664` / `0.23189431650723427`).

---

## 🔒 Integrity & Reproducibility

`run.py` fails loudly rather than producing subtly wrong output:

| Guard | On mismatch |
|---|---|
| Checkpoint SHA-256 `fac180b1…875cde30` | Hard error |
| Parameter count = 49,568 | Hard error |
| Normalization config present | Hard error |
| Per input: 2-D, even dims, finite | Reported; processing continues |
| Per output: exact 2× shape, finite | Assertion failure |

Deterministic for a fixed checkpoint, input set and hardware. Exit code `0` on full success,
`1` if any input failed — with every failure listed by filename. Use
`SEMICON_PRECISION=fp32` for bit-exact fp32 reproduction.

---

## 🧾 Environment

- **Python** 3.10+ (developed and tested on 3.12.5)
- **PyTorch** 2.5.1 · **NumPy** 2.1.3 · **Pillow** 11.2.1
- **GPU** CUDA auto-detected; clean CPU fallback
- **Target** NVIDIA H100 — fully offline at inference

`requirements.txt` pins the exact development-environment versions of the runtime dependency
closure. It is deliberately **not** a raw `pip freeze` dump: a full freeze of the development
machine carried Windows-only wheels (`pywin32`, `pywinpty`) that hard-fail on a Linux harness,
two `git+https` requirements needing git on the host, and ~10 GB of packages this submission
never imports. Verified by AST-parsing every shipped source file, the only third-party imports
are `torch`, `numpy` and `PIL`; everything else pinned is a transitive requirement of those.

---

## 🛠️ Troubleshooting

| Symptom | Resolution |
|---|---|
| `No module named torch` after install | Re-run with `--extra-index-url https://download.pytorch.org/whl/cu121`, or drop the `+cu121` suffix |
| `torch.cuda.is_available()` is `False` | Expected on CPU-only machines — falls back automatically |
| `Input dimensions must be even` | The network emits exactly 2H×2W; odd dims are rejected, never silently padded |
| `Checkpoint SHA-256 mismatch` | Weights altered or truncated — restore from the original package |
| CUDA out of memory | Try overriding with `SEMICON_BATCH_SIZE=32 python run.py <in> <out>` (Dynamic batching normally prevents this automatically) |
| Need exact fp32 numbers | `SEMICON_PRECISION=fp32 python run.py <in> <out>` |

---

## 📋 Example Run

```text
================================================================
SEMICON / KLA HACKATHON 2026
Unrolled K=3 + Degradation Estimator + FiLM
================================================================
  Device:           cuda (NVIDIA GeForce RTX 4050 Laptop GPU)
  Inference mode:   single-pass (TTA removed)
  Batch size:       Dynamic (Real-time VRAM allocation)
  Compute precision:fp16
  Writer threads:   4
  Input directory:  F:\my_data\competetion_projects\SEMICON\Final\Model_Round2\data\original\semicon_train_data_excluded\NoisyLR
  Output directory: F:\my_data\competetion_projects\SEMICON\Final\op
  Input files:      1197

Loading model...
  Checkpoint integrity: OK (f3497dc27fbe0f2d...)
  Parameters: 49,568

Loading 1197 inputs...

Processing 1197 valid inputs across 1 shape group(s)...
----------------------------------------------------------------
  [1197/1197] batch of 40 @ (128, 128)  (119.0 img/s, 10.1s elapsed)

================================================================
INFERENCE COMPLETE
================================================================
  Total inputs    : 1197
  Successful      : 1197
  Failed          : 0
  Total time      : 10.2s  (8 ms/img avg)
  Output directory: F:\my_data\competetion_projects\SEMICON\Final\op
  Output format   : .npy float32 (2x input size, [0,1])
================================================================
```

*Measured on an RTX 4050 Laptop GPU. An H100 will be substantially faster and automatically selects a maximum batch size near 1,800 depending on resolution and background VRAM usage.*

---

<div align="center">

### Built on measurement, not assumption.

**49,568 parameters · 3 unrolled stages · 8 ms/image · 100% offline**

*Reference: V. Monga, Y. Li, Y. C. Eldar, "Algorithm Unrolling: Interpretable, Efficient Deep
Learning for Signal and Image Processing," IEEE Signal Processing Magazine, 2021.*

</div>
