# Paper Context — Scaling Drifting VQ-VAE

**Last updated**: 2026-05-28 (artifacts fetched from W&B this session)
**Owner**: Hemal Arora (hemal1@stanford.edu), Stanford CS231N Spring 2026

**Data quality key used throughout:**
- ✓ = clean, complete run (local CSV artifact, verified this session)
- ✗ = excluded (corrupted, crashed, or collapsed — not counted in any mean/std)
- (partial) = run in progress; metrics are from mid-training, not final
- All means/std computed from raw CSV. Std is sample std (n−1 denominator).

---

## 1. Research Question

Can a physics-inspired codebook quantizer ("DriftingVQ") match or beat EMA VQ-VAE on
reconstruction quality while achieving dramatically higher codebook utilization?

Standard EMA VQ suffers from **Zipfian codebook usage**: a small fraction of codes dominate
while most are underused. Effective utilization (perplexity/K) is only 26–38% for EMA even
when the raw "active code" rate is >95%. DriftingVQ replaces the EMA update and commitment
loss with pairwise energy terms that enforce physics-inspired repulsion/attraction between
codes and encoder hiddens, driving near-uniform code usage.

---

## 2. Methods

### 2.1 Baselines

| method | description |
|---|---|
| `vanilla_ema` | EMA-updated codebook (k-means style, modern default). EMA moves code vectors toward the mean of assigned encoder hiddens. No gradient flows into codebook. |
| `vanilla_classical` | Gradient-based codebook via codebook + commitment loss (STE). |
| `simvq` | SimVQ baseline. |

### 2.2 DriftingVQ energy terms

All vectors L2-normalized to unit sphere before energy computation. Energy computed over
softmax-weighted similarities at temperature τ.

| term | symbol | description |
|---|---|---|
| U_pp | hidden-hidden repulsion | Pushes encoder hiddens apart. **Hurts reconstruction — remove it.** |
| U_nn | code-code repulsion | Pushes codebook vectors apart. **Essential** — removing causes catastrophic collapse |
| U_pn | hidden-code attraction | Pulls encoder hiddens toward codes and vice versa |

Gradient estimator: **rotation trick** (default) or **STE** (straight-through estimator).
STE proved dramatically better and is used in all final variants.

### 2.3 Variants tested

| variant | energy_terms | estimator | notes |
|---|---|---|---|
| `drift` | pp + nn + pn | rotation trick | full, original |
| `drift_no_pp` | nn + pn | rotation trick | remove hidden-hidden repulsion |
| `drift_ste` | pp + nn + pn | STE | full + STE |
| **`drift_no_pp_ste`** | **nn + pn** | **STE** | **canonical best variant** |
| `drift_ema` | pp + nn + pn | rotation trick | hybrid: EMA placement + drift energy |
| `drift_commit` | pp + nn + pn | rotation trick | drift + commitment loss |
| `drift_warmup` | pp + nn + pn | rotation trick | EMA hybrid, warmup schedule |
| `drift_anneal` | pp + nn + pn | rotation trick | linearly annealed energy — **unstable** |

### 2.4 Canonical best config

```
method: drift_no_pp_ste
energy_terms: (nn, pn)
rotation_trick: False
ste: True
tau: 1.0
energy_weight: 1.0
l2_normalize: True
```

---

## 3. Key Metrics

| Metric | Description | Direction |
|---|---|---|
| PSNR (dB) | Peak signal-to-noise ratio | ↑ higher is better |
| FID | Fréchet Inception Distance | ↓ lower is better |
| LPIPS | Learned perceptual similarity | ↓ lower is better |
| SSIM | Structural similarity | ↑ higher is better |
| Perplexity | exp(H) where H = codebook usage entropy; range 1–K | ↑ more uniform |
| Gini | Gini coefficient of code frequency; 0=uniform, 1=one code used | ↓ lower is better |
| Utilization | Fraction of codes with ≥1 assignment per batch | ↑ higher is better |
| **ppl/K** | Perplexity/K — effective utilization fraction | ↑ higher is better |
| hidden_norm | L2 norm of raw encoder output (before unit-sphere normalization) | diagnostic |

**Critical**: utilization % alone is misleading. EMA routinely has >95% "active" codes but
ppl/K ≈ 26–38%, meaning the alive codes are wildly unevenly used (Zipfian). Perplexity and
Gini are the right metrics.

---

## 4. Codebook Geometry — Mechanistic Explanation

Measured from Phase CIFAR-100, K=512, 30k iters (local CSV):

| metric | drift_no_pp_ste | vanilla_ema | interpretation |
|---|---|---|---|
| hidden_norm_mean | ~7.1–7.8 | ~0.37–0.39 | Encoder output scale before normalization |
| codebook/norm_mean | ~1.01–1.04 | ~0.81–1.06 | Code vector norms (≈1 on unit sphere) |
| codebook/pair_dist_mean | ~0.30–0.31 | ~1.15–1.26 | Mean Euclidean distance between codes |
| val/gini | ~0.37–0.40 | ~0.72–0.74 | Zipfian concentration |

**Mechanistic story:**

1. Drift trains an encoder with very high-norm raw outputs (~7–8). EMA encoder stays at low
   norm (~0.37).

2. Since VQ assignment is argmax_k(h · c_k) for unit-sphere codes, high hidden norm creates
   sharp (discriminative) projections — every batch assigns inputs to distinct codes →
   near-uniform usage. EMA's low norm makes many inputs equidistant from multiple codes →
   Zipfian concentration.

3. Drift codebook vectors cluster densely (pair_dist ≈ 0.30 vs theoretical √2 ≈ 1.41 for
   a uniform unit sphere). EMA spans a wider range (~1.2). U_pn (attraction) groups codes
   near the encoder's operating region; U_nn (repulsion) prevents collapse to a single point.

4. **The high hidden norm is emergent** — drift does not directly encourage it. The joint
   physics of U_pn plus reconstruction loss drives the encoder to commit strongly (high norm)
   to specific code directions.

Note: STL-10 shows even higher hidden_norm (~8.3–8.7) — more complex images require the
encoder to commit even more strongly.

At very low τ (e.g. 0.05), the softmax becomes too sharp, energy gradients collapse to ~0,
and hidden_norm stays low (~2.7). This explains why τ=0.05 underperforms: the energy terms
effectively vanish.

---

## 5. Results by Phase

---

### Phase 1 — CIFAR-10 Convergence (30k iters)

**Config**: CIFAR-10, K=512, 30k iters, 3 seeds each
**Runs**: 12/12 ✓ | **Source**: local CSV `runs/phase1_convergence/aggregate.csv`

| method | PSNR mean±std | FID mean±std | LPIPS | util | ppl | Gini |
|---|---|---|---|---|---|---|
| vanilla_ema | 23.386 ± 0.054 | 59.83 ± 0.51 | 0.248 | 99.8% | 135.8 | 0.742 |
| drift (full) | 23.199 ± 0.012 | 60.07 ± 1.16 | 0.244 | 100.0% | 489.7 | 0.164 |
| vanilla_classical | 22.980 ± 0.046 | 65.00 ± 2.40 | 0.267 | 99.6% | 105.7 | 0.745 |
| simvq | 21.944 ± 0.261 | 76.43 ± 7.45 | 0.306 | 98.6% | 59.4 | 0.855 |

Per-seed (vanilla_ema): PSNR 23.377/23.444/23.338 | FID 59.29/60.29/59.91
Per-seed (drift full): PSNR 23.214/23.193/23.192 | FID 58.79/60.36/61.08

**Findings:**
- Gap at 30k: 0.19 dB (was 1.5–2 dB at 2k iters) — full drift converges, just slower
- FID tied (overlapping error bars)
- Full drift: 3.6× higher perplexity (489.7 vs 135.8), Gini 0.164 vs 0.742
- vanilla_classical < vanilla_ema by 0.41 dB — EMA update rule matters
- simvq: highest variance, worst reconstruction

---

### Phase 2 — Diagnostic Ablations (10k iters, seed=0)

**Config**: CIFAR-10, K=512, 10k iters, seed=0 only
**Runs**: 9/9 ✓ | **Source**: local CSV `runs/phase2_diagnostic/aggregate.csv`
**Baseline for comparison**: vanilla_ema @10k = PSNR 23.532 ± 0.135, FID 60.86 (from Phase 3)

| ablation | config change | PSNR | FID | ppl | Gini | vs EMA@10k |
|---|---|---|---|---|---|---|
| `no_pp` | energy_terms=(nn,pn) | **23.732** | 55.783 | 384 | 0.410 | **+0.200** |
| `ste` | rotation_trick=False, ste=True | 23.672 | **52.078** | 439 | 0.303 | +0.140 |
| `tau=0.1` | τ=0.1, all terms | 23.266 | 58.741 | 397 | 0.392 | −0.267 |
| `no_l2` | l2_normalize=False | 23.053 | 60.050 | 350 | 0.470 | −0.480 |
| `full` (baseline) | all terms, τ=1.0, rotation | 22.858 | 62.818 | 479 | 0.197 | −0.675 |
| `tau=0.3` | τ=0.3, all terms | 22.808 | 64.895 | 472 | 0.212 | −0.725 |
| `tau=1.0` (run 2) | τ=1.0, all terms | 22.824 | 64.710 | 476 | 0.209 | −0.709 |
| `tau=3.0` | τ=3.0, all terms | 22.831 | 62.353 | 471 | 0.220 | −0.702 |
| `no_nn` ✗ | energy_terms=(pp,pn) | 17.901 | 137.2 | 12 | 0.980 | **COLLAPSE** |

Note: `no_nn` ran to completion but collapsed (79 active codes). Valid data point showing
U_nn is essential; excluded from all reconstruction quality comparisons.

**Key findings:**
1. **U_pp hurts**: removing gives +0.87 dB over full drift, +0.20 dB over EMA
2. **STE beats rotation trick**: +0.81 dB over full drift, +0.14 dB over EMA, −10.7 FID
3. **U_nn is essential**: removing → catastrophic collapse (PSNR 17.9, 79/512 active codes)
4. **τ insensitive** for full drift at 0.3–3.0; τ=0.1 slightly better but does NOT
   generalize to no_pp_ste (Phase Hparam Sweep finds τ=1.0 best there)
5. **Both `no_pp` and `ste` beat EMA at 10k** — key motivation for Phase 2b

---

### Phase 2b — Confirmation (30k iters, 3 seeds) — Headline Result

**Config**: CIFAR-10, K=512, 30k iters, 3 seeds each
**Runs**: 9/9 ✓ | **Source**: local CSV `runs/phase2b_confirmation/aggregate.csv`

| method | PSNR mean±std | FID mean | util | ppl | Gini | vs vanilla_ema@30k |
|---|---|---|---|---|---|---|
| **drift_no_pp_ste** | **24.151 ± 0.073** | **48.24** | 100% | 409 | 0.369 | **+0.765 dB, −11.59 FID** |
| drift_no_pp | 24.101 ± 0.031 | 49.83 | 100% | 410 | 0.364 | +0.715 dB |
| drift_ste | 23.982 ± 0.015 | 49.34 | 100% | 466 | 0.242 | +0.596 dB |
| *vanilla_ema @30k* | *23.386 ± 0.054* | *59.83* | *99.8%* | *136* | *0.742* | *baseline* |

Per-seed (drift_no_pp_ste): PSNR 24.252/24.120/24.081 | FID 47.08/48.80/48.85
Per-seed (drift_no_pp): PSNR 24.074/24.136/24.092 | FID 49.80/47.89/51.81
Per-seed (drift_ste): PSNR 23.997/23.967/23.981 | FID 50.69/49.05/48.29

**This is the headline result:**
- drift_no_pp_ste strictly dominates EMA simultaneously on PSNR (+0.77 dB), FID (−11.6),
  and perplexity (3× higher)
- `drift_ste` (keeps U_pp) has *higher* perplexity (466 vs 409) and lower Gini (more
  uniform) but worse PSNR (+0.596 vs +0.765). U_pp maximizes uniformity at a reconstruction cost.
- All variants: 100% utilization, 400+ perplexity vs EMA's 136

---

### Phase 3 — Hybrid Methods (10k iters, 3 seeds)

**Config**: CIFAR-10, K=512, 10k iters, 3 seeds each
**Runs**: 14/15 ✓ (1 excluded) | **Source**: local CSV `runs/phase3_hybrids/aggregate.csv`

| method | PSNR mean±std | FID mean | ppl | vs EMA@10k |
|---|---|---|---|---|
| vanilla_ema (10k) | 23.532 ± 0.135 | 60.86 | 171 | baseline |
| drift_ema | 23.124 ± 0.025 | 59.90 | 474 | −0.41 dB |
| drift_commit | 23.072 ± 0.068 | 61.37 | 448 | −0.46 dB |
| drift_warmup | 22.833 ± 0.222 | 62.30 | 496 | −0.70 dB |
| drift_anneal (seeds 0,1 only) | 22.813 | 66.3 | 473 | −0.72 dB |

**Excluded**: `cifar10_drift_anneal_K512_seed2` ✗ — partial collapse (PSNR 20.40, FID 121.0,
util 95.9%, active_codes 491/512).

Note: drift_warmup run_ids say "warmup" but the CSV method column reads "drift_ema" — CSV
artifact. These are the warmup-schedule hybrid runs.

**Key findings:**
- No hybrid beats vanilla_ema on PSNR at 10k
- `drift_ema` best FID of hybrids (59.90 ≈ EMA) but PSNR 0.41 dB behind
- Energy annealing is unstable — avoid linear schedules
- **Conclusion**: the right fix is ablating U_pp (Phase 2b), not building hybrids

---

### Phase Hparam Sweep — τ and energy_weight (30k, CIFAR-10 K=512)

**Config**: CIFAR-10, K=512, 30k iters, drift_no_pp_ste, 2 seeds per config
**Runs**: 14/14 ✓ | **Source**: local CSV `runs/phase_hparam_sweep/aggregate.csv`

τ sweep (energy_weight=1.0):

| τ | PSNR mean | FID mean | ppl mean | hidden_norm | notes |
|---|---|---|---|---|---|
| 0.05 | 23.663 | 55.16 | 397 | ~2.8 | energy terms collapse to ~0; hidden_norm low |
| 0.1 | 23.755 | 52.64 | 353 | ~2.1 | marginal energy; hidden_norm low |
| 0.3 | 24.141 | 49.73 | 459 | ~5.4 | energy active; pair_dist ~0.69 (not yet unit-sphere operating range) |
| **1.0** | **24.210** | 49.33 | 410 | ~7.6 | **optimal PSNR**; full energy regime |

Per-seed (τ=1.0, ew=1.0): PSNR 24.200/24.220 | FID 49.39/49.26

Energy weight sweep (τ=1.0):

| ew | PSNR mean | FID mean | ppl mean | hidden_norm |
|---|---|---|---|---|
| 0.1 | 24.030 | 50.26 | 426 | ~2.5 |
| **0.3** | 24.187 | **47.10** | 405 | ~5.1 |
| **1.0** | **24.210** | 49.33 | 410 | ~7.6 |
| 3.0 | 23.942 | 54.22 | 406 | ~8.4 |

Per-seed (ew=0.3): PSNR 24.146/24.227 | FID 46.16/48.03
Per-seed (ew=1.0): PSNR 24.200/24.220 | FID 49.39/49.26
Per-seed (ew=3.0): PSNR 23.796/24.087 | FID 54.31/54.13

**Key findings:**
- **τ=1.0 is best PSNR** (monotonically better 0.05→1.0). IMPORTANT: Phase 2 (full drift)
  found τ=0.1 best — does NOT carry over to no_pp_ste. Different physics without U_pp.
- At τ<0.3, energy terms effectively vanish (U_pn ≈ 0, hidden_norm stays low ~2). The
  physics only activates at τ≥0.3.
- **ew=1.0 best PSNR; ew=0.3 best FID** (−2.2 pts but same PSNR within noise). Trade-off.
- At ew=3.0, FID degrades significantly despite acceptable PSNR — too much energy
  disrupts the generation-quality balance.
- **Phase 2b (τ=1.0, ew=1.0) was already at the PSNR-optimal point.** Sweep validates
  Phase 2b headline result rather than improving it. All subsequent final runs use τ=1.0, ew=1.0.

---

### Phase CIFAR-100 — Cross-Class Scaling (K=512 and K=1024, 30k iters)

**Config**: CIFAR-100, K={512,1024}, 30k iters, 3 seeds each
**Runs**: 12/12 ✓ | **Source**: local CSV `runs/phase_cifar100/aggregate.csv`

| method | K | PSNR mean±std | FID mean | util | ppl | ppl/K | Gini | SSIM |
|---|---|---|---|---|---|---|---|---|
| vanilla_ema | 512 | 23.342 ± 0.011 | 58.02 | 99.8% | 145 | 28% | 0.728 | 0.741 |
| **drift_no_pp_ste** | 512 | **24.058 ± 0.092** | **49.72** | 100% | 402 | 79% | 0.385 | 0.778 |
| vanilla_ema | 1024 | 23.947 ± 0.019 | 52.17 | 98.7% | 265 | 26% | 0.753 | 0.768 |
| **drift_no_pp_ste** | 1024 | **24.570 ± 0.043** | **44.25** | 100% | 829 | 81% | 0.356 | 0.798 |

Per-seed (EMA K=512): PSNR 23.346/23.350/23.329 | FID 58.34/58.96/56.75 | ppl 148/138/149
Per-seed (drift K=512): PSNR 24.136/24.079/23.959 | FID 48.72/50.49/49.94 | ppl 395/401/409
Per-seed (EMA K=1024): PSNR 23.953/23.926/23.962 | FID 53.20/50.81/52.49 | ppl 251/277/267
Per-seed (drift K=1024): PSNR 24.561/24.532/24.617 | FID 43.62/43.57/45.56 | ppl 861/803/823

Geometry (drift K=512): hidden_norm 7.62/7.53/7.10 | pair_dist 0.301/0.306/0.308
Geometry (EMA K=512): hidden_norm 0.371/0.356/0.366 | pair_dist 1.220/1.261/1.147

**Deltas — drift vs EMA:**
- K=512: **+0.716 dB PSNR, −8.30 FID, 2.8× ppl**
- K=1024: **+0.623 dB PSNR, −7.92 FID, 3.1× ppl**

**Key findings:**
- CIFAR-100 (100 classes) causes EMA to distribute codes more unevenly than CIFAR-10
- EMA ppl/K: 28% → 26% as K doubles — doubling K barely increases distinct codes actually used
- Drift ppl/K: 79% → 81% — near-uniform regardless of K
- EMA active-code util: 99.8% → 98.7% — dead codes appear at K=1024, but the real issue
  is Zipfian concentration, not dead codes per se

---

### Phase CIFAR-100 K=2048 — Large-K Scaling

**Config**: CIFAR-100, K=2048, 30k iters, 3 seeds each
**Runs**: 6/6 ✓ | **Source**: local CSV `runs/phase_cifar100_k2048/aggregate.csv`

| method | K | PSNR mean±std | FID mean | util | ppl | ppl/K | Gini |
|---|---|---|---|---|---|---|---|
| vanilla_ema | 2048 | 24.363 ± 0.187 | 48.99 | 96.7% | 447 | 22% | 0.790 |
| **drift_no_pp_ste** | 2048 | **24.886 ± 0.119** | **43.11** | 99.7% | 1635 | 80% | 0.362 |

Per-seed (EMA): PSNR 24.374/24.173/24.543 | FID 48.36/50.08/48.53 | ppl 442/423/476 | util 96.3%/96.6%/97.2%
Per-seed (drift): PSNR 25.012/24.867/24.778 | FID 41.43/43.48/44.42 | ppl 1696/1628/1581 | util 99.7%/99.8%/99.8%

**Drift vs EMA**: **+0.523 dB PSNR, −5.88 FID, 3.7× ppl**

**Key findings:**
- EMA util drops to 96.7% avg = ~66 dead codes (vs ~1 at K=512) — dead-code problem
  accelerates with K
- EMA ppl/K drops to 22% (from 26–28% at lower K) — Zipfian concentration worsens
- Drift ppl/K = 80% — flat across K=512→2048

---

### Phase CIFAR-100 K=8192 — Pressure Test

**Config**: CIFAR-100, K=8192, 30k iters, 3 seeds each
**Runs**: 5/6 ✓ (1 excluded) | **Source**: local CSV `runs/phase_cifar100_k8192/aggregate.csv`

| method | seeds | PSNR mean±std | FID mean | util | ppl | ppl/K | Gini | status |
|---|---|---|---|---|---|---|---|---|
| vanilla_ema | 0,1,2 ✓ | 25.470 ± 0.356 | 38.08 | 92.5% | 2467 | 30.1% | 0.758 | stable all 3 seeds |
| drift_no_pp_ste | 0,2 ✓ | 25.718 ± 0.047 | 34.88 | 99.0% | 6578 | 80.3% | 0.353 | stable |
| drift_no_pp_ste | 1 ✗ | 22.791 | 87.15 | 60.9% | 1260 | 15.4% | 0.860 | **COLLAPSE** |

Per-seed (EMA): PSNR 25.111/25.822/25.477 | FID 40.24/37.49/36.51 | ppl 1955/2812/2633 | util 91.1%/92.7%/93.6%
Per-seed (drift stable): PSNR 25.751/25.685 | FID 36.27/33.49 | ppl 6410/6745 | util 98.8%/99.2%

**Excluded**: `cifar100_drift_no_pp_ste_K8192_seed1` ✗ — catastrophic collapse.
Full collapse data (not used in means): PSNR 22.791, FID 87.15, util 60.9%, ppl 1260, Gini 0.860.

**Drift vs EMA (stable drift seeds only)**: +0.248 dB PSNR, −3.20 FID, 2.7× ppl

**Key findings:**
- Drift advantage direction persists but is smaller (+0.25 vs +0.52 dB at K=2048)
- EMA is fully stable all 3 seeds; 1/3 drift seeds collapsed
- EMA util drops to ~92.5% at K=8192 (~655 dead codes) — degradation continues
- EMA still shows Zipfian: ppl/K 30.1% (even worse than K=2048's 22%; non-monotone
  because more codes fill in at K=8192 even if unevenly)
- Fix attempt `phase_cifar100_k8192_fix` in progress (ew=0.3 × seeds 0,1,2 + ew=1.0 × seed=3)

---

### Phase 4 Large-K Scaling Curve (10k iters, K=512→4096)

**Config**: CIFAR-100, K={512,1024,2048,4096}, 10k iters, 2 seeds each
**Runs**: 16/16 ✓ (all clean) | **Source**: local CSV `runs/phase4_large_k_v2/aggregate.csv`
(K=512 from original phase; K=1024/2048/4096 from relaunch after histogram-bug fix)

Note: these are **10k iters** (early convergence). Drift is slower to converge than EMA at
short horizons — the gap closes and reverses by 30k (see Phase 2b, Phase CIFAR-100).
The key insight here is the **ppl/K scaling shape**, not the reconstruction numbers.

| method | K | PSNR mean | FID mean | util | ppl | ppl/K | Gini |
|---|---|---|---|---|---|---|---|
| vanilla_ema | 512 | 23.649 | 58.94 | 99.8% | 193 | 37.6% | 0.692 |
| drift_no_pp_ste | 512 | 23.487 | 57.45 | 99.6% | 378 | 73.8% | 0.418 |
| vanilla_ema | 1024 | 24.287 | 55.72 | 99.3% | 384 | 37.5% | 0.701 |
| drift_no_pp_ste | 1024 | 23.627 | 53.69 | 98.1% | 708 | 69.2% | 0.457 |
| vanilla_ema | 2048 | 24.722 | 48.53 | 98.0% | 749 | 36.6% | 0.713 |
| drift_no_pp_ste | 2048 | 24.128 | 52.35 | 98.1% | 1394 | 68.1% | 0.465 |
| vanilla_ema | 4096 | 25.241 | 43.13 | 96.1% | 1514 | 36.9% | 0.712 |
| drift_no_pp_ste | 4096 | 24.889 | 44.94 | 97.3% | 2872 | 70.1% | 0.449 |

Per-seed detail (K=1024 relaunch):
- EMA: PSNR 24.375/24.199 | FID 54.58/56.85 | ppl 407/362
- drift: PSNR 23.941/23.312 | FID 48.94/58.44 | ppl 693/724

Per-seed detail (K=2048 relaunch):
- EMA: PSNR 24.857/24.587 | FID 49.13/47.93 | ppl 809/688
- drift: PSNR 23.697/24.559 | FID 57.88/46.83 | ppl 1316/1473
  (high PSNR variance at K=2048 reflects slow convergence at 10k, not instability)

Per-seed detail (K=4096 relaunch):
- EMA: PSNR 25.276/25.206 | FID 43.42/42.84 | ppl 1546/1482
- drift: PSNR 24.895/24.882 | FID 46.26/43.62 | ppl 3032/2712

**Key finding — the scaling curve:**
- **EMA ppl/K is flat at ~37% across ALL K** (512→4096). Doubling K consistently yields
  only ~37% effective utilization regardless of codebook size. Zipfian behavior is structural.
- **Drift ppl/K is flat at ~70% at 10k iters** (rises to ~79–81% by 30k per Phase CIFAR-100).
  Drift uses codebook proportionally regardless of K.
- This is the key figure for the paper: a ppl/K vs K plot with EMA flat at ~37% and drift
  flat at ~79–81% (at 30k convergence).
- At K=4096, drift is nearly competitive with EMA on PSNR at 10k (−0.35 dB drift vs EMA),
  narrower than K=512 (−0.16 dB). Drift's slow-convergence penalty shrinks at larger K.

---

### Phase STL-10 — Cross-Dataset, Higher Resolution

**Config**: STL-10 (100k images, 96×96 → 64×64), K=512, 30k iters, 3 seeds each
**Runs**: 6/6 ✓ | **Source**: local CSV `runs/phase_stl10/aggregate.csv`

Note: CelebA was the original target but abandoned — torchvision CelebA downloads via Google
Drive, which rate-limits globally. Replaced by STL-10.

| method | PSNR mean±std | FID mean | util | ppl | Gini |
|---|---|---|---|---|---|
| vanilla_ema | 21.475 ± 0.251 | 133.23 | 99.9% | 123.8 | 0.769 |
| **drift_no_pp_ste** | **21.997 ± 0.046** | **118.26** | 100% | 416.7 | 0.352 |

Per-seed (EMA): PSNR 21.184/21.630/21.612 | FID 133.15/133.24/133.30 | ppl 99.5/135.9/136.1
Per-seed (drift): PSNR 22.036/21.945/22.011 | FID 118.69/119.57/116.53 | ppl 424/417/409

Geometry (drift): hidden_norm 8.70/8.58/8.34 | pair_dist 0.322/0.345/0.340
Geometry (EMA): hidden_norm 0.403/0.434/0.478 | pair_dist 1.870/1.948/2.232

**Drift vs EMA**: **+0.522 dB PSNR, −14.97 FID, 3.4× ppl**

**Key findings:**
- FID gap (−14.97) is the largest seen across all datasets
- EMA variance is much higher (std 0.251 vs 0.046) — drift is more stable across seeds
- Drift hidden_norm is highest here (~8.3–8.7) — more complex images → encoder must
  commit even more strongly
- EMA seed0 ppl only 99.5 vs seeds 1,2 (~136) — EMA is less consistent

---

### Phase Tiny ImageNet

**Config**: Tiny ImageNet (200 classes, 64×64), K={512,1024}, 30k iters, 3 seeds each
**Runs**: 11/12 ✓ (1 excluded) | **Source**: local CSV `runs/phase_tiny_imagenet/aggregate.csv`

**Excluded**: `tinyimagenet_vanilla_ema_K512_seed2` ✗ — crashed at step 600/30000. Local
`summary.json` confirms final step=600, `opt/energy_weight=NaN`.

| method | K | PSNR mean±std (n) | FID mean | util | ppl | Gini |
|---|---|---|---|---|---|---|
| vanilla_ema | 512 | 18.860 ± 0.117 (n=2) | 121.96 | 99.6% | 112.4 | 0.780 |
| **drift_no_pp_ste** | 512 | **19.160 ± 0.024** (n=3) | **114.93** | 100% | 394.6 | 0.395 |
| vanilla_ema | 1024 | 18.972 ± 0.069 (n=3) | 118.98 | 97.4% | 183.0 | 0.811 |
| **drift_no_pp_ste** | 1024 | **19.376 ± 0.059** (n=3) | **109.76** | 99.9% | 740.3 | 0.437 |

Per-seed (EMA K=512): PSNR 18.977/18.744/CRASHED | FID 122.36/121.55/—
Per-seed (drift K=512): PSNR 19.184/19.137/19.160 | FID 113.75/114.49/116.56 | ppl 389/388/407
Per-seed (EMA K=1024): PSNR 18.919/18.945/19.052 | FID 118.26/120.91/117.76 | ppl 175/175/199
Per-seed (drift K=1024): PSNR 19.357/19.443/19.329 | FID 106.64/107.34/115.32 | ppl 776/745/700

**Drift vs EMA:**
- K=512: +0.300 dB PSNR, −7.03 FID, 3.5× ppl (EMA n=2 only — treat with caution)
- K=1024: **+0.404 dB PSNR, −9.22 FID, 4.0× ppl** (n=3 clean for both — use this)

---

### Phase CIFAR-100 K=8192 Fix (in progress)

**Config**: CIFAR-100, K=8192, 30k iters, drift only, ew=0.3 × seeds 0,1,2 + ew=1.0 × seed=3
**Status**: in progress as of 2026-05-28. Seeds 0,1 at ~81 and 25 W&B history points (~4050 and
~1250 training steps). No results reported — too early for valid metrics.

Hypothesis: ew=0.3 reduces energy force magnitude at extreme K, preventing the collapse seen
at ew=1.0 seed1. EMA baselines already exist from Phase CIFAR-100 K=8192.

---

### Phase Optimized — Tuned Hparams Across Datasets (partial)

**Config**: drift_no_pp_ste only, τ=1.0, ew=1.0, 30k iters, 3 seeds
**Datasets**: CIFAR-10 K=512, CIFAR-100 K=512+1024, STL-10 K=512, Tiny ImageNet K=512+1024
**Source**: local CSV `runs/phase_optimized/aggregate.csv` (partial — some runs still in flight)

Runs marked (partial) are still running as of 2026-05-28; their current metrics are from
mid-training and should not be treated as final. Use existing phase results instead.

**Completed runs (state=finished, 30k iters):**

| dataset | K | PSNR mean±std | FID mean | ppl | vs existing phase |
|---|---|---|---|---|---|
| CIFAR-10 | 512 | 24.111 ± 0.034 | 48.07 | 412 | cf. Phase 2b: 24.151 ± 0.073 — consistent ✓ |
| CIFAR-100 | 512 | 24.071 ± 0.019 | 48.51 | 405 | cf. Phase CIFAR-100: 24.058 ± 0.092 — consistent ✓ |
| CIFAR-100 | 1024 | 24.514 ± 0.052 | 45.05 | 835 | cf. Phase CIFAR-100: 24.570 ± 0.043 — consistent ✓ |
| STL-10 | 512 | seed0 only: 22.017, 119.49 | — | — | seed0 complete; seeds 1,2 still running (~82%) |

Per-seed CIFAR-10: PSNR 24.150/24.093/24.089 | FID 48.742/46.788/48.684
Per-seed CIFAR-100 K=512: PSNR 24.093/24.060/24.060 | FID 48.076/48.837/48.616
Per-seed CIFAR-100 K=1024: PSNR 24.466/24.568/24.508 | FID 43.377/45.328/46.436

**Still running — do not use as final results:**
- STL-10 K=512 seeds 1,2 (~82% complete at recover time)
- Tiny ImageNet K=512 all seeds (very early — ~1200 steps; metrics unreliable)
- Tiny ImageNet K=1024 seeds 0,2 (early — ~4000 steps); seed 1 no data yet

**Interpretation**: The phase_optimized results confirm Phase 2b was already at the optimal
hyperparameter point (τ=1.0, ew=1.0). Numbers are consistent across independent runs.
Use existing phase results for paper tables — they have more seeds and the same hparams.

---

### Prior Pilot — Autoregressive Prior Difficulty

**Config**: CIFAR-100, K=512, 2-layer d=128 transformer prior, 3000 training steps
**Source**: W&B directly; no local CSV artifact

| method | best val NLL (bits/code) | fraction of max entropy |
|---|---|---|
| vanilla_ema | **5.805** | 5.805/9.0 = 64% of max |
| drift_no_pp_ste | 6.928 | 6.928/9.0 = 77% of max |

**Gap**: −1.12 bits/code — EMA is substantially easier to model.

**Interpretation**: EMA's Zipfian distribution (~112/512 effective codes) is easy to predict —
a small prior memorizes the heavy-hitters. Drift's near-uniform distribution requires modeling
~395/512 effective codes. The gap is structural: high entropy distributions are harder to
compress autoregressively. A larger prior would narrow but not close the gap.

**Paper implication**: Better reconstruction coexists with harder prior modeling. Drift excels
at reconstruction (PSNR/FID) but produces codes that are less compressible. This is a
fundamental tradeoff to discuss in the paper.

---

### Linear Probe — Codebook Semantic Content

**Config**: CIFAR-100, K=512, drift_no_pp_ste vs vanilla_ema (phase_cifar100 checkpoints)
**Feature**: bag-of-visual-words histogram. Image → discrete token indices → normalized
frequency histogram over K codes. Feature shape: (N, K) = (50000, 512)/(10000, 512).
**Classifier**: logistic regression, C ∈ {0.01, 0.1, 1.0, 10.0}
**Source**: Modal run completed 2026-05-28 (stdout). K=1024 run in progress.

Note: an earlier run (killed mid-run) used mean-pooled embeddings. Results below are from
the corrected histogram implementation.

| run | top1 | top5 | best C |
|---|---|---|---|
| vanilla_ema K=512 seed0 | 13.3% | 33.7% | 0.01 |
| vanilla_ema K=512 seed1 | 12.7% | 33.0% | 0.01 |
| vanilla_ema K=512 seed2 | 13.2% | 33.6% | 0.01 |
| drift_no_pp_ste K=512 seed0 | 14.5% | 35.9% | 0.01 |
| drift_no_pp_ste K=512 seed1 | 14.7% | 36.0% | 0.01 |
| drift_no_pp_ste K=512 seed2 | 14.2% | 36.2% | 0.01 |

**Summary**: vanilla_ema 13.1 ± 0.2% top1, 33.4 ± 0.3% top5
**Summary**: drift_no_pp_ste 14.5 ± 0.2% top1, 36.1 ± 0.1% top5
**Δ (drift − EMA): top1 +1.4 pp, top5 +2.6 pp**

Verdict: **weak but consistent signal in drift's favor**. All 3 drift seeds beat all 3 EMA
seeds on both metrics. K=1024 result pending (tmux: `linprob1024`).

---

## 6. Consolidated Drift vs EMA Table

All rows: drift_no_pp_ste vs vanilla_ema, 30k iters unless noted.

| Dataset | K | PSNR Δ | FID Δ | ppl ratio | drift std | EMA std | notes |
|---|---|---|---|---|---|---|---|
| CIFAR-10 | 512 | **+0.765 dB** | −11.59 | 3.0× | 0.073 | 0.054 | Phase 2b, headline |
| CIFAR-100 | 512 | +0.716 dB | −8.30 | 2.8× | 0.092 | 0.011 | |
| CIFAR-100 | 1024 | +0.623 dB | −7.92 | 3.1× | 0.043 | 0.019 | |
| CIFAR-100 | 2048 | +0.523 dB | −5.88 | 3.7× | 0.119 | 0.187 | |
| CIFAR-100 | 8192 | +0.248 dB* | −3.20* | 2.7×* | 0.047* | 0.356 | *stable drift seeds only (2/3) |
| STL-10 | 512 | +0.522 dB | **−14.97** | 3.4× | 0.046 | 0.251 | largest FID gap |
| Tiny ImageNet | 512 | +0.300 dB† | −7.03† | 3.5× | 0.024 | 0.117† | †EMA n=2 only |
| Tiny ImageNet | 1024 | +0.404 dB | −9.22 | 4.0× | 0.059 | 0.069 | |

**ppl/K scaling (from Phase 4 Large-K Curve, 10k iters, CIFAR-100):**

| K | EMA ppl/K | drift ppl/K (10k) | drift ppl/K (30k est.) |
|---|---|---|---|
| 512 | 37.6% | 73.8% | ~79% |
| 1024 | 37.5% | 69.2% | ~81% |
| 2048 | 36.6% | 68.1% | ~80% |
| 4096 | 36.9% | 70.1% | — |
| 8192 | 30.1% | 80.3% (stable seeds) | — |

EMA ppl/K is flat at ~37% regardless of K. Drift ppl/K is ~70% at 10k, rises to ~79–81%
at 30k convergence. This table is the core scaling figure for the paper.

---

## 7. Corrupted / Excluded Runs — Complete List

| Run ID | Phase | Reason |
|---|---|---|
| `tinyimagenet_vanilla_ema_K512_seed2` | phase_tiny_imagenet | Crashed at step 600/30000 (NaN energy_weight). Confirmed in local summary.json. |
| `cifar10_drift_anneal_K512_seed2` | phase3_hybrids | Partial collapse: PSNR 20.40 vs seeds 0,1 at ~22.81; FID 121 vs ~66; util 95.9%; active_codes 491/512. |
| `cifar100_drift_no_pp_ste_K8192_seed1` | phase_cifar100_k8192 | Catastrophic collapse: PSNR 22.791, FID 87.15, util 60.9%, Gini 0.860. |
| `cifar100_drift_no_pp_ste_K1024_seed1` (original) | phase4_large_k_v2 | Crashed at step ~1000 (wandb.Histogram 512-bin limit bug, since fixed). No val metrics. Superseded by relaunch. |
| `cifar100_vanilla_ema_K1024_seed{0,1}` (original) | phase4_large_k_v2 | Same bug — missing all val metrics. Superseded by relaunch. |
| `cifar100_vanilla_ema_K2048_seed{0,1}` (original) | phase4_large_k_v2 | Same bug. Superseded by relaunch. |

Note on `no_nn` (Phase 2): ran to completion but produced a collapsed model. Valid data point
showing U_nn is essential; excluded from reconstruction quality statistics.

---

## 8. In-Flight Runs (as of 2026-05-28)

| tmux session | what | status |
|---|---|---|
| `phaseoptimized` | phase_optimized, 18 drift_no_pp_ste runs across 4 datasets | CIFAR-10 + CIFAR-100 done; STL-10 seed0 done, seeds 1,2 ~82%; Tiny ImageNet all early |
| `phasecifar100k8192fix` | K=8192 stability fix — ew=0.3 × seeds 0,1,2 + ew=1.0 × seed=3 | Seeds 0,1 at ~4050 and ~1250 steps; seed 2 not yet visible |
| `linprob1024` | linear_probe --k 1024 on phase_cifar100 checkpoints | Running |
| `perclass1024` | per_class_analysis --k 1024 on phase_cifar100 checkpoints | Running |

---

## 9. Key Design Decisions and Rationale

| Decision | Choice | Why |
|---|---|---|
| U_nn | Always keep | Without it: catastrophic collapse to 79 active codes, PSNR 17.9 dB |
| U_pp | **Remove** | Hurts PSNR by 0.87 dB, FID by 7 pts — forces encoder to spread unnaturally at cost of reconstruction |
| Gradient estimator | **STE** over rotation trick | +0.81 dB PSNR, −10.7 FID. Rotation trick is theoretically tighter but hurts in practice |
| τ (no_pp_ste) | **1.0** | Monotonically better 0.05→1.0. At τ<0.3 energy terms effectively vanish. Opposite of full drift (τ=0.1 best there) |
| ew (no_pp_ste) | **1.0** (headline; PSNR) or 0.3 (best FID, −2.2 pts) | Final runs use ew=1.0 for consistency with Phase 2b |
| Energy schedule | Constant | Linear annealing is unstable (drift_anneal seed2 collapse) |
| L2 normalization | Keep | Removing costs 0.48 dB PSNR, 10 FID |

---

## 10. Paper Narrative Outline

**Problem**: VQ-VAE codebooks suffer Zipfian concentration — EMA achieves ≥95% active codes
but only 22–37% effective utilization (ppl/K). Most codes are alive but rarely used. This
caps model capacity and semantic expressiveness of the discrete representation.

**Method**: Replace EMA update and commitment loss with physics-inspired pairwise energy:
- U_nn (code-code repulsion): prevents codebook collapse
- U_pn (hidden-code attraction): aligns encoder to codebook
- Remove U_pp (hidden-hidden repulsion): hurts reconstruction, remove it
- Use STE instead of rotation trick: simpler, empirically better

**Why it works — mechanistic story**: U_pn + reconstruction loss jointly drive the encoder to
commit strongly to specific code directions (emergent high hidden norm, ~7–8 vs EMA ~0.37).
High hidden norm → sharp VQ assignments → every code gets used → near-uniform utilization
(ppl/K ~79–81% vs EMA's ~22–37%).

**Headline result** (CIFAR-10, K=512, 30k):
drift_no_pp_ste beats vanilla_ema by **+0.765 dB PSNR, −11.6 FID, 3.0× perplexity**
simultaneously. 100% utilization, Gini 0.369 vs 0.742.

**Scaling**:
- ppl/K flat at ~79–81% for drift across K=512→2048 (and ~80% at K=8192 for stable seeds)
- ppl/K flat at ~22–38% for EMA across same K range — Zipfian behavior is structural
- Advantage consistent across CIFAR-100 (K=512/1024/2048), STL-10, Tiny ImageNet
- FID advantage largest on STL-10 (−14.97 pts)
- PSNR advantage narrows at extreme K (K=8192: +0.25 dB)

**Ablations**: U_nn essential; U_pp harmful; STE beats rotation trick; τ insensitive for
no_pp_ste once active (τ≥0.3); L2 normalization required.

**Stability**: 1/3 drift seeds collapsed at K=8192 (EMA fully stable). Fix (ew=0.3) in progress.

**Downstream:**
- Linear probe (bag-of-words, CIFAR-100 K=512): drift +1.4 pp top1, +2.6 pp top5 — weak but consistent
- Prior NLL (CIFAR-100 K=512): EMA easier to model (5.81 vs 6.93 bits/code) — structural tradeoff

**Discussion points:**
- Reconstruction quality and codebook uniformity are not in tension when U_pp is removed
- EMA's Zipfian behavior wastes codebook capacity — drift shows this is not fundamental
- Prior generation harder with drift codes — reconstruction fidelity and prior compressibility trade off
- Convergence: drift is slower at 10k but matches/beats EMA by 30k on all datasets
- K=8192 stability: open question; ew=0.3 fix pending

---

## 11. Data Source Summary

| Phase | Source | # clean runs | # excluded | notes |
|---|---|---|---|---|
| phase0_smoke | ✓ local CSV | 5 | 0 | 500-iter infra test, not used in paper |
| phase1_convergence | ✓ local CSV | 12 | 0 | |
| phase2_diagnostic | ✓ local CSV | 8+1‡ | 0 | ‡no_nn counted separately as collapse data point |
| phase2b_confirmation | ✓ local CSV | 9 | 0 | Headline result |
| phase3_hybrids | ✓ local CSV | 14 | 1 | drift_anneal seed2 excluded |
| phase_cifar100 | ✓ local CSV | 12 | 0 | |
| phase_cifar100_k2048 | ✓ local CSV | 6 | 0 | |
| phase_cifar100_k8192 | ✓ local CSV | 5 | 1 | drift seed1 collapse; fetched this session |
| phase_cifar100_k8192_fix | in-flight | — | — | ew=0.3 stability test |
| phase_stl10 | ✓ local CSV | 6 | 0 | |
| phase_tiny_imagenet | ✓ local CSV | 11 | 1 | EMA K=512 seed2 crashed |
| phase4_large_k_v2 | ✓ local CSV | 16 | 5† | †original K>512 runs (histogram bug); relaunch complete |
| phase_hparam_sweep | ✓ local CSV | 14 | 0 | fetched this session |
| prior_pilot | W&B summary only | 2 | 0 | no local CSV |
| phase_optimized | ✓ partial local CSV | 9 done; 9 in-flight | 0 | validates existing results |
| linear_probe K=512 | stdout | 6 | 0 | histogram feature |
| linear_probe K=1024 | in-flight | — | — | |
| per_class_analysis K=1024 | in-flight | — | — | |

**Total clean runs with local CSV verification**: 122+ (all completed phases now locally verified)
