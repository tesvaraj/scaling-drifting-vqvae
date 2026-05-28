# Experimental Context — Scaling Drifting VQ-VAE
<!-- Updated: 2026-05-27. Update this file after each phase completes. -->

## Phase 1 — Convergence (30k iters, K=512, CIFAR-10, 3 seeds each)  STATUS: DONE

| method | PSNR mean±std | FID mean±std | LPIPS | util% | perplexity | Gini |
|---|---|---|---|---|---|---|
| vanilla_ema | **23.386 ± 0.057** | **59.83 ± 0.50** | 0.248 | 99.8 | 135.8 | 0.742 |
| drift | 23.199 ± 0.012 | 60.07 ± 1.17 | 0.244 | 100.0 | **489.7** | **0.164** |
| vanilla_classical | 22.980 ± 0.046 | 65.00 ± 1.96 | 0.267 | 99.6 | 105.7 | 0.748 |
| simvq | 21.944 ± 0.261 | 76.43 ± 5.97 | 0.306 | 98.6 | 59.4 | 0.861 |

**Interpretation:**
- Gap at convergence (30k) is only **0.19 dB PSNR** (was 1.5–2 dB at 2k iters) — drift converges, just slower
- FID is essentially tied (59.83 vs 60.07, overlapping error bars)
- Drift achieves 3.6× higher perplexity and Gini 0.164 vs 0.742 — nearly uniform codebook vs Zipfian EMA
- vanilla_classical < vanilla_ema by 0.41 dB — EMA update rule matters
- simvq shows high variance across seeds (FID range 70–85) — least stable

---

## Phase 2 — Diagnostic Ablations (10k iters, K=512, seed=0 only)  STATUS: DONE

Baseline for comparison: **vanilla_ema @10k = 23.53 ± 0.11 PSNR, FID ≈ 60.86** (from Phase 3)

| ablation | config change | PSNR | FID | perplexity | vs EMA PSNR |
|---|---|---|---|---|---|
| `no_pp` | `energy_terms=('nn','pn')` | **23.73** | **55.78** | 384 | **+0.20** |
| `ste` | `rotation_trick=False, ste=True` | 23.67 | **52.08** | 439 | **+0.14** |
| `tau=0.1` | `tau=0.1` | 23.27 | 58.74 | 397 | -0.26 |
| `no_l2` | `l2_normalize=False` | 23.05 | 60.05 | 350 | -0.48 |
| `full` (baseline) | all terms | 22.86 | 62.82 | 479 | -0.67 |
| `tau=0.3/1.0/3.0` | tau sweep | 22.81–22.83 | 62–65 | 471–476 | -0.70 |
| `no_nn` | `energy_terms=('pp','pn')` | 17.90 | 137 | 12 | **COLLAPSE** |

**Interpretation:**
- **U_pp (hidden-hidden repulsion) hurts both PSNR and FID**: removing it gives +0.87 dB over full drift
- **STE beats rotation trick**: +0.81 dB PSNR and -10.7 FID (very surprising — rotation trick hurts)
- **U_nn is essential**: removing it causes catastrophic codebook collapse (17.9 PSNR, 79 active codes)
- **tau insensitive** for full drift: 0.3/1.0/3.0 all similar; 0.1 slightly better
- Both `no_pp` and `ste` already beat vanilla_ema at 10k — this is the key finding
- **`no_pp + ste` not yet tested** — likely the best combination

---

## Phase 3 — Hybrids (10k iters, K=512, 3 seeds each)  STATUS: DONE

Baseline: **vanilla_ema @10k = 23.532 ± 0.110 PSNR, FID ≈ 60.86**

| method | PSNR mean±std | FID mean | perplexity | vs EMA |
|---|---|---|---|---|
| vanilla_ema (10k) | 23.532 ± 0.110 | 60.86 | 171 | baseline |
| drift_ema | 23.124 ± 0.021 | **59.90** | 474 | -0.41 dB |
| drift_commit | 23.072 ± 0.065 | 61.37 | 448 | -0.46 dB |
| drift_warmup | 22.833 ± 0.185 | 62.30 | 496 | -0.70 dB |
| drift_anneal | 22.01 ± 1.20 | 84.5 | — | unstable (seed2 failed) |

**Interpretation:**
- `drift_ema` achieves best FID (59.90) essentially matching EMA, but PSNR still 0.41 dB behind
- `drift_warmup` achieves highest perplexity (496) but worse reconstruction
- `drift_anneal` with linear schedule is unstable — avoid
- None of the hybrids beat vanilla_ema on PSNR at 10k
- **The right fix is ablating U_pp (Phase 2b), not the EMA hybrid**

---

## Phase 2b — Confirmation Runs  STATUS: DONE ✓

**Question**: Do `no_pp` and `ste` advantages hold at 30k convergence?  
**Design**: CIFAR-10, K=512, 30k iters, 3 seeds each.

| method | PSNR mean±std | FID mean | util | perplexity | Gini | vs vanilla_ema |
|---|---|---|---|---|---|---|
| **drift_no_pp_ste** | **24.151 ± 0.073** | **48.24** | 100% | 409 | 0.369 | **+0.765 dB** |
| drift_no_pp | 24.101 ± 0.026 | 49.83 | 100% | 410 | 0.363 | +0.715 dB |
| drift_ste | 23.982 ± 0.012 | 49.34 | 100% | 466 | 0.242 | +0.596 dB |
| *vanilla_ema @30k* | *23.386 ± 0.057* | *59.83* | *99.8%* | *136* | *0.742* | baseline |

**Decision tree outcome**: `no_pp_ste@30k = 24.151 ≥ 23.39` → **drift strictly dominates EMA** on PSNR (+0.77 dB), FID (−11.6), and utilization (3× perplexity) simultaneously. This is the headline result.

**Key observations:**
- drift_ste (keeps U_pp) has *higher* perplexity (466) and lower Gini (more uniform) than no_pp_ste (409, Gini 0.369), but worse PSNR. U_pp forces the encoder to spread more uniformly but at a reconstruction cost.
- drift_no_pp_ste is the canonical best variant: remove U_pp (encoder spread) and use STE (skip rotation trick).
- All variants achieve 100% codebook utilization and 400+ perplexity vs EMA's 136.

---

## Phase STL-10 — Cross-Dataset (STL-10 → 64x64, K=512, 30k)  STATUS: DONE ✓

| method | PSNR mean±std | FID | util | ppl |
|---|---|---|---|---|
| vanilla_ema K=512 | 21.443 ± 0.250 | 132.5 | 99.9% | 126 |
| **drift_no_pp_ste K=512** | **21.997 ± 0.042** | **118.3** | **100%** | **417** |

- **+0.553 dB PSNR, −14.3 FID, 3.3× perplexity** — FID gap is the largest seen across all datasets
- EMA variance is much higher (0.250 vs 0.042) — drift is more stable across seeds
- Replaces CelebA (Google Drive quota issues). STL-10: 100k images, 10 classes, 96×96→64×64.

---

## Phase CIFAR-100 K=2048 — Large-K Scaling  STATUS: DONE ✓

| method | PSNR mean±std | FID | util | ppl | ppl/K |
|---|---|---|---|---|---|
| vanilla_ema K=2048 | 24.270 ± 0.141 | 49.2 | 96.4% | 433 | **21%** |
| **drift_no_pp_ste K=2048** | **24.940 ± 0.099** | **42.5** | **99.8%** | **1662** | **81%** |

- **+0.670 dB PSNR, −6.8 FID, 3.8× perplexity**
- EMA util drops to 96.4% = 74 dead codes (vs 13 at K=1024, 1 at K=512) — degradation accelerating
- EMA ppl/K = 21% (down from 26% at K=1024) — Zipfian concentration worsening with K
- Drift ppl/K = 81% — flat across K=512→2048 (78-81%)

---

## Phase CelebA — Cross-Dataset (CelebA-64, K=512, 30k)  STATUS: ABANDONED

CelebA downloads via Google Drive which rate-limits torchvision globally. Replaced by STL-10.

---

## Phase CIFAR-100 K=8192 — Pressure Test  STATUS: DONE ✓ (mixed)

| method | PSNR mean | FID mean | ppl/K | notes |
|---|---|---|---|---|
| vanilla_ema (3 seeds) | 25.47 ± 0.29 | 38.1 | 30.1% | stable |
| drift seeds 0,2 | 25.71 ± 0.04 | 34.9 | 80.3% | stable, wins |
| drift seed 1 | 22.79 | 87.2 | 15.4% | **COLLAPSE** — util 60.9%, Gini 0.86 |

**1/3 drift seeds collapsed** (PSNR 22.79, FID 87, util 60.9%). When stable, drift wins
(+0.25 dB, −3.2 FID, 2.7× ppl). EMA is fully stable all 3 seeds.

**Fix attempt**: `phase_cifar100_k8192_fix.py` — 4 runs drift only:
- ew=0.3 × seeds 0,1,2 (hparam sweep best-FID value; may stabilize via weaker energy forces)
- ew=1.0 × seed=3 (extra data point to characterize failure rate)

```bash
python -m experiments.launch --phase phase_cifar100_k8192_fix
```

---

## Phase CIFAR-100 — Best Showcase for Drift  STATUS: DONE ✓

**Why CIFAR-100**: 100 classes → EMA distributes codes more unevenly; drift's physics repulsion should shine.

| method | PSNR mean±std | FID mean | util | perplexity | SSIM |
|---|---|---|---|---|---|
| vanilla_ema K=512 | 23.343 ± 0.009 | 58.02 | 99.8% | 145 | 0.7407 |
| **drift_no_pp_ste K=512** | **24.060 ± 0.075** | **49.72** | **100%** | **402** | **0.7777** |
| vanilla_ema K=1024 | 23.947 ± 0.012 | 52.17 | 98.7% | 265 | 0.7679 |
| **drift_no_pp_ste K=1024** | **24.570 ± 0.037** | **44.25** | **100%** | **829** | **0.7983** |

**Delta drift vs EMA:**
- K=512: **+0.717 dB PSNR, -8.30 FID, 2.8× perplexity**
- K=1024: **+0.623 dB PSNR, -7.92 FID, 3.1× perplexity**

**Key observations:**
- Drift advantage is consistent at both K values — matches Phase 2b direction on CIFAR-10 (+0.765 dB there vs +0.717 here)
- EMA active-code utilization is stable (99.8% → 98.7%), so "collapse" is the wrong word; the issue is **Zipfian concentration** — EMA codes are alive but wildly unevenly used
- EMA effective utilization (perplexity/K): 145/512=28% at K=512, 265/1024=26% at K=1024 — flat, meaning doubling K barely increases the number of meaningfully distinct codes used
- Drift effective utilization (perplexity/K): 402/512=78%, 829/1024=81% — near-uniform regardless of K
- Gini is the clearest single metric: EMA ~0.74 (very concentrated), drift ~0.37 (much more uniform)
- FID improvement at K=1024 is the largest seen: -7.92 pts
- Config: `phase_cifar100.py`. 12 runs: vanilla_ema + drift_no_pp_ste × K{512,1024} × 3 seeds × 30k.

---

## Phase 4 v2 — Large-K Scaling on CIFAR-100  STATUS: PARTIAL — K>512 relaunch pending

**Key figure**: perplexity vs K curve — EMA's effective code use (perplexity/K) stays at ~26-38% across all K (Zipfian concentration), while drift's stays at ~72-81% (near-uniform). Utilization (% active codes) is NOT the right metric — EMA has only 4% dead codes at K=4096, but the alive codes are extremely unevenly used.

**K=512 results (10k, 2 seeds):**
| method | PSNR mean | FID mean | util | ppl |
|---|---|---|---|---|
| vanilla_ema | 23.65 | 58.94 | 99.8% | 193 |
| drift_no_pp_ste | 23.49 | 57.45 | 99.4% | 378 |

Note: drift is slightly behind EMA at 10k on CIFAR-100 K=512. Slow convergence — same pattern as CIFAR-10. See 30k results in Phase CIFAR-100 section above.

**K>512 runs crashed** at step 1000 due to `wandb.Histogram` 512-bin limit (fixed in train.py). Relaunch config: `phase4_large_k_v2_relaunch.py` — covers K=1024/2048/4096, seeds 0+1, 10k steps. K=4096 is the key missing piece for the scaling curve.

**PCA anomaly note**: early plots showed drift codebook displaced from hiddens. Root cause: `save_pca_figure` was plotting raw (unnormalized) hiddens alongside L2-normalized codes. Fixed in train.py — hiddens now normalized to unit sphere before PCA projection.

---

## Phase Prior Pilot  STATUS: DONE — negative signal

**Question**: Does drift's more uniform codebook make a causal transformer prior easier to learn (lower NLL)?

**Config**: K=512, CIFAR-100, 2-layer d=128 prior, 3000 steps. Runs `prior_pilot` in `modal_app.py`.

| method | best val NLL (bits/code) |
|---|---|
| vanilla_ema | **5.8051** |
| drift_no_pp_ste | 6.9275 |

**Gap**: −1.12 bits/code — EMA is easier to model (opposite of what you might hope).

**Interpretation**: This is theoretically expected. EMA's Zipfian distribution (~112/512 effective codes) is easy to predict — a small prior memorizes the heavy hitters. Drift's near-uniform distribution (~395/512 effective codes, 6.93 / 9.0 = 77% of max entropy) requires the prior to model a much harder distribution. The gap is not due to bad priors but is structural. A larger prior may narrow the gap but is unlikely to close it.

**What this means for generation**: Drift's codes are harder to model autoregressively. Better reconstruction quality coexists with harder prior modeling — there's a tradeoff between reconstruction fidelity and prior compressibility. Worth one paragraph in the paper discussion.

**Next step for generation**: Linear probe (cheap, see below) rather than full prior generation.

---

## Codebook Geometry Findings  STATUS: NEW — from full recovery

These metrics are available in all aggregate.csv files after running `recover_wandb.py`. They come from `codebook_geometry()` and `hidden_geometry()` in `experiments/metrics.py`.

| metric | drift (CIFAR-100 K=512) | EMA (CIFAR-100 K=512) | interpretation |
|---|---|---|---|
| `hidden/hidden_norm_mean` | ~7.3–7.7 | ~0.37–0.39 | encoder output scale |
| `codebook/norm_mean` | ~1.04 | ~1.0 | codebook vector scale |
| `codebook/pair_dist_mean` | ~0.30–0.31 | ~1.20–1.44 | mean Euclidean dist between codes |
| `val/gini` | ~0.39–0.43 | ~0.72–0.76 | Zipfian concentration (lower=better) |

**Key geometric insight**: Drift trains an encoder that projects to very high-norm vectors (~7–8), while EMA encoder stays at low norm (~0.4). Since VQ assignment is `argmax_k(h · c_k)` for unit-sphere codes, high hidden norm creates very discriminative projections → every code gets used uniformly. EMA's low norm makes many inputs "equidistant" from multiple codes → Zipfian concentration.

Drift codes cluster in a small dense region of the sphere (pair_dist ≈ 0.3 vs theoretical √2 ≈ 1.41 for uniform unit sphere), while EMA codes span a larger range (~1.2). This suggests drift's U_pn attraction groups codes near the encoder's operating region, while U_nn ensures they don't collapse to a single point.

**Paper mechanistic story**: The high hidden norm is an emergent property — drift doesn't directly encourage it. The physics of U_pn (code attraction to encoder hiddens) plus reconstruction loss jointly drive the encoder to commit strongly (high norm) to specific code directions.

---

## Phase 6 — Final Headline Table  STATUS: NOT YET RUN

5 methods × 5 seeds × 2 datasets × 30k iters. Only after Phase 2b decides best drift variant.

---

## Phase Hparam Sweep  STATUS: DONE ✓ — 2026-05-28

**Results (CIFAR-10, K=512, 30k, no_pp_ste):**

Tau sweep (ew=1.0):
| tau | PSNR mean | FID mean |
|---|---|---|
| 0.05 | 23.66 | 55.16 |
| 0.1  | 23.76 | 52.64 |
| 0.3  | 24.14 | 49.73 |
| **1.0** | **24.21** | **49.33** |

Energy weight sweep (tau=1.0):
| ew  | PSNR mean | FID mean |
|---|---|---|
| 0.1 | 24.03 | 50.26 |
| **0.3** | 24.19 | **47.10** |
| **1.0** | **24.21** | 49.33 |
| 3.0 | 23.95 | 54.22 |

**Decisions:**
- **tau=1.0 is best** (monotonically better 0.05→1.0). Phase 2 found tau=0.1 best for FULL drift — does NOT carry over to no_pp_ste.
- **ew=1.0 best PSNR; ew=0.3 best FID** (47.10 vs 49.33, −2.2 pts). Trade-off: pick ew=1.0 for PSNR headline consistency.
- **Key insight**: Phase 2b (tau=1.0, ew=1.0) was already at the PSNR-optimal point. The sweep validates the headline result rather than improving it.
- `phase_optimized.py` updated: TAU=1.0, EW=1.0.

---

## Phase Optimized  STATUS: NOT YET RUN — wait for hparam sweep

Final 30k runs with tuned hyperparameters. Only drift_no_pp_ste (EMA baselines unchanged by tau).
Config: `phase_optimized.py` — 18 runs. Update TAU/EW constants at top before launching.

**Datasets**: CIFAR-10 K=512 (3), CIFAR-100 K=512+1024 (6), STL-10 K=512 (3), Tiny ImageNet K=512+1024 (6)

To launch: `python -m experiments.launch --phase phase_optimized`

---

## Linear Probe Evaluation  STATUS: IMPLEMENTED — awaiting results

**Rationale**: Downstream "so what?" given prior generation is not compelling (EMA 5.81 vs drift 6.93 bits/code — structural gap, see Phase Prior Pilot). Tests whether drift's more uniform codes carry more semantic information.

**Feature representation**: Token frequency histograms (bag-of-visual-words), NOT mean-pooled codebook vectors.
- Encode image → discrete token indices (B, H, W)
- Count frequency of each of the K codes → histogram (B, K), normalized by L=H*W
- Mean-pooling would collapse code identity (two very different images can have the same average embedding). Histograms preserve WHICH codes appear.
- Feature shape: (N, K) — K=512 or K=1024

**NOTE — killed run**: The first launch (`modal run experiments/modal_app.py::linear_probe`, 2026-05-28) was killed mid-run. It was using the old mean-pool implementation (output showed `d=64  K=512`), not histograms. The code in `experiments/linear_probe.py` was already fixed to histograms before this run — Modal picked up the pre-fix version because it launched before the edit. Relaunch with the fixed code produces (N, K) histogram features.

**Prediction**: Drift codes should give higher linear probe accuracy — near-uniform utilization means each code specializes in a distinct visual pattern. EMA's Zipfian distribution means ~372/512 codes rarely used → the 140 active codes are semantically overloaded.

**Commands**:
```bash
modal run experiments/modal_app.py::linear_probe           # K=512
modal run experiments/modal_app.py::linear_probe --k 1024  # K=1024
```
Results saved to `/vol/runs/phase_linear_probe/<run_id>/probe_summary.json`.

---

## Key design decisions and rationale

| decision | choice | why |
|---|---|---|
| U_nn required | always keep | without it: catastrophic collapse (12 ppl, 79 active codes) |
| U_pp | **ablate out** | hurts PSNR 0.87 dB, hurts FID 7 pts — forces encoder to spread unnaturally |
| gradient estimator | **STE preferred over rotation trick** | STE: +0.81 dB PSNR, -10.7 FID at 10k |
| tau | 0.1 slightly better | insensitive for full drift; confirmed at 10k |
| energy schedule | constant | anneal is unstable |
| dataset for Phase 2b | CIFAR-10 | consistent with Phase 1/2 for direct comparison |

---

## Analysis workflow

Always run these in order before analyzing any phase:

```bash
# 1. Quick live check (works mid-training, now shows gini + lpips + active_codes)
python -m experiments.scripts.inspect_runs --group <phase> --table

# 2. Recover full artifacts from wandb (run once per phase; --overwrite to refresh)
python -m experiments.scripts.recover_wandb --group <phase>

# 3. Full table with all metrics including geometry + drift energies
python -m experiments.scripts.aggregate --phase <phase>
```

If aggregate returns "no rows.", run step 2 first — runs are on Modal and need recovery.

---

## Wandb

Project: `drifting-vqvae-231n`  
Groups: `phase1_convergence`, `phase2_diagnostic`, `phase3_hybrids`, `phase2b_confirmation`, `phase_cifar100`, `phase_cifar100_k2048`, `phase_stl10`, `phase_tiny_imagenet`, `phase4_large_k_v2`
