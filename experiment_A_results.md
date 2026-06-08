# Experiment A results (CNN probe)

CIFAR-100, K=512, frozen VQ-VAE, small CNN head on the latents (zq), 3 seeds.
Arrows show the better direction (↑ higher is better).

| method | top-1 ↑ | top-5 ↑ |
|---|---|---|
| vanilla_ema | 27.6 ± 0.3% | 56.0 ± 0.4% |
| drift_no_pp_ste | 28.2 ± 0.2% | 57.3 ± 0.5% |
| Δ (drift − EMA) | +0.63 pp | +1.3 pp |

Top-1 per seed:
- EMA: 27.9 / 27.3 / 27.7
- Drift: 28.3 / 28.0 / 28.5

Every drift seed beats every EMA seed.

Baselines (seed 0):
- random encoder: 13.4% top-1
- raw pixels: 54.1% top-1

Reconstruction figure (original / EMA / Drift): on the volume at
/runs/phase_downstream_figs/recon_compare_K512_seed0.png
Mean PSNR on those images: EMA 24.29 dB, Drift 25.19 dB.

Not run yet: ze and codes representations. Experiment B is running now.
