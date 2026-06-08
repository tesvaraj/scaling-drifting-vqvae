# Experiment B results (generation)

CIFAR-100, K=512, frozen VQ-VAE, transformer prior over the codes, 3 seeds.
Arrows show the better direction (↓ lower is better, ↑ higher is better).

| method | NLL bits ↓ | recon-FID ↓ | gen-FID ↓ | best T | sample Gini ↓ |
|---|---|---|---|---|---|
| vanilla_ema | 5.59 | 65.8 | 91.1 | 0.93 | 0.78 |
| drift_no_pp_ste | 6.72 | 56.9 | 79.6 | 0.80 | 0.47 |
| Δ (drift − EMA) | +1.13 | −8.8 | −11.6 | | −0.31 |

- NLL: bits per code to model the codes with the prior (↓).
- recon-FID: decode the real test codes, i.e. the tokenizer ceiling (↓).
- gen-FID: decode codes sampled from the prior, best over temperature (↓).
- best T: sampling temperature with the lowest gen-FID (not better/worse).
- sample Gini: code-usage spread of the samples, 0 = uniform (↓).

Drift has higher NLL (its codes are harder to model) but lower recon-FID and
gen-FID. Drift is lower on recon-FID by 8.8 and on gen-FID by 11.6.

Figures on the volume:
- per run: /runs/phase_prior/<run_id>/figures/samples_grid.png and recon_grid.png
- side by side recon: /runs/phase_downstream_figs/recon_compare_K512_seed0.png

Regenerate the tables with: modal run experiments/modal_app.py::downstream_tables
