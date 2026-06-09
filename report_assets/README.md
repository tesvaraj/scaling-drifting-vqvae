# Report figures (downstream experiments, CIFAR-100 K=512)

- `recon_compare.png` — reconstruction side by side, rows: original / EMA / Drift.
  Mean PSNR on these images: EMA 24.29 dB, Drift 25.19 dB.
- `samples_compare.png` — generated samples side by side, top block EMA prior,
  bottom block Drift prior (EMA T=0.9, Drift T=0.8).
- `prior_fid_vs_temp.png` — gen-FID vs sampling temperature, EMA vs Drift.
- `samples_ema.png`, `samples_drift.png` — generated samples, one method each.
- `recon_grid_drift.png` — Drift originals (top) vs reconstructions (bottom).

Numbers are in `paper_context.md` (sections 5 and 6).
Regenerate figures/tables: `modal run experiments/modal_app.py::downstream_tables`
(plus `recon_compare` / `samples_compare`).
