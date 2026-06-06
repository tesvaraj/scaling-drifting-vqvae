# Downstream evaluation — run guide

Two downstream experiments on the frozen VQ-VAEs, Drift* vs Vanilla EMA. Both
read the existing `phase_cifar100` checkpoints off the Modal volume and ship the
code automatically via the image mount, so just `git checkout tesvara` and run.

- **A — CNN probe (transfer / recognition):** freeze the encoder + codebook, train
  a small CNN head on the latent grid, measure CIFAR-100 classification. Tests
  whether Drift*'s codes are better *features*.
- **B — Generation (image quality):** train an autoregressive prior over the codes,
  sample, decode, measure FID. Tests whether Drift*'s reconstruction edge survives
  *generation*, given that its codes are harder to model.

## Prereqs
- Modal access to the workspace that owns the `drifting-vqvae` volume
  (`modal token new`); the `wandb` secret is already set.
- Checkpoints exist at
  `/vol/runs/phase_cifar100/cifar100_{vanilla_ema,drift_no_pp_ste}_K{512,1024}_seed{0,1,2}/checkpoints/final.pt`.

## Experiment A — CNN probe
```bash
# cheap smoke first: confirms volume + checkpoint paths resolve
modal run experiments/modal_app.py::cnn_probe --k 512 --epochs 2 --include-controls False

# primary (quantized latents) + the two ablation representations
modal run experiments/modal_app.py::cnn_probe --k 512  --representation zq
modal run experiments/modal_app.py::cnn_probe --k 512  --representation codes
modal run experiments/modal_app.py::cnn_probe --k 512  --representation ze
modal run experiments/modal_app.py::cnn_probe --k 1024 --representation zq
```
Each call runs `{EMA, Drift*} x 3 seeds` plus two reference baselines (`random`
untrained backbone, raw `pixels`). Representations: `zq` (quantized latents,
**primary** — about the codebook), `ze` (pre-quant encoder output — encoder only),
`codes` (discrete tokens via a learned embedding).

**Reading it.** The entrypoint prints per-method `top1/top5` mean±std, the
`Δ (drift − EMA)` in points, and a verdict. Per-run JSON lands at
`/vol/runs/phase_cnn_probe/<run_id>/cnn_probe_summary.json`
(`run_id = cnnprobe_{ema,drift,random,pixels}_<rep>_<head>_K<k>_seed<s>`). Positive
Δtop1 means Drift* features transfer better; `random` is the lower reference (how
much pretraining helps), `pixels` the upper reference (how much tokenizing costs).
This is the CNN upgrade of `linear_probe.py` (which gave a weak +1.4pp signal).

## Experiment B — generation
```bash
modal run experiments/modal_app.py::prior_pilot          # ~5 min, NLL signal check
modal run experiments/modal_app.py::prior_full --k 512   # 3 seeds, full eval
modal run experiments/modal_app.py::prior_full --k 1024
```
`prior_full` reports two FIDs per run so the tokenizer and the prior can be told
apart, and sweeps sampling temperature (FID is very temperature-sensitive):
- **recon-FID** — decode the *real* test codes. The ceiling a perfect prior could
  reach; isolates tokenizer quality (Drift* expected to win here).
- **gen-FID** — decode codes *sampled* from the prior, best over temperature. The
  full pipeline. `prior-gap = gen − recon` is what the prior costs.

**Reading it.** The entrypoint prints `recon-FID`, `gen-FID`, `best T`, the
`prior-gap`, and both `Δ recon-FID` and `Δ gen-FID (drift − EMA)` with a verdict.
The open question: Drift* should win recon-FID, but its near-uniform codes are
harder to model (prior NLL 6.93 vs 5.81 bits/code in the pilot), so the gen-FID
verdict tells us whether the reconstruction edge survives. Per-run JSON at
`/vol/runs/phase_prior/<run_id>/prior_summary.json` (includes `gen_fid_by_temp`).

## Pulling results back
The printed summary tables are the headline. To grab the JSON locally:
```bash
modal volume get drifting-vqvae /runs/phase_cnn_probe ./runs/phase_cnn_probe
modal volume get drifting-vqvae /runs/phase_prior     ./runs/phase_prior
```

## Notes
- Compute is small: A is a few minutes per config (features cached once, tiny head);
  B's prior is ~20–30 min per run as configured.
- `models.decode_indices` was fixed to support the EMA quantizer's index→code API
  (`get_output_from_indices`), without which generation could not run for the
  baseline.
