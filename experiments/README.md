# Experiments — Scaling Drifting VQ-VAE

This directory contains the experiment harness used for the final phase of
the CS 231N project. It is a complete rewrite of the original
`examples/autoencoder_drifting.py` script, organized so we can:

- Launch large parallel sweeps on Modal with one CLI invocation.
- Compare all four baselines (`vanilla_ema`, `vanilla_classical`, `simvq`,
  `drift`) plus the drift+EMA hybrid in the same harness.
- Track per-step training curves, validation metrics, and codebook
  geometry to disk *and* wandb.
- Introspect live runs (and dead ones) via small CLIs that read the
  on-disk artifacts and the Modal/wandb APIs.

## Layout

```
experiments/
├── data.py                     # dataset loaders
├── models.py                   # CNN encoder / decoder / VQAutoEncoder
├── quantizers.py               # factory + DriftEMAVQ hybrid
├── metrics.py                  # PSNR/SSIM/LPIPS/FID + codebook diagnostics
├── train.py                    # config-driven training loop
├── launch.py                   # CLI to launch a phase locally or on Modal
├── modal_app.py                # Modal app + remote run function
├── configs/
│   ├── base.py
│   ├── phase0_smoke.py         # 5-method smoke test (~5 min total, CPU/MPS)
│   ├── phase1_convergence.py   # 30k-iter convergence (the headline run)
│   ├── phase2_diagnostic.py    # ablate U_pp / U_nn / L2-norm / tau / STE
│   ├── phase3_hybrids.py       # drift_ema + warmup + commit + anneal
│   ├── phase4_large_k.py       # K ∈ {1024..8192} stress test
│   ├── phase5_downstream.py    # autoregressive prior on codes (TBD)
│   └── phase6_final.py         # multi-seed headline tables
└── scripts/
    ├── status.py               # what's running / done / failed
    ├── diagnose.py             # deep-dive one run (curves + figures)
    ├── inspect_runs.py         # pull live data from wandb
    ├── kill_run.py             # cancel a Modal call by run_id
    └── aggregate.py            # combine a phase into a markdown table
```

## Per-run output

Each run writes a self-contained directory:

```
runs/<phase>/<run_id>/
    config.json           # exact TrainConfig used (incl. defaults)
    summary.json          # final metrics (last validation pass)
    curves.csv            # per-log-step training metrics
    val_curves.csv        # per-validation-step metrics + geometry
    wandb_url.txt         # wandb URL or "disabled"
    figures/
        recons_<step>.png         # paired original vs reconstruction grid
        codebook_pca_<step>.png   # 2D PCA of codes + sample of hiddens
        usage_hist_<step>.png     # sorted code-usage bar chart
    checkpoints/
        ckpt_<step>.pt            # every cfg.ckpt_every steps
        final.pt
```

This format is meant to be machine-readable: a code assistant or user
can `cat`, `tail`, or parse any of these directly without scraping
wandb.

## One-time setup

```bash
# python deps
pip install -e .                      # the vector_quantize_pytorch package
pip install modal wandb torchmetrics lpips matplotlib scipy fire

# auth
wandb login                           # writes ~/.netrc
modal token new                       # writes ~/.modal.toml
modal secret create wandb WANDB_API_KEY=$(cat ~/.netrc | grep password | awk '{print $2}')
```

## Run a smoke test (~5 minutes, no Modal)

```bash
python -m experiments.launch --phase phase0_smoke --local
python -m experiments.scripts.aggregate --phase phase0_smoke
```

If that produces `runs/phase0_smoke/aggregate.md` with five rows, you are
ready to launch on Modal.

## Launch a phase on Modal

```bash
# parallel spawn, detached
python -m experiments.launch --phase phase1_convergence

# block until done
python -m experiments.launch --phase phase1_convergence --detach=False

# filter by substring (e.g. drift-only)
python -m experiments.launch --phase phase3_hybrids --only drift_ema

# limit for testing
python -m experiments.launch --phase phase4_large_k --limit 2
```

After spawning, function-call ids are saved to
`runs/<phase>/_handles.json` so individual runs can be cancelled.

## Monitor live runs

```bash
# local on-disk view (works any time)
python -m experiments.scripts.status --phase phase1_convergence

# add Modal state polling (running / done / failed)
python -m experiments.scripts.status --phase phase1_convergence --modal=True

# pull from wandb (works while training)
python -m experiments.scripts.inspect_runs --group phase1_convergence --table
python -m experiments.scripts.inspect_runs --run cifar10_drift_K512_seed0
```

## Drill into one run

```bash
# uses on-disk artifacts in runs/<phase>/<run_id>
python -m experiments.scripts.diagnose \
    --phase phase1_convergence --run_id cifar10_drift_K512_seed0 --tail 5

# pull the artifacts from the Modal volume first
python -m experiments.scripts.diagnose \
    --phase phase1_convergence --run_id cifar10_drift_K512_seed0 --fetch_modal
```

## Kill a bad run

```bash
python -m experiments.scripts.kill_run \
    --phase phase4_large_k --run_id cifar10_drift_K8192_seed0
```

## Aggregate a phase into a table

```bash
python -m experiments.scripts.aggregate --phase phase2_diagnostic
# writes runs/phase2_diagnostic/aggregate.csv + aggregate.md
```

## Adding a new run / new method

A method is added in three places:

1. Implement (or alias) the quantizer in `quantizers.py::build_quantizer`.
2. Reference it by string name from a config (`make_runs()` in a phase).
3. (Optional) add a token like `'<name>'` to `phase6_final.py::METHODS`
   when ready for the headline.

## What each phase answers

| Phase | Question | Decisive output |
|---|---|---|
| 1 (convergence) | Does the PSNR gap close at convergence? | `aggregate.md` PSNR/FID @ 30k vs current 2k |
| 2 (diagnostic) | Which drift term hurts PSNR? | Which ablation closes the gap most |
| 3 (hybrids) | Can drift_ema beat vanilla_ema? | PSNR / FID of `drift_ema` vs `vanilla_ema` |
| 4 (large-K) | Where does EMA collapse + drift survives? | Utilization curve vs K |
| 5 (downstream) | Does drift's coverage win sample-FID? | Sample-FID of decoded prior samples |
| 6 (final) | Headline numbers with ±std | Multi-seed table |

The plan, narrative, and per-phase rationale live in the milestone PDF
and the planning chat — not in this README.
