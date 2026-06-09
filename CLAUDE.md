# Scaling Drifting VQ-VAE — Agent Context

**CS231N Spring 2026 final project**

## What this project is

VQ-VAE with a physics-inspired codebook quantizer ("DriftingVQ") that replaces codebook+commitment loss with pairwise potential energy. Core question: can drift match or beat vanilla EMA VQ on reconstruction quality while achieving dramatically higher codebook utilization?

Key methods compared:
- `vanilla_ema` — EMA-updated codebook (k-means style), modern default
- `vanilla_classical` — gradient-based codebook
- `simvq` — SimVQ baseline
- `drift` — full DriftingVQ (U_pp + U_nn + U_pn)
- `drift_ema` — hybrid: EMA codebook placement + drift energy for encoder spread

## Repo layout

```
experiments/
  train.py          # TrainConfig dataclass + train() function
  launch.py         # CLI: python -m experiments.launch --phase <name> [--local] [--dry_run]
  modal_app.py      # Modal L40S GPU harness
  configs/          # phase configs — make_runs() returns list[TrainConfig]
  scripts/
    aggregate.py    # runs/<phase>/aggregate.{csv,md}
    status.py       # table of all runs in a phase
    diagnose.py     # deep-dive one run (--fetch_modal to pull from Modal)
    inspect_runs.py # live wandb queries
    kill_run.py     # cancel Modal call by run_id

runs/<phase>/<run_id>/
  config.json, summary.json, curves.csv, val_curves.csv
  figures/, checkpoints/
```

## Infrastructure

- **Modal**: authenticated as `emailhemalarora`, `wandb` secret present
- **W&B project**: `drifting-vqvae-231n`
- **Modal volume**: `drifting-vqvae` at `/vol` (runs write to `/vol/runs`)
- **Bug fixed**: first batch of Modal runs wrote to ephemeral `./runs` instead of `/vol/runs`. Fixed in `modal_app.py` (force-sets `out_root=/vol/runs`). Metrics recovered from W&B via `experiments/scripts/recover_wandb.py`.

## Common commands

```bash
# launch a phase on Modal (parallel, detached)
python -m experiments.launch --phase phase2b_confirmation

# dry run (print configs, no execution)
python -m experiments.launch --phase phase2b_confirmation --dry_run

# aggregate results to runs/<phase>/aggregate.md
python -m experiments.scripts.aggregate --phase phase2b_confirmation

# status table (local artifacts)
python -m experiments.scripts.status --phase phase2b_confirmation

# live wandb table
python -m experiments.scripts.inspect_runs --group phase2b_confirmation --table

# pull one run from Modal volume and diagnose
python -m experiments.scripts.diagnose --phase phase2b_confirmation \
    --run_id cifar10_drift_no_pp_K512_seed0 --fetch_modal
```

## Experimental status and next steps

See `experiments/CONTEXT.md` for full results, interpretations, and what to run next.

**TL;DR current state**: All phases complete. Headline result: `drift_no_pp_ste` beats `vanilla_ema` by +0.77 dB PSNR and −11.6 FID at 30k on CIFAR-10, with 3× codebook utilization. Result holds on CIFAR-100 (+0.72 dB) and STL-10 (+0.55 dB). Project submitted.
