# Teammate Guide — Running Experiments & Getting Metrics

**W&B project**: `drifting-vqvae-231n` (Hemal's account, team has read access)
**Compute**: Modal L40S GPUs, authenticated as `emailhemalarora`
**Results land in**: Modal volume `/vol/runs` → local `runs/<phase>/`

---

## 1. Running an Experiment

### Standard launch (Modal, parallel, detached)
```bash
conda activate vqvae
python -m experiments.launch --phase <phase_name>
```

### Dry run first — always check before burning GPU time
```bash
python -m experiments.launch --phase <phase_name> --dry_run
```
This prints all configs without submitting anything. Verify `run_id`, `method`,
`codebook_size`, `train_iter`, and `dataset` look right.

### Local run (single config, for debugging)
```bash
python -m experiments.launch --phase <phase_name> --local
```
Runs sequentially on your machine. Only use for smoke tests — slow.

---

## 2. Monitoring Live Runs

### Quick live table (works mid-training, no local artifacts needed)
```bash
python -m experiments.scripts.inspect_runs --group <phase_name> --table
```
Shows PSNR, FID, perplexity, Gini, active_codes, utilization for all runs in the group.
The `--group` flag matches the `phase` field in the config (not the config filename).

### W&B directly
Go to `wandb.ai → drifting-vqvae-231n → runs` and filter by Group = `<phase_name>`.
Key charts to watch: `val/psnr`, `val/rfid`, `val/perplexity`, `val/gini`.

### Check Modal run status
```bash
modal app list          # see running apps
modal app logs <app-id> # stream logs
```

---

## 3. Pulling Artifacts After Runs Finish

Runs on Modal write to the Modal volume. Pull them to local disk for analysis:

```bash
# Step 1: recover metrics from W&B into local runs/<phase>/
python -m experiments.scripts.recover_wandb --group <phase_name>

# Step 2: generate the aggregate CSV + markdown table
python -m experiments.scripts.aggregate --phase <phase_name>

# Results in: runs/<phase_name>/aggregate.csv  and  runs/<phase_name>/aggregate.md
```

Always do both steps in order. `aggregate` reads from `runs/`, which `recover_wandb` populates.

If a phase already has artifacts but you want fresh data (e.g. runs just finished):
```bash
python -m experiments.scripts.recover_wandb --group <phase_name> --overwrite
python -m experiments.scripts.aggregate --phase <phase_name>
```

---

## 4. What Gets Logged Automatically

All of these are logged to W&B and recovered by `recover_wandb`. Do not log them manually.

### Validation metrics (logged every `val_every` steps, default 1000)
| key | what it is |
|---|---|
| `val/psnr` | Reconstruction quality (dB) — primary metric |
| `val/rfid` | rFID (lower is better) |
| `val/ssim` | Structural similarity |
| `val/lpips` | Perceptual similarity |
| `val/perplexity` | exp(codebook entropy); range 1–K; higher = more uniform |
| `val/gini` | Gini coefficient of code usage; 0=uniform, 1=one code dominates |
| `val/utilization` | Fraction of codes with ≥1 assignment |
| `val/active_codes` | Raw count of codes used |

### Codebook geometry (logged at validation)
| key | what it is |
|---|---|
| `codebook/norm_mean` | Mean L2 norm of codebook vectors (should be ≈1 on unit sphere) |
| `codebook/pair_dist_mean` | Mean Euclidean distance between code pairs |
| `codebook/pair_dist_min` | Minimum inter-code distance |
| `codebook/pair_dist_std` | Std of inter-code distances |
| `hidden/hidden_norm_mean` | Mean L2 norm of encoder output **before** normalization |

### DriftingVQ energy terms (drift methods only)
| key | what it is |
|---|---|
| `drift/U_nn` | Code-code repulsion energy (should be positive/negative depending on convention) |
| `drift/U_pn` | Hidden-code attraction energy |
| `drift/U_pp` | Hidden-hidden repulsion energy (0.0 for no_pp variants) |
| `drift/U_total` | Sum of active terms |

### Training diagnostics
| key | what it is |
|---|---|
| `train/psnr` | Training PSNR (noisier than val, but useful for convergence check) |
| `train/rec_loss` | Reconstruction loss |
| `train/total_loss` | Total loss |
| `train/grad_norm` | Gradient norm — if this explodes, the run is likely crashing |
| `opt/lr` | Learning rate |
| `opt/energy_weight` | Energy weight (if NaN, the run has crashed) |

---

## 5. Defining a New Phase

Create `experiments/configs/<phase_name>.py`. Follow the existing config files as templates
(e.g. `phase_cifar100.py`). The key fields:

```python
PHASE = 'my_phase_name'   # must match --phase and --group flags

base = TrainConfig(
    phase=PHASE,
    dataset='cifar10',          # 'cifar10', 'cifar100', 'stl10', 'tiny_imagenet'
    train_iter=30000,
    codebook_size=512,
    batch_size=128,
    log_every=50,               # train metrics logged every N steps
    val_every=1000,             # val metrics logged every N steps
    eval_fid=True,              # FID is expensive; set False for quick runs
    eval_ssim=True,
    eval_lpips=True,
    tags=['cifar10', 'my_tag'], # W&B tags — add descriptive tags
)
```

**Naming convention for `run_id`**:
```
{dataset}_{method}_{variant}_{K}_{seed}
# e.g. cifar10_drift_no_pp_ste_K512_seed0
```
`run_id` is used as the W&B run name and the local directory name. Keep it unique and
descriptive — it's the primary identifier in aggregate tables.

---

## 6. Checking If a Run Is Clean

Before including a run in any table or figure, verify:

1. **`train_iter` reached target**: check the `step` field in `runs/<phase>/<run_id>/summary.json`
   ```bash
   cat runs/<phase>/<run_id>/summary.json | python3 -m json.tool | grep '"step"'
   ```
   Should equal `train_iter` (e.g. 30000).

2. **No NaN metrics**: if `val/psnr` is missing or `opt/energy_weight = NaN` in
   `summary.json`, the run crashed.

3. **Sanity check PSNR vs perplexity**: for drift methods at convergence (30k):
   - PSNR should be >23 dB on CIFAR-10, >23.5 dB on CIFAR-100
   - Perplexity should be >300 for K=512 (if <<100, the codebook may have collapsed)
   - util should be >99% (if <<90%, likely collapsing)

4. **Codebook collapse flag**: if `val/active_codes` drops below 50% of K and
   `val/gini` > 0.85, the run has collapsed. Exclude it.

5. **Cross-seed consistency**: for a method with 3 seeds, PSNR std > 0.5 dB is a red flag
   (suggests at least one seed diverged). Investigate before including the mean.

---

## 7. Common Pitfalls — Learn From These

| Pitfall | What happened | How to avoid |
|---|---|---|
| Runs wrote to ephemeral disk | Early Modal runs wrote to `./runs` instead of `/vol/runs`; data lost after container shut down | Fixed in `modal_app.py`. Don't override `out_root` unless you know what you're doing. |
| `wandb.Histogram` 512-bin limit | K>512 runs crashed at step ~1000 because the histogram logger couldn't handle codebooks larger than 512 | Fixed in `train.py`. Make sure you're on the latest main. |
| EMA K=512 seed crash (Tiny ImageNet) | One run hit a NaN `energy_weight` at step 600 and silently failed | Check `summary.json` step count after runs finish. |
| drift_anneal instability | Linear energy annealing caused 1/3 seeds to partially collapse | Use constant energy schedule for all serious runs. |
| Linear probe with mean-pool features | First probe run used wrong feature type | Features must be bag-of-visual-words histograms (N, K), not mean-pooled (N, d). |
| Low τ energy terms vanishing | At τ≤0.1, U_pn ≈ 0 and hidden_norm stays ~2 — drift is effectively off | Use τ=1.0 for all no_pp_ste runs. |
| Reporting utilization % instead of perplexity | EMA has >95% "active" codes but most are barely used | Always report `val/perplexity` and `val/gini` alongside or instead of `val/utilization`. |

---

## 8. Figures and Tables Workflow

### Generating an aggregate table for a finished phase
```bash
python -m experiments.scripts.aggregate --phase <phase_name>
# → runs/<phase_name>/aggregate.csv   (machine-readable)
# → runs/<phase_name>/aggregate.md    (human-readable markdown table)
```

### Deep-diving a single run
```bash
python -m experiments.scripts.diagnose --phase <phase_name> --run_id <run_id>
# Add --fetch_modal to pull full checkpoints from Modal volume (needed for inference)
```

### Checking status of all runs in a phase
```bash
python -m experiments.scripts.status --phase <phase_name>
```

### Killing a stuck Modal run
```bash
python -m experiments.scripts.kill_run --run_id <run_id>
```

---

## 9. What the Paper Needs

When running new experiments for the paper, make sure:

1. **At least 3 seeds** for any result that goes in a table. Single-seed ablations are
   fine for Phase 2-style diagnostics but headline numbers need 3 seeds.

2. **30k iters for final results**. 10k is fine for ablations and scaling curves (ppl/K
   shape), but any PSNR/FID comparison in the main table should be 30k.

3. **Always include a vanilla_ema baseline** at the same K and dataset, with the same
   number of seeds, run in the same phase config. This ensures apples-to-apples comparison
   even if the dataset loading or normalization changes.

4. **Log both `eval_ssim=True` and `eval_lpips=True`** in final-table phases. These are
   disabled by default in some early configs to save time, but reviewers may ask for them.

5. **Use the run_id convention** from Section 5. The aggregate script groups by
   (method, dataset, K, seed) from the CSV columns, not from the run_id. But the run_id
   must be unique or overwrite will silently corrupt artifacts.

6. **Tag runs** with at least `[dataset, phase_name]` in the config tags list. This makes
   W&B filtering much easier when you have 100+ runs.

---

## 10. Quick Reference

```bash
# Launch
python -m experiments.launch --phase <phase> --dry_run   # check first
python -m experiments.launch --phase <phase>              # submit to Modal

# Monitor
python -m experiments.scripts.inspect_runs --group <phase> --table

# After runs finish
python -m experiments.scripts.recover_wandb --group <phase>
python -m experiments.scripts.aggregate --phase <phase>

# Dig into one run
python -m experiments.scripts.diagnose --phase <phase> --run_id <id>

# Verify a run completed
cat runs/<phase>/<run_id>/summary.json | python3 -m json.tool | grep -E '"step"|"val/psnr"'
```

W&B project URL: https://wandb.ai/hemal1-stanford-university/drifting-vqvae-231n
