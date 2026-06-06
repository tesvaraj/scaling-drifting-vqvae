"""Modal harness for the Scaling Drifting VQ-VAE project.

Architecture
------------
* A single Modal Volume ``drifting-vqvae`` holds the dataset cache (``/vol/data``)
  and run outputs (``/vol/runs``). Datasets download once and persist.
* A single image carries all deps: torch, einops, einx, lpips, torchmetrics,
  wandb, plus the local source code mounted into ``/root/scaling_drifting_vqvae``.
* ``run_experiment`` runs one ``TrainConfig`` to completion on an L40S.
* ``launch_phase`` is a local CLI entrypoint that imports a phase config,
  builds the ``TrainConfig`` list, and dispatches them as Modal calls.

Setup
-----
    pip install modal
    modal token new                # one-time
    modal secret create wandb WANDB_API_KEY=...
    modal volume create drifting-vqvae   # auto-created by from_name(create_if_missing=True)

Run
---
    # smoke test (no Modal)
    python -m experiments.launch --phase phase0_smoke --local

    # phase 1, on Modal
    python -m experiments.launch --phase phase1_convergence

The two paths share the same ``TrainConfig`` plumbing — locally we call
``experiments.train.train``; remotely we call ``run_experiment.remote``
with the same dataclass serialized to a dict.
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = 'drifting-vqvae-231n'
VOLUME_NAME = 'drifting-vqvae'

# ----- image -----

LOCAL_ROOT = Path(__file__).resolve().parent.parent  # repo root

image = (
    modal.Image.debian_slim(python_version = '3.11')
    .apt_install('git')
    .pip_install(
        'torch>=2.4',
        'torchvision',
        'einops>=0.8.0',
        'einx>=0.3.0',
        'fire',
        'tqdm',
        'wandb',
        'numpy',
        'matplotlib',
        'lpips',
        'torchmetrics[image]',
        'scipy',
        'gdown',
        'scikit-learn',
    )
    # add the local repo so `experiments` and `vector_quantize_pytorch` are importable
    .add_local_dir(
        str(LOCAL_ROOT),
        '/root/scaling_drifting_vqvae',
        ignore = ['runs/*', '.git/*', '.pytest_cache/*', '__pycache__/*',
                  '**/__pycache__/*', '*.pdf'],
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing = True)

app = modal.App(APP_NAME)


# ----- remote function -----

@app.function(
    image = image,
    gpu = os.environ.get('MODAL_GPU', 'L40S'),
    volumes = {'/vol': volume},
    secrets = [modal.Secret.from_name('wandb')],
    timeout = 6 * 60 * 60,  # 6h
)
def run_experiment(config_dict: dict) -> dict:
    """Run one TrainConfig (serialized as dict) inside a Modal container."""
    import sys
    sys.path.insert(0, '/root/scaling_drifting_vqvae')

    from experiments.train import TrainConfig, train

    # Rewrite paths to live on the persistent volume so data and run artifacts
    # survive after the ephemeral Modal container exits. ``asdict`` includes the
    # local defaults, so this must override rather than setdefault.
    config_dict['data_root'] = '/vol/data'
    config_dict['out_root'] = '/vol/runs'

    # cast list back to tuple for fields that expect it
    if 'energy_terms' in config_dict and isinstance(config_dict['energy_terms'], list):
        config_dict['energy_terms'] = tuple(config_dict['energy_terms'])

    cfg = TrainConfig(**config_dict)
    summary = train(cfg)

    # commit volume so writes are persisted
    try:
        volume.commit()
    except Exception:
        pass
    return summary


# ----- helpers to pull results back locally -----

@app.function(image = image, volumes = {'/vol': volume}, timeout = 600)
def list_runs(phase: str = '') -> list:
    """List run directories under /vol/runs/<phase> (or all phases if empty)."""
    base = Path('/vol/runs')
    if not base.exists():
        return []
    out = []
    for ph_dir in sorted(base.iterdir()):
        if not ph_dir.is_dir():
            continue
        if phase and ph_dir.name != phase:
            continue
        for run_dir in sorted(ph_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            summary_path = run_dir / 'summary.json'
            ckpt_dir = run_dir / 'checkpoints'
            curves = run_dir / 'curves.csv'
            out.append({
                'phase': ph_dir.name,
                'run_id': run_dir.name,
                'has_summary': summary_path.exists(),
                'n_checkpoints': len(list(ckpt_dir.glob('*.pt'))) if ckpt_dir.exists() else 0,
                'curves_size': curves.stat().st_size if curves.exists() else 0,
            })
    return out


@app.function(image = image, volumes = {'/vol': volume}, timeout = 600)
def fetch_run_summary(phase: str, run_id: str) -> dict:
    import json
    path = Path('/vol/runs') / phase / run_id / 'summary.json'
    if not path.exists():
        return {'error': f'no summary at {path}'}
    with open(path) as f:
        return json.load(f)


@app.function(image = image, volumes = {'/vol': volume}, timeout = 600)
def fetch_curves(phase: str, run_id: str) -> str:
    path = Path('/vol/runs') / phase / run_id / 'curves.csv'
    if not path.exists():
        return ''
    return path.read_text()


@app.function(image = image, volumes = {'/vol': volume}, timeout = 600)
def fetch_val_curves(phase: str, run_id: str) -> str:
    path = Path('/vol/runs') / phase / run_id / 'val_curves.csv'
    if not path.exists():
        return ''
    return path.read_text()


@app.function(image = image, volumes = {'/vol': volume}, timeout = 1800)
def preload_dataset(name: str) -> str:
    """Download a dataset to the persistent volume so training runs don't race to download it.

    Usage:
        modal run experiments/modal_app.py::preload_dataset --name celeba
    """
    import sys
    sys.path.insert(0, '/root/scaling_drifting_vqvae')
    from experiments.data import build_dataset
    ds = build_dataset(name, data_root = '/vol/data')
    volume.commit()
    return f'ok: {name} train={len(ds.train)} val={len(ds.val)}'


@app.local_entrypoint()
def preload(name: str = 'celeba'):
    result = preload_dataset.remote(name)
    print(result)


@app.local_entrypoint()
def preload_tiny_imagenet():
    """Download + reorganise Tiny ImageNet into the Modal volume (run once before training).

    Usage:
        modal run experiments/modal_app.py::preload_tiny_imagenet
    """
    result = preload_dataset.remote('tiny_imagenet')
    print(result)


# ----- per-class reconstruction analysis -----

@app.function(image = image, gpu = 'L40S', volumes = {'/vol': volume}, timeout = 600)
def _run_per_class_analysis(phase: str, ema_run_id: str, drift_run_id: str) -> dict:
    import sys, math
    import torch, torch.nn.functional as F
    from torch.utils.data import DataLoader
    sys.path.insert(0, '/root/scaling_drifting_vqvae')
    from experiments.prior import load_vqvae_frozen
    from experiments.data import build_dataset

    device = 'cuda'
    results = {}

    for label, run_id in [('ema', ema_run_id), ('drift', drift_run_id)]:
        ckpt = f'/vol/runs/{phase}/{run_id}/checkpoints/final.pt'
        model, cfg = load_vqvae_frozen(ckpt, device)
        ds = build_dataset('cifar100', '/vol/data')
        K = cfg.codebook_size

        # Val set: 10k images, 100 classes x 100 images each (ordered by class)
        loader = DataLoader(ds.val, batch_size=200, shuffle=False, num_workers=2)

        per_class_psnr = []       # shape: (100,)
        per_class_ppl  = []       # effective codes used per class
        global_code_counts = torch.zeros(K, dtype=torch.long)

        for batch_idx, batch in enumerate(loader):
            x, class_labels = batch[0].to(device), batch[1]
            with torch.no_grad():
                recon, indices, _ = model(x)
                recon = recon.clamp(-1, 1)

            # per-image PSNR
            mse = ((recon - x) ** 2).mean(dim=(1, 2, 3))
            img_psnr = -10 * torch.log10(mse + 1e-8) + 20 * math.log10(2.0)

            # group by class within this batch
            for cls in class_labels.unique():
                mask = class_labels == cls
                cls_psnr = img_psnr[mask].mean().item()
                cls_codes = indices[mask].reshape(-1)
                counts = cls_codes.bincount(minlength=K).float()
                # perplexity of code distribution for this class
                probs = counts / counts.sum().clamp(min=1)
                ent = -(probs * (probs + 1e-8).log()).sum()
                ppl = math.exp(ent.item())
                per_class_psnr.append((cls.item(), cls_psnr))
                per_class_ppl.append((cls.item(), ppl))

            global_code_counts += indices.reshape(-1).cpu().bincount(minlength=K)

        # sort by class id
        per_class_psnr.sort()
        per_class_ppl.sort()
        psnr_vals = [v for _, v in per_class_psnr]
        ppl_vals  = [v for _, v in per_class_ppl]

        # correlation: do classes that get fewer distinct codes have worse PSNR?
        import numpy as np
        corr = float(np.corrcoef(ppl_vals, psnr_vals)[0, 1])

        # Gini of global code distribution
        c = global_code_counts.float()
        c_sorted, _ = c.sort()
        n = c.shape[0]
        idx = torch.arange(1, n + 1, dtype=torch.float)
        gini = float((2 * (idx * c_sorted).sum() / (n * c_sorted.sum()) - (n + 1) / n).item())

        results[label] = {
            'psnr_per_class': psnr_vals,
            'ppl_per_class': ppl_vals,
            'psnr_mean': float(np.mean(psnr_vals)),
            'psnr_std': float(np.std(psnr_vals)),
            'psnr_p10': float(np.percentile(psnr_vals, 10)),   # worst-10% classes
            'ppl_class_mean': float(np.mean(ppl_vals)),
            'corr_ppl_psnr': corr,
            'global_gini': gini,
            'global_ppl': float(2 ** (-(c / c.sum().clamp(min=1) * (c / c.sum().clamp(min=1) + 1e-8).log2()).sum().item())),
        }
        print(f'[{label}] mean PSNR={results[label]["psnr_mean"]:.3f}  '
              f'p10 PSNR={results[label]["psnr_p10"]:.3f}  '
              f'Gini={gini:.3f}  corr(ppl,psnr)={corr:.3f}')

    # save to volume
    import json
    from pathlib import Path
    out = Path(f'/vol/runs/analysis')
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f'per_class_{phase}.json', 'w') as f:
        json.dump(results, f, indent=2)
    volume.commit()
    return results


@app.local_entrypoint()
def per_class_analysis(phase: str = 'phase_cifar100', k: int = 1024, seed: int = 0):
    """Per-class PSNR analysis: does EMA serve rare classes worse than drift?

    Usage:
        modal run experiments/modal_app.py::per_class_analysis
        modal run experiments/modal_app.py::per_class_analysis --k 512
    """
    ema_run   = f'cifar100_vanilla_ema_K{k}_seed{seed}'
    drift_run = f'cifar100_drift_no_pp_ste_K{k}_seed{seed}'

    print(f'Running per-class analysis: {phase}  K={k}  seed={seed}')
    res = _run_per_class_analysis.remote(phase, ema_run, drift_run)

    import numpy as np
    print(f"\n{'='*60}")
    print(f"PER-CLASS ANALYSIS  K={k}  (100 CIFAR-100 classes x 100 val images)")
    print(f"{'metric':<35} {'EMA':>10} {'Drift':>10}")
    print('-'*60)

    metrics = [
        ('mean PSNR (all classes)', 'psnr_mean'),
        ('std PSNR across classes', 'psnr_std'),
        ('p10 PSNR (worst 10% classes)', 'psnr_p10'),
        ('mean per-class code ppl', 'ppl_class_mean'),
        ('corr(class ppl, class PSNR)', 'corr_ppl_psnr'),
        ('global Gini', 'global_gini'),
    ]
    for label, key in metrics:
        ev = res['ema'][key]
        dv = res['drift'][key]
        print(f"  {label:<33} {ev:>10.3f} {dv:>10.3f}")

    print()
    print('Interpretation:')
    psnr_gap = res['drift']['psnr_p10'] - res['ema']['psnr_p10']
    std_gap  = res['ema']['psnr_std'] - res['drift']['psnr_std']
    print(f'  Worst-10%-class PSNR gap: drift {psnr_gap:+.3f} dB over EMA')
    print(f'  EMA std across classes {std_gap:+.3f} higher than drift (more uneven per-class quality)')
    ema_corr = res['ema']['corr_ppl_psnr']
    print(f'  EMA corr(class ppl, class PSNR) = {ema_corr:.3f}  '
          + ('(classes with fewer distinct codes → worse reconstruction)' if ema_corr > 0.1 else '(weak signal)'))
    print('='*60)


# ----- token prior experiment -----

@app.function(
    image = image,
    gpu = os.environ.get('MODAL_GPU', 'L40S'),
    volumes = {'/vol': volume},
    secrets = [modal.Secret.from_name('wandb')],
    timeout = 2 * 60 * 60,
)
def run_prior(config_dict: dict) -> dict:
    """Train one token prior (PriorConfig serialized as dict) inside a Modal container."""
    import sys
    sys.path.insert(0, '/root/scaling_drifting_vqvae')
    from experiments.prior import PriorConfig, train_prior

    config_dict['data_root'] = '/vol/data'
    config_dict['out_root'] = '/vol/runs'
    config_dict['vqvae_root'] = '/vol/runs'

    cfg = PriorConfig(**config_dict)
    summary = train_prior(cfg)
    try:
        volume.commit()
    except Exception:
        pass
    return summary


@app.local_entrypoint()
def prior_pilot(phase: str = 'phase_cifar100', k: int = 512,
                pilot_steps: int = 3000):
    """3k-step pilot on a tiny 2-layer prior: does drift give lower NLL than EMA?

    Runs EMA and drift in parallel, prints a VERDICT in ~5 minutes.

    Usage:
        modal run experiments/modal_app.py::prior_pilot
        modal run experiments/modal_app.py::prior_pilot --k 1024
    """
    ema_run_id = f'cifar100_vanilla_ema_K{k}_seed0'
    drift_run_id = f'cifar100_drift_no_pp_ste_K{k}_seed0'

    base = dict(
        phase = 'phase_prior_pilot',
        dataset = 'cifar100',
        train_steps = pilot_steps,
        d_model = 128, n_layers = 2, n_heads = 4,  # tiny — just checking signal
        batch_size = 256, lr = 3e-4, seed = 0,
        log_every = 100, val_every = 500,
        eval_gen_fid = False,
        wandb_mode = 'disabled',
        vqvae_phase = phase, vqvae_ckpt = 'final.pt',
    )

    ema_cfg   = {**base, 'run_id': f'pilot_ema_K{k}',   'vqvae_run_id': ema_run_id}
    drift_cfg = {**base, 'run_id': f'pilot_drift_K{k}', 'vqvae_run_id': drift_run_id}

    print(f'Launching pilot: EMA vs drift_no_pp_ste  K={k}  {pilot_steps} steps each...')
    ema_call   = run_prior.spawn(ema_cfg)
    drift_call = run_prior.spawn(drift_cfg)

    ema_result   = ema_call.get()
    drift_result = drift_call.get()

    ema_nll   = ema_result['best_val_nll']
    drift_nll = drift_result['best_val_nll']
    gap = ema_nll - drift_nll

    print(f"\n{'='*58}")
    print(f"PRIOR PILOT  (K={k}, {pilot_steps} steps, 2-layer d=128 prior)")
    print(f"  EMA   best val NLL : {ema_nll:.4f} bits/code")
    print(f"  Drift best val NLL : {drift_nll:.4f} bits/code")
    print(f"  Gap                : {gap:+.4f} bits/code  (+ = drift easier to model)")
    ema_curve_str   = ['{:.3f}'.format(r['val_nll']) for r in ema_result['nll_curve']]
    drift_curve_str = ['{:.3f}'.format(r['val_nll']) for r in drift_result['nll_curve']]
    print(f"  NLL curves (EMA)   : {ema_curve_str}")
    print(f"  NLL curves (drift) : {drift_curve_str}")
    print()
    if gap > 0.05:
        pct = gap / ema_nll * 100
        print(f"  VERDICT: ✓  Signal present ({pct:.1f}% lower NLL) — run full experiment")
    elif gap > 0.01:
        print(f"  VERDICT: ~  Weak signal — run full experiment but temper expectations")
    else:
        print(f"  VERDICT: ✗  No signal at pilot scale — full experiment unlikely compelling")
    print('='*58)


@app.local_entrypoint()
def prior_full(phase: str = 'phase_cifar100', k: int = 512):
    """Full 3-seed prior experiment with gen FID. Run after prior_pilot confirms signal.

    Usage:
        modal run experiments/modal_app.py::prior_full
        modal run experiments/modal_app.py::prior_full --k 1024
    """
    configs = []
    for method, vqvae_suffix in [('ema', f'vanilla_ema'), ('drift', f'drift_no_pp_ste')]:
        for seed in [0, 1, 2]:
            vqvae_run_id = f'cifar100_{vqvae_suffix}_K{k}_seed{seed}'
            configs.append(dict(
                run_id = f'prior_{method}_K{k}_seed{seed}',
                phase = 'phase_prior',
                dataset = 'cifar100',
                vqvae_phase = phase,
                vqvae_run_id = vqvae_run_id,
                vqvae_ckpt = 'final.pt',
                d_model = 256, n_layers = 4, n_heads = 8, dropout = 0.0,
                train_steps = 20000, batch_size = 256, lr = 3e-4,
                seed = seed,
                log_every = 200, val_every = 1000,
                eval_gen_fid = True, n_gen_samples = 2000, gen_temperature = 1.0,
                wandb_mode = 'online',
            ))

    print(f'Launching {len(configs)} prior runs (K={k}, 20k steps, gen FID)...')
    calls = [run_prior.spawn(cfg) for cfg in configs]
    results = [c.get() for c in calls]

    # summary table
    import numpy as np

    def _g(r, key):
        v = r.get(key)
        return float('nan') if v is None else v

    print(f"\n{'='*84}")
    print(f"PRIOR FULL RESULTS  (K={k}, phase={phase}, 20k steps)")
    print(f"{'run_id':<32} {'val NLL':>9} {'recon FID':>10} {'gen FID':>9} "
          f"{'best T':>7} {'smp Gini':>9}")
    print('-'*84)
    for r in results:
        print(f"{r['run_id']:<32} {_g(r,'best_val_nll'):>9.4f} {_g(r,'recon_fid'):>10.2f} "
              f"{_g(r,'gen_fid'):>9.2f} {_g(r,'best_temp'):>7.2f} {_g(r,'sample_gini'):>9.3f}")

    # aggregate by method: NLL, recon-FID (tokenizer), gen-FID (tokenizer+prior),
    # and the prior gap (gen - recon = what the prior costs)
    agg = {}
    for method, label in [('ema', 'vanilla_ema'), ('drift', 'drift_no_pp_ste')]:
        mr = [r for r in results if method in r['run_id']]
        nlls = [r['best_val_nll'] for r in mr]
        rfid = [r['recon_fid'] for r in mr if r.get('recon_fid') is not None]
        gfid = [r['gen_fid'] for r in mr if r.get('gen_fid') is not None]
        agg[method] = {'nll': nlls, 'recon': rfid, 'gen': gfid}
        line = f"\n{label}: NLL {np.mean(nlls):.3f}±{np.std(nlls):.3f}"
        if rfid:
            line += f"  recon-FID {np.mean(rfid):.2f}±{np.std(rfid):.2f}"
        if gfid:
            line += f"  gen-FID {np.mean(gfid):.2f}±{np.std(gfid):.2f}"
        if rfid and gfid:
            line += f"  prior-gap {np.mean(gfid)-np.mean(rfid):+.2f}"
        print(line)

    # the headline tension: drift usually wins recon-FID but its codes are harder
    # to model, so the gen-FID verdict is the interesting open question
    if agg['ema']['recon'] and agg['drift']['recon']:
        d_recon = np.mean(agg['drift']['recon']) - np.mean(agg['ema']['recon'])
        print(f"\nΔ recon-FID (drift − EMA): {d_recon:+.2f}  "
              "(negative = drift's tokenizer reconstructs better)")
    if agg['ema']['gen'] and agg['drift']['gen']:
        d_gen = np.mean(agg['drift']['gen']) - np.mean(agg['ema']['gen'])
        print(f"Δ gen-FID   (drift − EMA): {d_gen:+.2f}  "
              "(negative = drift also generates better despite harder-to-model codes)")
        print("VERDICT: " + ("✓ drift's reconstruction edge survives generation"
                             if d_gen < 0 else
                             "~ drift reconstructs better but the harder prior eats the gen-FID gain"))
    print('='*84)


# ----- linear probe experiment -----

@app.function(
    image = image,
    gpu = 'L40S',
    volumes = {'/vol': volume},
    timeout = 30 * 60,
)
def run_linear_probe(config_dict: dict) -> dict:
    """Encode a dataset with a frozen VQ-VAE and run a logistic regression probe."""
    import sys
    sys.path.insert(0, '/root/scaling_drifting_vqvae')
    from experiments.linear_probe import ProbeConfig, run_probe

    cfg = ProbeConfig(**{k: v for k, v in config_dict.items()
                         if k in ProbeConfig.__dataclass_fields__})
    cfg.data_root = '/vol/data'
    cfg.vqvae_root = '/vol/runs'
    cfg.out_root = '/vol/runs'
    result = run_probe(cfg)
    volume.commit()
    return result


@app.local_entrypoint()
def linear_probe(phase: str = 'phase_cifar100', k: int = 512):
    """Linear probe: do drift's codes carry more semantic information than EMA's?

    Encodes CIFAR-100 with frozen VQ-VAEs (3 seeds each) and trains a logistic
    regression on token frequency histograms (K-dim bag-of-visual-words). Reports top-1 and top-5 accuracy.

    Usage:
        modal run experiments/modal_app.py::linear_probe
        modal run experiments/modal_app.py::linear_probe --k 1024
        modal run experiments/modal_app.py::linear_probe --k 512 --phase phase_cifar100
    """
    import numpy as np

    configs = []
    for method_label, method_suffix in [('ema', 'vanilla_ema'), ('drift', 'drift_no_pp_ste')]:
        for seed in [0, 1, 2]:
            run_id = f'cifar100_{method_suffix}_K{k}_seed{seed}'
            configs.append({
                'vqvae_phase': phase,
                'vqvae_run_id': run_id,
                'vqvae_ckpt': 'final.pt',
                'dataset': 'cifar100',
            })

    print(f'Launching {len(configs)} linear probe runs (K={k}, phase={phase})...')
    calls = [run_linear_probe.spawn(cfg) for cfg in configs]
    results = [c.get() for c in calls]

    # print per-run results
    print(f"\n{'='*70}")
    print(f"LINEAR PROBE RESULTS  (K={k}, phase={phase})")
    print(f"{'run_id':<40} {'top1':>8} {'top5':>8} {'best_C':>8}")
    print('-'*70)
    for r in results:
        print(f"{r['run_id']:<40} {r['best_top1']*100:>7.1f}% {r['best_top5']*100:>7.1f}% {r['best_C']:>8.2f}")

    # aggregate by method
    print()
    for method_label, method_suffix in [('vanilla_ema', 'ema'), ('drift_no_pp_ste', 'drift')]:
        method_results = [r for r in results if method_label in r['run_id']]
        top1s = [r['best_top1'] for r in method_results]
        top5s = [r['best_top5'] for r in method_results]
        ppls  = [r.get('K', k) for r in method_results]
        print(f"{method_label} K={k}:  top1 {np.mean(top1s)*100:.1f}±{np.std(top1s)*100:.1f}%"
              f"  top5 {np.mean(top5s)*100:.1f}±{np.std(top5s)*100:.1f}%")

    # delta
    ema_r = [r for r in results if 'vanilla_ema' in r['run_id']]
    drift_r = [r for r in results if 'drift_no_pp_ste' in r['run_id']]
    if ema_r and drift_r:
        delta_top1 = np.mean([r['best_top1'] for r in drift_r]) - np.mean([r['best_top1'] for r in ema_r])
        delta_top5 = np.mean([r['best_top5'] for r in drift_r]) - np.mean([r['best_top5'] for r in ema_r])
        print(f"\nΔ (drift − EMA): top1 {delta_top1*100:+.1f}pp  top5 {delta_top5*100:+.1f}pp")
        if delta_top1 > 0.02:
            print("VERDICT: ✓  Drift codes carry more semantic information")
        elif delta_top1 > 0:
            print("VERDICT: ~  Weak signal in drift's favour")
        else:
            print("VERDICT: ✗  No semantic advantage — uniform codes ≠ semantically richer codes")
    print('='*70)


# ----- CNN probe experiment (Experiment A: transfer / recognition) -----

@app.function(
    image = image,
    gpu = 'L40S',
    volumes = {'/vol': volume},
    timeout = 60 * 60,
)
def run_cnn_probe(config_dict: dict) -> dict:
    """Train one CNN probe head on a frozen VQ-VAE's latent grid."""
    import sys
    sys.path.insert(0, '/root/scaling_drifting_vqvae')
    from experiments.cnn_probe import CNNProbeConfig, train_probe

    cfg = CNNProbeConfig(**{k: v for k, v in config_dict.items()
                            if k in CNNProbeConfig.__dataclass_fields__})
    cfg.data_root = '/vol/data'
    cfg.vqvae_root = '/vol/runs'
    cfg.out_root = '/vol/runs'
    result = train_probe(cfg)
    volume.commit()
    return result


@app.local_entrypoint()
def cnn_probe(phase: str = 'phase_cifar100', k: int = 512,
              representation: str = 'zq', head: str = 'cnn',
              epochs: int = 50, include_controls: bool = True):
    """CNN probe: train a small conv head on frozen VQ-VAE latents, drift vs EMA.

    The stronger, spatial version of the linear probe. Freezes the encoder +
    codebook and trains only the head, so accuracy differences come from the
    frozen features alone (transfer learning / feature extraction).

    Representations: 'zq' (quantized latents, primary), 'ze' (pre-quant encoder
    output), 'codes' (discrete tokens via a learned embedding).

    Usage:
        modal run experiments/modal_app.py::cnn_probe --k 512
        modal run experiments/modal_app.py::cnn_probe --k 512 --representation codes
        modal run experiments/modal_app.py::cnn_probe --k 1024 --head linear
    """
    import numpy as np

    configs = []
    for method_suffix in ['vanilla_ema', 'drift_no_pp_ste']:
        for seed in [0, 1, 2]:
            vqvae_run_id = f'cifar100_{method_suffix}_K{k}_seed{seed}'
            tag = 'ema' if 'ema' in method_suffix else 'drift'
            configs.append({
                'run_id': f'cnnprobe_{tag}_{representation}_{head}_K{k}_seed{seed}',
                'backbone': 'vqvae',
                'vqvae_phase': phase,
                'vqvae_run_id': vqvae_run_id,
                'vqvae_ckpt': 'final.pt',
                'dataset': 'cifar100',
                'representation': representation,
                'head': head,
                'epochs': epochs,
                'seed': seed,
            })

    # reference baselines: random (untrained) backbone and raw pixels
    if include_controls:
        configs.append({
            'run_id': f'cnnprobe_random_{representation}_{head}_K{k}_seed0',
            'backbone': 'random', 'random_method': 'vanilla_ema',
            'random_codebook_size': k, 'dataset': 'cifar100',
            'representation': representation, 'head': head, 'epochs': epochs, 'seed': 0,
        })
        configs.append({
            'run_id': f'cnnprobe_pixels_{head}_seed0',
            'backbone': 'vqvae', 'dataset': 'cifar100',
            'representation': 'pixels', 'head': head, 'epochs': epochs, 'seed': 0,
        })

    print(f'Launching {len(configs)} CNN probe runs '
          f'(K={k}, rep={representation}, head={head}, {epochs} epochs)...')
    calls = [run_cnn_probe.spawn(cfg) for cfg in configs]
    results = [c.get() for c in calls]

    print(f"\n{'='*72}")
    print(f"CNN PROBE  (K={k}, rep={representation}, head={head}, phase={phase})")
    print(f"{'run_id':<42} {'top1':>8} {'top5':>8}")
    print('-'*72)
    for r in results:
        print(f"{r['run_id']:<42} {r['test_top1']*100:>7.1f}% {r['test_top5']*100:>7.1f}%")

    def agg(tag):
        rs = [r for r in results if r['run_id'].startswith(f'cnnprobe_{tag}_')]
        return ([r['test_top1'] for r in rs], [r['test_top5'] for r in rs])

    print()
    for tag, label in [('ema', 'vanilla_ema'), ('drift', 'drift_no_pp_ste')]:
        t1, t5 = agg(tag)
        if t1:
            print(f"{label} K={k}:  top1 {np.mean(t1)*100:.1f}±{np.std(t1)*100:.1f}%"
                  f"  top5 {np.mean(t5)*100:.1f}±{np.std(t5)*100:.1f}%  (n={len(t1)})")

    e1, _ = agg('ema'); d1, _ = agg('drift')
    if e1 and d1:
        dt = np.mean(d1) - np.mean(e1)
        print(f"\nΔ (drift − EMA): top1 {dt*100:+.2f}pp")
        print("VERDICT: " + ("✓ drift features transfer better" if dt > 0.01
                              else "~ no clear advantage at this head capacity"))
    if include_controls:
        for tag in ['random', 'pixels']:
            rs = [r for r in results if r['run_id'].startswith(f'cnnprobe_{tag}_')]
            if rs:
                print(f"  reference [{tag}]: top1 {rs[0]['test_top1']*100:.1f}%")
    print('='*72)
