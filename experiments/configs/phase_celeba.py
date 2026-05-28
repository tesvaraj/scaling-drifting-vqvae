"""CelebA-64 comparison — harder benchmark, bigger images, 160k training faces.

CIFAR-10 may be too easy for EMA (K=512 barely collapses at 99.8% util).
CelebA has a richer visual distribution and more structured variation — a better
test of whether drift's uniform coverage translates to reconstruction gains.

6 runs: 2 methods x 3 seeds x K=512 x 30k iters.

Trimmed from 12→6 after Phase 2b confirmed drift_no_pp_ste is the winner.
Ablations (drift, drift_no_pp) are already settled on CIFAR-10; no need to
repeat them on every dataset.

Note: CelebA downloads ~1.4GB via torchvision; first run will cache to /vol/data.
If download fails on Modal, run: modal volume put drifting-vqvae <local_celeba_dir> /data/celeba
"""
from __future__ import annotations

from experiments.train import TrainConfig
from experiments.configs.base import run_id, variants


PHASE = 'phase_celeba'


def make_runs() -> list[TrainConfig]:
    base = TrainConfig(
        phase = PHASE,
        dataset = 'celeba',
        train_iter = 30000,
        batch_size = 64,       # 64x64 images — halve batch to keep GPU memory
        codebook_size = 512,
        log_every = 50,
        val_every = 1000,
        image_every = 2500,
        ckpt_every = 5000,
        eval_ssim = True,
        eval_lpips = True,
        eval_fid = True,
        tags = ['celeba', 'comparison'],
    )

    seeds = [0, 1, 2]
    overrides = []
    for s in seeds:
        overrides.append({
            'method': 'vanilla_ema',
            'seed': s,
            'run_id': f'celeba_vanilla_ema_K512_seed{s}',
        })
        overrides.append({
            'method': 'drift',
            'seed': s,
            'energy_terms': ('nn', 'pn'),
            'rotation_trick': False,
            'run_id': f'celeba_drift_no_pp_ste_K512_seed{s}',
        })

    return variants(base, overrides)
