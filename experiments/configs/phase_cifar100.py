"""CIFAR-100 — the dataset with the best chance of showing drift's advantage.

CIFAR-10 has 10 classes → K=512 gives ~100 codes/class → EMA fills them easily (99.8% util).
CIFAR-100 has 100 classes → K=1024 gives ~10 codes/class on average. EMA's greedy
k-means-style update will concentrate codes in high-density regions, leaving rare
classes underrepresented. Drift's physics repulsion forces codes to spread regardless
of density — this is where the coverage advantage should translate to PSNR/FID gains.

K=512 included as direct comparison to Phase 1 numbers.
K=1024 is the real test: drift coverage advantage should be clearest here.

12 runs: 2 K values x 2 methods x 3 seeds x 30k iters.
"""
from __future__ import annotations

from experiments.train import TrainConfig
from experiments.configs.base import variants


PHASE = 'phase_cifar100'


def make_runs() -> list[TrainConfig]:
    base = TrainConfig(
        phase = PHASE,
        dataset = 'cifar100',
        train_iter = 30000,
        batch_size = 128,
        log_every = 50,
        val_every = 1000,
        image_every = 2500,
        ckpt_every = 5000,
        eval_ssim = True,
        eval_lpips = True,
        eval_fid = True,
        tags = ['cifar100', 'generalization'],
    )

    seeds = [0, 1, 2]
    overrides = []
    for K in [512, 1024]:
        for s in seeds:
            overrides.append({
                'method': 'vanilla_ema',
                'codebook_size': K,
                'seed': s,
                'run_id': f'cifar100_vanilla_ema_K{K}_seed{s}',
            })
            overrides.append({
                'method': 'drift',
                'codebook_size': K,
                'seed': s,
                'energy_terms': ('nn', 'pn'),
                'rotation_trick': False,
                'run_id': f'cifar100_drift_no_pp_ste_K{K}_seed{s}',
            })

    return variants(base, overrides)
