"""CIFAR-100 K=2048 at 30k — extend the scaling story.

Phase 4 (10k) showed drift at K=2048 is -0.30 dB behind EMA, matching the
same slow-convergence pattern seen at K=512 (-0.16 dB @10k → +0.72 dB @30k)
and K=1024 (-0.84 dB @10k → +0.62 dB @30k). This run confirms whether drift
overtakes EMA at K=2048 by 30k.

6 runs: vanilla_ema + drift_no_pp_ste × 3 seeds × K=2048 × 30k.
"""
from __future__ import annotations

from experiments.train import TrainConfig
from experiments.configs.base import variants


PHASE = 'phase_cifar100_k2048'


def make_runs() -> list[TrainConfig]:
    base = TrainConfig(
        phase = PHASE,
        dataset = 'cifar100',
        train_iter = 30000,
        batch_size = 128,
        codebook_size = 2048,
        log_every = 50,
        val_every = 1000,
        image_every = 2500,
        ckpt_every = 5000,
        eval_ssim = True,
        eval_lpips = True,
        eval_fid = True,
        tags = ['cifar100', 'large_k', 'scaling'],
    )

    seeds = [0, 1, 2]
    overrides = []
    for s in seeds:
        overrides.append({
            'method': 'vanilla_ema',
            'seed': s,
            'run_id': f'cifar100_vanilla_ema_K2048_seed{s}',
        })
        overrides.append({
            'method': 'drift',
            'seed': s,
            'energy_terms': ('nn', 'pn'),
            'rotation_trick': False,
            'run_id': f'cifar100_drift_no_pp_ste_K2048_seed{s}',
        })

    return variants(base, overrides)
