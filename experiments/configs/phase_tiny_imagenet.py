"""Tiny ImageNet — 200 classes, 64x64, 100k train images.

The hardest standard benchmark in this project. 200 classes at K=512
gives ~2.5 codes/class on average — EMA's greedy centroid update will
struggle to cover all 200 categories, producing visible Zipfian skew.
Drift's repulsion forces coverage regardless of class frequency.

Same architecture as STL-10 (n_downsample=3, 8x8 latent at 64x64).
Direct comparison: CIFAR-10 (10 cls) → CIFAR-100 (100 cls) → TinyIN (200 cls).

12 runs: vanilla_ema + drift_no_pp_ste x K{512,1024} x 3 seeds x 30k.
"""
from __future__ import annotations

from experiments.train import TrainConfig
from experiments.configs.base import variants


PHASE = 'phase_tiny_imagenet'


def make_runs() -> list[TrainConfig]:
    base = TrainConfig(
        phase = PHASE,
        dataset = 'tiny_imagenet',
        train_iter = 30000,
        batch_size = 64,       # 64x64 images — halve batch vs CIFAR
        log_every = 50,
        val_every = 1000,
        image_every = 2500,
        ckpt_every = 5000,
        eval_ssim = True,
        eval_lpips = True,
        eval_fid = True,
        tags = ['tiny_imagenet', 'generalization', 'pressure_test'],
    )

    seeds = [0, 1, 2]
    overrides = []
    for K in [512, 1024]:
        for s in seeds:
            overrides.append({
                'method': 'vanilla_ema',
                'codebook_size': K,
                'seed': s,
                'run_id': f'tinyimagenet_vanilla_ema_K{K}_seed{s}',
            })
            overrides.append({
                'method': 'drift',
                'codebook_size': K,
                'seed': s,
                'energy_terms': ('nn', 'pn'),
                'rotation_trick': False,
                'run_id': f'tinyimagenet_drift_no_pp_ste_K{K}_seed{s}',
            })

    return variants(base, overrides)
