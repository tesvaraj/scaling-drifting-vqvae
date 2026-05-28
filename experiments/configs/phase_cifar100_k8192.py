"""CIFAR-100 K=8192 pressure test — stress EMA's Zipfian failure mode.

At K=8192, CIFAR-100 has ~6 training images per code on average. EMA's
batch-level assignment is so sparse (128/8192 ≈ 1.6% of codes per step) that
most codes get stale centroids. Prediction: EMA ppl/K drops to ~12-15%
(heavily Zipfian), drift ppl/K stays ~75% (near-uniform).

Note on drift stability: K=4096 seed1 collapsed at 10k iters (util=31%).
Running 3 seeds to characterize; if >1 seed collapses, the instability is
systematic and may require energy_weight tuning.

6 runs: vanilla_ema + drift_no_pp_ste x 3 seeds x K=8192 x 30k.
"""
from __future__ import annotations

from experiments.train import TrainConfig
from experiments.configs.base import variants


PHASE = 'phase_cifar100_k8192'


def make_runs() -> list[TrainConfig]:
    base = TrainConfig(
        phase = PHASE,
        dataset = 'cifar100',
        train_iter = 30000,
        batch_size = 128,
        codebook_size = 8192,
        log_every = 50,
        val_every = 1000,
        image_every = 5000,
        ckpt_every = 10000,
        eval_ssim = True,
        eval_lpips = False,
        eval_fid = True,
        tags = ['cifar100', 'large_k', 'pressure_test'],
    )

    seeds = [0, 1, 2]
    overrides = []
    for s in seeds:
        overrides.append({
            'method': 'vanilla_ema',
            'seed': s,
            'run_id': f'cifar100_vanilla_ema_K8192_seed{s}',
        })
        overrides.append({
            'method': 'drift',
            'seed': s,
            'energy_terms': ('nn', 'pn'),
            'rotation_trick': False,
            'run_id': f'cifar100_drift_no_pp_ste_K8192_seed{s}',
        })

    return variants(base, overrides)
