"""Phase 4 — Large-K stress test.

Question: at K >> 1024 does EMA vanilla start to collapse where drift
survives? Modern EMA solves collapse at moderate K; we want to find the
regime where the physics-based approach wins.

24 runs: 3 methods x 4 K values x 2 seeds on CIFAR-10, 10k iters each.
"""
from __future__ import annotations

from experiments.train import TrainConfig
from experiments.configs.base import run_id, variants


PHASE = 'phase4_large_k'


def make_runs() -> list[TrainConfig]:
    base = TrainConfig(
        phase = PHASE,
        dataset = 'cifar10',
        train_iter = 10000,
        batch_size = 128,
        log_every = 50,
        val_every = 1000,
        image_every = 2500,
        ckpt_every = 5000,
        eval_ssim = True,
        eval_lpips = False,
        eval_fid = True,
        tags = ['phase4', 'stress'],
    )
    methods = ['vanilla_ema', 'drift', 'drift_ema']
    Ks = [1024, 2048, 4096, 8192]
    seeds = [0, 1]
    overrides = []
    for m in methods:
        for K in Ks:
            for s in seeds:
                overrides.append({
                    'method': m,
                    'codebook_size': K,
                    'seed': s,
                    'run_id': run_id(m, K, s),
                })
    return variants(base, overrides)
