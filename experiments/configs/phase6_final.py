"""Phase 6 — Final multi-seed headline runs.

After phases 1-5 we know which methods deserve to be in the headline tables.
This phase produces the publication numbers: 5 seeds per method on both
CIFAR-10 and CelebA at K=512, 30k iters.

Edit ``METHODS`` after the earlier phases land to include the best hybrid.
"""
from __future__ import annotations

from experiments.train import TrainConfig
from experiments.configs.base import run_id, variants


PHASE = 'phase6_final'

# methods to include in the final headline. Update after phase 1-5 lands.
METHODS = ['vanilla_ema', 'vanilla_classical', 'simvq', 'drift', 'drift_ema']


def make_runs() -> list[TrainConfig]:
    base = TrainConfig(
        phase = PHASE,
        train_iter = 30000,
        batch_size = 128,
        codebook_size = 512,
        log_every = 100,
        val_every = 2000,
        image_every = 5000,
        ckpt_every = 10000,
        eval_ssim = True,
        eval_lpips = True,
        eval_fid = True,
        tags = ['phase6', 'final'],
    )

    overrides = []
    for ds in ['cifar10', 'celeba']:
        for m in METHODS:
            for s in range(5):
                overrides.append({
                    'method': m, 'dataset': ds, 'seed': s,
                    'run_id': run_id(m, base.codebook_size, s, dataset = ds),
                })
    return variants(base, overrides)
