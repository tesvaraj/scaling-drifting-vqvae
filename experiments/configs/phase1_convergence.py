"""Phase 1 — Convergence experiment.

Question: does the PSNR gap close at convergence, or is it structural?

12 runs: 4 methods x 3 seeds x K=512 on CIFAR-10, 30k iters each. The most
important single experiment in the project.
"""
from __future__ import annotations

from experiments.train import TrainConfig
from experiments.configs.base import run_id, variants


PHASE = 'phase1_convergence'


def make_runs() -> list[TrainConfig]:
    base = TrainConfig(
        phase = PHASE,
        dataset = 'cifar10',
        train_iter = 30000,
        batch_size = 128,
        codebook_size = 512,
        log_every = 50,
        val_every = 1000,
        image_every = 2500,
        ckpt_every = 5000,
        eval_ssim = True,
        eval_lpips = True,
        eval_fid = True,
        tags = ['phase1', 'convergence'],
    )
    methods = ['vanilla_ema', 'vanilla_classical', 'simvq', 'drift']
    seeds = [0, 1, 2]
    overrides = []
    for m in methods:
        for s in seeds:
            overrides.append({
                'method': m,
                'seed': s,
                'run_id': run_id(m, base.codebook_size, s),
            })
    return variants(base, overrides)
