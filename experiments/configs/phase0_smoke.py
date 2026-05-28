"""Phase 0 — local smoke test.

Goal: verify the pipeline end-to-end (data loading, all methods, metrics,
checkpointing, wandb-disabled mode) in under 5 minutes on CPU/MPS.
"""
from __future__ import annotations

from experiments.train import TrainConfig
from experiments.configs.base import run_id, variants


PHASE = 'phase0_smoke'


def make_runs() -> list[TrainConfig]:
    base = TrainConfig(
        phase = PHASE,
        dataset = 'cifar10',
        train_iter = 500,
        batch_size = 64,
        codebook_size = 64,
        log_every = 25,
        val_every = 250,
        image_every = 250,
        ckpt_every = 500,
        n_hidden_subsample = 256,
        eval_ssim = True,
        eval_lpips = False,
        eval_fid = False,
        wandb_mode = 'disabled',
        num_workers = 0,
        tags = ['smoke'],
    )
    methods = ['vanilla_ema', 'vanilla_classical', 'simvq', 'drift', 'drift_ema']
    return variants(base, [
        {'method': m, 'run_id': run_id(m, base.codebook_size, 0, suffix = 'smoke')}
        for m in methods
    ])
