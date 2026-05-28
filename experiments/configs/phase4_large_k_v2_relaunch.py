"""Phase 4 v2 relaunch — K>512 only.

K=512 runs completed successfully in the original phase4_large_k_v2 launch.
K=1024/2048/4096 runs crashed at step 1000 due to wandb.Histogram's 512-bin
limit (fixed in train.py). This config relaunches only the 12 missing runs
into the same 'phase4_large_k_v2' wandb group and output directory.
"""
from __future__ import annotations

from experiments.train import TrainConfig
from experiments.configs.base import variants

PHASE = 'phase4_large_k_v2'


def make_runs() -> list[TrainConfig]:
    base = TrainConfig(
        phase = PHASE,
        dataset = 'cifar100',
        train_iter = 10000,
        batch_size = 128,
        log_every = 50,
        val_every = 1000,
        image_every = 5000,
        ckpt_every = 10000,
        eval_ssim = True,
        eval_lpips = False,
        eval_fid = True,
        tags = ['large_k', 'scaling', 'cifar100'],
    )

    Ks = [1024, 2048, 4096]
    seeds = [0, 1]
    overrides = []
    for K in Ks:
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
