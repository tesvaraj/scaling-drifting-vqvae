"""Phase 4 v2 — Large-K scaling on CIFAR-100 (best showcase for drift's coverage advantage).

CIFAR-10 at K=512 barely stresses EMA (99.8% util). CIFAR-100 with K=2048+
is the regime where EMA's greedy k-means update will produce Zipfian code usage
while drift's physics repulsion maintains uniform coverage.

Design:
  Dataset: CIFAR-100 (100 classes, 50k images, 32x32 — same as CIFAR-10 but 10x diverse)
  K:       512, 1024, 2048, 4096  (increasing stress on EMA)
  Methods: vanilla_ema vs drift_no_pp_ste (best variant from Phase 2)
  Iters:   10k (fast sweep to show utilization collapse curve)
  Seeds:   2

16 runs total. The utilization vs K curve is the key figure:
  vanilla_ema: flat at ~100% for K≤512, then drops for K=1024+
  drift:       flat at 100% across all K  ← the point

After this phase, run 30k on the K where EMA collapses most to get full metrics.
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

    Ks = [512, 1024, 2048, 4096]
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
