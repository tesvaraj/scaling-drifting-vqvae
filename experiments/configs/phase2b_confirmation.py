"""Phase 2b — 30k confirmation of the two best diagnostic ablations.

Phase 2 showed at 10k (single seed):
  no_pp (drop U_pp):  PSNR 23.73, FID 55.78  — beats vanilla_ema@10k
  ste   (STE grad):   PSNR 23.67, FID 52.08  — beats vanilla_ema@10k

This phase runs both ablations + their combination at 30k with 3 seeds to
determine if the advantage holds at convergence.

Compare against existing runs:
  vanilla_ema@30k:  PSNR 23.386 ± 0.057, FID 59.83 ± 0.50
  drift_full@30k:   PSNR 23.199 ± 0.012, FID 60.07 ± 1.17
"""
from __future__ import annotations

from experiments.train import TrainConfig
from experiments.configs.base import run_id, variants


PHASE = 'phase2b_confirmation'


def make_runs() -> list[TrainConfig]:
    base = TrainConfig(
        phase = PHASE,
        dataset = 'cifar10',
        method = 'drift',
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
        tags = ['phase2b', 'confirmation'],
    )

    seeds = [0, 1, 2]
    overrides = []
    for s in seeds:
        # ablation 1: remove U_pp (hidden-hidden repulsion)
        overrides.append({
            'run_id': f'cifar10_drift_no_pp_K512_seed{s}',
            'seed': s,
            'energy_terms': ('nn', 'pn'),
        })
        # ablation 2: STE gradient estimator instead of rotation trick
        overrides.append({
            'run_id': f'cifar10_drift_ste_K512_seed{s}',
            'seed': s,
            'rotation_trick': False,
            'ste': True,
        })
        # ablation 3: both — best candidate to beat vanilla_ema
        overrides.append({
            'run_id': f'cifar10_drift_no_pp_ste_K512_seed{s}',
            'seed': s,
            'energy_terms': ('nn', 'pn'),
            'rotation_trick': False,
            'ste': True,
        })

    return variants(base, overrides)
