"""Hyperparameter sweep for drift_no_pp_ste — tau and energy_weight.

Phase 2 showed tau=0.1 is best for FULL drift (+0.44 dB vs tau=1.0 at 10k).
That sweep was on the full energy (pp+nn+pn). This phase repeats it for the
final variant (no_pp_ste: nn+pn only) to get principled choices.

energy_weight is swept independently at tau=1.0 (our current default).
The interaction between tau and ew is checked with the best tau once known.

Baseline for comparison: vanilla_ema@10k = 23.53 ± 0.11 PSNR (from Phase 3 runs)
Existing no_pp_ste@10k (tau=1.0): ~23.73 PSNR (Phase 2, single seed extrapolation)
Existing no_pp_ste@30k (tau=1.0): 24.151 ± 0.073 PSNR (Phase 2b, gold standard)

14 runs: tau sweep (8) + ew sweep (6), all CIFAR-10 K=512 30k.
Must run to convergence — tau affects convergence speed so 10k would just measure
which tau converges fastest, not which achieves the best asymptotic quality.
"""
from __future__ import annotations

from experiments.train import TrainConfig
from experiments.configs.base import variants


PHASE = 'phase_hparam_sweep'

_DRIFT_BASE = dict(
    method = 'drift',
    energy_terms = ('nn', 'pn'),
    rotation_trick = False,
    ste = True,
)


def make_runs() -> list[TrainConfig]:
    base = TrainConfig(
        phase = PHASE,
        dataset = 'cifar10',
        codebook_size = 512,
        train_iter = 30000,
        batch_size = 128,
        log_every = 50,
        val_every = 1000,
        image_every = 5000,
        ckpt_every = 30000,
        eval_ssim = True,
        eval_lpips = False,
        eval_fid = True,
        tags = ['hparam_sweep', 'tau', 'energy_weight'],
    )

    overrides = []

    # --- tau sweep (energy_weight=1.0 default) ---
    for tau in [0.05, 0.1, 0.3, 1.0]:
        for seed in [0, 1]:
            overrides.append({
                **_DRIFT_BASE,
                'tau': tau,
                'energy_weight': 1.0,
                'seed': seed,
                'run_id': f'cifar10_noppste_tau{tau}_ew1.0_seed{seed}',
            })

    # --- energy_weight sweep (tau=1.0 default) ---
    for ew in [0.1, 0.3, 3.0]:
        for seed in [0, 1]:
            overrides.append({
                **_DRIFT_BASE,
                'tau': 1.0,
                'energy_weight': ew,
                'seed': seed,
                'run_id': f'cifar10_noppste_tau1.0_ew{ew}_seed{seed}',
            })

    return variants(base, overrides)
