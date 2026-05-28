"""Final optimized runs — drift_no_pp_ste with tuned tau and energy_weight.

DO NOT LAUNCH until phase_hparam_sweep results are in. Update TAU and EW
below with the best values from the sweep, then launch.

Default guess based on Phase 2 evidence: tau=0.1 (best for full drift),
energy_weight=1.0 (not swept yet; keeping default until sweep says otherwise).

Only runs drift_no_pp_ste (EMA baselines are unchanged by tau — no need to rerun).
Compare against existing EMA runs from phase_cifar100, phase2b_confirmation, etc.

Datasets covered: CIFAR-10, CIFAR-100 (K=512 + K=1024), STL-10, Tiny ImageNet (K=512 + K=1024)
Total: 21 runs × 30k iters (drop STL-10 and TinyImageNet to 15 if compute-constrained).
"""
from __future__ import annotations

from experiments.train import TrainConfig
from experiments.configs.base import variants


PHASE = 'phase_optimized'

# ---- UPDATE THESE after phase_hparam_sweep results ----
TAU = 1.0           # best from sweep (tau sweep: 0.05→0.1→0.3→1.0 monotonically better)
EW  = 1.0           # best PSNR from sweep (ew=0.3 gives better FID by ~2pts but -0.02 PSNR)
# -------------------------------------------------------

_DRIFT = dict(
    method = 'drift',
    energy_terms = ('nn', 'pn'),
    rotation_trick = False,
    ste = True,
    tau = TAU,
    energy_weight = EW,
)


def make_runs() -> list[TrainConfig]:
    runs = []

    # ---- CIFAR-10 K=512 (replication of Phase 2b with optimal tau) ----
    base10 = TrainConfig(
        phase = PHASE,
        dataset = 'cifar10',
        codebook_size = 512,
        train_iter = 30000,
        batch_size = 128,
        log_every = 50,
        val_every = 1000,
        image_every = 5000,
        ckpt_every = 5000,
        eval_ssim = True,
        eval_lpips = True,
        eval_fid = True,
        tags = ['optimized', 'cifar10'],
    )
    runs += variants(base10, [
        {**_DRIFT, 'seed': s, 'run_id': f'cifar10_drift_noppste_opt_K512_seed{s}'}
        for s in [0, 1, 2]
    ])

    # ---- CIFAR-100 K=512 and K=1024 ----
    base100 = TrainConfig(
        phase = PHASE,
        dataset = 'cifar100',
        train_iter = 30000,
        batch_size = 128,
        log_every = 50,
        val_every = 1000,
        image_every = 5000,
        ckpt_every = 5000,
        eval_ssim = True,
        eval_lpips = True,
        eval_fid = True,
        tags = ['optimized', 'cifar100'],
    )
    for K in [512, 1024]:
        runs += variants(base100, [
            {**_DRIFT, 'codebook_size': K, 'seed': s,
             'run_id': f'cifar100_drift_noppste_opt_K{K}_seed{s}'}
            for s in [0, 1, 2]
        ])

    # ---- STL-10 K=512 ----
    base_stl = TrainConfig(
        phase = PHASE,
        dataset = 'stl10',
        codebook_size = 512,
        train_iter = 30000,
        batch_size = 64,
        log_every = 50,
        val_every = 1000,
        image_every = 5000,
        ckpt_every = 5000,
        eval_ssim = True,
        eval_lpips = True,
        eval_fid = True,
        tags = ['optimized', 'stl10'],
    )
    runs += variants(base_stl, [
        {**_DRIFT, 'seed': s, 'run_id': f'stl10_drift_noppste_opt_K512_seed{s}'}
        for s in [0, 1, 2]
    ])

    # ---- Tiny ImageNet K=512 and K=1024 ----
    base_tiny = TrainConfig(
        phase = PHASE,
        dataset = 'tiny_imagenet',
        train_iter = 30000,
        batch_size = 64,
        log_every = 50,
        val_every = 1000,
        image_every = 5000,
        ckpt_every = 5000,
        eval_ssim = True,
        eval_lpips = False,
        eval_fid = True,
        tags = ['optimized', 'tiny_imagenet'],
    )
    for K in [512, 1024]:
        runs += variants(base_tiny, [
            {**_DRIFT, 'codebook_size': K, 'seed': s,
             'run_id': f'tinyimagenet_drift_noppste_opt_K{K}_seed{s}'}
            for s in [0, 1, 2]
        ])

    return runs
