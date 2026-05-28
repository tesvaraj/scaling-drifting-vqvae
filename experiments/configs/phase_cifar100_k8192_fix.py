"""K=8192 stability fix — test whether ew=0.3 prevents the collapse seen in seed1.

Original K=8192 run (ew=1.0): 2/3 drift seeds stable, 1/3 catastrophic collapse
(seed1: PSNR 22.79, FID 87, util 60.9%, Gini 0.86). EMA was stable all 3 seeds.

Hypothesis: ew=1.0 makes U_nn/U_pn too aggressive when only ~6 CIFAR-100
images/code exist at K=8192. ew=0.3 was best FID at K=512 (47.1 vs 49.3 for ew=1.0)
and should reduce the energy forces that can destabilize training at large K.

4 runs (drift only — EMA already stable at K=8192):
  - ew=0.3  seeds 0,1,2  (does lower energy weight prevent collapse?)
  - ew=1.0  seed=3       (extra data point on the ew=1.0 failure rate: is 1/3 reliable?)

EMA baselines already done in phase_cifar100_k8192.
"""
from __future__ import annotations

from experiments.train import TrainConfig
from experiments.configs.base import variants


PHASE = 'phase_cifar100_k8192_fix'

_DRIFT = dict(
    method='drift',
    energy_terms=('nn', 'pn'),
    rotation_trick=False,
    ste=True,
    tau=1.0,
)


def make_runs() -> list[TrainConfig]:
    base = TrainConfig(
        phase=PHASE,
        dataset='cifar100',
        train_iter=30000,
        batch_size=128,
        codebook_size=8192,
        log_every=50,
        val_every=1000,
        image_every=5000,
        ckpt_every=30000,
        eval_ssim=True,
        eval_lpips=False,
        eval_fid=True,
        tags=['cifar100', 'large_k', 'stability_fix'],
    )

    overrides = []

    # ew=0.3: does reduced energy weight prevent the K=8192 collapse?
    for s in [0, 1, 2]:
        overrides.append({
            **_DRIFT,
            'energy_weight': 0.3,
            'seed': s,
            'run_id': f'cifar100_drift_no_pp_ste_K8192_ew0.3_seed{s}',
        })

    # ew=1.0 seed=3: extra sample to characterize failure rate of original config
    overrides.append({
        **_DRIFT,
        'energy_weight': 1.0,
        'seed': 3,
        'run_id': 'cifar100_drift_no_pp_ste_K8192_ew1.0_seed3',
    })

    return variants(base, overrides)
