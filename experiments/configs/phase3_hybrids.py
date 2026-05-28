"""Phase 3 — Hybrid methods. Try to beat EMA vanilla.

Question: can we combine drift's spreading with EMA's placement quality?

Primary contender: drift_ema (codebook updated by EMA, physics provides
encoder gradient). Three seeds each at K=512, 10k iters on CIFAR-10.
"""
from __future__ import annotations

from experiments.train import TrainConfig
from experiments.configs.base import run_id, variants


PHASE = 'phase3_hybrids'


def make_runs() -> list[TrainConfig]:
    base = TrainConfig(
        phase = PHASE,
        dataset = 'cifar10',
        train_iter = 10000,
        batch_size = 128,
        codebook_size = 512,
        log_every = 50,
        val_every = 1000,
        image_every = 2500,
        ckpt_every = 5000,
        eval_ssim = True,
        eval_lpips = True,
        eval_fid = True,
        tags = ['phase3', 'hybrid'],
    )

    overrides = []
    seeds = [0, 1, 2]

    # 1) drift + EMA codebook (primary hybrid)
    for s in seeds:
        overrides.append({
            'method': 'drift_ema', 'seed': s,
            'run_id': run_id('drift_ema', 512, s),
        })

    # 2) drift + commitment loss
    for s in seeds:
        overrides.append({
            'method': 'drift_commit', 'seed': s,
            'commitment_weight': 0.25,
            'run_id': run_id('drift_commit', 512, s),
        })

    # 3) drift warmup -> vanilla_ema (we approximate via warmup_off schedule:
    #    drift energy active for warmup_steps, then 0, leaving vanilla-style
    #    training. We use the drift_ema method so EMA is always available.)
    for s in seeds:
        overrides.append({
            'method': 'drift_ema', 'seed': s,
            'energy_schedule': 'warmup_off',
            'energy_warmup_steps': 2000,
            'run_id': run_id('drift_warmup', 512, s),
            'tags': ['phase3', 'hybrid', 'warmup'],
        })

    # 4) energy_weight annealing: 1.0 -> 0 linear
    for s in seeds:
        overrides.append({
            'method': 'drift', 'seed': s,
            'energy_schedule': 'linear',
            'energy_weight': 1.0,
            'energy_final': 0.0,
            'run_id': run_id('drift_anneal', 512, s),
            'tags': ['phase3', 'hybrid', 'anneal'],
        })

    # vanilla EMA baseline at the same iter count, for direct comparison
    for s in seeds:
        overrides.append({
            'method': 'vanilla_ema', 'seed': s,
            'run_id': run_id('vanilla_ema', 512, s, suffix = '10k'),
            'tags': ['phase3', 'baseline'],
        })

    return variants(base, overrides)
