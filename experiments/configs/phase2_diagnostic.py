"""Phase 2 — Diagnostic ablations of DriftingVQ.

Question: which design choices in drift cost reconstruction quality?

Tests each component in isolation at K=512, 10k iters, 1 seed for speed.
The hypothesis to confirm or reject: U_pp (hidden-hidden repulsion) is the
term that distorts the encoder.
"""
from __future__ import annotations

from experiments.train import TrainConfig
from experiments.configs.base import run_id, variants


PHASE = 'phase2_diagnostic'


def make_runs() -> list[TrainConfig]:
    base = TrainConfig(
        phase = PHASE,
        dataset = 'cifar10',
        train_iter = 10000,
        batch_size = 128,
        codebook_size = 512,
        method = 'drift',
        log_every = 50,
        val_every = 1000,
        image_every = 2500,
        ckpt_every = 5000,
        eval_ssim = True,
        eval_lpips = False,
        eval_fid = True,
        tags = ['phase2', 'diagnostic'],
    )

    overrides = [
        # baseline (full drift)
        {'run_id': run_id('drift', 512, 0, suffix = 'full'),
         'energy_terms': ('pp', 'nn', 'pn')},
        # drop one term at a time
        {'run_id': run_id('drift', 512, 0, suffix = 'no_pp'),
         'energy_terms': ('nn', 'pn')},
        {'run_id': run_id('drift', 512, 0, suffix = 'no_nn'),
         'energy_terms': ('pp', 'pn')},
        # ablation: L2-norm off
        {'run_id': run_id('drift', 512, 0, suffix = 'no_l2'),
         'l2_normalize': False},
        # ablation: STE instead of rotation trick
        {'run_id': run_id('drift', 512, 0, suffix = 'ste'),
         'rotation_trick': False, 'ste': True},
        # tau sweep
        {'run_id': run_id('drift', 512, 0, suffix = 'tau0.1'), 'tau': 0.1},
        {'run_id': run_id('drift', 512, 0, suffix = 'tau0.3'), 'tau': 0.3},
        {'run_id': run_id('drift', 512, 0, suffix = 'tau1.0'), 'tau': 1.0},
        {'run_id': run_id('drift', 512, 0, suffix = 'tau3.0'), 'tau': 3.0},
    ]
    return variants(base, overrides)
