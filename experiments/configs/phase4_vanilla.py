"""Milestone 3 — Production Run Suite for Scaling Drifting VQ-VAE.

The core vanilla formulation variant across all four core datasets.
"""
from __future__ import annotations

import copy

from experiments.train import TrainConfig
from experiments.configs.base import variants


def _add_dataset_runs(
    runs: list[TrainConfig],
    base_cfg: TrainConfig,
    target_datasets: list[str],
    run_suffix: str,
) -> None:
    for dataset in target_datasets:
        clean_base = copy.deepcopy(base_cfg)
        
        clean_base.wandb_project = f'star_ablations_{dataset}'
        clean_base.phase = f'm3_table_vanilla_{dataset}'
        clean_base.tags = ['m3_report', 'vanilla', dataset]

        runs.extend(
            variants(
                clean_base,
                [{
                    'dataset': dataset,
                    'run_id': f'{dataset}_{run_suffix}_run'
                }]
            )
        )


def make_vanilla_runs() -> list[TrainConfig]:
    """True Vanilla formulation runs across all four target datasets.

    Configurations:
      - method: 'vanilla' (Strictly NO DRIFT)
      - rotation_trick: True (Rotation Trick = YES)
      - ste: False (No STE)
      - energy_terms: ('pp', 'nn', 'pn') (All three forces used)
    """
    runs: list[TrainConfig] = []
    target_datasets = ['omniglot', 'pcam', 'galaxy', 'dtd']

    base_vanilla = TrainConfig(
        phase='m3_table_vanilla_placeholder',
        train_iter=30000,
        batch_size=128,
        codebook_size=512,
        seed=0,
        log_every=100,
        val_every=2000,
        eval_fid=True,
        eval_ssim=True,
        method='vanilla',
        energy_terms=('pp', 'nn', 'pn'),
        rotation_trick=True,
        ste=False,
        wandb_project='star_ablations_placeholder',
        tags=['m3_report', 'vanilla'],
    )
    
    _add_dataset_runs(runs, base_vanilla, target_datasets, 'vanilla')

    return runs


def make_runs(phase: str = 'ablations') -> list[TrainConfig]:
    """Unified entry point for the experiment launcher router."""
    if phase in ('ablations', 'phase4_star'):
        return make_vanilla_runs()
    raise ValueError(f"Unknown sub-phase suite target: {phase}")