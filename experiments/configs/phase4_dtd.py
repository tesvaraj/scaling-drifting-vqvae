"""Milestone 3 — Production Run Suite for Scaling Drifting VQ-VAE.

Replicates the complete experiment matrix presented in the Milestone 3 report,
focusing specifically on isolating the core formulation variants across all 4 datasets
into separate, clean tables inside a dedicated workspace.
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

        runs.extend(
            variants(
                clean_base,
                [{
                    'dataset': dataset,
                    'run_id': f'{dataset}_{run_suffix}_run'
                }]
            )
        )


def make_ablation_runs() -> list[TrainConfig]:
    """2. FORMULATION COMPONENT ABLATIONS (Slide 4 Table).

    Tables:
      - ste_only: no rotation trick, STE on, U_pp present
      - no_upp: rotation trick on, U_pp removed
      - star: no rotation trick, STE on, U_pp removed
      - ema: rotation trick on, U_pp present
    """
    runs: list[TrainConfig] = []
    target_datasets = ['dtd']

    # 1. Table: ste_only
    base_ste_only = TrainConfig(
        phase='m3_table_ste_only',
        train_iter=30000,
        batch_size=128,
        codebook_size=512,
        seed=0,
        log_every=100,
        val_every=2000,
        eval_fid=True,
        eval_ssim=True,
        method='drift',
        energy_terms=('pp', 'nn', 'pn'),
        rotation_trick=False,
        ste=True,
        wandb_project='star_ablations_dtd',
        tags=['m3_report', 'slide4_ablation', 'ste_only'],
    )
    _add_dataset_runs(runs, base_ste_only, target_datasets, 'ste_only')

    # 2. Table: no_upp
    base_no_upp = TrainConfig(
        phase='m3_table_no_upp',
        train_iter=30000,
        batch_size=128,
        codebook_size=512,
        seed=0,
        log_every=100,
        val_every=2000,
        eval_fid=True,
        eval_ssim=True,
        method='drift',
        energy_terms=('nn', 'pn'),
        rotation_trick=True,
        ste=False,
        wandb_project='star_ablations_dtd',
        tags=['m3_report', 'slide4_ablation', 'no_upp'],
    )
    _add_dataset_runs(runs, base_no_upp, target_datasets, 'no_upp')

    # 3. Table: star
    base_star = TrainConfig(
        phase='m3_table_star',
        train_iter=30000,
        batch_size=128,
        codebook_size=512,
        seed=0,
        log_every=100,
        val_every=2000,
        eval_fid=True,
        eval_ssim=True,
        method='drift',
        energy_terms=('nn', 'pn'),
        rotation_trick=False,
        ste=True,
        tau=1.0,
        wandb_project='star_ablations_dtd',
        tags=['m3_report', 'slide4_ablation', 'star'],
    )
    _add_dataset_runs(runs, base_star, target_datasets, 'star')

    # 4. Table: ema
    base_ema = TrainConfig(
        phase='m3_table_ema',
        train_iter=30000,
        batch_size=128,
        codebook_size=512,
        seed=0,
        log_every=100,
        val_every=2000,
        eval_fid=True,
        eval_ssim=True,
        method='drift',
        energy_terms=('pp', 'nn', 'pn'),
        rotation_trick=True,
        ste=False,
        wandb_project='star_ablations_dtd',
        tags=['m3_report', 'slide4_ablation', 'ema'],
    )
    _add_dataset_runs(runs, base_ema, target_datasets, 'ema')

    return runs


def make_runs(phase: str = 'ablations') -> list[TrainConfig]:
    """Unified entry point for the experiment launcher router."""
    if phase in ('ablations', 'phase4_star'):
        return make_ablation_runs()
    raise ValueError(f"Unknown sub-phase suite target: {phase}")