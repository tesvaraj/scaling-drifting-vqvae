"""Convergence curves for the paper's main figure.

Combines phase1_convergence (vanilla_ema, drift) and phase2b_confirmation
(drift_no_pp_ste) to show PSNR, FID, and perplexity vs training step with
mean ± 1σ bands across 3 seeds.

Usage
-----
    python -m experiments.scripts.plot_convergence
    python -m experiments.scripts.plot_convergence --out runs/figures/convergence.png
    python -m experiments.scripts.plot_convergence --methods vanilla_ema drift_no_pp_ste
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import fire
import numpy as np


METHODS = {
    'vanilla_ema': {
        'phase': 'phase1_convergence',
        'run_ids': ['cifar10_vanilla_ema_K512_seed0',
                    'cifar10_vanilla_ema_K512_seed1',
                    'cifar10_vanilla_ema_K512_seed2'],
        'label': 'vanilla EMA',
        'color': '#2166ac',
        'ls': '-',
    },
    'drift': {
        'phase': 'phase1_convergence',
        'run_ids': ['cifar10_drift_K512_seed0',
                    'cifar10_drift_K512_seed1',
                    'cifar10_drift_K512_seed2'],
        'label': 'drift (full)',
        'color': '#999999',
        'ls': '--',
    },
    'drift_no_pp_ste': {
        'phase': 'phase2b_confirmation',
        'run_ids': ['cifar10_drift_no_pp_ste_K512_seed0',
                    'cifar10_drift_no_pp_ste_K512_seed1',
                    'cifar10_drift_no_pp_ste_K512_seed2'],
        'label': 'drift (no U_pp + STE)',
        'color': '#d6604d',
        'ls': '-',
    },
}

METRICS = [
    ('val/psnr',        'PSNR (dB)',       True),   # higher-is-better
    ('val/rfid',        'FID',             False),  # lower-is-better
    ('val/perplexity',  'Perplexity',      True),
]


def _load_val_curves(phase: str, run_id: str, runs_root: Path) -> dict[int, dict]:
    """Load val_curves.csv, dedup by step, return {step: row}."""
    p = runs_root / phase / run_id / 'val_curves.csv'
    if not p.exists():
        return {}
    seen: dict[int, dict] = {}
    for row in csv.DictReader(open(p)):
        raw = row.get('step') or row.get('_step') or ''
        try:
            step = int(float(raw))
        except (ValueError, TypeError):
            continue
        if step not in seen:
            seen[step] = row
    return seen


def _collect(method_key: str, cfg: dict, runs_root: Path,
             metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values_per_seed) arrays. Values are NaN where missing."""
    all_steps: set[int] = set()
    per_seed: list[dict[int, float]] = []
    for run_id in cfg['run_ids']:
        curves = _load_val_curves(cfg['phase'], run_id, runs_root)
        seed_vals: dict[int, float] = {}
        for step, row in curves.items():
            v = row.get(metric, '')
            try:
                seed_vals[step] = float(v)
                all_steps.add(step)
            except (ValueError, TypeError):
                pass
        per_seed.append(seed_vals)

    steps = np.array(sorted(all_steps))
    values = np.full((len(cfg['run_ids']), len(steps)), np.nan)
    for i, sv in enumerate(per_seed):
        for j, st in enumerate(steps):
            if st in sv:
                values[i, j] = sv[st]
    return steps, values


def main(
    out: Optional[str] = None,
    methods: Optional[str] = None,
    runs_root: str = 'runs',
    show: bool = False,
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rr = Path(runs_root)
    method_keys = methods.split(',') if methods else list(METHODS.keys())
    plot_metrics = [m for m in METRICS if m[0] in
                    ['val/psnr', 'val/rfid', 'val/perplexity']]

    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(5 * len(plot_metrics), 4.5))
    if len(plot_metrics) == 1:
        axes = [axes]

    for ax, (metric, ylabel, higher_better) in zip(axes, plot_metrics):
        for key in method_keys:
            cfg = METHODS[key]
            steps, values = _collect(key, cfg, rr, metric)
            if steps.size == 0:
                continue
            mean = np.nanmean(values, axis=0)
            std  = np.nanstd(values, axis=0)
            steps_k = steps / 1000

            ax.plot(steps_k, mean,
                    label=cfg['label'], color=cfg['color'],
                    ls=cfg['ls'], lw=2.0)
            ax.fill_between(steps_k, mean - std, mean + std,
                            color=cfg['color'], alpha=0.15)

        ax.set_xlabel('Training step (k)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlim(left=0)
        ax.grid(alpha=0.3, lw=0.7)
        ax.legend(fontsize=9, framealpha=0.8)
        if not higher_better:
            ax.invert_yaxis()
        arrow = '↑ better' if higher_better else '↓ better'
        ax.set_title(f'{ylabel}  ({arrow})', fontsize=10)

    fig.suptitle('CIFAR-10 K=512 — convergence (mean ± 1σ, 3 seeds)', fontsize=12)
    fig.tight_layout()

    out_path = Path(out) if out else rr / 'figures' / 'convergence_cifar10_k512.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'saved → {out_path}')
    if show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    fire.Fire(main)
