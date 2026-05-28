"""Diagnose one run — print curves, final metrics, and key plots' paths.

Usage
-----
    python -m experiments.scripts.diagnose --phase phase1_convergence --run_id cifar10_drift_K512_seed0
    python -m experiments.scripts.diagnose --phase phase1_convergence --run_id cifar10_drift_K512_seed0 --tail 5
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import fire


def _read_csv(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main(phase: str, run_id: str, tail: int = 10, fetch_modal: bool = False):
    if fetch_modal:
        _fetch_from_modal(phase, run_id)

    run_dir = Path('runs') / phase / run_id
    if not run_dir.exists():
        print(f'no local run at {run_dir}')
        print('(try --fetch_modal to pull from the Modal volume)')
        return

    cfg_path = run_dir / 'config.json'
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        print('=== config ===')
        for k in ('method', 'codebook_size', 'train_iter', 'seed', 'dataset',
                  'energy_terms', 'tau', 'energy_weight', 'l2_normalize',
                  'rotation_trick', 'commitment_weight', 'energy_schedule'):
            if k in cfg:
                print(f'  {k}: {cfg[k]}')
        print()

    summary_path = run_dir / 'summary.json'
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print('=== summary ===')
        print(json.dumps(summary.get('final', summary), indent = 2))
        print()
    else:
        print('summary.json not present yet — run not finished.\n')

    curves = _read_csv(run_dir / 'curves.csv')
    val_curves = _read_csv(run_dir / 'val_curves.csv')

    print(f'=== training rows: {len(curves)} ===')
    if curves:
        cols = ['step', 'train/rec_loss', 'train/psnr', 'codebook/utilization',
                'codebook/perplexity']
        widths = [10, 12, 9, 14, 14]
        header = ' '.join(c.ljust(w) for c, w in zip(cols, widths))
        print(header)
        for r in curves[-tail:]:
            print(' '.join((r.get(c, '')[:w]).ljust(w) for c, w in zip(cols, widths)))
        print()

    print(f'=== validation rows: {len(val_curves)} ===')
    if val_curves:
        cols = ['step', 'val/psnr', 'val/ssim', 'val/rfid', 'val/perplexity',
                'val/utilization']
        widths = [10, 9, 9, 9, 14, 14]
        print(' '.join(c.ljust(w) for c, w in zip(cols, widths)))
        for r in val_curves:
            print(' '.join((r.get(c, '')[:w]).ljust(w) for c, w in zip(cols, widths)))
        print()

    print('=== artifacts ===')
    for fname in ('config.json', 'summary.json', 'curves.csv', 'val_curves.csv',
                  'wandb_url.txt'):
        p = run_dir / fname
        if p.exists():
            print(f'  {p}')
    fig_dir = run_dir / 'figures'
    if fig_dir.exists():
        figs = sorted(fig_dir.glob('*.png'))
        print(f'  figures/: {len(figs)} files')
        for p in figs[-6:]:
            print(f'    {p}')
    ckpt_dir = run_dir / 'checkpoints'
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob('*.pt'))
        print(f'  checkpoints/: {len(ckpts)} files')
        for p in ckpts[-3:]:
            print(f'    {p}')


def _fetch_from_modal(phase: str, run_id: str):
    """Download curves / summary / config from the Modal volume to local disk."""
    print(f'fetching {phase}/{run_id} from Modal...')
    from experiments.modal_app import app, fetch_run_summary, fetch_curves, fetch_val_curves

    target = Path('runs') / phase / run_id
    target.mkdir(parents = True, exist_ok = True)

    with app.run():
        summary = fetch_run_summary.remote(phase, run_id)
        curves = fetch_curves.remote(phase, run_id)
        val_curves = fetch_val_curves.remote(phase, run_id)

    if summary and 'error' not in summary:
        (target / 'summary.json').write_text(json.dumps(summary, indent = 2))
        print(f'  saved summary.json')
    if curves:
        (target / 'curves.csv').write_text(curves)
        print(f'  saved curves.csv ({len(curves.splitlines())} lines)')
    if val_curves:
        (target / 'val_curves.csv').write_text(val_curves)
        print(f'  saved val_curves.csv ({len(val_curves.splitlines())} lines)')


if __name__ == '__main__':
    fire.Fire(main)
