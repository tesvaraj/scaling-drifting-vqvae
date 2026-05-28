"""Aggregate finished runs in a phase into a single summary table.

Walks ``runs/<phase>/*/summary.json`` (and curves.csv for any final-step
metrics missing from summary), produces:
    runs/<phase>/aggregate.csv
    runs/<phase>/aggregate.md
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import fire


METRIC_COLS = [
    # reconstruction quality
    'val/psnr', 'val/ssim', 'val/lpips', 'val/rfid',
    # codebook utilization
    'val/utilization', 'val/perplexity', 'val/entropy_nats', 'val/gini',
    'val/active_codes',
    # codebook geometry (distance structure)
    'codebook/norm_mean', 'codebook/pair_dist_mean', 'codebook/pair_dist_min',
    'codebook/pair_dist_std',
    # hidden geometry
    'hidden/hidden_norm_mean',
    # drift energy breakdown (nan for non-drift methods)
    'drift/U_nn', 'drift/U_pn', 'drift/U_pp', 'drift/U_total',
]


def _last_row(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    rows = list(csv.DictReader(open(csv_path)))
    return rows[-1] if rows else {}


def main(phase: str):
    runs_root = Path('runs') / phase
    if not runs_root.exists():
        print(f'no such phase dir: {runs_root}')
        return

    rows = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith('_'):
            continue
        summary_path = run_dir / 'summary.json'
        cfg_path = run_dir / 'config.json'

        summary = None
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
        cfg = {}
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)

        if summary:
            # recover_wandb.py wraps metrics in 'final'; train.py writes them at top level
            final = summary.get('final') or {k: v for k, v in summary.items()
                                             if isinstance(v, (int, float, str, bool))}
        else:
            final = _last_row(run_dir / 'val_curves.csv')

        row = {
            'run_id': run_dir.name,
            'method': cfg.get('method', ''),
            'dataset': cfg.get('dataset', ''),
            'K': cfg.get('codebook_size', ''),
            'seed': cfg.get('seed', ''),
            'tau': cfg.get('tau', ''),
            'energy_weight': cfg.get('energy_weight', ''),
            'energy_terms': '|'.join(cfg.get('energy_terms', [])),
            'energy_schedule': cfg.get('energy_schedule', ''),
            'train_iter': cfg.get('train_iter', ''),
        }
        for c in METRIC_COLS:
            v = final.get(c)
            try:
                row[c] = float(v) if v is not None and v != '' else ''
            except Exception:
                row[c] = v if v is not None else ''
        rows.append(row)

    if not rows:
        print('no rows.')
        return

    out_csv = runs_root / 'aggregate.csv'
    cols = list(rows[0].keys())
    with open(out_csv, 'w', newline = '') as f:
        w = csv.DictWriter(f, fieldnames = cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'wrote {out_csv}')

    # markdown
    out_md = runs_root / 'aggregate.md'
    with open(out_md, 'w') as f:
        f.write(f'# {phase} — aggregate results\n\n')
        f.write('| ' + ' | '.join(cols) + ' |\n')
        f.write('| ' + ' | '.join('---' for _ in cols) + ' |\n')
        for r in rows:
            def fmt(v):
                if isinstance(v, float):
                    return f'{v:.4f}' if abs(v) < 100 else f'{v:.2f}'
                return str(v)
            f.write('| ' + ' | '.join(fmt(r.get(c, '')) for c in cols) + ' |\n')
    print(f'wrote {out_md}')


if __name__ == '__main__':
    fire.Fire(main)
