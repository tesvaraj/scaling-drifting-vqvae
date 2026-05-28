"""Recover local run artifacts from W&B.

This is a fallback for runs whose Modal containers finished before their
``runs/<phase>/<run_id>/`` files were persisted to the Modal volume. It rebuilds
the machine-readable files used by the local diagnosis scripts:

    config.json
    summary.json
    curves.csv
    val_curves.csv
    wandb_url.txt

The recovered CSVs come from W&B history, so they may not include every local
detail if a key was never logged to W&B, but they preserve the metrics needed
for aggregation and iteration.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import fire


DEFAULT_PROJECT = 'drifting-vqvae-231n'

TRAIN_PREFIXES = (
    'train/',
    'codebook/',
    'drift/',
    'hidden/',
)

VAL_PREFIXES = (
    'val/',
    'codebook/',
    'drift/',
    'hidden/',
)


def _api():
    import wandb
    return wandb.Api(timeout = 60)


def _jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def _summary_dict(run) -> dict:
    if hasattr(run.summary, '_json_dict'):
        return dict(run.summary._json_dict)
    return dict(run.summary)


def _config_dict(run) -> dict:
    return {k: _jsonable(v) for k, v in dict(run.config).items()
            if not str(k).startswith('_')}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    cols = []
    seen = set()
    preferred = ['step', '_step']
    for c in preferred:
        if any(c in r for r in rows):
            cols.append(c)
            seen.add(c)

    for row in rows:
        for k in row:
            if k not in seen:
                cols.append(k)
                seen.add(k)

    with open(path, 'w', newline = '') as f:
        writer = csv.DictWriter(f, fieldnames = cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _clean_history_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if k.startswith('_') and k not in ('_step',):
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
    if '_step' in out and 'step' not in out:
        out['step'] = out['_step']
    return out


def _split_history(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    train_rows = []
    val_rows = []
    for row in rows:
        clean = _clean_history_row(row)
        if not clean:
            continue

        has_val = any(k.startswith('val/') for k in clean)
        if has_val:
            val_rows.append({
                k: v for k, v in clean.items()
                if k in ('step', '_step') or k.startswith(VAL_PREFIXES)
            })

        has_train = any(k.startswith('train/') for k in clean)
        if has_train:
            train_rows.append({
                k: v for k, v in clean.items()
                if k in ('step', '_step') or k.startswith(TRAIN_PREFIXES)
            })
    return train_rows, val_rows


def _recover_run(run, out_root: Path, overwrite: bool = False) -> dict:
    phase = run.group or run.tags[0] if run.tags else 'wandb_recovered'
    run_dir = out_root / phase / run.name
    run_dir.mkdir(parents = True, exist_ok = True)

    cfg = _config_dict(run)
    summary = _summary_dict(run)

    paths = {
        'config': run_dir / 'config.json',
        'summary': run_dir / 'summary.json',
        'wandb': run_dir / 'wandb_url.txt',
        'curves': run_dir / 'curves.csv',
        'val_curves': run_dir / 'val_curves.csv',
    }

    if overwrite or not paths['config'].exists():
        paths['config'].write_text(json.dumps(cfg, indent = 2))

    final = {k: v for k, v in summary.items()
             if isinstance(v, (int, float, str, bool)) and not k.startswith('_')}
    recovered_summary = {
        'run_id': cfg.get('run_id', run.name),
        'phase': phase,
        'method': cfg.get('method', ''),
        'dataset': cfg.get('dataset', ''),
        'codebook_size': cfg.get('codebook_size', ''),
        'train_iter': cfg.get('train_iter', ''),
        'seed': cfg.get('seed', ''),
        'final': final,
        'wandb_url': run.url,
        'recovered_from_wandb': True,
    }
    if overwrite or not paths['summary'].exists():
        paths['summary'].write_text(json.dumps(recovered_summary, indent = 2))
    if overwrite or not paths['wandb'].exists():
        paths['wandb'].write_text(run.url + '\n')

    history_rows = list(run.scan_history(page_size = 1000))
    # deduplicate by wandb's internal _step counter — scan_history can return
    # the same row multiple times when pages overlap
    seen: dict = {}
    for row in history_rows:
        key = row.get('_step') if row.get('_step') is not None else row.get('step')
        if key not in seen:
            seen[key] = row
    history_rows = list(seen.values())
    train_rows, val_rows = _split_history(history_rows)
    if overwrite or not paths['curves'].exists():
        _write_csv(paths['curves'], train_rows)
    if overwrite or not paths['val_curves'].exists():
        _write_csv(paths['val_curves'], val_rows)

    return {
        'phase': phase,
        'run_id': run.name,
        'state': run.state,
        'history_rows': len(history_rows),
        'train_rows': len(train_rows),
        'val_rows': len(val_rows),
        'url': run.url,
    }


def main(
    group: Optional[str] = None,
    run: Optional[str] = None,
    project: str = DEFAULT_PROJECT,
    entity: Optional[str] = None,
    out_root: str = 'runs',
    overwrite: bool = False,
):
    """Recover runs by W&B group or exact/substring run name."""
    if not group and not run:
        print('use --group <phase> or --run <name>')
        return

    api = _api()
    path = f'{entity}/{project}' if entity else project
    if group:
        runs = list(api.runs(path, filters = {'group': group}, per_page = 500))
    else:
        runs = list(api.runs(path, filters = {'display_name': run}, per_page = 500))
        if not runs:
            runs = [r for r in api.runs(path, per_page = 500) if run in r.name]

    if not runs:
        print('no matching W&B runs')
        return

    out = Path(out_root)
    results = [_recover_run(r, out, overwrite = overwrite) for r in runs]
    for r in results:
        print(
            f"{r['phase']}/{r['run_id']}: state={r['state']} "
            f"history={r['history_rows']} train={r['train_rows']} val={r['val_rows']}"
        )


if __name__ == '__main__':
    fire.Fire(main)
