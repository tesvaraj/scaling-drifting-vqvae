"""Show status across all spawned Modal runs and on-disk run directories.

For each spawned function call, queries Modal for state (running / done /
failed) and prints a single-line summary. Also walks ``runs/`` and shows
local outputs.

Usage
-----
    python -m experiments.scripts.status
    python -m experiments.scripts.status --phase phase1_convergence
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import fire


def _scan_local(phase_filter: Optional[str] = None):
    runs_root = Path('runs')
    if not runs_root.exists():
        return []
    rows = []
    for phase_dir in sorted(runs_root.iterdir()):
        if not phase_dir.is_dir():
            continue
        if phase_filter and phase_dir.name != phase_filter:
            continue
        handles_path = phase_dir / '_handles.json'
        handles = []
        if handles_path.exists():
            with open(handles_path) as f:
                handles = json.load(f)
        handle_by_id = {h['run_id']: h for h in handles}

        for run_dir in sorted(phase_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith('_'):
                continue
            summary_path = run_dir / 'summary.json'
            summary = None
            if summary_path.exists():
                try:
                    with open(summary_path) as f:
                        summary = json.load(f)
                except Exception:
                    pass
            ckpt_dir = run_dir / 'checkpoints'
            n_ckpts = len(list(ckpt_dir.glob('*.pt'))) if ckpt_dir.exists() else 0
            curves_path = run_dir / 'curves.csv'
            curves_lines = 0
            if curves_path.exists():
                try:
                    curves_lines = sum(1 for _ in open(curves_path)) - 1
                except Exception:
                    pass
            h = handle_by_id.get(run_dir.name, {})
            rows.append({
                'phase': phase_dir.name,
                'run_id': run_dir.name,
                'fc_id': h.get('function_call_id', ''),
                'method': summary.get('method') if summary else h.get('config_summary', {}).get('method', ''),
                'K': summary.get('codebook_size') if summary else h.get('config_summary', {}).get('codebook_size', ''),
                'iter': summary.get('train_iter') if summary else h.get('config_summary', {}).get('train_iter', ''),
                'seed': summary.get('seed') if summary else h.get('config_summary', {}).get('seed', ''),
                'has_summary': summary_path.exists(),
                'n_ckpts': n_ckpts,
                'curves_rows': curves_lines,
                'psnr': summary.get('final', {}).get('val/psnr') if summary else None,
                'fid': summary.get('final', {}).get('val/rfid') if summary else None,
                'perplexity': summary.get('final', {}).get('val/perplexity') if summary else None,
            })
        # also include un-started handles
        for h in handles:
            if not (phase_dir / h['run_id']).exists():
                rows.append({
                    'phase': phase_dir.name,
                    'run_id': h['run_id'],
                    'fc_id': h.get('function_call_id', ''),
                    'method': h.get('config_summary', {}).get('method', ''),
                    'K': h.get('config_summary', {}).get('codebook_size', ''),
                    'iter': h.get('config_summary', {}).get('train_iter', ''),
                    'seed': h.get('config_summary', {}).get('seed', ''),
                    'has_summary': False,
                    'n_ckpts': 0,
                    'curves_rows': 0,
                    'psnr': None,
                    'fid': None,
                    'perplexity': None,
                })
    return rows


def _modal_status(fc_id: str) -> str:
    if not fc_id:
        return ''
    try:
        from modal import FunctionCall
        fc = FunctionCall.from_id(fc_id)
        # there is no first-class "status" API on FunctionCall in modal>=0.66;
        # poll with timeout=0 — raises TimeoutError if still running
        try:
            fc.get(timeout = 0)
            return 'DONE'
        except Exception as e:
            name = type(e).__name__
            if 'Timeout' in name:
                return 'RUNNING'
            return f'ERR:{name}'
    except Exception as e:
        return f'ERR:{type(e).__name__}'


def main(phase: Optional[str] = None, modal: bool = False):
    rows = _scan_local(phase)
    if not rows:
        print('no runs found.')
        return

    fmt = '{:<22} {:<35} {:<14} {:<7} {:<6} {:<5} {:<8} {:<8} {:>7} {:>7} {:>7}'
    print(fmt.format(
        'phase', 'run_id', 'method', 'K', 'iter', 'seed',
        'summary', 'modal', 'psnr', 'fid', 'ppl',
    ))
    print('-' * 140)
    for r in rows:
        status = 'DONE' if r['has_summary'] else f"part({r['curves_rows']})"
        modal_status = _modal_status(r['fc_id']) if modal else ''
        print(fmt.format(
            r['phase'][:22], r['run_id'][:35], (r['method'] or '')[:14],
            str(r['K'])[:7], str(r['iter'])[:6], str(r['seed'])[:5],
            status, modal_status,
            f"{r['psnr']:.2f}" if r['psnr'] is not None else '-',
            f"{r['fid']:.2f}" if r['fid'] is not None else '-',
            f"{r['perplexity']:.1f}" if r['perplexity'] is not None else '-',
        ))


if __name__ == '__main__':
    fire.Fire(main)
