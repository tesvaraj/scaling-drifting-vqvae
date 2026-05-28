"""Launch a phase locally or on Modal.

Usage
-----
    # local smoke test
    python -m experiments.launch --phase phase0_smoke --local

    # remote on Modal (parallel)
    python -m experiments.launch --phase phase1_convergence

    # dry run: print configs without executing
    python -m experiments.launch --phase phase2_diagnostic --dry_run

    # limit to first N runs (useful for testing)
    python -m experiments.launch --phase phase3_hybrids --limit 2

After submitting to Modal, function-call IDs are written to
``runs/<phase>/_handles.json`` so individual runs can be killed via
``experiments/scripts/kill_run.py``.
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import fire


def _to_jsonable(obj):
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def _load_phase(phase: str):
    mod = importlib.import_module(f'experiments.configs.{phase}')
    if not hasattr(mod, 'make_runs'):
        raise RuntimeError(f'config module {phase} has no make_runs()')
    return mod


def main(
    phase: str,
    local: bool = False,
    dry_run: bool = False,
    limit: Optional[int] = None,
    only: Optional[str] = None,
    wandb_mode: Optional[str] = None,
    detach: bool = True,
):
    """Launch a phase. ``--only`` is a substring filter on ``run_id``."""
    mod = _load_phase(phase)
    runs = mod.make_runs()
    if only:
        runs = [r for r in runs if only in r.run_id]
    if limit:
        runs = runs[:limit]
    if wandb_mode:
        for r in runs:
            r.wandb_mode = wandb_mode

    print(f'phase={phase}  runs={len(runs)}  local={local}  dry_run={dry_run}')
    for r in runs:
        print(f'  - {r.run_id}  method={r.method}  K={r.codebook_size}  '
              f'iter={r.train_iter}  seed={r.seed}')

    if dry_run:
        return

    out_dir = Path('runs') / phase
    out_dir.mkdir(parents = True, exist_ok = True)

    if local:
        from experiments.train import train
        for cfg in runs:
            print(f'\n=== {cfg.run_id} ===')
            train(cfg)
        return

    # ----- Modal path -----
    from experiments.modal_app import app, run_experiment

    handles_path = out_dir / '_handles.json'
    handles = []

    print('\nspawning Modal calls...')
    with app.run(detach = detach):
        # spawn all in parallel (no blocking)
        for cfg in runs:
            cfg_dict = _to_jsonable(asdict(cfg))
            call = run_experiment.spawn(cfg_dict)
            handles.append({
                'run_id': cfg.run_id,
                'function_call_id': call.object_id,
                'submitted_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'config_summary': {
                    'method': cfg.method,
                    'codebook_size': cfg.codebook_size,
                    'train_iter': cfg.train_iter,
                    'seed': cfg.seed,
                    'dataset': cfg.dataset,
                },
            })
            print(f'  spawned {cfg.run_id} -> {call.object_id}')

        # save handles before waiting, so user can inspect / kill
        with open(handles_path, 'w') as f:
            json.dump(handles, f, indent = 2)
        print(f'\nsaved handles to {handles_path}')

        if detach:
            print('detach=True: runs continue on Modal after this exits.')
            print('Use `python -m experiments.scripts.status` to monitor.')
        else:
            print('\nblocking until all runs finish...')
            for h in handles:
                from modal import FunctionCall
                fc = FunctionCall.from_id(h['function_call_id'])
                try:
                    res = fc.get()
                    print(f'  done {h["run_id"]}: psnr={res.get("final", {}).get("val/psnr", "?")}')
                except Exception as e:
                    print(f'  FAILED {h["run_id"]}: {e}')


if __name__ == '__main__':
    fire.Fire(main)
