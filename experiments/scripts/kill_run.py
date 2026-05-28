"""Kill a Modal function call by run_id.

Reads ``runs/<phase>/_handles.json`` to find the function-call id, then
calls ``modal.FunctionCall.from_id(...).cancel()``.

Usage
-----
    python -m experiments.scripts.kill_run --phase phase1_convergence --run_id cifar10_drift_K512_seed0
    python -m experiments.scripts.kill_run --phase phase4_large_k --run_id ALL  # cancel everything in phase
"""
from __future__ import annotations

import json
from pathlib import Path

import fire


def main(phase: str, run_id: str):
    handles_path = Path('runs') / phase / '_handles.json'
    if not handles_path.exists():
        print(f'no handles file at {handles_path}')
        return
    with open(handles_path) as f:
        handles = json.load(f)

    matches = handles if run_id == 'ALL' else [h for h in handles if h['run_id'] == run_id]
    if not matches:
        print(f'no handle for {run_id} in {phase}')
        return

    from modal import FunctionCall
    for h in matches:
        fc_id = h['function_call_id']
        try:
            fc = FunctionCall.from_id(fc_id)
            fc.cancel()
            print(f'cancelled {h["run_id"]} ({fc_id})')
        except Exception as e:
            print(f'failed to cancel {h["run_id"]} ({fc_id}): {e}')


if __name__ == '__main__':
    fire.Fire(main)
