"""Pull live training data from wandb for analysis.

This is the script the user (and the code assistant) calls when they want to
inspect *live* training, not just on-disk artifacts. It uses ``wandb.Api``
to fetch the latest metrics from the cloud — works even mid-training.

Usage
-----
    # list all runs in a wandb group
    python -m experiments.scripts.inspect_runs --group phase1_convergence

    # detailed metrics for one run
    python -m experiments.scripts.inspect_runs --run cifar10_drift_K512_seed0

    # all phase1 runs, last value of each key metric
    python -m experiments.scripts.inspect_runs --group phase1_convergence --table
"""
from __future__ import annotations

from typing import Optional

import fire


DEFAULT_PROJECT = 'drifting-vqvae-231n'


def _api():
    import wandb
    return wandb.Api(timeout = 60)


def list_group(group: str, project: str = DEFAULT_PROJECT, entity: Optional[str] = None,
               table: bool = False):
    api = _api()
    path = f'{entity}/{project}' if entity else project
    runs = api.runs(path, filters = {'group': group}, per_page = 200)
    runs = list(runs)
    if not runs:
        print(f'no runs in group {group}')
        return

    if table:
        cols = ['name', 'state', 'step', 'val/psnr', 'val/ssim', 'val/rfid',
                'val/utilization', 'val/perplexity']
        widths = [40, 10, 8, 9, 9, 9, 14, 14]
        print(' '.join(c.ljust(w) for c, w in zip(cols, widths)))
        for r in runs:
            s = r.summary._json_dict if hasattr(r.summary, '_json_dict') else dict(r.summary)
            vals = [
                r.name[:40], r.state[:10], str(s.get('_step', '')),
                f"{s.get('val/psnr', float('nan')):.2f}",
                f"{s.get('val/ssim', float('nan')):.4f}",
                f"{s.get('val/rfid', float('nan')):.2f}",
                f"{s.get('val/utilization', float('nan')):.3f}",
                f"{s.get('val/perplexity', float('nan')):.1f}",
            ]
            print(' '.join(v.ljust(w) for v, w in zip(vals, widths)))
    else:
        for r in runs:
            print(f'{r.name}  state={r.state}  url={r.url}')


def one_run(run: str, project: str = DEFAULT_PROJECT, entity: Optional[str] = None,
            n_last: int = 5):
    api = _api()
    path = f'{entity}/{project}' if entity else project
    matches = api.runs(path, filters = {'display_name': run})
    matches = list(matches)
    if not matches:
        # try substring search
        matches = [r for r in api.runs(path, per_page = 500) if run in r.name]
    if not matches:
        print(f'no run named {run}')
        return
    r = matches[0]
    print(f'name: {r.name}')
    print(f'state: {r.state}')
    print(f'url: {r.url}')
    print(f'group: {r.group}')
    print('--- summary ---')
    s = r.summary._json_dict if hasattr(r.summary, '_json_dict') else dict(r.summary)
    for k in sorted(s.keys()):
        v = s[k]
        if isinstance(v, (int, float, str, bool)):
            print(f'  {k}: {v}')

    print(f'--- last {n_last} validation rows ---')
    hist = r.history(keys = ['val/psnr', 'val/ssim', 'val/rfid',
                             'val/utilization', 'val/perplexity'],
                     pandas = False)
    rows = [h for h in hist if any(k in h for k in
                                    ('val/psnr', 'val/ssim', 'val/rfid'))]
    for row in rows[-n_last:]:
        print(' ', row)


def main(group: Optional[str] = None, run: Optional[str] = None,
         project: str = DEFAULT_PROJECT, entity: Optional[str] = None,
         table: bool = False, n_last: int = 5):
    if run:
        return one_run(run, project = project, entity = entity, n_last = n_last)
    if group:
        return list_group(group, project = project, entity = entity, table = table)
    print('use --group <name> or --run <name>')


if __name__ == '__main__':
    fire.Fire(main)
