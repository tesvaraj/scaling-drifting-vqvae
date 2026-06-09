"""Fetch final reconstruction figures from the Modal volume.

Each run saves figures/recons_XXXXXX.png during training.  This script
pulls the last (highest-step) recons figure for each key comparison run
and saves it locally with a descriptive filename.

Usage:
    modal run experiments/scripts/fetch_recon_figures.py
    modal run experiments/scripts/fetch_recon_figures.py --out_dir figures/paper_recons
"""
from __future__ import annotations

from pathlib import Path
import sys

import modal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.modal_app import app, image, volume

# ---------- runs to fetch ----------
# (output_filename_stem, phase, run_id)
# Using seed0 throughout; one EMA + one Drift★ per dataset at K=512 30k.
RUNS = [
    # CIFAR-10
    ('cifar10_ema',        'phase1_convergence',   'cifar10_vanilla_ema_K512_seed0'),
    ('cifar10_drift',      'phase2b_confirmation', 'cifar10_drift_no_pp_ste_K512_seed0'),
    # CIFAR-100
    ('cifar100_ema',       'phase_cifar100',        'cifar100_vanilla_ema_K512_seed0'),
    ('cifar100_drift',     'phase_cifar100',        'cifar100_drift_no_pp_ste_K512_seed0'),
    # STL-10
    ('stl10_ema',          'phase_stl10',           'stl10_vanilla_ema_K512_seed0'),
    ('stl10_drift',        'phase_stl10',           'stl10_drift_no_pp_ste_K512_seed0'),
    # Tiny ImageNet
    ('tinyimagenet_ema',   'phase_tiny_imagenet',   'tinyimagenet_vanilla_ema_K512_seed0'),
    ('tinyimagenet_drift', 'phase_tiny_imagenet',   'tinyimagenet_drift_no_pp_ste_K512_seed0'),
]


@app.function(image=image, volumes={'/vol': volume}, timeout=120)
def fetch_last_recon(phase: str, run_id: str) -> bytes | None:
    """Return bytes of the highest-step recons_*.png, or None if not found."""
    import glob
    pattern = f'/vol/runs/{phase}/{run_id}/figures/recons_*.png'
    figs = sorted(glob.glob(pattern))
    if not figs:
        print(f'  [MISSING] no recons figures found at {pattern}')
        return None
    chosen = figs[-1]
    print(f'  fetching {chosen}')
    with open(chosen, 'rb') as f:
        return f.read()


@app.local_entrypoint()
def main(out_dir: str = 'figures/paper_recons'):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f'Fetching {len(RUNS)} reconstruction figures → {out}\n')

    saved, missing = [], []
    for stem, phase, run_id in RUNS:
        data = fetch_last_recon.remote(phase, run_id)
        if data:
            fpath = out / f'{stem}_recon_K512_30k.png'
            fpath.write_bytes(data)
            saved.append(fpath.name)
            print(f'  ✓  {fpath.name}  ({len(data)//1024} KB)')
        else:
            missing.append(f'{phase}/{run_id}')
            print(f'  ✗  {phase}/{run_id}  (no figure found)')

    print(f'\nDone.  {len(saved)} saved, {len(missing)} missing.')
    if missing:
        print('Missing runs:', missing)
    print(f'\nNote: each PNG has original images on top row, reconstructions on bottom.')
    print(f'Files are named  <dataset>_<method>_recon_K512_30k.png')
