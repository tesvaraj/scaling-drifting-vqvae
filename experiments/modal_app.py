"""Modal harness for the Scaling Drifting VQ-VAE project.

Architecture
------------
* A single Modal Volume ``drifting-vqvae`` holds the dataset cache (``/vol/data``)
  and run outputs (``/vol/runs``). Datasets download once and persist.
* A single image carries all deps: torch, einops, einx, lpips, torchmetrics,
  wandb, plus the local source code mounted into ``/root/scaling_drifting_vqvae``.
* ``run_experiment`` runs one ``TrainConfig`` to completion on an L40S.
* ``launch_phase`` is a local CLI entrypoint that imports a phase config,
  builds the ``TrainConfig`` list, and dispatches them as Modal calls.

Setup
-----
    pip install modal
    modal token new                # one-time
    modal secret create wandb WANDB_API_KEY=...
    modal volume create drifting-vqvae   # auto-created by from_name(create_if_missing=True)

Run
---
    # smoke test (no Modal)
    python -m experiments.launch --phase phase0_smoke --local

    # phase 1, on Modal
    python -m experiments.launch --phase phase1_convergence

The two paths share the same ``TrainConfig`` plumbing — locally we call
``experiments.train.train``; remotely we call ``run_experiment.remote``
with the same dataclass serialized to a dict.
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = 'drifting-vqvae-231n'
VOLUME_NAME = 'drifting-vqvae'

# ----- image -----

LOCAL_ROOT = Path(__file__).resolve().parent.parent  # repo root

image = (
    modal.Image.debian_slim(python_version = '3.11')
    .apt_install('git')
    .pip_install(
        'torch>=2.4',
        'torchvision',
        'einops>=0.8.0',
        'einx>=0.3.0',
        'fire',
        'tqdm',
        'wandb',
        'numpy',
        'matplotlib',
        'lpips',
        'torchmetrics[image]',
        'scipy',
    )
    # add the local repo so `experiments` and `vector_quantize_pytorch` are importable
    .add_local_dir(
        str(LOCAL_ROOT),
        '/root/scaling_drifting_vqvae',
        ignore = ['runs/*', '.git/*', '.pytest_cache/*', '__pycache__/*',
                  '**/__pycache__/*', '*.pdf'],
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing = True)

app = modal.App(APP_NAME)


# ----- remote function -----

@app.function(
    image = image,
    gpu = os.environ.get('MODAL_GPU', 'L40S'),
    volumes = {'/vol': volume},
    secrets = [modal.Secret.from_name('wandb')],
    timeout = 6 * 60 * 60,  # 6h
)
def run_experiment(config_dict: dict) -> dict:
    """Run one TrainConfig (serialized as dict) inside a Modal container."""
    import sys
    sys.path.insert(0, '/root/scaling_drifting_vqvae')

    from experiments.train import TrainConfig, train

    # rewrite paths to live on the persistent volume so checkpoints survive
    config_dict.setdefault('data_root', '/vol/data')
    config_dict.setdefault('out_root', '/vol/runs')

    # cast list back to tuple for fields that expect it
    if 'energy_terms' in config_dict and isinstance(config_dict['energy_terms'], list):
        config_dict['energy_terms'] = tuple(config_dict['energy_terms'])

    cfg = TrainConfig(**config_dict)
    summary = train(cfg)

    # commit volume so writes are persisted
    try:
        volume.commit()
    except Exception:
        pass
    return summary


# ----- helpers to pull results back locally -----

@app.function(image = image, volumes = {'/vol': volume}, timeout = 600)
def list_runs(phase: str = '') -> list:
    """List run directories under /vol/runs/<phase> (or all phases if empty)."""
    base = Path('/vol/runs')
    if not base.exists():
        return []
    out = []
    for ph_dir in sorted(base.iterdir()):
        if not ph_dir.is_dir():
            continue
        if phase and ph_dir.name != phase:
            continue
        for run_dir in sorted(ph_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            summary_path = run_dir / 'summary.json'
            ckpt_dir = run_dir / 'checkpoints'
            curves = run_dir / 'curves.csv'
            out.append({
                'phase': ph_dir.name,
                'run_id': run_dir.name,
                'has_summary': summary_path.exists(),
                'n_checkpoints': len(list(ckpt_dir.glob('*.pt'))) if ckpt_dir.exists() else 0,
                'curves_size': curves.stat().st_size if curves.exists() else 0,
            })
    return out


@app.function(image = image, volumes = {'/vol': volume}, timeout = 600)
def fetch_run_summary(phase: str, run_id: str) -> dict:
    import json
    path = Path('/vol/runs') / phase / run_id / 'summary.json'
    if not path.exists():
        return {'error': f'no summary at {path}'}
    with open(path) as f:
        return json.load(f)


@app.function(image = image, volumes = {'/vol': volume}, timeout = 600)
def fetch_curves(phase: str, run_id: str) -> str:
    path = Path('/vol/runs') / phase / run_id / 'curves.csv'
    if not path.exists():
        return ''
    return path.read_text()


@app.function(image = image, volumes = {'/vol': volume}, timeout = 600)
def fetch_val_curves(phase: str, run_id: str) -> str:
    path = Path('/vol/runs') / phase / run_id / 'val_curves.csv'
    if not path.exists():
        return ''
    return path.read_text()
