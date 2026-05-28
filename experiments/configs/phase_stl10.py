"""STL-10 cross-dataset comparison — replaces CelebA (Google Drive quota issues).

STL-10: 96x96 natural images from 10 semantic categories, resized to 64x64.
105k training images (train+unlabeled split), 8k test. Downloads from AWS CDN.
Same encoder depth as planned CelebA (n_downsample=3 → 8x8 latent at 64x64).

6 runs: vanilla_ema + drift_no_pp_ste × 3 seeds × K=512 × 30k.
"""
from __future__ import annotations

from experiments.train import TrainConfig
from experiments.configs.base import variants


PHASE = 'phase_stl10'


def make_runs() -> list[TrainConfig]:
    base = TrainConfig(
        phase = PHASE,
        dataset = 'stl10',
        train_iter = 30000,
        batch_size = 64,
        codebook_size = 512,
        log_every = 50,
        val_every = 1000,
        image_every = 2500,
        ckpt_every = 5000,
        eval_ssim = True,
        eval_lpips = True,
        eval_fid = True,
        tags = ['stl10', 'generalization'],
    )

    seeds = [0, 1, 2]
    overrides = []
    for s in seeds:
        overrides.append({
            'method': 'vanilla_ema',
            'seed': s,
            'run_id': f'stl10_vanilla_ema_K512_seed{s}',
        })
        overrides.append({
            'method': 'drift',
            'seed': s,
            'energy_terms': ('nn', 'pn'),
            'rotation_trick': False,
            'run_id': f'stl10_drift_no_pp_ste_K512_seed{s}',
        })

    return variants(base, overrides)
