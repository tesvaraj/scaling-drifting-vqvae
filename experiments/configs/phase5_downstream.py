"""Phase 5 — Downstream generation. Train autoregressive priors on codes.

This phase doesn't reuse the autoencoder training script; it's its own
sub-pipeline that:
    1. Loads a trained VQ-VAE checkpoint from phase 1 or 6.
    2. Caches the codes for the training set (8x8 int sequences).
    3. Trains a small autoregressive transformer over those codes.
    4. Samples code sequences, decodes them through the VQ-VAE, and
       computes sample-FID against the test set.

This file just documents which (method, seed) checkpoints to use. The
actual implementation lives in ``experiments/scripts/train_prior.py``,
which is built out only after we have phase 1 checkpoints to feed it.
"""
from __future__ import annotations

PHASE = 'phase5_downstream'

# Which checkpoints to train priors on. Filled in after Phase 1 finishes.
# Each entry: {'phase': str, 'run_id': str, 'ckpt': str (e.g. 'final.pt')}
CHECKPOINTS = [
    # {'phase': 'phase1_convergence', 'run_id': 'cifar10_vanilla_ema_K512_seed0', 'ckpt': 'final.pt'},
    # {'phase': 'phase1_convergence', 'run_id': 'cifar10_drift_K512_seed0', 'ckpt': 'final.pt'},
]
