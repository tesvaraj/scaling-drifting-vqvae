"""Linear probe for frozen VQ-VAE representations.

For each image:
  1. Encode to L=H*W discrete tokens via the frozen VQ-VAE encoder
  2. Look up the codebook vector for each token
  3. Mean-pool across tokens → d-dim feature vector
  4. Normalize (StandardScaler) → train logistic regression
  5. Report top-1 and top-5 accuracy on CIFAR-100 test set

Hypothesis: drift's near-uniform codebook (~400 effective codes vs EMA's ~140)
produces more semantically distinct codes, so each code carries less ambiguous
meaning → higher linear probe accuracy.

Usage
-----
    # run on Modal (reads checkpoints from /vol/runs/phase_cifar100)
    modal run experiments/modal_app.py::linear_probe
    modal run experiments/modal_app.py::linear_probe --k 1024
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class ProbeConfig:
    vqvae_phase: str = 'phase_cifar100'
    vqvae_run_id: str = 'cifar100_vanilla_ema_K512_seed0'
    vqvae_ckpt: str = 'final.pt'
    dataset: str = 'cifar100'

    data_root: str = '~/data'
    vqvae_root: str = './runs'
    out_root: str = './runs'

    # sklearn LogisticRegression C sweep (small grid)
    C_values: tuple = (0.01, 0.1, 1.0, 10.0)
    max_iter: int = 1000

    num_workers: int = 2
    device: Optional[str] = None


@torch.no_grad()
def encode_features(model, loader, device: str, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode all images to normalised token-frequency histograms.

    Feature for image i: fraction of its L tokens assigned to each of the K codes.
    Shape (N, K). This preserves code identity — which codes are used — rather than
    collapsing everything into a mean embedding.

    Returns (features, labels) arrays of shape (N, K) and (N,).
    """
    features, labels = [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        else:
            x = batch
            y = torch.zeros(x.shape[0], dtype=torch.long)

        x = x.to(device)
        indices = model.encode_indices(x)          # (B, H, W)
        B, H, W = indices.shape
        flat_idx = indices.reshape(B, H * W)       # (B, L)

        hist = torch.zeros(B, K, device=device)
        hist.scatter_add_(1, flat_idx,
                          torch.ones_like(flat_idx, dtype=torch.float))
        hist = hist / (H * W)                      # normalise to frequencies

        features.append(hist.cpu())
        labels.append(y if torch.is_tensor(y) else torch.tensor(y))

    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def run_probe(cfg: ProbeConfig) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import sys
    sys.path.insert(0, '/root/scaling_drifting_vqvae')
    from experiments.prior import load_vqvae_frozen
    from experiments.data import build_dataset

    device = cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()

    ckpt_path = str(Path(cfg.vqvae_root) / cfg.vqvae_phase / cfg.vqvae_run_id
                    / 'checkpoints' / cfg.vqvae_ckpt)
    print(f'Loading {ckpt_path}')
    model, vqvae_cfg = load_vqvae_frozen(ckpt_path, device)
    K = vqvae_cfg.codebook_size

    ds = build_dataset(cfg.dataset, cfg.data_root)
    train_loader = DataLoader(ds.train, batch_size=256, shuffle=False,
                              num_workers=cfg.num_workers)
    val_loader = DataLoader(ds.val, batch_size=256, shuffle=False,
                            num_workers=cfg.num_workers)

    print('Encoding train set...')
    X_train, y_train = encode_features(model, train_loader, device, K)
    print('Encoding val/test set...')
    X_test, y_test = encode_features(model, val_loader, device, K)

    print(f'Features: train={X_train.shape}  test={X_test.shape}  K={K}')

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    best_top1, best_C = 0.0, None
    results_by_C = {}
    for C in cfg.C_values:
        clf = LogisticRegression(C=C, max_iter=cfg.max_iter, random_state=0,
                                 solver='lbfgs')
        clf.fit(X_train_s, y_train)

        top1 = float(clf.score(X_test_s, y_test))

        proba = clf.predict_proba(X_test_s)
        top5_hits = np.sort(proba, axis=1)[:, -5:]
        top5_labels = np.argsort(proba, axis=1)[:, -5:]
        top5 = float((top5_labels == y_test[:, None]).any(axis=1).mean())

        results_by_C[C] = {'top1': top1, 'top5': top5}
        print(f'  C={C:.2f}  top1={top1*100:.1f}%  top5={top5*100:.1f}%')
        if top1 > best_top1:
            best_top1 = top1
            best_C = C

    best = results_by_C[best_C]
    summary = {
        'run_id': cfg.vqvae_run_id,
        'method': vqvae_cfg.method,
        'K': K,
        'feature_dim': K,
        'best_C': best_C,
        'best_top1': best['top1'],
        'best_top5': best['top5'],
        'results_by_C': results_by_C,
        'wallclock_s': time.time() - t0,
    }

    out_dir = Path(cfg.out_root) / 'phase_linear_probe' / cfg.vqvae_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'probe_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary
