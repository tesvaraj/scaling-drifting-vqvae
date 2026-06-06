"""CNN probe on frozen VQ-VAE latents — downstream transfer-learning evaluation.

This is the CNN upgrade of ``linear_probe.py``. The question is the same but the
probe is stronger: instead of a logistic regression on a bag-of-codewords
histogram, we train a small convolutional head on the *spatial* latent grid the
tokenizer produces, with the VQ-VAE encoder and codebook frozen.

Transfer-learning framing
-------------------------
The VQ-VAE is pretrained with a reconstruction objective. We freeze it and treat
its encoder + codebook as a fixed feature extractor, then train only a small head
on a new task (CIFAR-100 / Tiny ImageNet classification). The head capacity and
training recipe are held fixed across backbones, so any accuracy difference comes
from the frozen features alone. That isolates the question: do Drift* codes carry
more class-discriminative structure than EMA codes?

Representations (a built-in ablation)
-------------------------------------
'zq'    quantized latents the decoder actually consumes, gathered from the
        codebook by index -> (B, d, H, W). PRIMARY: this makes the comparison
        about the codebook.
'ze'    pre-quantization encoder output -> (B, d, H, W). Isolates the encoder
        from the codebook (no quantization).
'codes' raw discrete indices -> (B, H, W), embedded by a learned table in the
        head. Tests the discrete tokens directly.
'pixels' raw images (no VQ-VAE). Reference for how much tokenizing costs.

Backbones
---------
'vqvae'  a trained checkpoint (the real experiment).
'random' the same architecture with random (untrained) weights. Lower reference:
         how much of the probe accuracy is due to pretraining vs. architecture.
         Also used as a no-checkpoint smoke test.

Usage
-----
    # local smoke test (no checkpoint, random backbone, tiny):
    python -m experiments.cnn_probe --smoke

    # on Modal (reads checkpoints from /vol/runs), via modal_app entrypoint:
    modal run experiments/modal_app.py::cnn_probe --k 512
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------- config ----------

@dataclass
class CNNProbeConfig:
    # which frozen backbone to probe
    backbone: str = 'vqvae'              # 'vqvae' | 'random'
    vqvae_phase: str = 'phase_cifar100'
    vqvae_run_id: str = 'cifar100_vanilla_ema_K512_seed0'
    vqvae_ckpt: str = 'final.pt'
    # only used when backbone == 'random' (build arch without loading weights)
    random_method: str = 'vanilla_ema'
    random_codebook_size: int = 512

    dataset: str = 'cifar100'

    # representation + head
    representation: str = 'zq'           # 'zq' | 'ze' | 'codes' | 'pixels'
    head: str = 'cnn'                    # 'cnn' | 'linear'
    head_width: int = 128
    code_embed_dim: int = 64

    # training
    epochs: int = 50
    batch_size: int = 256
    lr: float = 3e-3
    weight_decay: float = 5e-4
    label_smoothing: float = 0.0
    val_frac: float = 0.1                # carved from train for model selection
    seed: int = 0

    # paths / io
    run_id: str = 'cnnprobe_debug'
    out_phase: str = 'phase_cnn_probe'
    data_root: str = '~/data'
    vqvae_root: str = './runs'
    out_root: str = './runs'

    num_workers: int = 2
    device: Optional[str] = None
    max_images: Optional[int] = None     # cap dataset size (smoke tests)


# ---------- probe head ----------

class ProbeNet(nn.Module):
    """Small classifier over a latent grid (or raw pixels).

    For 'codes' the input is integer indices, embedded by a learned table; for
    'zq'/'ze'/'pixels' the input is already a (B, C, H, W) feature map. The conv
    trunk is identical across representations so capacity is matched.
    """

    def __init__(self, representation: str, head: str, in_ch: int, grid: int,
                 n_classes: int, width: int, codebook_size: int = 0,
                 code_embed_dim: int = 64):
        super().__init__()
        self.representation = representation
        self.head = head

        if representation == 'codes':
            self.embed = nn.Embedding(codebook_size, code_embed_dim)
            in_ch = code_embed_dim
        else:
            self.embed = None

        if head == 'linear':
            self.trunk = None
            self.classifier = nn.Linear(in_ch * grid * grid, n_classes)
        elif head == 'cnn':
            self.trunk = nn.Sequential(
                nn.Conv2d(in_ch, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                nn.Conv2d(width, 2 * width, 3, padding=1), nn.BatchNorm2d(2 * width), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Linear(2 * width, n_classes)
        else:
            raise ValueError(f'unknown head: {head}')

    def forward(self, x):
        if self.embed is not None:                    # x: (B, H, W) long indices
            x = self.embed(x)                         # (B, H, W, e)
            x = x.permute(0, 3, 1, 2).contiguous()    # (B, e, H, W)
        if self.head == 'linear':
            return self.classifier(x.flatten(1))
        h = self.trunk(x).flatten(1)                  # (B, 2*width)
        return self.classifier(h)


# ---------- backbone + feature extraction ----------

def build_backbone(cfg: CNNProbeConfig, device: str):
    """Return (model_or_None, K, in_channels, grid).

    model is a frozen VQ-VAE for representations that need it; None for 'pixels'.
    """
    import sys
    sys.path.insert(0, '/root/scaling_drifting_vqvae')
    from experiments.data import build_dataset

    ds = build_dataset(cfg.dataset, cfg.data_root)

    if cfg.representation == 'pixels':
        return None, 0, ds.in_channels, ds.image_size, ds

    if cfg.backbone == 'vqvae':
        from experiments.prior import load_vqvae_frozen
        ckpt = str(Path(cfg.vqvae_root) / cfg.vqvae_phase / cfg.vqvae_run_id
                   / 'checkpoints' / cfg.vqvae_ckpt)
        print(f'Loading frozen VQ-VAE: {ckpt}')
        model, vqvae_cfg = load_vqvae_frozen(ckpt, device)
        K = vqvae_cfg.codebook_size
        dim = vqvae_cfg.dim
    elif cfg.backbone == 'random':
        # same architecture, random (untrained) weights -> lower reference
        from experiments.models import VQAutoEncoder
        from experiments.quantizers import build_quantizer
        K = cfg.random_codebook_size
        dim = 64
        quant = build_quantizer(cfg.random_method, dim=dim, codebook_size=K)
        model = VQAutoEncoder(ds.in_channels, 128, dim, ds.n_downsample, quant)
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print(f'Built RANDOM (untrained) backbone: {cfg.random_method} K={K}')
    else:
        raise ValueError(f'unknown backbone: {cfg.backbone}')

    # infer grid size from a dummy image
    dummy = torch.zeros(1, ds.in_channels, ds.image_size, ds.image_size, device=device)
    with torch.no_grad():
        idx = model.encode_indices(dummy)             # (1, H, W)
    grid = idx.shape[-1]
    in_ch = dim if cfg.representation in ('zq', 'ze') else 0
    return model, K, in_ch, grid, ds


def _zq_from_indices(quantizer, idx):
    """Gather quantized code vectors (B, d, H, W) from indices, robust to the
    different quantizer APIs: drift quantizers expose ``indices_to_codes``,
    the library's VectorQuantize / SimVQ expose ``get_output_from_indices``."""
    if hasattr(quantizer, 'indices_to_codes'):
        return quantizer.indices_to_codes(idx)
    if hasattr(quantizer, 'get_output_from_indices'):
        return quantizer.get_output_from_indices(idx)
    raise AttributeError(f'{type(quantizer).__name__} has no index->code method')


@torch.no_grad()
def featurize(model, x, representation: str):
    """Map a batch of images to the requested frozen representation."""
    if representation == 'pixels':
        return x
    if representation == 'ze':
        return model.encoder(x)                       # (B, d, H, W)
    idx = model.encode_indices(x)                     # (B, H, W)
    if representation == 'codes':
        return idx                                    # (B, H, W) long
    if representation == 'zq':
        return _zq_from_indices(model.quantizer, idx) # (B, d, H, W)
    raise ValueError(f'unknown representation: {representation}')


@torch.no_grad()
def cache_features(model, dataset, cfg: CNNProbeConfig, device: str):
    """Encode a whole dataset once with the frozen backbone. Returns (feats, labels)."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=cfg.num_workers)
    feats, labels = [], []
    seen = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        else:
            x, y = batch, torch.zeros(len(batch), dtype=torch.long)
        x = x.to(device)
        f = featurize(model, x, cfg.representation).cpu()
        feats.append(f)
        labels.append(y if torch.is_tensor(y) else torch.tensor(y))
        seen += x.shape[0]
        if cfg.max_images is not None and seen >= cfg.max_images:
            break
    feats = torch.cat(feats)[: cfg.max_images] if cfg.max_images else torch.cat(feats)
    labels = torch.cat(labels)[: cfg.max_images] if cfg.max_images else torch.cat(labels)
    return feats, labels


# ---------- train / eval ----------

def _topk_acc(logits, y, k=5):
    topk = logits.topk(k, dim=1).indices
    return (topk == y[:, None]).any(dim=1).float().mean().item()


def train_probe(cfg: CNNProbeConfig) -> dict:
    import random
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()

    model, K, in_ch, grid, ds = build_backbone(cfg, device)
    n_classes = len(getattr(ds.train, 'classes', range(200))) if hasattr(ds.train, 'classes') \
        else int(max(_infer_labels(ds.train)) + 1)

    print('Caching train features...')
    Xtr, ytr = cache_features(model, ds.train, cfg, device)
    print('Caching test features...')
    Xte, yte = cache_features(model, ds.val, cfg, device)
    n_classes = int(max(ytr.max().item(), yte.max().item()) + 1)
    print(f'features: train={tuple(Xtr.shape)} test={tuple(Xte.shape)}  '
          f'classes={n_classes}  in_ch={in_ch} grid={grid}')

    # carve a validation split from train for model selection
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(len(Xtr), generator=g)
    n_val = int(len(Xtr) * cfg.val_frac)
    vidx, tidx = perm[:n_val], perm[n_val:]
    Xval, yval = Xtr[vidx], ytr[vidx]
    Xtr, ytr = Xtr[tidx], ytr[tidx]

    is_codes = cfg.representation == 'codes'
    feat_in = in_ch if not is_codes else 0
    if cfg.representation == 'pixels':
        feat_in = ds.in_channels
        grid = ds.image_size
    net = ProbeNet(cfg.representation, cfg.head, feat_in, grid, n_classes,
                   cfg.head_width, codebook_size=K, code_embed_dim=cfg.code_embed_dim).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'probe head: {cfg.head} {cfg.representation}  {n_params/1e3:.0f}k params')

    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    def loader(X, y, shuffle):
        return DataLoader(TensorDataset(X, y), batch_size=cfg.batch_size,
                          shuffle=shuffle, drop_last=False)

    @torch.no_grad()
    def evaluate(X, y):
        net.eval()
        top1s, top5s, ns = 0.0, 0.0, 0
        for xb, yb in loader(X, y, False):
            xb, yb = xb.to(device), yb.to(device)
            logits = net(xb)
            top1s += _topk_acc(logits, yb, 1) * len(yb)
            top5s += _topk_acc(logits, yb, 5) * len(yb)
            ns += len(yb)
        return top1s / ns, top5s / ns

    best_val, best = -1.0, {}
    curve = []
    for epoch in range(cfg.epochs):
        net.train()
        for xb, yb in loader(Xtr, ytr, True):
            xb, yb = xb.to(device), yb.to(device)
            loss = lossf(net(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        vt1, vt5 = evaluate(Xval, yval)
        curve.append({'epoch': epoch, 'val_top1': vt1, 'val_top5': vt5})
        if vt1 > best_val:
            best_val = vt1
            tt1, tt5 = evaluate(Xte, yte)
            best = {'epoch': epoch, 'val_top1': vt1, 'test_top1': tt1, 'test_top5': tt5}
        if epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs - 1:
            print(f'  epoch {epoch:3d}  val top1 {vt1*100:.2f}  '
                  f'(best test top1 {best["test_top1"]*100:.2f})')

    summary = {
        'run_id': cfg.run_id,
        'backbone': cfg.backbone,
        'vqvae_run_id': cfg.vqvae_run_id if cfg.backbone == 'vqvae' else None,
        'representation': cfg.representation,
        'head': cfg.head,
        'K': K,
        'n_params': n_params,
        'n_classes': n_classes,
        'best_epoch': best['epoch'],
        'test_top1': best['test_top1'],
        'test_top5': best['test_top5'],
        'val_top1': best['val_top1'],
        'curve': curve,
        'wallclock_s': time.time() - t0,
    }
    out_dir = Path(cfg.out_root) / cfg.out_phase / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'cnn_probe_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'DONE  {cfg.run_id}  test top1 {best["test_top1"]*100:.2f}  '
          f'top5 {best["test_top5"]*100:.2f}')
    return summary


def _infer_labels(dataset):
    ys = getattr(dataset, 'targets', None)
    if ys is not None:
        return np.asarray(ys)
    return np.array([dataset[i][1] for i in range(min(len(dataset), 1000))])


# ---------- local smoke test ----------

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true',
                   help='tiny CPU run with a random backbone, no checkpoint needed')
    args = p.parse_args()

    if args.smoke:
        cfg = CNNProbeConfig(
            backbone='random', random_method='vanilla_ema', random_codebook_size=256,
            dataset='cifar10', representation='zq', head='cnn',
            epochs=2, batch_size=128, max_images=512, num_workers=0,
            run_id='smoke_zq_cnn',
        )
        train_probe(cfg)
    else:
        print('Use --smoke for a local test, or launch via '
              'modal_app.py::cnn_probe for the real experiment.')
