#!/usr/bin/env python
"""
CNN VQ-VAE training harness for the Scaling Drifting VQ-VAE project.

Methods
-------
vanilla : nearest-neighbor VectorQuantize (van den Oord et al., 2017)
simvq   : SimVQ reparam (Zhu et al., 2025)
drift   : Drifting VQ-VAE (Liu, 2026), our method under test

Run
---
# quick local sanity test on MPS / CPU (no wandb)
python examples/autoencoder_drifting.py \
    --dataset cifar10 --method drift --train_iter 200 --no_wandb --batch_size 64

# full GPU run with wandb
python examples/autoencoder_drifting.py \
    --dataset cifar10 --method drift --codebook_size 512 \
    --train_iter 30000 --wandb_project drifting-vqvae
"""

from __future__ import annotations

import math
import os
from collections import deque
from pathlib import Path
from typing import Optional

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm.auto import trange

from vector_quantize_pytorch import VectorQuantize, SimVQ, DriftingVQ


# ---------- CNN encoder / decoder ----------

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding = 1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding = 1),
        )

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, in_ch, hidden, dim, n_downsample):
        super().__init__()
        layers = [nn.Conv2d(in_ch, hidden, 3, padding = 1)]
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(hidden, hidden, 4, stride = 2, padding = 1),
                ResBlock(hidden),
            ]
        layers += [ResBlock(hidden), nn.GroupNorm(8, hidden), nn.SiLU(),
                   nn.Conv2d(hidden, dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_ch, hidden, dim, n_upsample):
        super().__init__()
        layers = [nn.Conv2d(dim, hidden, 1), ResBlock(hidden)]
        for _ in range(n_upsample):
            layers += [
                nn.ConvTranspose2d(hidden, hidden, 4, stride = 2, padding = 1),
                ResBlock(hidden),
            ]
        layers += [nn.GroupNorm(8, hidden), nn.SiLU(),
                   nn.Conv2d(hidden, out_ch, 3, padding = 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VQAutoEncoder(nn.Module):
    def __init__(self, in_ch, hidden, dim, n_downsample, quantizer):
        super().__init__()
        self.encoder = Encoder(in_ch, hidden, dim, n_downsample)
        self.quantizer = quantizer
        self.decoder = Decoder(in_ch, hidden, dim, n_downsample)

    def forward(self, x):
        h = self.encoder(x)
        q, indices, vq_loss = self.quantizer(h)
        recon = self.decoder(q)
        return recon, indices, vq_loss


# ---------- quantizer factory ----------

def build_quantizer(method, dim, codebook_size, tau, energy_weight, n_hidden_subsample,
                    l2_normalize):
    if method == 'vanilla':
        return VectorQuantize(
            dim = dim,
            codebook_size = codebook_size,
            accept_image_fmap = True,
            rotation_trick = True,
        )
    if method == 'simvq':
        return SimVQ(
            dim = dim,
            codebook_size = codebook_size,
            channel_first = True,
            rotation_trick = True,
        )
    if method == 'drift':
        return DriftingVQ(
            dim = dim,
            codebook_size = codebook_size,
            channel_first = True,
            tau = tau,
            energy_weight = energy_weight,
            n_hidden_subsample = n_hidden_subsample,
            l2_normalize = l2_normalize,
            rotation_trick = True,
        )
    raise ValueError(f'unknown method: {method}')


# ---------- dataset ----------

def build_dataset(dataset_name, data_root):
    name = dataset_name.lower()
    if name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train = datasets.CIFAR10(root = data_root, train = True, download = True, transform = transform)
        val = datasets.CIFAR10(root = data_root, train = False, download = True, transform = transform)
        return train, val, 3, 2  # in_ch, n_downsample (32 -> 8)
    if name == 'celeba':
        transform = transforms.Compose([
            transforms.Resize(64), transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train = datasets.CelebA(root = data_root, split = 'train', download = True, transform = transform)
        val = datasets.CelebA(root = data_root, split = 'valid', download = True, transform = transform)
        return train, val, 3, 3  # 64 -> 8
    if name == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        train = datasets.FashionMNIST(root = data_root, train = True, download = True, transform = transform)
        val = datasets.FashionMNIST(root = data_root, train = False, download = True, transform = transform)
        return train, val, 1, 2
    raise ValueError(f'unknown dataset: {dataset_name}')


def cycle(loader, device):
    while True:
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            yield x.to(device, non_blocking = True)


# ---------- metrics ----------

def code_usage_counts(indices, codebook_size):
    """Histogram of code assignments. Uses CPU for MPS compatibility."""
    flat = indices.reshape(-1).detach().cpu()
    return torch.bincount(flat, minlength = codebook_size).float()


def perplexity_from_counts(counts):
    probs = counts / counts.sum().clamp(min = 1)
    nz = probs > 0
    entropy = -(probs[nz] * probs[nz].log()).sum()
    return entropy.exp().item()


def utilization_from_counts(counts):
    return (counts > 0).float().mean().item()


def psnr(recon, x):
    mse = F.mse_loss(recon, x).clamp(min = 1e-12)
    return (10.0 * torch.log10(4.0 / mse)).item()  # peak = 2 on [-1, 1] range


def grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


# ---------- evaluation pass ----------

@torch.no_grad()
def evaluate(model, val_loader, device, codebook_size, max_batches = 20):
    model.eval()
    rec_losses, psnrs = [], []
    counts_total = torch.zeros(codebook_size)
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        recon, indices, _ = model(x)
        recon = recon.clamp(-1.0, 1.0)
        rec_losses.append(F.mse_loss(recon, x).item())
        psnrs.append(psnr(recon, x))
        counts_total += code_usage_counts(indices, codebook_size)
    model.train()
    return {
        'val/rec_loss': sum(rec_losses) / len(rec_losses),
        'val/psnr': sum(psnrs) / len(psnrs),
        'val/utilization': utilization_from_counts(counts_total),
        'val/perplexity': perplexity_from_counts(counts_total),
    }, counts_total


# ---------- train ----------

def train(
    # core
    dataset = 'cifar10',
    method = 'drift',
    train_iter = 30000,
    batch_size = 128,
    lr = 3e-4,
    seed = 0,
    # model
    dim = 64,
    hidden = 128,
    codebook_size = 512,
    # drifting hyperparams
    tau = 1.0,
    energy_weight = 1.0,
    n_hidden_subsample = 1024,
    l2_normalize = True,
    # loss weights
    alpha = 1.0,  # weight on the quantizer loss (rec_loss + alpha * vq_loss)
    # I/O
    data_root = '~/data',
    out_dir = './runs',
    # logging cadence
    log_every = 25,
    eval_every = 1000,
    image_every = 1000,
    metric_window = 50,
    # wandb
    wandb_project = 'drifting-vqvae',
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[str] = None,
    no_wandb = False,
    # misc
    num_workers = 2,
    device: Optional[str] = None,
    compile = False,
):
    torch.manual_seed(seed)

    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    print(f'device: {device}')

    run_name = wandb_run_name or f'{dataset}_{method}_K{codebook_size}_d{dim}_tau{tau}_ew{energy_weight}'
    run_dir = Path(out_dir).expanduser() / run_name
    run_dir.mkdir(parents = True, exist_ok = True)

    # ----- data -----
    train_ds, val_ds, in_ch, n_downsample = build_dataset(dataset, os.path.expanduser(data_root))
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True,
                              num_workers = num_workers, drop_last = True,
                              pin_memory = (device == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size = batch_size, shuffle = False,
                            num_workers = num_workers, drop_last = False,
                            pin_memory = (device == 'cuda'))

    # ----- model -----
    quantizer = build_quantizer(method, dim, codebook_size, tau, energy_weight,
                                n_hidden_subsample, l2_normalize)
    model = VQAutoEncoder(in_ch, hidden, dim, n_downsample, quantizer).to(device)
    opt = AdamW(model.parameters(), lr = lr)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable params: {n_params/1e6:.2f}M')

    if compile and device == 'cuda':
        model = torch.compile(model)

    # ----- wandb -----
    use_wandb = not no_wandb
    if use_wandb:
        try:
            import wandb
        except ImportError:
            print('wandb not installed; running without it. `pip install wandb`')
            use_wandb = False
    if use_wandb:
        tags = wandb_tags.split(',') if wandb_tags else []
        wandb.init(
            project = wandb_project,
            entity = wandb_entity,
            name = run_name,
            tags = tags,
            config = dict(
                dataset = dataset, method = method, train_iter = train_iter,
                batch_size = batch_size, lr = lr, seed = seed,
                dim = dim, hidden = hidden, codebook_size = codebook_size,
                tau = tau, energy_weight = energy_weight,
                n_hidden_subsample = n_hidden_subsample, l2_normalize = l2_normalize,
                alpha = alpha, n_downsample = n_downsample, n_params = n_params,
                device = device,
            ),
            dir = str(run_dir),
        )

    # ----- fixed eval batch for reconstruction visualization -----
    fixed_batch = next(iter(val_loader))
    fixed_batch = (fixed_batch[0] if isinstance(fixed_batch, (list, tuple)) else fixed_batch)[:16].to(device)

    # ----- windowed running metrics -----
    win = dict(rec_loss = deque(maxlen = metric_window),
               vq_loss = deque(maxlen = metric_window),
               psnr = deque(maxlen = metric_window))
    # rolling code-usage counts for a stable utilization/perplexity signal
    rolling_counts = torch.zeros(codebook_size)
    rolling_window = deque(maxlen = metric_window)

    # ----- train loop -----
    dl_iter = cycle(train_loader, device)
    pbar = trange(train_iter)
    for step in pbar:
        x = next(dl_iter)
        recon, indices, vq_loss = model(x)
        recon = recon.clamp(-1.0, 1.0)

        rec_loss = F.mse_loss(recon, x)
        loss = rec_loss + alpha * vq_loss

        opt.zero_grad()
        loss.backward()
        gn = grad_norm(model.parameters())
        opt.step()

        with torch.no_grad():
            counts = code_usage_counts(indices, codebook_size)
            win['rec_loss'].append(rec_loss.item())
            win['vq_loss'].append(vq_loss.item())
            win['psnr'].append(psnr(recon, x))
            rolling_window.append(counts)
            rolling_counts = torch.stack(list(rolling_window)).sum(0)

        if step % log_every == 0:
            mean_rec = sum(win['rec_loss']) / len(win['rec_loss'])
            mean_vq = sum(win['vq_loss']) / len(win['vq_loss'])
            mean_psnr = sum(win['psnr']) / len(win['psnr'])
            util = utilization_from_counts(rolling_counts)
            ppl = perplexity_from_counts(rolling_counts)
            dead_frac = 1.0 - util

            pbar.set_description(
                f'rec {mean_rec:.4f} | vq {mean_vq:.4f} | psnr {mean_psnr:.2f} | '
                f'util {util*100:.1f}% | ppl {ppl:.1f}'
            )

            if use_wandb:
                log = {
                    'train/rec_loss': mean_rec,
                    'train/vq_loss': mean_vq,
                    'train/total_loss': mean_rec + alpha * mean_vq,
                    'train/psnr': mean_psnr,
                    'train/grad_norm': gn,
                    'codebook/utilization': util,
                    'codebook/perplexity': ppl,
                    'codebook/dead_fraction': dead_frac,
                    'codebook/active_codes': int((rolling_counts > 0).sum().item()),
                    'opt/lr': opt.param_groups[0]['lr'],
                    'step': step,
                }
                # drifting-specific: U_pp, U_nn, U_pn breakdown
                bd = getattr(model.quantizer, 'last_breakdown', None)
                if bd:
                    for k, v in bd.items():
                        log[f'drift/{k}'] = v.item() if torch.is_tensor(v) else v
                wandb.log(log, step = step)

        # ---- eval ----
        if step > 0 and (step % eval_every == 0 or step == train_iter - 1):
            val_metrics, val_counts = evaluate(model, val_loader, device, codebook_size)
            print(f'[step {step}] ' + ' | '.join(f'{k.split("/")[-1]} {v:.4f}' for k, v in val_metrics.items()))
            if use_wandb:
                val_metrics['step'] = step
                # code-usage histogram across val set
                val_metrics['codebook/usage_hist'] = wandb.Histogram(
                    np_histogram = (val_counts.numpy(),
                                    list(range(codebook_size + 1)))
                )
                wandb.log(val_metrics, step = step)

        # ---- image logging ----
        if step > 0 and (step % image_every == 0 or step == train_iter - 1):
            with torch.no_grad():
                rec_fixed, _, _ = model(fixed_batch)
                rec_fixed = rec_fixed.clamp(-1.0, 1.0)
                grid = make_grid(torch.cat([fixed_batch, rec_fixed], dim = 0) * 0.5 + 0.5, nrow = 16)
            out_img = run_dir / f'recon_{step:06d}.png'
            from torchvision.utils import save_image
            save_image(grid, out_img)
            if use_wandb:
                wandb.log({'samples/reconstructions': wandb.Image(str(out_img))}, step = step)

    # ----- final checkpoint -----
    ckpt = {
        'model': model.state_dict(),
        'config': dict(
            dataset = dataset, method = method, dim = dim, hidden = hidden,
            codebook_size = codebook_size, tau = tau, energy_weight = energy_weight,
            l2_normalize = l2_normalize, n_downsample = n_downsample, in_ch = in_ch,
        ),
    }
    torch.save(ckpt, run_dir / 'final.pt')
    print(f'saved checkpoint to {run_dir / "final.pt"}')

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    fire.Fire(train)
