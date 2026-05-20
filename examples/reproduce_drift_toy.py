#!/usr/bin/env python
"""
Reproduce Liu's K=2 toy experiment with the production DriftingVQ class.

Liu (2026) shows that on a 2D two-cluster dataset, vanilla VQ-VAE with K=2
often leaves one code dead while Drifting VQ-VAE uses both. This script runs
both methods across many seeds, prints a per-seed table, saves one scatter
plot per seed, and writes a CSV plus a markdown summary.

Drifting config matches the Colab: tau=0.2, energy_weight=0.25, no L2
normalization, no rotation trick, no STE.

Usage:
    python examples/reproduce_drift_toy.py
    python examples/reproduce_drift_toy.py --n_seeds 50
"""

from __future__ import annotations

import csv
from pathlib import Path

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F

from vector_quantize_pytorch import DriftingVQ, VectorQuantize


def generate_data(n_samples: int, sigma: float = 0.2, seed: int = 0):
    """Two clusters at x = ±0.25, y = 0, with uniform noise of width sigma."""
    g = torch.Generator().manual_seed(seed)
    cluster_id = torch.randint(0, 2, (n_samples,), generator = g)
    x = torch.zeros(n_samples, 2)
    x[:, 0] = cluster_id.float() - 0.5
    x = x / 2 + (torch.rand(n_samples, 2, generator = g) - 0.5) * sigma
    return x, cluster_id


class ToyAutoencoder(nn.Module):
    """Linear encoder/decoder around an arbitrary quantizer."""

    def __init__(self, encoder, quantizer, decoder):
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder

    def forward(self, x):
        h = self.encoder(x).unsqueeze(0)  # quantizers expect (B, N, D)
        q, indices, vq_loss = self.quantizer(h)
        return self.decoder(q.squeeze(0)), indices.squeeze(0), vq_loss


def build_quantizer(method: str, vanilla_mode: str = 'ema'):
    if method == 'vanilla':
        if vanilla_mode == 'classical':
            # Gradient-updated learnable codebook, no EMA. Matches Liu's Colab
            # vanilla setup (the original van den Oord 2017 formulation).
            return VectorQuantize(
                dim = 2, codebook_size = 2,
                learnable_codebook = True,
                ema_update = False,
            )
        return VectorQuantize(dim = 2, codebook_size = 2)
    if method == 'drift':
        return DriftingVQ(
            dim = 2, codebook_size = 2,
            tau = 0.2,
            energy_weight = 0.25,
            l2_normalize = False,
            rotation_trick = False,
            ste = False,
            n_hidden_subsample = 10_000,  # > batch, so no subsampling
        )
    raise ValueError(f'unknown method: {method}')


def get_codes(quantizer) -> torch.Tensor:
    """Return codes as a (K, D) tensor across both quantizer types."""
    cb = quantizer.codebook.detach()
    return cb.squeeze(0) if cb.ndim == 3 else cb


def train_one(method: str, seed: int, vanilla_mode: str = 'ema',
              n_iter: int = 1000, lr: float = 3e-3, batch: int = 1000):
    """Train one quantizer for one seed and return end-of-run state.

    Encoder/decoder are built first under a fixed seed so both methods see the
    same encoder initialization at the same seed. The quantizer is built after
    another reseed so each method is reproducible across reruns.
    """
    data, cluster_id = generate_data(batch, seed = seed)

    torch.manual_seed(seed)
    encoder = nn.Linear(2, 2)
    decoder = nn.Linear(2, 2)

    torch.manual_seed(seed)
    quantizer = build_quantizer(method, vanilla_mode = vanilla_mode)

    model = ToyAutoencoder(encoder, quantizer, decoder)
    opt = torch.optim.Adam(model.parameters(), lr = lr)

    for _ in range(n_iter):
        opt.zero_grad()
        recon, _, vq_loss = model(data)
        loss = F.mse_loss(recon, data) + vq_loss
        loss.backward()
        opt.step()

    with torch.no_grad():
        recon, indices, _ = model(data)
        counts = torch.bincount(indices.reshape(-1), minlength = 2)
        return dict(
            hidden = model.encoder(data),
            codes = get_codes(model.quantizer),
            indices = indices,
            counts = counts,
            rec_mse = F.mse_loss(recon, data).item(),
        )


def save_seed_plot(seed_results: dict, seed: int, out_path: Path, methods: tuple):
    """Save a side-by-side scatter plot for one seed."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize = (10, 5), sharex = True, sharey = True)
    for ax, m in zip(axes, methods):
        r = seed_results[m]
        h, c, idx = r['hidden'].numpy(), r['codes'].numpy(), r['indices'].numpy()
        ax.scatter(h[:, 0], h[:, 1], c = idx, cmap = 'tab10', s = 10, alpha = 0.6, vmin = 0, vmax = 1)
        ax.scatter(c[:, 0], c[:, 1], c = 'k', s = 200, marker = '*',
                   edgecolors = 'yellow', linewidths = 1.5, zorder = 5)
        c0, c1 = int(r['counts'][0]), int(r['counts'][1])
        is_dead = c0 == 0 or c1 == 0
        title = f'{m}  counts=[{c0}, {c1}]'
        if is_dead:
            title += '  DEAD'
        ax.set_title(title)
        ax.set_aspect('equal')
    fig.suptitle(f'K=2 toy, seed {seed}')
    fig.tight_layout()
    fig.savefig(out_path / f'seed_{seed:02d}.png', dpi = 120)
    plt.close(fig)


def save_results_csv(rows: list, out_path: Path):
    with open(out_path / 'results.csv', 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'seed', 'count_0', 'count_1', 'dead', 'rec_mse'])
        for method, seed, c0, c1, dead, rec_mse in rows:
            writer.writerow([method, seed, c0, c1, int(dead), f'{rec_mse:.6f}'])


def save_summary_md(rows: list, dead_counts: dict, n_seeds: int, out_path: Path):
    """Markdown summary suitable for pasting into project notes."""
    lines = [
        f'# K=2 toy reproduction ({n_seeds} seeds)',
        '',
        '## Dead-code rate',
        '',
        '| method | dead seeds | rate |',
        '| --- | --- | --- |',
    ]
    for m, d in dead_counts.items():
        lines.append(f'| {m} | {d} / {n_seeds} | {d / n_seeds:.0%} |')
    lines += [
        '',
        '## Per-seed counts',
        '',
        '| method | seed | count_0 | count_1 | dead | rec_mse |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for method, seed, c0, c1, dead, rec_mse in rows:
        lines.append(f'| {method} | {seed} | {c0} | {c1} | {"yes" if dead else "no"} | {rec_mse:.4f} |')
    (out_path / 'summary.md').write_text('\n'.join(lines) + '\n')


def main(n_seeds: int = 30, vanilla_mode: str = 'ema', out_dir: str = None):
    """vanilla_mode: 'ema' (lucidrains default) or 'classical' (matches Liu's Colab)."""
    if vanilla_mode not in ('ema', 'classical'):
        raise ValueError(f"vanilla_mode must be 'ema' or 'classical', got {vanilla_mode!r}")

    methods = ('vanilla', 'drift')
    suffix = '' if vanilla_mode == 'ema' else f'_classical_vanilla'
    out_path = Path(out_dir or f'./figures/drift_toy_k2{suffix}').expanduser()
    out_path.mkdir(parents = True, exist_ok = True)

    try:
        import matplotlib.pyplot  # noqa: F401
        plot_enabled = True
    except ImportError:
        print('matplotlib not installed; skipping per-seed plots.')
        plot_enabled = False

    rows = []
    dead_counts = {m: 0 for m in methods}

    print(f'{"method":<8}{"seed":>5}{"count_0":>10}{"count_1":>10}{"dead?":>8}{"rec_mse":>12}')
    for seed in range(n_seeds):
        seed_results = {m: train_one(m, seed, vanilla_mode = vanilla_mode) for m in methods}
        for m in methods:
            r = seed_results[m]
            c0, c1 = int(r['counts'][0]), int(r['counts'][1])
            is_dead = c0 == 0 or c1 == 0
            dead_counts[m] += int(is_dead)
            rows.append((m, seed, c0, c1, is_dead, r['rec_mse']))
            print(f'{m:<8}{seed:>5}{c0:>10}{c1:>10}{"yes" if is_dead else "no":>8}{r["rec_mse"]:>12.4f}')
        if plot_enabled:
            save_seed_plot(seed_results, seed, out_path, methods)

    print()
    for m in methods:
        rate = dead_counts[m] / n_seeds
        print(f'{m}: {dead_counts[m]} / {n_seeds} seeds had a dead code ({rate:.0%})')

    save_results_csv(rows, out_path)
    save_summary_md(rows, dead_counts, n_seeds, out_path)
    print(f'\nwrote: {out_path}/results.csv, summary.md, seed_NN.png x {n_seeds}')


if __name__ == '__main__':
    fire.Fire(main)
