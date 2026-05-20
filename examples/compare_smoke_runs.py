#!/usr/bin/env python
"""
Build a side-by-side comparison of vanilla / SimVQ / drift smoke runs.

Reads the final eval metrics from runs/_logs/*.log (parsed via regex), stacks
the final reconstruction images from each run directory, and writes a markdown
summary plus a utilization+perplexity bar chart to figures/smoke_runs_k64/.

Expects three runs to already exist under runs/cifar10_{vanilla,simvq,drift}_K64_*
with a recon_001999.png and a sibling log at runs/_logs/<method>.log.

Usage:
    python examples/compare_smoke_runs.py
"""

from __future__ import annotations

import re
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


METHODS = ('vanilla', 'simvq', 'drift')
EVAL_RE = re.compile(
    r'\[step (\d+)\] rec_loss ([\d.]+) \| psnr ([\d.]+) \| utilization ([\d.]+) \| perplexity ([\d.]+)'
)


def parse_log(log_path: Path) -> dict:
    """Return the last eval line's metrics, or None if no eval line found."""
    text = log_path.read_text(errors = 'ignore')
    matches = EVAL_RE.findall(text)
    if not matches:
        return None
    step, rec, psnr, util, ppl = matches[-1]
    return dict(step = int(step), rec_loss = float(rec), psnr = float(psnr),
                utilization = float(util), perplexity = float(ppl))


def find_run_dir(method: str, runs_root: Path) -> Path:
    """Pick the run directory matching this method (assumes one per method)."""
    matches = sorted(runs_root.glob(f'cifar10_{method}_*'))
    if not matches:
        raise FileNotFoundError(f'no run directory for {method} under {runs_root}')
    return matches[-1]


def stack_recons(run_dirs: dict, out_path: Path):
    """Vertically stack the final recon image from each run, labeled."""
    images = {m: Image.open(d / 'recon_001999.png') for m, d in run_dirs.items()}
    w = max(img.width for img in images.values())
    h = max(img.height for img in images.values())

    fig, axes = plt.subplots(len(METHODS), 1, figsize = (w / 80, 1.2 * h * len(METHODS) / 80))
    for ax, m in zip(axes, METHODS):
        ax.imshow(np.asarray(images[m]))
        ax.set_ylabel(m, rotation = 0, ha = 'right', va = 'center', fontsize = 14, fontweight = 'bold')
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_title('top row: originals | bottom row: reconstructions  (CIFAR-10, K=64, 2000 iter)')
    fig.tight_layout()
    fig.savefig(out_path, dpi = 120, bbox_inches = 'tight')
    plt.close(fig)


def metrics_bar_chart(metrics: dict, out_path: Path):
    """Bar chart comparing utilization, perplexity, and PSNR across methods."""
    fig, axes = plt.subplots(1, 3, figsize = (12, 4))
    labels = list(METHODS)

    util = [metrics[m]['utilization'] * 100 for m in labels]
    ppl = [metrics[m]['perplexity'] for m in labels]
    psnr = [metrics[m]['psnr'] for m in labels]

    axes[0].bar(labels, util, color = ['#4477aa', '#ee7733', '#228833'])
    axes[0].set_title('codebook utilization (%)')
    axes[0].set_ylim(0, 105)
    axes[1].bar(labels, ppl, color = ['#4477aa', '#ee7733', '#228833'])
    axes[1].set_title('perplexity (max = K = 64)')
    axes[1].axhline(64, color = 'gray', linestyle = '--', alpha = 0.5, label = 'K=64 (uniform use)')
    axes[1].legend()
    axes[2].bar(labels, psnr, color = ['#4477aa', '#ee7733', '#228833'])
    axes[2].set_title('PSNR (dB) — reconstruction quality')

    for ax, vals in zip(axes, (util, ppl, psnr)):
        for i, v in enumerate(vals):
            ax.text(i, v, f'{v:.1f}', ha = 'center', va = 'bottom')

    fig.suptitle('CIFAR-10 smoke runs, K=64, 2000 iterations')
    fig.tight_layout()
    fig.savefig(out_path, dpi = 120)
    plt.close(fig)


def save_summary_md(metrics: dict, out_path: Path):
    lines = [
        '# CIFAR-10 smoke runs (K=64, 2000 iterations)',
        '',
        '| method | rec_loss | PSNR (dB) | utilization | perplexity (max 64) |',
        '| --- | --- | --- | --- | --- |',
    ]
    for m in METHODS:
        x = metrics[m]
        lines.append(f'| {m} | {x["rec_loss"]:.4f} | {x["psnr"]:.2f} | '
                     f'{x["utilization"] * 100:.1f}% | {x["perplexity"]:.2f} |')
    out_path.write_text('\n'.join(lines) + '\n')


def main(runs_dir: str = './runs', out_dir: str = './figures/smoke_runs_k64'):
    runs_root = Path(runs_dir).expanduser()
    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents = True, exist_ok = True)

    run_dirs = {m: find_run_dir(m, runs_root) for m in METHODS}
    log_dir = runs_root / '_logs'
    metrics = {m: parse_log(log_dir / f'{m}.log') for m in METHODS}
    missing = [m for m, x in metrics.items() if x is None]
    if missing:
        raise RuntimeError(f'no eval metrics found for: {missing}')

    print(f'{"method":<10}{"rec":>10}{"psnr":>10}{"util":>10}{"ppl":>10}')
    for m in METHODS:
        x = metrics[m]
        print(f'{m:<10}{x["rec_loss"]:>10.4f}{x["psnr"]:>10.2f}{x["utilization"]*100:>9.1f}%{x["perplexity"]:>10.2f}')

    stack_recons(run_dirs, out_path / 'recons.png')
    metrics_bar_chart(metrics, out_path / 'metrics.png')
    save_summary_md(metrics, out_path / 'summary.md')
    print(f'\nwrote: {out_path}/recons.png, metrics.png, summary.md')


if __name__ == '__main__':
    fire.Fire(main)
