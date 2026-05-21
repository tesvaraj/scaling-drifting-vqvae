#!/usr/bin/env python
"""
Plot drift's reconstruction quality vs energy_weight at fixed K.

Reads drift logs from runs/_logs/drift_K{K}_ew{ew}.log and plots PSNR and
perplexity against energy_weight. The vanilla baseline at the same K is drawn
as a horizontal reference line.

Usage:
    python examples/compare_beta_sweep.py
    python examples/compare_beta_sweep.py --K 512
"""

from __future__ import annotations

import re
from pathlib import Path

import fire
import matplotlib.pyplot as plt


EVAL_RE = re.compile(
    r'\[step (\d+)\] rec_loss ([\d.]+) \| psnr ([\d.]+) \| utilization ([\d.]+) \| perplexity ([\d.]+)'
)
DRIFT_LOG_RE = re.compile(r'^drift_K(?P<k>\d+)_ew(?P<ew>[\d.]+)\.log$')


def parse_log(log_path: Path):
    """Return the last eval line's metrics, or None if no eval line found."""
    text = log_path.read_text(errors = 'ignore')
    matches = EVAL_RE.findall(text)
    if not matches:
        return None
    step, rec, psnr, util, ppl = matches[-1]
    return dict(step = int(step), rec_loss = float(rec), psnr = float(psnr),
                utilization = float(util), perplexity = float(ppl))


def collect_drift(log_dir: Path, K: int) -> dict:
    """Return {ew: metrics} for drift at the requested K."""
    out = {}
    for path in log_dir.glob(f'drift_K{K}_ew*.log'):
        match = DRIFT_LOG_RE.match(path.name)
        if not match:
            continue
        ew = float(match.group('ew'))
        x = parse_log(path)
        if x is not None:
            out[ew] = x
    return out


def save_summary_md(drift_by_ew: dict, vanilla: dict, K: int, out_path: Path):
    lines = [
        f'# Drift energy_weight sweep at K={K} (2000 iter, CIFAR-10)',
        '',
        '| energy_weight | PSNR (dB) | Utilization | Perplexity (max=K) |',
        '| --- | --- | --- | --- |',
    ]
    for e in sorted(drift_by_ew.keys()):
        x = drift_by_ew[e]
        lines.append(f'| {e} | {x["psnr"]:.2f} | {x["utilization"]*100:.1f}% | {x["perplexity"]:.1f} |')
    if vanilla is not None:
        lines += [
            '',
            f'## Vanilla baseline at K={K}',
            '',
            f'| metric | value |',
            f'| --- | --- |',
            f'| PSNR | {vanilla["psnr"]:.2f} |',
            f'| Perplexity | {vanilla["perplexity"]:.1f} |',
        ]
    out_path.write_text('\n'.join(lines) + '\n')


def plot_sweep(drift_by_ew: dict, vanilla: dict, K: int, out_path: Path):
    ews = sorted(drift_by_ew.keys())
    psnr = [drift_by_ew[e]['psnr'] for e in ews]
    ppl = [drift_by_ew[e]['perplexity'] for e in ews]

    fig, axes = plt.subplots(1, 2, figsize = (12, 5))

    axes[0].plot(ews, psnr, '-o', color = '#228833', label = 'drift')
    if vanilla is not None:
        axes[0].axhline(vanilla['psnr'], color = '#4477aa', linestyle = '--',
                        label = f'vanilla K={K} ({vanilla["psnr"]:.2f} dB)')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('energy_weight (drifting power)')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('PSNR vs drifting power')
    axes[0].grid(alpha = 0.3)
    axes[0].legend()

    axes[1].plot(ews, ppl, '-o', color = '#228833', label = 'drift')
    if vanilla is not None:
        axes[1].axhline(vanilla['perplexity'], color = '#4477aa', linestyle = '--',
                        label = f'vanilla K={K} ({vanilla["perplexity"]:.1f})')
    axes[1].axhline(K, color = 'gray', linestyle = ':', alpha = 0.5,
                    label = f'K={K} (uniform use)')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('energy_weight (drifting power)')
    axes[1].set_ylabel('perplexity')
    axes[1].set_title('perplexity vs drifting power')
    axes[1].grid(alpha = 0.3)
    axes[1].legend()

    fig.suptitle(f'Drift energy_weight sweep, K={K}, CIFAR-10, 2000 iter')
    fig.tight_layout()
    fig.savefig(out_path, dpi = 120)
    plt.close(fig)


def main(K: int = 512, runs_dir: str = './runs', out_dir: str = None):
    log_dir = Path(runs_dir).expanduser() / '_logs'
    out_path = Path(out_dir or f'./figures/beta_sweep_K{K}').expanduser()
    out_path.mkdir(parents = True, exist_ok = True)

    drift_by_ew = collect_drift(log_dir, K)
    if not drift_by_ew:
        raise RuntimeError(f'no drift logs found for K={K} under {log_dir}')

    vanilla_log = log_dir / f'vanilla_K{K}.log'
    vanilla = parse_log(vanilla_log) if vanilla_log.exists() else None

    print(f'{"ew":>8}{"psnr":>10}{"util":>10}{"ppl":>10}')
    for e in sorted(drift_by_ew.keys()):
        x = drift_by_ew[e]
        print(f'{e:>8.2f}{x["psnr"]:>10.2f}{x["utilization"]*100:>9.1f}%{x["perplexity"]:>10.2f}')
    if vanilla is not None:
        print(f'\nvanilla K={K} baseline: PSNR {vanilla["psnr"]:.2f}, perplexity {vanilla["perplexity"]:.2f}')

    plot_sweep(drift_by_ew, vanilla, K, out_path / 'sweep.png')
    save_summary_md(drift_by_ew, vanilla, K, out_path / 'summary.md')
    print(f'\nwrote: {out_path}/sweep.png, summary.md')


if __name__ == '__main__':
    fire.Fire(main)
