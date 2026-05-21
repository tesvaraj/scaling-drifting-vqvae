#!/usr/bin/env python
"""
Plot CIFAR-10 results across codebook sizes.

Reads per-method per-K logs under runs/_logs/ and plots PSNR, perplexity, and
utilization vs K for each method. Saves a three-panel figure and a markdown
summary table to figures/k_sweep/.

Usage:
    python examples/compare_k_sweep.py
"""

from __future__ import annotations

import re
from pathlib import Path

import fire
import matplotlib.pyplot as plt


METHODS = ('vanilla', 'simvq', 'drift')
METHOD_COLORS = {'vanilla': '#4477aa', 'simvq': '#ee7733', 'drift': '#228833'}
EVAL_RE = re.compile(
    r'\[step (\d+)\] rec_loss ([\d.]+) \| psnr ([\d.]+) \| utilization ([\d.]+) \| perplexity ([\d.]+)'
)
LOG_NAME_RE = re.compile(r'^(?P<method>\w+)_K(?P<k>\d+)\.log$')


def discover_logs(log_dir: Path) -> dict:
    """Group log files by method and K. Returns {method: {K: log_path}}."""
    grouped = {m: {} for m in METHODS}
    for path in log_dir.glob('*_K*.log'):
        match = LOG_NAME_RE.match(path.name)
        if not match:
            continue
        method = match.group('method')
        K = int(match.group('k'))
        if method in grouped:
            grouped[method][K] = path
    return grouped


def parse_log(log_path: Path):
    """Return the last eval line's metrics, or None if no eval line found."""
    text = log_path.read_text(errors = 'ignore')
    matches = EVAL_RE.findall(text)
    if not matches:
        return None
    step, rec, psnr, util, ppl = matches[-1]
    return dict(step = int(step), rec_loss = float(rec), psnr = float(psnr),
                utilization = float(util), perplexity = float(ppl))


def plot_metrics_vs_k(metrics: dict, ks_all: list, out_path: Path):
    """Three-panel line plot: PSNR, perplexity, and utilization vs K."""
    fig, axes = plt.subplots(1, 3, figsize = (15, 5))

    for method in METHODS:
        method_data = metrics[method]
        if not method_data:
            continue
        ks = sorted(method_data.keys())
        psnr = [method_data[k]['psnr'] for k in ks]
        ppl = [method_data[k]['perplexity'] for k in ks]
        util = [method_data[k]['utilization'] * 100 for k in ks]
        c = METHOD_COLORS[method]
        axes[0].plot(ks, psnr, '-o', label = method, color = c)
        axes[1].plot(ks, ppl, '-o', label = method, color = c)
        axes[2].plot(ks, util, '-o', label = method, color = c)

    # reference line: perfectly uniform code usage has perplexity = K
    axes[1].plot(ks_all, ks_all, '--', color = 'gray', alpha = 0.5, label = 'perplexity = K (uniform)')

    titles = ['PSNR (dB) — reconstruction quality',
              'perplexity — effective code count',
              'utilization (%) — fraction of codes used']
    for ax, t in zip(axes, titles):
        ax.set_xscale('log')
        ax.set_xticks(ks_all)
        ax.set_xticklabels([str(k) for k in ks_all])
        ax.set_xlabel('codebook size K')
        ax.set_title(t)
        ax.legend()
        ax.grid(alpha = 0.3)

    fig.suptitle('CIFAR-10 sweep: PSNR, perplexity, and utilization vs codebook size')
    fig.tight_layout()
    fig.savefig(out_path, dpi = 120)
    plt.close(fig)


def md_table(metrics: dict, ks_all: list, value_fn) -> str:
    header = '| method | ' + ' | '.join(f'K={k}' for k in ks_all) + ' |'
    sep = '| --- | ' + ' | '.join('---' for _ in ks_all) + ' |'
    rows = [header, sep]
    for m in METHODS:
        row = [m]
        for k in ks_all:
            x = metrics[m].get(k)
            row.append(value_fn(x) if x else '-')
        rows.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(rows)


def save_summary_md(metrics: dict, ks_all: list, out_path: Path):
    """Three markdown tables: PSNR, perplexity, utilization."""
    sections = [
        '# CIFAR-10 K-sweep (2000 iterations per run)',
        '',
        '## PSNR (dB) — higher is better',
        '',
        md_table(metrics, ks_all, lambda x: f'{x["psnr"]:.2f}'),
        '',
        '## Perplexity — max possible value is K (uniform code use)',
        '',
        md_table(metrics, ks_all, lambda x: f'{x["perplexity"]:.1f}'),
        '',
        '## Utilization (%) — fraction of codes with nonzero use',
        '',
        md_table(metrics, ks_all, lambda x: f'{x["utilization"]*100:.1f}%'),
        '',
    ]
    out_path.write_text('\n'.join(sections))


def main(runs_dir: str = './runs', out_dir: str = './figures/k_sweep'):
    log_dir = Path(runs_dir).expanduser() / '_logs'
    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents = True, exist_ok = True)

    logs = discover_logs(log_dir)
    raw = {m: {k: parse_log(p) for k, p in v.items()} for m, v in logs.items()}
    metrics = {m: {k: x for k, x in v.items() if x is not None} for m, v in raw.items()}

    ks_all = sorted({K for method_data in metrics.values() for K in method_data})
    if not ks_all:
        raise RuntimeError(f'no parseable logs under {log_dir}')

    print(f'discovered K values: {ks_all}')
    print(f'\n{"method":<10}{"K":>8}{"psnr":>10}{"util":>10}{"ppl":>10}')
    for m in METHODS:
        for k in ks_all:
            x = metrics[m].get(k)
            if x:
                print(f'{m:<10}{k:>8}{x["psnr"]:>10.2f}{x["utilization"]*100:>9.1f}%{x["perplexity"]:>10.2f}')

    plot_metrics_vs_k(metrics, ks_all, out_path / 'sweep.png')
    save_summary_md(metrics, ks_all, out_path / 'summary.md')
    print(f'\nwrote: {out_path}/sweep.png, summary.md')


if __name__ == '__main__':
    fire.Fire(main)
