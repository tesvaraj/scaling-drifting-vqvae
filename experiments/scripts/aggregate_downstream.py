"""Aggregate the downstream experiments into paper-ready tables and figures.

Reads the per-run JSON the experiments write (pull them down first with
`modal volume get drifting-vqvae /runs/phase_cnn_probe ./runs/phase_cnn_probe`
and likewise for `/runs/phase_prior`):

    runs/phase_cnn_probe/*/cnn_probe_summary.json
    runs/phase_prior/*/prior_summary.json

Writes to runs/downstream_summary/:
    cnn_probe.{md,csv}        per (representation, head, K): EMA vs Drift top1/top5
                              mean±std over seeds, Δ, all-seeds-beat, + references
    prior.md                  per method: NLL, recon-FID, gen-FID, best T, prior-gap, Gini
    prior_fid_vs_temp.png     gen-FID vs sampling temperature, EMA vs Drift

Run with an env that has numpy + matplotlib (e.g. the `vqvae` conda env).
"""
from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# DOWNSTREAM_RUNS lets this run on Modal (/vol/runs) or locally (repo runs/).
RUNS = os.environ.get('DOWNSTREAM_RUNS', os.path.join(ROOT, 'runs'))
# which phase dirs to read (per-dataset for non-cifar100 runs)
CNN_PHASE = os.environ.get('DOWNSTREAM_CNN_PHASE', 'phase_cnn_probe')
PRIOR_PHASE = os.environ.get('DOWNSTREAM_PRIOR_PHASE', 'phase_prior')
OUT = os.path.join(RUNS, 'downstream_summary')
os.makedirs(OUT, exist_ok=True)

EMA, DRIFT = '#3b6ea5', '#8c1515'


def _load(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        try:
            rows.append(json.load(open(f)))
        except Exception as e:
            print('skip', f, e)
    return rows


def _tag(run_id):
    for t in ('random', 'pixels', 'drift', 'ema'):
        if t in run_id:
            return t
    return 'other'


def aggregate_cnn_probe():
    rows = _load(os.path.join(RUNS, CNN_PHASE, '*', 'cnn_probe_summary.json'))
    if not rows:
        print('[cnn_probe] no results found under runs/phase_cnn_probe/')
        return
    groups = defaultdict(lambda: defaultdict(list))   # (rep,head,K) -> tag -> [runs]
    refs = []
    for r in rows:
        tag = _tag(r['run_id'])
        key = (r.get('representation'), r.get('head'), r.get('K'))
        if tag in ('ema', 'drift'):
            groups[key][tag].append(r)
        else:
            refs.append((tag, key, r))

    md = ['# CNN probe — CIFAR-100 transfer (frozen VQ-VAE)', '',
          '| rep | head | K | method | top1 % | top5 % | n |',
          '|---|---|---|---|---|---|---|']
    rows_csv = [['rep', 'head', 'K', 'method', 'top1_mean', 'top1_std',
                 'top5_mean', 'top5_std', 'n', 'all_seeds_beat']]
    for key in sorted(groups, key=lambda x: (str(x[0]), str(x[1]), x[2] or 0)):
        rep, head, K = key
        stat = {}
        for tag in ('ema', 'drift'):
            rs = groups[key][tag]
            if not rs:
                continue
            t1 = [x['test_top1'] * 100 for x in rs]
            t5 = [x['test_top5'] * 100 for x in rs]
            stat[tag] = (t1, t5)
            md.append(f"| {rep} | {head} | {K} | {tag} | "
                      f"{np.mean(t1):.2f}±{np.std(t1):.2f} | "
                      f"{np.mean(t5):.2f}±{np.std(t5):.2f} | {len(rs)} |")
        if 'ema' in stat and 'drift' in stat:
            e1, _ = stat['ema']; d1, _ = stat['drift']
            dt = np.mean(d1) - np.mean(e1)
            beat = min(d1) > max(e1)
            md.append(f"| {rep} | {head} | {K} | **Δ drift−ema** | "
                      f"**{dt:+.2f}pp** | all-seeds-beat={beat} | |")
            rows_csv.append([rep, head, K, 'drift_minus_ema', dt, '', '', '',
                             len(d1), beat])
        for tag in ('ema', 'drift'):
            if tag in stat:
                t1, t5 = stat[tag]
                rows_csv.append([rep, head, K, tag, np.mean(t1), np.std(t1),
                                 np.mean(t5), np.std(t5), len(t1), ''])

    if refs:
        md += ['', '## reference baselines', '',
               '| ref | rep | head | K | top1 % | top5 % |',
               '|---|---|---|---|---|---|']
        for tag, (rep, head, K), r in refs:
            md.append(f"| {tag} | {rep} | {head} | {K} | "
                      f"{r['test_top1']*100:.2f} | {r['test_top5']*100:.2f} |")

    open(os.path.join(OUT, 'cnn_probe.md'), 'w').write('\n'.join(md) + '\n')
    with open(os.path.join(OUT, 'cnn_probe.csv'), 'w', newline='') as f:
        csv.writer(f).writerows(rows_csv)
    print('[cnn_probe] wrote cnn_probe.md / cnn_probe.csv\n')
    text = '\n'.join(md)
    print(text)
    return text


def aggregate_prior():
    rows = _load(os.path.join(RUNS, PRIOR_PHASE, '*', 'prior_summary.json'))
    if not rows:
        print('\n[prior] no results found under runs/phase_prior/')
        return
    by = defaultdict(list)
    for r in rows:
        by[_tag(r['run_id'])].append(r)

    def col(rs, k):
        return [r[k] for r in rs if r.get(k) is not None]

    md = ['# Prior / generation — CIFAR-100', '',
          '| method | NLL bits | recon-FID | gen-FID | best T | prior-gap | sample Gini | n |',
          '|---|---|---|---|---|---|---|---|']
    agg = {}
    for tag in ('ema', 'drift'):
        rs = by.get(tag, [])
        if not rs:
            continue
        nll, rf, gf = col(rs, 'best_val_nll'), col(rs, 'recon_fid'), col(rs, 'gen_fid')
        bt, sg = col(rs, 'best_temp'), col(rs, 'sample_gini')
        gap = [g - r for g, r in zip(gf, rf)] if (gf and rf) else []
        agg[tag] = {'recon': rf, 'gen': gf}
        m = lambda xs: np.mean(xs) if xs else float('nan')
        md.append(f"| {tag} | {m(nll):.3f}±{np.std(nll) if nll else 0:.3f} | {m(rf):.2f} | "
                  f"{m(gf):.2f} | {m(bt):.2f} | {m(gap):+.2f} | {m(sg):.3f} | {len(rs)} |")

    if agg.get('ema', {}).get('recon') and agg.get('drift', {}).get('recon'):
        d = np.mean(agg['drift']['recon']) - np.mean(agg['ema']['recon'])
        md.append(f"\n**Δ recon-FID (drift−ema): {d:+.2f}**  (negative = drift tokenizer better)")
    if agg.get('ema', {}).get('gen') and agg.get('drift', {}).get('gen'):
        d = np.mean(agg['drift']['gen']) - np.mean(agg['ema']['gen'])
        md.append(f"**Δ gen-FID (drift−ema): {d:+.2f}**  (negative = drift also generates better)")

    open(os.path.join(OUT, 'prior.md'), 'w').write('\n'.join(md) + '\n')
    print('\n[prior] wrote prior.md\n')
    print('\n'.join(md))

    # gen-FID vs temperature, averaged across seeds
    plt.figure(figsize=(4, 3))
    plotted = False
    for tag, color in [('ema', EMA), ('drift', DRIFT)]:
        acc = defaultdict(list)
        for r in by.get(tag, []):
            for t, v in (r.get('gen_fid_by_temp') or {}).items():
                acc[float(t)].append(v['gen_fid'])
        if not acc:
            continue
        ts = sorted(acc)
        plt.plot(ts, [np.mean(acc[t]) for t in ts], 'o-', color=color, label=tag)
        plotted = True
    if plotted:
        plt.xlabel('sampling temperature'); plt.ylabel('gen-FID')
        plt.title('Generation FID vs temperature'); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, 'prior_fid_vs_temp.png'), dpi=200)
        print('[prior] wrote prior_fid_vs_temp.png')
    return '\n'.join(md)


if __name__ == '__main__':
    aggregate_cnn_probe()
    aggregate_prior()
    print(f'\nAll outputs in {OUT}')
