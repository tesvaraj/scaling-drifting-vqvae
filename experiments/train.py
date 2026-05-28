"""Config-driven training script for the Scaling Drifting VQ-VAE project.

Single entrypoint ``train(config)`` runs one experiment and writes a
self-contained output directory:

    runs/<phase>/<run_id>/
        config.json           # exact config used (incl. all defaults)
        summary.json          # final metrics from the last validation pass
        curves.csv            # per-log-step training metrics
        val_curves.csv        # per-validation-step val metrics
        wandb_url.txt         # wandb run URL (or "disabled")
        figures/
            recons_<step>.png
            codebook_pca_<step>.png
            usage_hist_<step>.png
        checkpoints/
            ckpt_<step>.pt
            final.pt

Everything in this folder is meant to be machine-readable, so the
project's introspection scripts (and a code assistant) can analyze runs
without scraping wandb.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import trange

from experiments.data import build_dataset
from experiments.metrics import (
    MetricAccumulator,
    code_usage_counts,
    codebook_geometry,
    entropy_from_counts,
    fid_compute,
    fid_update,
    grad_norm,
    hidden_geometry,
    perplexity_from_counts,
    psnr,
    reset_fid,
    ssim,
    utilization_from_counts,
)
from experiments.models import VQAutoEncoder
from experiments.quantizers import build_quantizer, make_energy_weight_schedule


# ---------- config ----------

@dataclass
class TrainConfig:
    # identity
    run_id: str = 'debug'
    phase: str = 'phase0_smoke'
    tags: list = field(default_factory = list)

    # core
    dataset: str = 'cifar10'
    method: str = 'drift'
    train_iter: int = 2000
    batch_size: int = 128
    lr: float = 3e-4
    seed: int = 0

    # model
    dim: int = 64
    hidden: int = 128
    codebook_size: int = 512

    # drift / hybrid hyperparams
    tau: float = 1.0
    energy_weight: float = 1.0
    n_hidden_subsample: int = 1024
    l2_normalize: bool = True
    energy_terms: tuple = ('pp', 'nn', 'pn')
    rotation_trick: bool = True
    ste: bool = True

    # vanilla hyperparams
    commitment_weight: float = 0.25
    ema_decay: float = 0.8

    # drift_ema hyperparams
    drift_ema_decay: float = 0.99
    drift_ema_dead_threshold: int = 0

    # energy_weight schedule (for drift_warmup / drift_anneal style experiments)
    energy_schedule: str = 'constant'       # constant | linear | warmup_off | warmup_const
    energy_final: Optional[float] = None
    energy_warmup_steps: int = 0

    # loss weights
    alpha: float = 1.0   # weight on quantizer loss

    # I/O
    data_root: str = '~/data'
    out_root: str = './runs'

    # cadence
    log_every: int = 50
    val_every: int = 1000
    image_every: int = 1000
    ckpt_every: int = 5000
    metric_window: int = 50
    val_max_batches: int = 40

    # heavy metrics
    eval_ssim: bool = True
    eval_lpips: bool = False
    eval_fid: bool = False
    eval_fid_max_batches: int = 40  # capped for cost

    # wandb
    wandb_project: str = 'drifting-vqvae-231n'
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_mode: str = 'online'              # online | offline | disabled

    # misc
    num_workers: int = 2
    device: Optional[str] = None
    compile: bool = False


# ---------- utilities ----------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(cfg) -> str:
    if cfg.device is not None:
        return cfg.device
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def cycle(loader, device):
    while True:
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            yield x.to(device, non_blocking = True)


def codebook_view(quantizer) -> Optional[torch.Tensor]:
    """Return a (K, d) view of the current codebook for any quantizer family."""
    # DriftingVQ / DriftEMAVQ have a .codebook attribute (Parameter or buffer)
    if hasattr(quantizer, 'codebook') and torch.is_tensor(quantizer.codebook):
        return quantizer.codebook.detach()
    # VectorQuantize has _codebook with .embed [(1, K, d)] (1 codebook head)
    inner = getattr(quantizer, '_codebook', None)
    if inner is not None and hasattr(inner, 'embed'):
        return inner.embed.detach().squeeze(0)
    # SimVQ: codebook is registered via .codebook (frozen) projected by linear
    if hasattr(quantizer, 'codebook'):
        return quantizer.codebook.detach()
    return None


# ---------- figures ----------

@torch.no_grad()
def save_pca_figure(quantizer, hidden_sample: torch.Tensor, counts: torch.Tensor,
                    out_path: Path, title: str):
    """2D PCA of codebook vectors + a sample of encoder outputs.

    Dead codes (count == 0) are drawn faded; alive codes solid.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cb = codebook_view(quantizer)
    if cb is None:
        return
    cb = cb.float().cpu()
    h = hidden_sample.float().cpu()
    # normalize hiddens to unit sphere so they're on the same scale as the
    # L2-normalized codebook vectors; without this the scale mismatch (~8× raw
    # hidden norm vs ~1 code norm) makes the PCA misleading
    h = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # fit PCA on codebook ∪ hiddens to share the projection axes
    combined = torch.cat([cb, h], dim = 0)
    combined = combined - combined.mean(0, keepdim = True)
    try:
        U, S, V = torch.pca_lowrank(combined, q = min(2, combined.shape[1]))
    except Exception:
        return
    proj = combined @ V[:, :2]
    cb2 = proj[:cb.shape[0]]
    h2 = proj[cb.shape[0]:]

    alive_mask = counts > 0

    fig, ax = plt.subplots(1, 1, figsize = (6, 6))
    ax.scatter(h2[:, 0], h2[:, 1], s = 6, alpha = 0.15, color = 'tab:blue', label = 'hiddens')
    if (~alive_mask).any():
        ax.scatter(cb2[~alive_mask, 0], cb2[~alive_mask, 1], s = 30, alpha = 0.3,
                   color = 'gray', marker = 'x', label = f'dead codes ({(~alive_mask).sum().item()})')
    ax.scatter(cb2[alive_mask, 0], cb2[alive_mask, 1], s = 24, alpha = 0.9,
               color = 'crimson', label = f'alive codes ({alive_mask.sum().item()})')
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(loc = 'best', fontsize = 8)
    ax.grid(alpha = 0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi = 120)
    plt.close(fig)


def save_usage_hist(counts: torch.Tensor, out_path: Path, title: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sorted_counts, _ = counts.sort(descending = True)
    fig, ax = plt.subplots(1, 1, figsize = (8, 4))
    ax.bar(range(sorted_counts.shape[0]), sorted_counts.cpu().numpy(),
           color = 'tab:purple', width = 1.0)
    ax.set_yscale('symlog')
    ax.set_xlabel('code rank (sorted by usage)')
    ax.set_ylabel('count (symlog)')
    ax.set_title(title)
    ax.grid(alpha = 0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi = 120)
    plt.close(fig)


def save_recon_grid(x, recon, out_path: Path, n: int = 16):
    from torchvision.utils import make_grid, save_image
    x = x[:n].detach().cpu()
    recon = recon[:n].detach().cpu()
    grid = make_grid(torch.cat([x, recon], dim = 0) * 0.5 + 0.5, nrow = n)
    save_image(grid, out_path)


# ---------- evaluation ----------

@torch.no_grad()
def evaluate(model, val_loader, device, codebook_size, cfg, step: int) -> tuple:
    model.eval()
    acc = MetricAccumulator()
    reset_fid()

    use_fid = cfg.eval_fid
    hidden_sample = None
    for i, batch in enumerate(val_loader):
        if i >= cfg.val_max_batches:
            break
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        recon, indices, _ = model(x)
        recon = recon.clamp(-1.0, 1.0)
        acc.add(recon, x, codebook_size, indices,
                want_ssim = cfg.eval_ssim,
                want_lpips = cfg.eval_lpips,
                want_fid = use_fid and i < cfg.eval_fid_max_batches)
        if hidden_sample is None:
            h = model.encoder(x)
            if h.ndim == 4:
                h = h.permute(0, 2, 3, 1).reshape(-1, h.shape[1])
            hidden_sample = h.detach()
    summary = acc.summarize()
    if use_fid:
        v = fid_compute()
        if v is not None:
            summary['rfid'] = v
    model.train()
    return summary, acc.counts, hidden_sample


# ---------- train ----------

def _to_jsonable(obj):
    """Best-effort conversion of nested containers to JSON-serializable types."""
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def train(cfg: TrainConfig) -> dict:
    """Run one experiment. Returns the final summary dict."""
    set_seed(cfg.seed)
    device = pick_device(cfg)

    # run directory
    out_dir = Path(cfg.out_root).expanduser() / cfg.phase / cfg.run_id
    (out_dir / 'figures').mkdir(parents = True, exist_ok = True)
    (out_dir / 'checkpoints').mkdir(parents = True, exist_ok = True)

    # persist config
    cfg_dict = _to_jsonable(asdict(cfg))
    cfg_dict['device'] = device
    cfg_dict['start_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(cfg_dict, f, indent = 2)

    # ----- data -----
    ds = build_dataset(cfg.dataset, cfg.data_root)
    train_loader = DataLoader(ds.train, batch_size = cfg.batch_size, shuffle = True,
                              num_workers = cfg.num_workers, drop_last = True,
                              pin_memory = (device == 'cuda'))
    val_loader = DataLoader(ds.val, batch_size = cfg.batch_size, shuffle = False,
                            num_workers = cfg.num_workers, drop_last = False,
                            pin_memory = (device == 'cuda'))

    # ----- quantizer -----
    quantizer = build_quantizer(
        cfg.method,
        dim = cfg.dim,
        codebook_size = cfg.codebook_size,
        tau = cfg.tau,
        energy_weight = cfg.energy_weight,
        n_hidden_subsample = cfg.n_hidden_subsample,
        l2_normalize = cfg.l2_normalize,
        energy_terms = tuple(cfg.energy_terms),
        rotation_trick = cfg.rotation_trick,
        ste = cfg.ste,
        commitment_weight = cfg.commitment_weight,
        ema_decay = cfg.ema_decay,
        drift_ema_decay = cfg.drift_ema_decay,
        drift_ema_dead_threshold = cfg.drift_ema_dead_threshold,
    )

    # ----- model -----
    model = VQAutoEncoder(ds.in_channels, cfg.hidden, cfg.dim, ds.n_downsample,
                          quantizer).to(device)
    opt = AdamW(model.parameters(), lr = cfg.lr)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[{cfg.run_id}] device={device} params={n_params/1e6:.2f}M method={cfg.method} K={cfg.codebook_size}')

    if cfg.compile and device == 'cuda':
        model = torch.compile(model)

    # energy weight schedule (for drift methods)
    energy_fn = make_energy_weight_schedule(
        cfg.energy_schedule,
        base = cfg.energy_weight,
        final = cfg.energy_final,
        total_steps = cfg.train_iter,
        warmup_steps = cfg.energy_warmup_steps,
    )

    # ----- wandb -----
    use_wandb = cfg.wandb_mode != 'disabled'
    wandb = None
    wandb_url = 'disabled'
    if use_wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
        except ImportError:
            print('[warn] wandb not installed; running without it')
            use_wandb = False

    if use_wandb:
        run = wandb.init(
            project = cfg.wandb_project,
            entity = cfg.wandb_entity,
            group = cfg.wandb_group or cfg.phase,
            name = cfg.run_id,
            tags = list(cfg.tags),
            config = cfg_dict,
            dir = str(out_dir),
            mode = cfg.wandb_mode,
        )
        wandb_url = run.url if hasattr(run, 'url') and run.url else 'offline'
    with open(out_dir / 'wandb_url.txt', 'w') as f:
        f.write(wandb_url + '\n')

    # ----- fixed eval batch for reconstruction visualization -----
    fixed_batch = next(iter(val_loader))
    fixed_batch = (fixed_batch[0] if isinstance(fixed_batch, (list, tuple)) else fixed_batch)[:16].to(device)

    # ----- CSV writers -----
    curves_path = out_dir / 'curves.csv'
    val_curves_path = out_dir / 'val_curves.csv'
    curves_f = open(curves_path, 'w', newline = '')
    val_curves_f = open(val_curves_path, 'w', newline = '')
    curves_writer = None     # lazy initialized once we know columns
    val_writer = None

    def write_csv(path_writer_pair, row: dict):
        f, writer = path_writer_pair
        if writer is None:
            writer = csv.DictWriter(f, fieldnames = list(row.keys()))
            writer.writeheader()
            path_writer_pair[1] = writer
        writer.writerow(row)
        f.flush()

    curves_pair = [curves_f, curves_writer]
    val_pair = [val_curves_f, val_writer]

    # ----- windowed running metrics -----
    win = dict(rec_loss = deque(maxlen = cfg.metric_window),
               vq_loss = deque(maxlen = cfg.metric_window),
               psnr = deque(maxlen = cfg.metric_window))
    rolling_window = deque(maxlen = cfg.metric_window)

    # ----- train loop -----
    dl_iter = cycle(train_loader, device)
    pbar = trange(cfg.train_iter, dynamic_ncols = True)
    last_summary = {}
    final_counts = None
    final_hidden_sample = None
    t0 = time.time()
    try:
        for step in pbar:
            # update energy weight per schedule
            if hasattr(model.quantizer, 'energy_weight'):
                model.quantizer.energy_weight = float(energy_fn(step))

            x = next(dl_iter)
            recon, indices, vq_loss = model(x)
            recon = recon.clamp(-1.0, 1.0)

            rec_loss = F.mse_loss(recon, x)
            loss = rec_loss + cfg.alpha * vq_loss

            opt.zero_grad()
            loss.backward()
            gn = grad_norm(model.parameters())
            opt.step()

            with torch.no_grad():
                counts = code_usage_counts(indices, cfg.codebook_size)
                win['rec_loss'].append(rec_loss.item())
                win['vq_loss'].append(vq_loss.item())
                win['psnr'].append(psnr(recon, x))
                rolling_window.append(counts)
                rolling_counts = torch.stack(list(rolling_window)).sum(0)

            if step % cfg.log_every == 0:
                mean_rec = sum(win['rec_loss']) / len(win['rec_loss'])
                mean_vq = sum(win['vq_loss']) / len(win['vq_loss'])
                mean_psnr = sum(win['psnr']) / len(win['psnr'])
                util = utilization_from_counts(rolling_counts)
                ppl = perplexity_from_counts(rolling_counts)
                ent = entropy_from_counts(rolling_counts)
                pbar.set_description(
                    f'rec {mean_rec:.4f} | vq {mean_vq:.4f} | psnr {mean_psnr:.2f} | '
                    f'util {util*100:.1f}% | ppl {ppl:.1f}'
                )

                row = {
                    'step': step,
                    'time': time.time() - t0,
                    'train/rec_loss': mean_rec,
                    'train/vq_loss': mean_vq,
                    'train/total_loss': mean_rec + cfg.alpha * mean_vq,
                    'train/psnr': mean_psnr,
                    'train/grad_norm': gn,
                    'codebook/utilization': util,
                    'codebook/perplexity': ppl,
                    'codebook/entropy_nats': ent,
                    'codebook/active_codes': int((rolling_counts > 0).sum().item()),
                    'opt/lr': opt.param_groups[0]['lr'],
                    'opt/energy_weight': getattr(model.quantizer, 'energy_weight', float('nan')),
                }

                bd = getattr(model.quantizer, 'last_breakdown', None) or {}
                for k, v in bd.items():
                    row[f'drift/{k}'] = v.item() if torch.is_tensor(v) else float(v)

                write_csv(curves_pair, row)
                if use_wandb:
                    wandb.log(row, step = step)

            # ----- validation -----
            if step > 0 and (step % cfg.val_every == 0 or step == cfg.train_iter - 1):
                val_summary, val_counts, hidden_sample = evaluate(
                    model, val_loader, device, cfg.codebook_size, cfg, step,
                )
                final_counts = val_counts
                final_hidden_sample = hidden_sample
                cb_geom = codebook_geometry(codebook_view(model.quantizer)) \
                    if codebook_view(model.quantizer) is not None else {}
                h_geom = hidden_geometry(hidden_sample) if hidden_sample is not None else {}

                row = {'step': step, 'time': time.time() - t0}
                row.update({f'val/{k}': v for k, v in val_summary.items()})
                row.update({f'codebook/{k}': v for k, v in cb_geom.items()})
                row.update({f'hidden/{k}': v for k, v in h_geom.items()})
                write_csv(val_pair, row)
                if use_wandb:
                    wandb_row = dict(row)
                    _counts = val_counts.numpy()
                    if cfg.codebook_size > 512:
                        import numpy as np
                        bs = (cfg.codebook_size + 511) // 512
                        pad = (-cfg.codebook_size) % bs
                        _counts = np.pad(_counts, (0, pad)).reshape(-1, bs).sum(axis=1)
                    _bins = list(range(len(_counts) + 1))
                    wandb_row['codebook/usage_hist'] = wandb.Histogram(
                        np_histogram=(_counts, _bins)
                    )
                    wandb.log(wandb_row, step = step)
                last_summary = row

                pbar.write(f'[step {step}] ' + ' | '.join(
                    f'{k.split("/")[-1]} {v:.4f}'
                    for k, v in val_summary.items() if isinstance(v, (int, float))
                ))

            # ----- figures -----
            if step > 0 and (step % cfg.image_every == 0 or step == cfg.train_iter - 1):
                with torch.no_grad():
                    rec_fixed, _, _ = model(fixed_batch)
                    rec_fixed = rec_fixed.clamp(-1.0, 1.0)
                rec_path = out_dir / 'figures' / f'recons_{step:06d}.png'
                save_recon_grid(fixed_batch, rec_fixed, rec_path)
                if final_counts is not None:
                    save_usage_hist(
                        final_counts, out_dir / 'figures' / f'usage_hist_{step:06d}.png',
                        title = f'step {step} | {cfg.method} K={cfg.codebook_size}',
                    )
                if final_hidden_sample is not None:
                    save_pca_figure(
                        model.quantizer,
                        final_hidden_sample[:2000],
                        final_counts if final_counts is not None
                        else torch.ones(cfg.codebook_size),
                        out_dir / 'figures' / f'codebook_pca_{step:06d}.png',
                        title = f'step {step} | {cfg.method} K={cfg.codebook_size}',
                    )
                if use_wandb:
                    wandb_log = {'samples/reconstructions': wandb.Image(str(rec_path))}
                    pca_path = out_dir / 'figures' / f'codebook_pca_{step:06d}.png'
                    if pca_path.exists():
                        wandb_log['samples/codebook_pca'] = wandb.Image(str(pca_path))
                    hist_path = out_dir / 'figures' / f'usage_hist_{step:06d}.png'
                    if hist_path.exists():
                        wandb_log['samples/usage_hist'] = wandb.Image(str(hist_path))
                    wandb.log(wandb_log, step = step)

            # ----- checkpoint -----
            if step > 0 and step % cfg.ckpt_every == 0:
                ckpt = {
                    'step': step,
                    'model': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'config': cfg_dict,
                }
                torch.save(ckpt, out_dir / 'checkpoints' / f'ckpt_{step:06d}.pt')

    finally:
        curves_pair[0].close()
        val_pair[0].close()

    # ----- final save -----
    torch.save({
        'step': cfg.train_iter,
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'config': cfg_dict,
    }, out_dir / 'checkpoints' / 'final.pt')

    summary = {
        'run_id': cfg.run_id,
        'phase': cfg.phase,
        'method': cfg.method,
        'codebook_size': cfg.codebook_size,
        'dataset': cfg.dataset,
        'seed': cfg.seed,
        'train_iter': cfg.train_iter,
        'wallclock_s': time.time() - t0,
        'final': last_summary,
        'wandb_url': wandb_url,
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(_to_jsonable(summary), f, indent = 2)

    if use_wandb:
        wandb.finish()

    return summary


# ---------- CLI ----------

def _config_from_kwargs(**kwargs) -> TrainConfig:
    # Allow passing JSON path via config=<path>
    if 'config' in kwargs:
        path = kwargs.pop('config')
        with open(path) as f:
            data = json.load(f)
        data.update(kwargs)
        return TrainConfig(**data)
    return TrainConfig(**kwargs)


if __name__ == '__main__':
    import fire
    def main(**kwargs):
        cfg = _config_from_kwargs(**kwargs)
        return train(cfg)
    fire.Fire(main)
