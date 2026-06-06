"""Autoregressive token prior for downstream tokenizer evaluation.

Train a small causal transformer on the discrete token sequences produced
by a frozen VQ-VAE encoder. The key question: does drift's more uniform
codebook distribution make the prior easier to learn?

Metrics
-------
prior NLL (bits/code)  — lower = codebook distribution easier to model
generation FID         — sample tokens from prior, decode, compare to real images
sample Gini            — Gini coefficient of code frequencies in generated samples
                         (lower = prior reproduces drift's uniform distribution)

Usage
-----
    # quick pilot (~5 min): does signal exist?
    modal run experiments/modal_app.py::prior_pilot

    # full experiment (3 seeds, gen FID, ~30 min wall time):
    modal run experiments/modal_app.py::prior_full
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange


# ---------- config ----------

@dataclass
class PriorConfig:
    # identity
    run_id: str = 'prior_debug'
    phase: str = 'phase_prior'

    # which frozen VQ-VAE to load
    vqvae_phase: str = 'phase_cifar100'
    vqvae_run_id: str = 'cifar100_vanilla_ema_K512_seed0'
    vqvae_ckpt: str = 'final.pt'

    # dataset (must match the VQ-VAE's training dataset)
    dataset: str = 'cifar100'

    # prior architecture
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.0

    # training
    train_steps: int = 20000
    batch_size: int = 256
    lr: float = 3e-4
    seed: int = 0

    # I/O paths (overridden to /vol/* on Modal)
    data_root: str = '~/data'
    out_root: str = './runs'
    vqvae_root: str = './runs'

    # cadence
    log_every: int = 200
    val_every: int = 1000

    # generation eval (expensive — skip for pilot)
    eval_gen_fid: bool = False
    n_gen_samples: int = 2000
    gen_temperature: float = 1.0
    # sampling-temperature sweep for gen-FID (FID is very temperature-sensitive,
    # so a fair drift-vs-EMA comparison must report the curve / best point)
    eval_temperatures: tuple = (0.8, 0.9, 1.0, 1.1)

    # wandb
    wandb_project: str = 'drifting-vqvae-231n'
    wandb_mode: str = 'online'

    num_workers: int = 2
    device: Optional[str] = None


# ---------- model ----------

class TokenPrior(nn.Module):
    """Causal transformer prior over VQ-VAE discrete token sequences.

    Prepends a learned [BOS] token so every real token has at least one
    token of context. At position t the model predicts token t given
    tokens 0..t-1.
    """

    def __init__(self, vocab_size: int, seq_len: int,
                 d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.bos_idx = vocab_size  # BOS lives at index vocab_size

        # +1 for BOS in both token and position embeddings
        self.embed = nn.Embedding(vocab_size + 1, d_model)
        self.pos_embed = nn.Embedding(seq_len + 1, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # causal mask: upper-triangular bool, True = block attention
        L = seq_len + 1
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1),
        )
        self._init_weights()

    def _init_weights(self):
        for emb in (self.embed, self.pos_embed):
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def _fwd(self, idx: torch.Tensor) -> torch.Tensor:
        """Raw forward: idx (B, T) already includes BOS at position 0."""
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        h = self.embed(idx) + self.pos_embed(pos)
        h = self.transformer(h, mask=self.causal_mask[:T, :T])
        return self.head(h)  # (B, T, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) real tokens (no BOS).
        Returns logits (B, T, V) where logit[b, t] predicts idx[b, t].
        """
        B = idx.shape[0]
        bos = idx.new_full((B, 1), self.bos_idx)
        x = torch.cat([bos, idx], dim=1)   # (B, T+1)
        return self._fwd(x)[:, :-1]        # (B, T, V) — drop last unused position

    def nll_bits(self, idx: torch.Tensor) -> torch.Tensor:
        """Mean per-token NLL in bits/code (scalar)."""
        logits = self(idx)
        B, T, V = logits.shape
        nll_nats = F.cross_entropy(logits.reshape(-1, V), idx.reshape(-1))
        return nll_nats / math.log(2)

    @torch.no_grad()
    def sample(self, n: int, temperature: float = 1.0,
               device: str = 'cuda') -> torch.Tensor:
        """Autoregressively sample n sequences → (n, seq_len)."""
        idx = torch.full((n, 1), self.bos_idx, dtype=torch.long, device=device)
        for _ in range(self.seq_len):
            logits = self._fwd(idx)[:, -1, :] / max(temperature, 1e-8)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx[:, 1:]  # strip BOS


# ---------- helpers ----------

def _gini(counts: torch.Tensor) -> float:
    """Gini coefficient of a count vector (0 = perfectly uniform, 1 = all mass on one code)."""
    c = counts.float() + 1e-8
    c = c / c.sum()
    c, _ = c.sort()
    n = c.shape[0]
    idx = torch.arange(1, n + 1, dtype=torch.float, device=c.device)
    return (2 * (idx * c).sum() / (n * c.sum()) - (n + 1) / n).item()


def load_vqvae_frozen(ckpt_path: str, device: str):
    """Load a VQ-VAE checkpoint, freeze all weights. Returns (model, TrainConfig)."""
    import sys
    sys.path.insert(0, '/root/scaling_drifting_vqvae')
    from experiments.train import TrainConfig
    from experiments.models import VQAutoEncoder
    from experiments.quantizers import build_quantizer
    from experiments.data import build_dataset

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt['config']

    # Reconstruct config, only passing known fields (guards against schema drift)
    known = set(TrainConfig.__dataclass_fields__)
    cfg = TrainConfig(**{k: v for k, v in cfg_dict.items() if k in known})
    if isinstance(cfg.energy_terms, list):
        cfg.energy_terms = tuple(cfg.energy_terms)

    ds = build_dataset(cfg.dataset, cfg_dict.get('data_root', '/vol/data'))
    quantizer = build_quantizer(
        cfg.method, dim=cfg.dim, codebook_size=cfg.codebook_size,
        tau=cfg.tau, energy_weight=cfg.energy_weight,
        n_hidden_subsample=cfg.n_hidden_subsample,
        l2_normalize=cfg.l2_normalize,
        energy_terms=tuple(cfg.energy_terms),
        rotation_trick=cfg.rotation_trick,
        ste=cfg.ste,
        commitment_weight=cfg.commitment_weight,
        ema_decay=cfg.ema_decay,
        drift_ema_decay=cfg.drift_ema_decay,
        drift_ema_dead_threshold=cfg.drift_ema_dead_threshold,
    )
    model = VQAutoEncoder(ds.in_channels, cfg.hidden, cfg.dim, ds.n_downsample, quantizer)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


@torch.no_grad()
def encode_to_tokens(model, loader, device: str) -> torch.Tensor:
    """Encode a full dataset to flat token sequences → (N, seq_len) LongTensor."""
    all_tokens = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        indices = model.encode_indices(x.to(device))  # (B, H, W)
        all_tokens.append(indices.reshape(indices.shape[0], -1).cpu())
    return torch.cat(all_tokens, dim=0)


# ---------- training ----------

def train_prior(cfg: PriorConfig) -> dict:
    """Train one token prior. Returns summary dict."""
    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cfg.out_root) / cfg.phase / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- wandb -----
    wandb = None
    if cfg.wandb_mode != 'disabled':
        try:
            import wandb as _w
            _w.init(project=cfg.wandb_project, group=cfg.phase, name=cfg.run_id,
                    config=asdict(cfg), dir=str(out_dir), mode=cfg.wandb_mode)
            wandb = _w
        except Exception as e:
            print(f'[warn] wandb init failed: {e}')

    # ----- load frozen VQ-VAE -----
    ckpt_path = (Path(cfg.vqvae_root) / cfg.vqvae_phase
                 / cfg.vqvae_run_id / 'checkpoints' / cfg.vqvae_ckpt)
    print(f'Loading VQ-VAE: {ckpt_path}')
    vqvae, vqvae_cfg = load_vqvae_frozen(str(ckpt_path), device)
    K = vqvae_cfg.codebook_size

    # ----- encode dataset to tokens (one-time) -----
    from experiments.data import build_dataset
    ds = build_dataset(cfg.dataset, cfg.data_root)
    enc_loader = DataLoader(ds.train, batch_size=256, shuffle=False,
                            num_workers=cfg.num_workers)
    val_enc_loader = DataLoader(ds.val, batch_size=256, shuffle=False,
                                num_workers=cfg.num_workers)

    print('Encoding training set...')
    train_tokens = encode_to_tokens(vqvae, enc_loader, device)    # (N, L)
    val_tokens = encode_to_tokens(vqvae, val_enc_loader, device)  # (N, L)
    seq_len = train_tokens.shape[1]

    # Gini of training token distribution — baseline for comparison to samples
    train_counts = torch.zeros(K, dtype=torch.long)
    for tok in train_tokens.reshape(-1):
        train_counts[tok] += 1
    train_gini = _gini(train_counts)
    print(f'Encoded: train={train_tokens.shape}, val={val_tokens.shape}  '
          f'K={K} seq_len={seq_len}  train Gini={train_gini:.3f}')

    # ----- build prior -----
    prior = TokenPrior(vocab_size=K, seq_len=seq_len,
                       d_model=cfg.d_model, n_heads=cfg.n_heads,
                       n_layers=cfg.n_layers, dropout=cfg.dropout).to(device)
    n_params = sum(p.numel() for p in prior.parameters())
    print(f'Prior: {n_params/1e6:.2f}M params  d_model={cfg.d_model} n_layers={cfg.n_layers}')

    opt = torch.optim.AdamW(prior.parameters(), lr=cfg.lr, weight_decay=1e-4)

    token_loader = DataLoader(TensorDataset(train_tokens), batch_size=cfg.batch_size,
                              shuffle=True, drop_last=True)

    def cycle(ldr):
        while True:
            for (b,) in ldr:
                yield b.to(device)

    tok_iter = cycle(token_loader)

    # ----- train loop -----
    nll_curve = []
    best_val_nll = float('inf')
    t0 = time.time()

    pbar = trange(cfg.train_steps, dynamic_ncols=True)
    for step in pbar:
        batch = next(tok_iter)
        nll = prior.nll_bits(batch)
        nll.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        if step % cfg.log_every == 0:
            pbar.set_description(f'train nll {nll.item():.4f} bits')
            if wandb:
                wandb.log({'train/nll_bits': nll.item()}, step=step)

        if step % cfg.val_every == 0 or step == cfg.train_steps - 1:
            prior.eval()
            with torch.no_grad():
                total, count = 0.0, 0
                for (vb,) in DataLoader(TensorDataset(val_tokens), batch_size=512):
                    vb = vb.to(device)
                    total += prior.nll_bits(vb).item() * vb.shape[0]
                    count += vb.shape[0]
            val_nll = total / count
            prior.train()

            best_val_nll = min(best_val_nll, val_nll)
            nll_curve.append({'step': step, 'val_nll': val_nll})
            pbar.write(f'[step {step}] val NLL = {val_nll:.4f} bits/code')
            if wandb:
                wandb.log({'val/nll_bits': val_nll}, step=step)

    # ----- generation eval (optional) -----
    # We report two FIDs so the tokenizer and the prior can be told apart:
    #   recon-FID : decode the REAL val tokens -> the ceiling a perfect prior could
    #               reach. Isolates tokenizer quality.
    #   gen-FID   : decode tokens SAMPLED from the prior, swept over sampling
    #               temperature. The gap (gen - recon) is what the prior costs.
    gen_fid = None
    sample_gini = None
    recon_fid = None
    best_temp = None
    gen_fid_by_temp = {}
    if cfg.eval_gen_fid:
        import copy
        from torchmetrics.image.fid import FrechetInceptionDistance

        H = W = int(math.sqrt(seq_len))
        n_eval = min(cfg.n_gen_samples, val_tokens.shape[0])

        # Real-image FID statistics, accumulated once and reused for every fake set.
        print('Building real FID statistics...')
        fid_base = FrechetInceptionDistance(normalize=True).to(device)
        with torch.no_grad():
            for batch in DataLoader(ds.val, batch_size=128):
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                fid_base.update((x.to(device) * 0.5 + 0.5).clamp(0, 1), real=True)

        def _fake_fid(tokens_2d_iter):
            """Clone the real-stats metric, stream fake images through it, compute.
            Returns (fid, sample_gini)."""
            fid = copy.deepcopy(fid_base)
            counts = torch.zeros(K, dtype=torch.long)
            with torch.no_grad():
                for tokens_2d in tokens_2d_iter:
                    counts += tokens_2d.reshape(-1).cpu().bincount(minlength=K)
                    imgs = vqvae.decode_indices(tokens_2d.to(device)).clamp(-1, 1)
                    fid.update((imgs * 0.5 + 0.5).clamp(0, 1), real=False)
            return fid.compute().item(), _gini(counts)

        # (1) reconstruction-FID reference (decode the real val tokens)
        print('Computing reconstruction-FID reference...')
        def _recon_iter():
            for start in range(0, n_eval, 128):
                yield val_tokens[start:start + 128].reshape(-1, H, W)
        recon_fid, _ = _fake_fid(_recon_iter())
        print(f'  recon-FID (perfect-prior ceiling): {recon_fid:.2f}')

        # (2) generation-FID swept over sampling temperature
        prior.eval()
        for temp in cfg.eval_temperatures:
            def _gen_iter(temp=temp):
                for start in range(0, n_eval, 128):
                    n = min(128, n_eval - start)
                    yield prior.sample(n, temperature=temp, device=device).reshape(n, H, W)
            fid_t, gini_t = _fake_fid(_gen_iter())
            gen_fid_by_temp[float(temp)] = {'gen_fid': fid_t, 'sample_gini': gini_t}
            print(f'  gen-FID @ T={temp:.2f}: {fid_t:.2f}  sample Gini {gini_t:.3f}')
        prior.train()

        best_temp = min(gen_fid_by_temp, key=lambda t: gen_fid_by_temp[t]['gen_fid'])
        gen_fid = gen_fid_by_temp[best_temp]['gen_fid']
        sample_gini = gen_fid_by_temp[best_temp]['sample_gini']
        print(f'Best gen-FID {gen_fid:.2f} at T={best_temp:.2f}  '
              f'(recon-FID {recon_fid:.2f}, prior gap {gen_fid - recon_fid:+.2f}, '
              f'train Gini {train_gini:.3f})')
        if wandb:
            wandb.log({'eval/gen_fid': gen_fid, 'eval/recon_fid': recon_fid,
                       'eval/best_temp': best_temp, 'eval/sample_gini': sample_gini,
                       'eval/train_gini': train_gini})

    if wandb:
        wandb.finish()

    summary = {
        'run_id': cfg.run_id,
        'vqvae_run_id': cfg.vqvae_run_id,
        'K': K,
        'seq_len': seq_len,
        'n_params': n_params,
        'best_val_nll': best_val_nll,
        'final_val_nll': nll_curve[-1]['val_nll'] if nll_curve else None,
        'train_gini': train_gini,
        'gen_fid': gen_fid,
        'recon_fid': recon_fid,
        'best_temp': best_temp,
        'gen_fid_by_temp': gen_fid_by_temp,
        'sample_gini': sample_gini,
        'nll_curve': nll_curve,
        'wallclock_s': time.time() - t0,
    }
    with open(out_dir / 'prior_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary
