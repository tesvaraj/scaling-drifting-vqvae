"""Quantizer factory + hybrid quantizers for the experiment harness.

Supported methods
-----------------
vanilla_ema       : VectorQuantize with EMA codebook updates (modern default).
vanilla_classical : VectorQuantize with learnable codebook (van den Oord, 2017
                    style) — codebook is a Parameter trained by gradient.
simvq             : SimVQ (Zhu, 2025) reparameterization.
drift             : DriftingVQ — physics potential replaces VQ losses.
drift_ema         : Hybrid — drift energy provides encoder gradient, codebook
                    is updated by EMA (placement) on the same assignments.
                    Codebook is a buffer; U_nn has no gradient path (effectively
                    dropped). This is the primary "best of both worlds" hybrid.

The factory returns a quantizer that follows the (quantized, indices, vq_loss)
calling convention so the VQAutoEncoder can swap them freely.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, pack, unpack
from einx import get_at

from vector_quantize_pytorch import VectorQuantize, SimVQ, DriftingVQ
from vector_quantize_pytorch.vector_quantize_pytorch import rotate_to
from vector_quantize_pytorch.drifting_vq import drifting_potential


# ----- helpers -----

def _pack_one(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        out, = unpack(out, packed_shape, inv_pattern if inv_pattern is not None else pattern)
        return out
    return packed, inverse


# ----- drift + EMA hybrid -----

class DriftEMAVQ(nn.Module):
    """Drift energy on the encoder, EMA codebook updates from assignments.

    Mechanics
    ---------
    * The codebook is a buffer (no grad).
    * Each forward computes nearest-neighbor assignments, accumulates batch
      sums per code, and applies an EMA update of the codebook toward the
      mean of assigned hiddens (matches VectorQuantize's EMA path).
    * The drift energy (U_pn + U_pp by default) is also computed and returned
      as ``vq_loss``. Because the codebook is a buffer, energy gradients only
      flow through ``hidden`` — so the energy term provides the encoder's
      "spread + attract" signal, while EMA provides the codebook's placement.

    This decouples placement (EMA) from coverage (physics), testing the
    hypothesis that vanilla's PSNR win comes from EMA's placement quality
    while drift's coverage win comes from the U_pp / U_pn pressure on
    encoder outputs.
    """

    def __init__(
        self,
        dim,
        codebook_size,
        tau = 1.0,
        channel_first = False,
        accept_image_fmap = False,
        n_hidden_subsample = 1024,
        l2_normalize = True,
        rotation_trick = True,
        ste = True,
        energy_weight = 1.0,
        commitment_weight = 0.0,
        energy_terms = ('pp', 'pn'),   # U_nn would have no gradient path; default to dropping it
        decay = 0.99,                    # EMA decay (VectorQuantize default ~0.8; we use stronger)
        eps_ema = 1e-5,
        threshold_dead_code = 0,         # if > 0, reinit dead codes from batch hiddens
        eps = 1e-8,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.tau = tau
        self.channel_first = channel_first or accept_image_fmap
        self.n_hidden_subsample = n_hidden_subsample
        self.l2_normalize = l2_normalize
        self.rotation_trick = rotation_trick
        self.ste = ste
        self.energy_weight = energy_weight
        self.commitment_weight = commitment_weight
        valid = {'pp', 'nn', 'pn'}
        terms = tuple(energy_terms)
        assert set(terms).issubset(valid), terms
        self.energy_terms = terms
        self.decay = decay
        self.eps_ema = eps_ema
        self.threshold_dead_code = threshold_dead_code
        self.eps = eps

        codebook = torch.randn(codebook_size, dim) * (dim ** -0.5)
        self.register_buffer('codebook', codebook)
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('cluster_sum', codebook.clone())
        self.register_buffer('initted', torch.tensor(False))

        self.last_breakdown: dict = {}

    def indices_to_codes(self, indices):
        codes = self.codebook
        if self.l2_normalize:
            codes = F.normalize(codes, dim = -1)
        q = get_at('[c] d, b ... -> b ... d', codes, indices)
        if self.channel_first:
            q = rearrange(q, 'b ... d -> b d ...')
        return q

    def _energy(self, hidden, codes):
        N, K = hidden.shape[0], codes.shape[0]
        tau = self.tau
        device = hidden.device
        q_h = 1.0 / N
        q_c = -1.0 / K

        zero = torch.zeros((), device = device)
        U_pn = zero
        U_nn = zero
        U_pp = zero

        d_hc = torch.cdist(hidden, codes)
        if 'pn' in self.energy_terms:
            U_pn = drifting_potential(d_hc, q_h * q_c, tau, self.eps).sum()

        if 'nn' in self.energy_terms:
            d_cc = torch.cdist(codes, codes)
            V_cc = drifting_potential(d_cc, q_c * q_c, tau, self.eps)
            U_nn = (V_cc.sum() - V_cc.diagonal().sum()) / 2

        if 'pp' in self.energy_terms:
            S = min(self.n_hidden_subsample, N)
            if S < N:
                idx = torch.randperm(N, device = device)[:S]
                h_sub = hidden[idx]
                scale = (N * (N - 1)) / max(S * (S - 1), 1)
            else:
                h_sub = hidden
                scale = 1.0
            d_hh = torch.cdist(h_sub, h_sub)
            V_hh = drifting_potential(d_hh, q_h * q_h, tau, self.eps)
            U_pp = scale * (V_hh.sum() - V_hh.diagonal().sum()) / 2

        total = U_pp + U_nn + U_pn
        self.last_breakdown = {
            'U_pp': U_pp.detach() if torch.is_tensor(U_pp) else U_pp,
            'U_nn': U_nn.detach() if torch.is_tensor(U_nn) else U_nn,
            'U_pn': U_pn.detach() if torch.is_tensor(U_pn) else U_pn,
            'U_total': total.detach() if torch.is_tensor(total) else total,
            'mean_d_hc': d_hc.mean().detach(),
        }
        return total

    @torch.no_grad()
    def _ema_update(self, hidden_flat, indices_flat):
        """Per-step EMA accumulation of code centroids and counts."""
        K = self.codebook_size
        # accumulate sum-of-hiddens per code
        one_hot = F.one_hot(indices_flat, num_classes = K).type(hidden_flat.dtype)  # (N, K)
        batch_sum = one_hot.t() @ hidden_flat                                       # (K, d)
        batch_count = one_hot.sum(dim = 0)                                          # (K,)

        # standard EMA: new = decay * old + (1 - decay) * batch
        self.cluster_sum.mul_(self.decay).add_(batch_sum, alpha = 1.0 - self.decay)
        self.cluster_size.mul_(self.decay).add_(batch_count, alpha = 1.0 - self.decay)

        # update codebook from running centroids; smooth small clusters
        n = self.cluster_size + self.eps_ema
        self.codebook.copy_(self.cluster_sum / n.unsqueeze(-1))

        # optionally reinit dead codes from a batch hidden
        if self.threshold_dead_code > 0:
            dead = self.cluster_size < self.threshold_dead_code
            if dead.any():
                N = hidden_flat.shape[0]
                pick = torch.randint(0, N, (int(dead.sum().item()),), device = hidden_flat.device)
                self.codebook[dead] = hidden_flat[pick]
                self.cluster_size[dead] = 1.0
                self.cluster_sum[dead] = hidden_flat[pick]

    def forward(self, x):
        if self.channel_first:
            x = rearrange(x, 'b d ... -> b ... d')
        x, inv = _pack_one(x, 'b * d')

        codes = self.codebook
        if self.l2_normalize:
            x = F.normalize(x, dim = -1)
            codes = F.normalize(codes, dim = -1)

        hidden_flat = rearrange(x, 'b n d -> (b n) d')

        with torch.no_grad():
            dist = torch.cdist(x, codes)
            indices = dist.argmin(dim = -1)

        if self.training:
            self._ema_update(hidden_flat.detach(), indices.reshape(-1))
            # refresh codes view after the update (l2 norm if applicable)
            codes = self.codebook
            if self.l2_normalize:
                codes = F.normalize(codes, dim = -1)

        quantized = get_at('[c] d, b n -> b n d', codes, indices)
        energy = self._energy(hidden_flat, codes)
        vq_loss = self.energy_weight * energy

        if self.commitment_weight > 0.0:
            vq_loss = vq_loss + self.commitment_weight * F.mse_loss(x, quantized.detach())

        if self.rotation_trick:
            quantized = rotate_to(x, quantized)
        elif self.ste:
            quantized = x + (quantized - x).detach()

        quantized = inv(quantized)
        indices = inv(indices, 'b *')
        if self.channel_first:
            quantized = rearrange(quantized, 'b ... d -> b d ...')
        return quantized, indices, vq_loss


# ----- factory -----

def build_quantizer(
    method: str,
    dim: int,
    codebook_size: int,
    *,
    # drift hyperparams (also used by drift_ema)
    tau: float = 1.0,
    energy_weight: float = 1.0,
    n_hidden_subsample: int = 1024,
    l2_normalize: bool = True,
    energy_terms: tuple = ('pp', 'nn', 'pn'),
    rotation_trick: bool = True,
    ste: bool = True,
    # vanilla/sim hyperparams
    commitment_weight: float = 0.25,
    ema_decay: float = 0.8,
    # drift_ema hyperparams
    drift_ema_decay: float = 0.99,
    drift_ema_dead_threshold: int = 0,
):
    """Construct any of the supported quantizers by string method name."""
    m = method.lower()

    if m == 'vanilla_ema':
        return VectorQuantize(
            dim = dim,
            codebook_size = codebook_size,
            accept_image_fmap = True,
            rotation_trick = rotation_trick,
            ema_update = True,
            learnable_codebook = False,
            decay = ema_decay,
            commitment_weight = commitment_weight,
        )

    if m == 'vanilla_classical':
        # van den Oord (2017) — codebook is a learnable parameter, no EMA
        return VectorQuantize(
            dim = dim,
            codebook_size = codebook_size,
            accept_image_fmap = True,
            rotation_trick = rotation_trick,
            ema_update = False,
            learnable_codebook = True,
            commitment_weight = commitment_weight,
        )

    if m == 'simvq':
        return SimVQ(
            dim = dim,
            codebook_size = codebook_size,
            channel_first = True,
            rotation_trick = rotation_trick,
        )

    if m == 'drift':
        return DriftingVQ(
            dim = dim,
            codebook_size = codebook_size,
            channel_first = True,
            tau = tau,
            energy_weight = energy_weight,
            n_hidden_subsample = n_hidden_subsample,
            l2_normalize = l2_normalize,
            rotation_trick = rotation_trick,
            ste = ste,
            energy_terms = energy_terms,
            commitment_weight = 0.0,
        )

    if m == 'drift_commit':
        # drift + standard commitment loss (β‖z_e − sg(e)‖²)
        return DriftingVQ(
            dim = dim,
            codebook_size = codebook_size,
            channel_first = True,
            tau = tau,
            energy_weight = energy_weight,
            n_hidden_subsample = n_hidden_subsample,
            l2_normalize = l2_normalize,
            rotation_trick = rotation_trick,
            ste = ste,
            energy_terms = energy_terms,
            commitment_weight = commitment_weight,
        )

    if m == 'drift_ema':
        return DriftEMAVQ(
            dim = dim,
            codebook_size = codebook_size,
            channel_first = True,
            tau = tau,
            energy_weight = energy_weight,
            n_hidden_subsample = n_hidden_subsample,
            l2_normalize = l2_normalize,
            rotation_trick = rotation_trick,
            ste = ste,
            energy_terms = energy_terms,
            decay = drift_ema_decay,
            threshold_dead_code = drift_ema_dead_threshold,
            commitment_weight = 0.0,
        )

    raise ValueError(f'unknown method: {method}')


# ----- helpers for train-time effects (warmup / annealing) -----

def make_energy_weight_schedule(
    schedule: str = 'constant',
    base: float = 1.0,
    final: Optional[float] = None,
    total_steps: int = 1,
    warmup_steps: int = 0,
):
    """Return a function ``step -> energy_weight``.

    Supported schedules:
        constant   : always ``base``.
        linear     : decay from ``base`` to ``final`` over ``total_steps``.
        warmup_off : ``base`` for first ``warmup_steps``, then 0 (drift-warmup).
        warmup_const : 0 for first ``warmup_steps``, then ``base``.
    """
    s = schedule.lower()
    final = base if final is None else final

    if s == 'constant':
        return lambda step: base
    if s == 'linear':
        def fn(step):
            t = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
            return base + (final - base) * t
        return fn
    if s == 'warmup_off':
        return lambda step: base if step < warmup_steps else 0.0
    if s == 'warmup_const':
        return lambda step: 0.0 if step < warmup_steps else base
    raise ValueError(f'unknown schedule: {schedule}')
