from __future__ import annotations

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from einx import get_at
from einops import rearrange, pack, unpack

from vector_quantize_pytorch.vector_quantize_pytorch import rotate_to

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        out, = unpack(out, packed_shape, inv_pattern)
        return out

    return packed, inverse

# drifting potential
# U(r) = tau * (r + tau) * exp(-r / tau) * q1 * q2
# force f(r) = r * exp(-r / tau)  (radial; sign set by q1*q2)
# opposite charges attract, same charges repel — see Liu (2026), "Drifting VQ-VAE"

def drifting_potential(r, q_prod, tau, eps = 1e-8):
    r = r + eps
    return tau * (r + tau) * torch.exp(-r / tau) * q_prod


class DriftingVQ(Module):
    """
    Drifting VQ-VAE quantizer (Liu, 2026).

    Encoder outputs are positive charges, codewords are negative charges. The
    drifting potential energy term replaces the usual codebook + commitment
    losses: attraction pulls codes toward encoded data (reviving dead codes),
    repulsion spreads codes and hiddens apart.

    Returns ``(quantized, indices, drift_loss)`` matching the ``VectorQuantize``
    / ``SimVQ`` calling convention so it is a drop-in replacement.
    """

    def __init__(
        self,
        dim,
        codebook_size,
        tau = 1.0,                # interaction length scale; with l2_normalize=True, distances live in [0, 2]
        channel_first = False,
        accept_image_fmap = False,
        n_hidden_subsample = 1024,
        l2_normalize = True,      # project hidden & codes to the unit sphere so tau is interpretable
        rotation_trick = True,    # gradient path for encoder; False -> plain straight-through
        ste = True,               # if both are False, no recon gradient reaches encoder (matches Liu's toy code)
        energy_weight = 1.0,
        commitment_weight = 0.0,
        eps = 1e-8,
    ):
        super().__init__()

        self.dim = dim
        self.codebook_size = codebook_size
        self.tau = tau

        # accept_image_fmap is the VectorQuantize-style alias for channel_first
        # on (B, C, H, W) inputs
        self.channel_first = channel_first or accept_image_fmap

        self.n_hidden_subsample = n_hidden_subsample
        self.l2_normalize = l2_normalize
        self.eps = eps

        # learnable codebook, initialized like SimVQ's frozen codebook
        codebook = torch.randn(codebook_size, dim) * (dim ** -0.5)
        self.codebook = nn.Parameter(codebook)

        self.rotation_trick = rotation_trick
        self.ste = ste

        self.energy_weight = energy_weight
        self.commitment_weight = commitment_weight

        # populated each forward() so callers (logging) can read per-term energies
        self.last_breakdown: dict[str, torch.Tensor] = {}

    def indices_to_codes(self, indices):
        codes = self.codebook
        if self.l2_normalize:
            codes = F.normalize(codes, dim = -1)
        quantized = get_at('[c] d, b ... -> b ... d', codes, indices)
        if self.channel_first:
            quantized = rearrange(quantized, 'b ... d -> b d ...')
        return quantized

    def _energy(self, hidden, codes):
        """
        Drifting potential energy: U_pp + U_nn + U_pn.

        hidden: (N, d) flattened encoder outputs.
        codes:  (K, d) codebook.

        Charges:
          q_h = +1/N per hidden  (total +1)
          q_c = -1/K per code    (total -1)
        Sign of q_h * q_c gives attraction vs repulsion automatically.

        U_pp uses a random subset of size S = ``n_hidden_subsample`` to keep
        the N x N cost bounded; the subsampled pair sum is rescaled to be an
        unbiased estimator of the full sum.
        """
        N, K = hidden.shape[0], codes.shape[0]
        tau = self.tau

        q_h = 1.0 / N
        q_c = -1.0 / K

        # hidden <-> code attraction (full, cheap: N*K)
        d_hc = torch.cdist(hidden, codes)
        U_pn = drifting_potential(d_hc, q_h * q_c, tau, self.eps).sum()

        # code <-> code repulsion (full, cheap: K*K)
        d_cc = torch.cdist(codes, codes)
        V_cc = drifting_potential(d_cc, q_c * q_c, tau, self.eps)
        # strict upper triangle, exclude self-pairs (r=0)
        U_nn = (V_cc.sum() - V_cc.diagonal().sum()) / 2

        # hidden <-> hidden repulsion: O(N^2) — subsample
        S = min(self.n_hidden_subsample, N)
        if S < N:
            idx = torch.randperm(N, device = hidden.device)[:S]
            h_sub = hidden[idx]
            # unbiased rescale for sampling-without-replacement pair sum:
            # E[ sum_{i<j in S} f(r_ij) ] = (S(S-1) / (N(N-1))) * sum_{i<j in N} f(r_ij)
            scale = (N * (N - 1)) / max(S * (S - 1), 1)
        else:
            h_sub = hidden
            scale = 1.0

        d_hh = torch.cdist(h_sub, h_sub)
        V_hh = drifting_potential(d_hh, q_h * q_h, tau, self.eps)
        U_pp = scale * (V_hh.sum() - V_hh.diagonal().sum()) / 2

        total = U_pp + U_nn + U_pn

        # stash detached scalars for outside logging
        self.last_breakdown = {
            'U_pp': U_pp.detach(),
            'U_nn': U_nn.detach(),
            'U_pn': U_pn.detach(),
            'U_total': total.detach(),
            'mean_d_hc': d_hc.mean().detach(),
            'mean_d_cc': d_cc.mean().detach(),
        }
        return total

    def forward(self, x):
        if self.channel_first:
            x = rearrange(x, 'b d ... -> b ... d')

        # flatten the spatial / sequence dims to a (B, N, d) layout
        x, inverse_pack = pack_one(x, 'b * d')

        codes = self.codebook
        if self.l2_normalize:
            x = F.normalize(x, dim = -1)
            codes = F.normalize(codes, dim = -1)

        # flatten the batch into one big set of hiddens for the energy term —
        # codewords serve a single global codebook, so all hiddens contribute
        # to the same physical system.
        hidden_flat = rearrange(x, 'b n d -> (b n) d')

        # nearest-neighbor assignment (no grad through argmin)
        with torch.no_grad():
            dist = torch.cdist(x, codes)
            indices = dist.argmin(dim = -1)

        quantized = get_at('[c] d, b n -> b n d', codes, indices)

        # drifting potential energy — the "codebook + commitment" replacement
        energy = self._energy(hidden_flat, codes)
        drift_loss = self.energy_weight * energy

        # optional plain commit loss to anchor encoder outputs to chosen code;
        # off by default since the energy already drives encoder & codes together
        if self.commitment_weight > 0.0:
            drift_loss = drift_loss + self.commitment_weight * F.mse_loss(x, quantized.detach())

        # gradient path encoder -> decoder
        if self.rotation_trick:
            quantized = rotate_to(x, quantized)
        elif self.ste:
            quantized = x + (quantized - x).detach()
        # else: argmin output is used as-is (matches Liu's toy code; no recon
        # gradient reaches the encoder — only useful for ablation)

        quantized = inverse_pack(quantized)
        indices = inverse_pack(indices, 'b *')

        if self.channel_first:
            quantized = rearrange(quantized, 'b ... d -> b d ...')

        return quantized, indices, drift_loss


if __name__ == '__main__':
    # quick shape / gradient sanity check
    torch.manual_seed(0)
    x = torch.randn(2, 32, 8, 8, requires_grad = True)
    vq = DriftingVQ(dim = 32, codebook_size = 64, channel_first = True, n_hidden_subsample = 64)
    quantized, indices, loss = vq(x)
    assert quantized.shape == x.shape, (quantized.shape, x.shape)
    assert indices.shape == (2, 8, 8), indices.shape
    (quantized.sum() + loss).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert vq.codebook.grad is not None and torch.isfinite(vq.codebook.grad).all()
    print('DriftingVQ smoke check passed. loss =', loss.item())
