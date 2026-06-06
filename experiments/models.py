"""CNN encoder / decoder / autoencoder used across all methods.

The autoencoder is intentionally generic: the quantizer is a module passed
in at construction time, so methods are swapped by replacing the middle
module only. Encoder/decoder weights are identical across methods within
a given dataset, so headline comparisons are apples-to-apples.
"""
from __future__ import annotations

import torch.nn as nn


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

    def encode_indices(self, x):
        """Encode an image batch to (B, H, W) code indices — for downstream prior training."""
        h = self.encoder(x)
        _, indices, _ = self.quantizer(h)
        return indices

    def decode_indices(self, indices):
        """Decode code indices back to images — for downstream prior sampling.

        Quantizers expose different index->code methods: the drift quantizers
        have ``indices_to_codes``; the library's VectorQuantize (EMA) and SimVQ
        expose ``get_output_from_indices``. Support both so the EMA baseline can
        be decoded for generation, not just the drift variants.
        """
        quant = self.quantizer
        if hasattr(quant, 'indices_to_codes'):
            q = quant.indices_to_codes(indices)
        elif hasattr(quant, 'get_output_from_indices'):
            q = quant.get_output_from_indices(indices)
        else:
            raise AttributeError(
                f'{type(quant).__name__} has no index->code method for decoding')
        return self.decoder(q)
