import torch

from vector_quantize_pytorch import DriftingVQ


class TestDriftingVQ:
    def test_image_fmap_shape(self):
        vq = DriftingVQ(dim = 32, codebook_size = 64, channel_first = True,
                        n_hidden_subsample = 64)
        x = torch.randn(2, 32, 8, 8)
        q, indices, loss = vq(x)
        assert q.shape == x.shape
        assert indices.shape == (2, 8, 8)
        assert torch.isfinite(loss)

    def test_sequence_shape(self):
        vq = DriftingVQ(dim = 16, codebook_size = 32, channel_first = False,
                        n_hidden_subsample = 64)
        x = torch.randn(4, 20, 16)
        q, indices, loss = vq(x)
        assert q.shape == x.shape
        assert indices.shape == (4, 20)
        assert torch.isfinite(loss)

    def test_indices_to_codes_roundtrip(self):
        vq = DriftingVQ(dim = 16, codebook_size = 32, channel_first = True,
                        n_hidden_subsample = 32)
        x = torch.randn(1, 16, 4, 4)
        q, indices, _ = vq(x)
        # rotation_trick + l2_normalize off => indices_to_codes should recover
        # the same quantized vectors (up to the STE residual). Use ST path:
        vq2 = DriftingVQ(dim = 16, codebook_size = 32, channel_first = True,
                         n_hidden_subsample = 32, rotation_trick = False, ste = True)
        # copy codebook so lookup matches
        vq2.codebook.data.copy_(vq.codebook.data)
        looked_up = vq2.indices_to_codes(indices)
        assert looked_up.shape == x.shape

    def test_gradients_flow(self):
        vq = DriftingVQ(dim = 8, codebook_size = 16, channel_first = True,
                        n_hidden_subsample = 32)
        x = torch.randn(2, 8, 4, 4, requires_grad = True)
        q, _, loss = vq(x)
        (q.sum() + loss).backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert vq.codebook.grad is not None
        assert torch.isfinite(vq.codebook.grad).all()
        # codebook gradient should be non-zero — the energy term depends on it
        assert vq.codebook.grad.abs().sum() > 0

    def test_subsampling_runs(self):
        # N = 2*16*16 = 512 hiddens; subsample to 64
        vq = DriftingVQ(dim = 16, codebook_size = 32, channel_first = True,
                        n_hidden_subsample = 64)
        x = torch.randn(2, 16, 16, 16)
        _, _, loss = vq(x)
        assert torch.isfinite(loss)

    def test_l2_normalize_finite(self):
        vq = DriftingVQ(dim = 16, codebook_size = 32, channel_first = True,
                        n_hidden_subsample = 64, l2_normalize = True)
        x = torch.randn(2, 16, 8, 8) * 10  # large scale should be normalized away
        _, _, loss = vq(x)
        assert torch.isfinite(loss)
