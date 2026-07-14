"""Test that score_mod with aux_tensors works with FA4 varlen + page_table."""

import torch
import pytest
from flash_attn.cute.interface import flash_attn_varlen_func
from ssd.layers.tree_mask import create_tree_score_mod, build_tree_mask_bias

DEVICE = "cuda"
DTYPE = torch.bfloat16


class TestScoreModBasic:
    """Verify score_mod compiles and runs with FA4 varlen + page_table."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.B = 2
        self.MQ_LEN = 6
        self.num_heads = 4
        self.num_kv_heads = 2
        self.head_dim = 128
        self.num_pages = 200
        self.max_pages_per_seq = 50
        self.page_size = 1

    def _make_inputs(self, kv_lens):
        total_q = self.B * self.MQ_LEN
        q = torch.randn(total_q, self.num_heads, self.head_dim, dtype=DTYPE, device=DEVICE)
        k_cache = torch.randn(self.num_pages, self.page_size, self.num_kv_heads, self.head_dim, dtype=DTYPE, device=DEVICE)
        v_cache = torch.randn(self.num_pages, self.page_size, self.num_kv_heads, self.head_dim, dtype=DTYPE, device=DEVICE)
        cu_seqlens_q = torch.arange(self.B + 1, dtype=torch.int32, device=DEVICE) * self.MQ_LEN
        page_table = torch.zeros(self.B, self.max_pages_per_seq, dtype=torch.int32, device=DEVICE)
        for b in range(self.B):
            n = kv_lens[b]
            page_table[b, :n] = torch.arange(n, dtype=torch.int32, device=DEVICE) + b * 50
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=DEVICE)
        return q, k_cache, v_cache, cu_seqlens_q, page_table, seqused_k

    def test_zero_bias_matches_no_scoremod(self):
        """A score_mod that adds zero should produce identical output."""
        kv_lens = [10, 5]
        max_kv_stride = 50
        q, k, v, cu, pt, sk = self._make_inputs(kv_lens)

        out_base, _ = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu, cu_seqlens_k=None,
            max_seqlen_q=self.MQ_LEN, max_seqlen_k=max(kv_lens),
            seqused_k=sk, page_table=pt,
            softmax_scale=self.head_dim ** -0.5, causal=False,
        )

        score_mod = create_tree_score_mod(max_kv_stride)
        # All-zero bias = no masking
        bias = torch.zeros(self.B * self.MQ_LEN * max_kv_stride, dtype=torch.float32, device=DEVICE)

        out_mod, _ = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu, cu_seqlens_k=None,
            max_seqlen_q=self.MQ_LEN, max_seqlen_k=max(kv_lens),
            seqused_k=sk, page_table=pt,
            softmax_scale=self.head_dim ** -0.5, causal=False,
            score_mod=score_mod, aux_tensors=[bias],
        )

        assert torch.allclose(out_base, out_mod, atol=1e-2), \
            f"Zero bias should match base, max diff: {(out_base - out_mod).abs().max().item()}"

    def test_full_mask_produces_uniform_attention(self):
        """Masking all but one KV position should concentrate attention there."""
        kv_lens = [10, 5]
        max_kv_stride = 50
        q, k, v, cu, pt, sk = self._make_inputs(kv_lens)

        score_mod = create_tree_score_mod(max_kv_stride)
        # Mask everything except KV position 0 for all queries
        bias = torch.full((self.B * self.MQ_LEN * max_kv_stride,), -1e6, dtype=torch.float32, device=DEVICE)
        for b in range(self.B):
            for qi in range(self.MQ_LEN):
                flat_idx = (b * self.MQ_LEN + qi) * max_kv_stride + 0  # only attend to kv_idx=0
                bias[flat_idx] = 0.0

        out, _ = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu, cu_seqlens_k=None,
            max_seqlen_q=self.MQ_LEN, max_seqlen_k=max(kv_lens),
            seqused_k=sk, page_table=pt,
            softmax_scale=self.head_dim ** -0.5, causal=False,
            score_mod=score_mod, aux_tensors=[bias],
        )

        assert not torch.isnan(out).any(), "Masked output has NaN"
        assert not torch.isinf(out).any(), "Masked output has Inf"


class TestTreeMaskBuild:
    """Test build_tree_mask_bias produces correct mask structure."""

    def test_prefix_unmasked(self):
        """All prefix positions should have bias=0 (attend)."""
        B, K, MQ_LEN = 1, 2, 6
        fol = [2, 2, 2]
        context_lens = torch.tensor([20], dtype=torch.int32)  # prefix = 20 - (1*6 + 3) = 11
        cache_hits = torch.tensor([1])
        max_kv_stride = 50

        bias = build_tree_mask_bias(
            context_lens, step=0, K=K, MQ_LEN=MQ_LEN,
            fan_out_list=fol, fan_out_list_miss=fol,
            cache_hits=cache_hits, max_kv_stride=max_kv_stride,
            device="cpu",
        )
        bias_2d = bias.reshape(MQ_LEN, max_kv_stride)
        prefix_len = 20 - (1 * MQ_LEN + K + 1)
        # All prefix columns should be 0.0 (unmasked)
        assert (bias_2d[:, :prefix_len] == 0.0).all(), "Prefix should be unmasked"

    def test_masked_positions_negative(self):
        """Positions beyond the valid KV should be masked (large negative)."""
        B, K, MQ_LEN = 1, 2, 6
        fol = [2, 2, 2]
        context_lens = torch.tensor([20], dtype=torch.int32)
        cache_hits = torch.tensor([1])
        max_kv_stride = 50

        bias = build_tree_mask_bias(
            context_lens, step=0, K=K, MQ_LEN=MQ_LEN,
            fan_out_list=fol, fan_out_list_miss=fol,
            cache_hits=cache_hits, max_kv_stride=max_kv_stride,
            device="cpu",
        )
        bias_2d = bias.reshape(MQ_LEN, max_kv_stride)
        # Beyond context_lens should be masked
        assert (bias_2d[:, 20:] < -1e5).all(), "Beyond context_lens should be masked"

    def test_diagonal_pattern(self):
        """At step 0, each query should attend to its own diagonal position."""
        B, K, MQ_LEN = 1, 2, 6
        fol = [2, 2, 2]
        # context_lens at step 0 needs to be at least ttl_added = 1*MQ_LEN + K+1 = 9
        context_lens = torch.tensor([15], dtype=torch.int32)
        cache_hits = torch.tensor([1])
        max_kv_stride = 50

        bias = build_tree_mask_bias(
            context_lens, step=0, K=K, MQ_LEN=MQ_LEN,
            fan_out_list=fol, fan_out_list_miss=fol,
            cache_hits=cache_hits, max_kv_stride=max_kv_stride,
            device="cpu",
        )
        bias_2d = bias.reshape(MQ_LEN, max_kv_stride)
        prefix_len = 15 - (1 * MQ_LEN + K + 1)  # = 6
        diag_start = prefix_len + K + 1  # = 9
        # At step 0, block 0: bias_2d[q, diag_start + q] should be 0.0
        for q in range(MQ_LEN):
            col = diag_start + q
            assert bias_2d[q, col].item() == 0.0, f"Diagonal at q={q}, col={col} should be unmasked"
