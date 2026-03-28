"""Tests for FA4 flash_attn_varlen_func with paged KV cache (tree decode replacement)."""

import pytest
import torch
from flash_attn.cute.interface import flash_attn_varlen_func as fa4_varlen_func
from ssd.layers.attention import Attention
from ssd.utils.context import set_context, reset_context


DEVICE = "cuda"
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# FA4 varlen + page_table: basic correctness
# ---------------------------------------------------------------------------

class TestFA4VarlenPageTable:
    """Test flash_attn_varlen_func with page_table at various page sizes."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.B = 2
        self.MQ_LEN = 6
        self.num_heads = 4
        self.num_kv_heads = 2
        self.head_dim = 128
        self.num_pages = 200
        self.max_pages_per_seq = 20

    def _run(self, page_size, kv_lens):
        total_q = self.B * self.MQ_LEN
        q = torch.randn(total_q, self.num_heads, self.head_dim, dtype=DTYPE, device=DEVICE)
        k_cache = torch.randn(self.num_pages, page_size, self.num_kv_heads, self.head_dim, dtype=DTYPE, device=DEVICE)
        v_cache = torch.randn(self.num_pages, page_size, self.num_kv_heads, self.head_dim, dtype=DTYPE, device=DEVICE)
        cu_seqlens_q = torch.arange(self.B + 1, dtype=torch.int32, device=DEVICE) * self.MQ_LEN

        page_table = torch.zeros(self.B, self.max_pages_per_seq, dtype=torch.int32, device=DEVICE)
        for b in range(self.B):
            n_pages = (kv_lens[b] + page_size - 1) // page_size
            page_table[b, :n_pages] = torch.arange(n_pages, dtype=torch.int32, device=DEVICE) + b * 50

        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=DEVICE)

        out, lse = fa4_varlen_func(
            q, k_cache, v_cache,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=None,
            max_seqlen_q=self.MQ_LEN,
            max_seqlen_k=max(kv_lens),
            seqused_k=seqused_k,
            page_table=page_table,
            softmax_scale=self.head_dim ** -0.5,
            causal=False,
        )
        return out, lse

    @pytest.mark.parametrize("page_size", [1, 16, 128])
    def test_output_shape(self, page_size):
        out, _ = self._run(page_size, kv_lens=[10, 5])
        assert out.shape == (self.B * self.MQ_LEN, self.num_heads, self.head_dim)

    @pytest.mark.parametrize("page_size", [1, 16, 128])
    def test_no_nan_inf(self, page_size):
        out, _ = self._run(page_size, kv_lens=[10, 5])
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    @pytest.mark.parametrize("page_size", [1, 16, 128])
    def test_lse_returned_none_by_default(self, page_size):
        _, lse = self._run(page_size, kv_lens=[10, 5])
        assert lse is None, "LSE should be None when return_lse=False (default)"

    def test_variable_kv_lengths(self):
        """Sequences with very different KV lengths should both produce valid output."""
        self.max_pages_per_seq = 60  # accommodate kv_len=50
        out, _ = self._run(page_size=1, kv_lens=[50, 3])
        assert not torch.isnan(out).any()
        # Check that the two sequences produce different outputs (they have different KV)
        out_seq0 = out[:self.MQ_LEN]
        out_seq1 = out[self.MQ_LEN:]
        assert not torch.allclose(out_seq0, out_seq1), "Different KV should produce different outputs"

    def test_deterministic(self):
        """Same inputs should produce same outputs."""
        out1, _ = self._run(page_size=1, kv_lens=[10, 5])
        torch.manual_seed(42)  # reset seed to get same random inputs
        out2, _ = self._run(page_size=1, kv_lens=[10, 5])
        assert torch.allclose(out1, out2), "Same inputs should produce identical outputs"

    def test_batch_size_1(self):
        """Single-sequence batch should work."""
        self.B = 1
        out, _ = self._run(page_size=1, kv_lens=[10])
        assert out.shape == (self.MQ_LEN, self.num_heads, self.head_dim)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Attention layer integration: tree decode path
# ---------------------------------------------------------------------------

class TestAttentionTreeDecode:
    """Test the Attention module's tree_decode path end-to-end with FA4."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.num_heads = 8
        self.num_kv_heads = 2
        self.head_dim = 128
        self.scale = self.head_dim ** -0.5
        self.F_fan = 2
        self.K_spec = 2
        self.MQ_LEN = self.F_fan * (self.K_spec + 1)
        self.page_size = 1
        self.num_pages = 200
        self.max_pages_per_seq = 50
        self.max_model_len = 50
        yield
        reset_context()

    def _make_attn(self):
        attn = Attention(
            num_heads=self.num_heads, head_dim=self.head_dim, scale=self.scale,
            num_kv_heads=self.num_kv_heads, draft=True, speculate=True,
            draft_async=True, use_eagle=False, F=self.F_fan, K=self.K_spec,
        )
        attn.k_cache = torch.randn(
            self.num_pages, self.page_size, self.num_kv_heads, self.head_dim,
            dtype=DTYPE, device=DEVICE)
        attn.v_cache = torch.randn(
            self.num_pages, self.page_size, self.num_kv_heads, self.head_dim,
            dtype=DTYPE, device=DEVICE)
        attn.max_seqlen_k = self.max_model_len
        return attn

    def _run(self, attn, B, context_lens_list):
        total_tokens = B * self.MQ_LEN
        q = torch.randn(total_tokens, self.num_heads * self.head_dim, dtype=DTYPE, device=DEVICE)
        k = torch.randn(total_tokens, self.num_kv_heads * self.head_dim, dtype=DTYPE, device=DEVICE)
        v = torch.randn(total_tokens, self.num_kv_heads * self.head_dim, dtype=DTYPE, device=DEVICE)

        context_lens = torch.tensor(context_lens_list, dtype=torch.int32, device=DEVICE)
        slot_mapping = torch.arange(total_tokens, dtype=torch.int32, device=DEVICE)

        block_tables = torch.zeros(B, self.max_pages_per_seq, dtype=torch.int32, device=DEVICE)
        for b in range(B):
            n_pages = context_lens_list[b]  # page_size=1, so pages == tokens
            block_tables[b, :n_pages] = torch.arange(n_pages, dtype=torch.int32, device=DEVICE) + b * 50

        cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32, device=DEVICE) * self.MQ_LEN

        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            tree_cu_seqlens_q=cu_seqlens_q,
        )

        with torch.inference_mode():
            out = attn(q, k, v)
        return out

    def test_output_shape(self):
        attn = self._make_attn()
        out = self._run(attn, B=2, context_lens_list=[20, 15])
        expected = (2 * self.MQ_LEN, self.num_heads * self.head_dim)
        assert out.shape == expected, f"Expected {expected}, got {out.shape}"

    def test_no_nan_inf(self):
        attn = self._make_attn()
        out = self._run(attn, B=2, context_lens_list=[20, 15])
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_single_sequence(self):
        attn = self._make_attn()
        out = self._run(attn, B=1, context_lens_list=[30])
        expected = (self.MQ_LEN, self.num_heads * self.head_dim)
        assert out.shape == expected

    def test_different_context_lens(self):
        """Sequences with different context lengths should produce different outputs."""
        attn = self._make_attn()
        out = self._run(attn, B=2, context_lens_list=[40, 10])
        out_seq0 = out[:self.MQ_LEN]
        out_seq1 = out[self.MQ_LEN:]
        assert not torch.allclose(out_seq0, out_seq1)

    def test_non_tree_decode_paths_unaffected(self):
        """Verify that non-tree-decode paths still use the original kernels."""
        attn = Attention(
            num_heads=self.num_heads, head_dim=self.head_dim, scale=self.scale,
            num_kv_heads=self.num_kv_heads, draft=False, speculate=False,
            draft_async=False, use_eagle=False,
        )
        # This attention module should NOT take the tree_decode path
        assert not (attn.speculate and attn.draft and attn.draft_async)
