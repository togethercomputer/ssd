"""Tests for all Attention code paths after migration from sgl_kernel to FA4.

Covers:
  1. Prefill (contiguous Q/K/V with cu_seqlens)
  2. Verify/glue decode (paged KV cache with cu_seqlens_q)
  3. Single query decode (paged KV cache, 1 query per sequence)
  4. Tree decode is already covered in test_fa4_tree_decode.py
"""

import pytest
import torch
from ssd.layers.attention import Attention
from ssd.utils.context import set_context, reset_context


DEVICE = "cuda"
DTYPE = torch.bfloat16


@pytest.fixture(autouse=True)
def cleanup_context():
    yield
    reset_context()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_attention(
    num_heads=8, num_kv_heads=2, head_dim=128,
    draft=False, speculate=False, draft_async=False,
    F=1, K=1,
):
    scale = head_dim ** -0.5
    return Attention(
        num_heads=num_heads, head_dim=head_dim, scale=scale,
        num_kv_heads=num_kv_heads, draft=draft, speculate=speculate,
        draft_async=draft_async, use_eagle=False, F=F, K=K,
    )


def make_paged_kv_cache(num_pages, page_size, num_kv_heads, head_dim):
    k_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, dtype=DTYPE, device=DEVICE)
    v_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, dtype=DTYPE, device=DEVICE)
    return k_cache, v_cache


def make_block_tables(batch_size, context_lens_list, page_size, max_pages_per_seq, page_offset=0):
    block_tables = torch.zeros(batch_size, max_pages_per_seq, dtype=torch.int32, device=DEVICE)
    for b in range(batch_size):
        n_pages = (context_lens_list[b] + page_size - 1) // page_size
        block_tables[b, :n_pages] = torch.arange(n_pages, dtype=torch.int32, device=DEVICE) + b * page_offset
    return block_tables


# ===========================================================================
# 1. Prefill path
# ===========================================================================

class TestPrefill:
    """context.is_prefill=True, no paged KV cache (contiguous Q/K/V)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(0)
        self.num_heads = 8
        self.num_kv_heads = 2
        self.head_dim = 128
        self.hidden = self.num_heads * self.head_dim
        self.kv_hidden = self.num_kv_heads * self.head_dim

    def _run(self, seq_lens):
        attn = make_attention(
            num_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )
        # No KV cache for prefill without paging
        total_tokens = sum(seq_lens)
        q = torch.randn(total_tokens, self.hidden, dtype=DTYPE, device=DEVICE)
        k = torch.randn(total_tokens, self.kv_hidden, dtype=DTYPE, device=DEVICE)
        v = torch.randn(total_tokens, self.kv_hidden, dtype=DTYPE, device=DEVICE)

        cu_seqlens = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=DEVICE)
        for i, sl in enumerate(seq_lens):
            cu_seqlens[i + 1] = cu_seqlens[i] + sl
        max_seqlen = max(seq_lens)
        slot_mapping = torch.arange(total_tokens, dtype=torch.int32, device=DEVICE)

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            slot_mapping=slot_mapping,
        )

        with torch.inference_mode():
            out = attn(q, k, v)
        return out

    def test_output_shape(self):
        out = self._run([10, 15])
        assert out.shape == (25, self.hidden)

    def test_no_nan_inf(self):
        out = self._run([10, 15])
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_single_sequence(self):
        out = self._run([20])
        assert out.shape == (20, self.hidden)
        assert not torch.isnan(out).any()

    def test_different_seq_lens(self):
        out = self._run([5, 30])
        out_seq0 = out[:5]
        out_seq1 = out[5:]
        assert not torch.allclose(out_seq0.mean(), out_seq1.mean())

    def test_deterministic(self):
        torch.manual_seed(0)
        out1 = self._run([10, 15])
        torch.manual_seed(0)
        out2 = self._run([10, 15])
        assert torch.allclose(out1, out2)


# ===========================================================================
# 2. Prefill with paged KV cache
# ===========================================================================

class TestPrefillPaged:
    """context.is_prefill=True with block_tables set (paged KV)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(1)
        self.num_heads = 8
        self.num_kv_heads = 2
        self.head_dim = 128
        self.hidden = self.num_heads * self.head_dim
        self.kv_hidden = self.num_kv_heads * self.head_dim
        self.page_size = 1
        self.num_pages = 200
        self.max_pages_per_seq = 50

    def _run(self, seq_lens):
        attn = make_attention(
            num_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )
        k_cache, v_cache = make_paged_kv_cache(
            self.num_pages, self.page_size, self.num_kv_heads, self.head_dim,
        )
        attn.k_cache = k_cache
        attn.v_cache = v_cache

        total_tokens = sum(seq_lens)
        q = torch.randn(total_tokens, self.hidden, dtype=DTYPE, device=DEVICE)
        k = torch.randn(total_tokens, self.kv_hidden, dtype=DTYPE, device=DEVICE)
        v = torch.randn(total_tokens, self.kv_hidden, dtype=DTYPE, device=DEVICE)

        cu_seqlens = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=DEVICE)
        for i, sl in enumerate(seq_lens):
            cu_seqlens[i + 1] = cu_seqlens[i] + sl
        max_seqlen = max(seq_lens)

        slot_mapping = torch.arange(total_tokens, dtype=torch.int32, device=DEVICE)
        block_tables = make_block_tables(
            len(seq_lens), seq_lens, self.page_size, self.max_pages_per_seq, page_offset=50,
        )

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
        )

        with torch.inference_mode():
            out = attn(q, k, v)
        return out

    def test_output_shape(self):
        out = self._run([10, 15])
        assert out.shape == (25, self.hidden)

    def test_no_nan_inf(self):
        out = self._run([10, 15])
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ===========================================================================
# 3. Verify/glue decode path
# ===========================================================================

class TestVerifyGlueDecode:
    """speculate=True, cu_seqlens_q is not None → verify_or_glue path."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(2)
        self.num_heads = 8
        self.num_kv_heads = 2
        self.head_dim = 128
        self.hidden = self.num_heads * self.head_dim
        self.kv_hidden = self.num_kv_heads * self.head_dim
        self.page_size = 1
        self.num_pages = 200
        self.max_pages_per_seq = 50
        self.max_model_len = 100

    def _make_attn(self):
        attn = make_attention(
            num_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim, speculate=True,
        )
        k_cache, v_cache = make_paged_kv_cache(
            self.num_pages, self.page_size, self.num_kv_heads, self.head_dim,
        )
        attn.k_cache = k_cache
        attn.v_cache = v_cache
        attn.max_seqlen_k = self.max_model_len
        return attn

    def _run(self, query_lens, context_lens_list):
        """
        query_lens: list of query tokens per sequence (e.g. [K+1, K+1] for verify)
        context_lens_list: list of KV context lengths per sequence
        """
        attn = self._make_attn()
        B = len(query_lens)
        total_q = sum(query_lens)
        q = torch.randn(total_q, self.hidden, dtype=DTYPE, device=DEVICE)
        k = torch.randn(total_q, self.kv_hidden, dtype=DTYPE, device=DEVICE)
        v = torch.randn(total_q, self.kv_hidden, dtype=DTYPE, device=DEVICE)

        cu_seqlens_q = torch.zeros(B + 1, dtype=torch.int32, device=DEVICE)
        for i, ql in enumerate(query_lens):
            cu_seqlens_q[i + 1] = cu_seqlens_q[i] + ql
        max_seqlen_q = max(query_lens)

        context_lens = torch.tensor(context_lens_list, dtype=torch.int32, device=DEVICE)
        slot_mapping = torch.arange(total_q, dtype=torch.int32, device=DEVICE)
        block_tables = make_block_tables(
            B, context_lens_list, self.page_size, self.max_pages_per_seq, page_offset=50,
        )

        set_context(
            is_prefill=False,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )

        with torch.inference_mode():
            out = attn(q, k, v)
        return out

    def test_output_shape(self):
        # 2 sequences, each with K+1=4 query tokens, context 20 and 15
        out = self._run([4, 4], [20, 15])
        assert out.shape == (8, self.hidden)

    def test_no_nan_inf(self):
        out = self._run([4, 4], [20, 15])
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_single_sequence(self):
        out = self._run([8], [30])
        assert out.shape == (8, self.hidden)
        assert not torch.isnan(out).any()

    def test_variable_query_lens(self):
        out = self._run([3, 6], [25, 10])
        assert out.shape == (9, self.hidden)
        assert not torch.isnan(out).any()

    def test_deterministic(self):
        torch.manual_seed(2)
        out1 = self._run([4, 4], [20, 15])
        torch.manual_seed(2)
        out2 = self._run([4, 4], [20, 15])
        assert torch.allclose(out1, out2)


# ===========================================================================
# 4. Single query decode path
# ===========================================================================

class TestSingleQueryDecode:
    """decode=True, not verify_or_glue, not tree_decode → single query decode."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(3)
        self.num_heads = 8
        self.num_kv_heads = 2
        self.head_dim = 128
        self.hidden = self.num_heads * self.head_dim
        self.kv_hidden = self.num_kv_heads * self.head_dim
        self.page_size = 1
        self.num_pages = 200
        self.max_pages_per_seq = 50
        self.max_model_len = 100

    def _make_attn(self):
        # speculate=False (or draft=False, draft_async=False) so we don't enter
        # verify_or_glue or tree_decode
        attn = make_attention(
            num_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim, speculate=False,
        )
        k_cache, v_cache = make_paged_kv_cache(
            self.num_pages, self.page_size, self.num_kv_heads, self.head_dim,
        )
        attn.k_cache = k_cache
        attn.v_cache = v_cache
        attn.max_seqlen_k = self.max_model_len
        return attn

    def _run(self, batch_size, context_lens_list):
        attn = self._make_attn()
        # Single query decode: 1 query token per sequence
        total_q = batch_size
        q = torch.randn(total_q, self.hidden, dtype=DTYPE, device=DEVICE)
        k = torch.randn(total_q, self.kv_hidden, dtype=DTYPE, device=DEVICE)
        v = torch.randn(total_q, self.kv_hidden, dtype=DTYPE, device=DEVICE)

        context_lens = torch.tensor(context_lens_list, dtype=torch.int32, device=DEVICE)
        slot_mapping = torch.arange(total_q, dtype=torch.int32, device=DEVICE)
        block_tables = make_block_tables(
            batch_size, context_lens_list, self.page_size, self.max_pages_per_seq, page_offset=50,
        )

        set_context(
            is_prefill=False,
            cu_seqlens_q=None,  # None → not verify_or_glue
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )

        with torch.inference_mode():
            out = attn(q, k, v)
        return out

    def test_output_shape(self):
        out = self._run(2, [20, 15])
        assert out.shape == (2, self.hidden)

    def test_no_nan_inf(self):
        out = self._run(2, [20, 15])
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_single_sequence(self):
        out = self._run(1, [30])
        assert out.shape == (1, self.hidden)
        assert not torch.isnan(out).any()

    def test_large_batch(self):
        B = 16
        ctx_lens = [5 + i * 2 for i in range(B)]  # max = 5 + 15*2 = 35 < max_pages_per_seq
        out = self._run(B, ctx_lens)
        assert out.shape == (B, self.hidden)
        assert not torch.isnan(out).any()

    def test_different_context_lens_produce_different_outputs(self):
        out = self._run(2, [50, 5])
        assert not torch.allclose(out[0], out[1])

    def test_deterministic(self):
        torch.manual_seed(3)
        out1 = self._run(2, [20, 15])
        torch.manual_seed(3)
        out2 = self._run(2, [20, 15])
        assert torch.allclose(out1, out2)
