"""Correctness tests: verify FA4 tree mask matches the original flashinfer mask logic."""

import torch
import numpy as np
import pytest
from flash_attn.cute.interface import flash_attn_varlen_func
from ssd.layers.tree_mask import create_tree_score_mod, build_tree_mask_bias
from ssd.engine.helpers.mask_helpers import get_custom_mask

DEVICE = "cuda"
DTYPE = torch.bfloat16


class FakeConfig:
    """Minimal config for get_custom_mask."""
    def __init__(self, K, fan_out_list, fan_out_list_miss, max_model_len):
        self.speculate_k = K
        self.fan_out_list = fan_out_list
        self.fan_out_list_miss = fan_out_list_miss
        self.max_model_len = max_model_len


class TestTreeMaskMatchesOriginal:
    """Verify that build_tree_mask_bias produces masks equivalent to get_custom_mask."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.K = 2
        self.F = 2
        self.fan_out_list = [2, 2, 2]  # F=2, K+1=3 groups
        self.fan_out_list_miss = [2, 2, 2]
        self.MQ_LEN = sum(self.fan_out_list)  # = 6

    def _compare_masks(self, B, context_lens_list, step, cache_hits_list):
        """Compare old (get_custom_mask) vs new (build_tree_mask_bias) for one step."""
        context_lens = torch.tensor(context_lens_list, dtype=torch.int32, device=DEVICE)
        cache_hits = torch.tensor(cache_hits_list, dtype=torch.float32, device=DEVICE)
        max_model_len = 100

        config = FakeConfig(self.K, self.fan_out_list, self.fan_out_list_miss, max_model_len)

        # Old mask: 1D bool tensor, concatenation of per-seq (MQ_LEN x kv_len) masks
        old_mask = get_custom_mask(
            config, context_lens, step, self.K, self.F, B,
            device=DEVICE, cache_hits=cache_hits,
        )

        # New mask bias: (B * MQ_LEN * max_model_len,) float32
        new_bias = build_tree_mask_bias(
            context_lens, step=step, K=self.K, MQ_LEN=self.MQ_LEN,
            fan_out_list=self.fan_out_list,
            fan_out_list_miss=self.fan_out_list_miss,
            cache_hits=cache_hits,
            max_kv_stride=max_model_len,
            device=DEVICE,
        )
        new_bias_2d = new_bias.reshape(B * self.MQ_LEN, max_model_len)

        # Extract per-batch masks from old format and compare
        old_offset = 0
        for b in range(B):
            kv_len = context_lens_list[b]
            old_mask_b = old_mask[old_offset:old_offset + self.MQ_LEN * kv_len].reshape(self.MQ_LEN, kv_len)
            new_mask_b = new_bias_2d[b * self.MQ_LEN:(b + 1) * self.MQ_LEN, :kv_len]

            # Old: True = attend, False = mask
            # New: 0.0 = attend, -1e6 = mask
            new_attend = (new_mask_b == 0.0)
            old_attend = old_mask_b.bool()

            mismatches = (new_attend != old_attend).sum().item()
            assert mismatches == 0, (
                f"Mask mismatch at batch={b}, step={step}: {mismatches} positions differ\n"
                f"  old attend count: {old_attend.sum().item()}, new attend count: {new_attend.sum().item()}\n"
                f"  context_len={kv_len}, cache_hit={cache_hits_list[b]}"
            )
            old_offset += self.MQ_LEN * kv_len

    @pytest.mark.parametrize("step", [0, 1])
    def test_single_seq_cache_hit(self, step):
        # context_lens must be >= ttl_added = (step+1)*MQ_LEN + K+1
        cl = 30 + step * self.MQ_LEN
        self._compare_masks(B=1, context_lens_list=[cl], step=step, cache_hits_list=[1])

    @pytest.mark.parametrize("step", [0, 1])
    def test_single_seq_cache_miss(self, step):
        cl = 30 + step * self.MQ_LEN
        self._compare_masks(B=1, context_lens_list=[cl], step=step, cache_hits_list=[0])

    @pytest.mark.parametrize("step", [0, 1])
    def test_multi_seq_mixed_hits(self, step):
        base = 25 + step * self.MQ_LEN
        self._compare_masks(
            B=3,
            context_lens_list=[base, base + 10, base + 5],
            step=step,
            cache_hits_list=[1, 0, 1],
        )

    def test_step_2(self):
        cl = 40 + 2 * self.MQ_LEN
        self._compare_masks(B=2, context_lens_list=[cl, cl - 5], step=2, cache_hits_list=[1, 0])


class TestFA4WithTreeMask:
    """End-to-end: verify FA4 attention with tree mask produces valid, masked output."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.B = 2
        self.K = 2
        self.MQ_LEN = 6
        self.num_heads = 4
        self.num_kv_heads = 2
        self.head_dim = 128
        self.num_pages = 200
        self.page_size = 1
        self.max_pages_per_seq = 50
        self.max_kv_stride = 50
        self.fan_out_list = [2, 2, 2]
        self.fan_out_list_miss = [2, 2, 2]

    def test_masked_vs_unmasked_differ(self):
        """Masked attention should produce different output than unmasked."""
        kv_lens = [20, 15]
        total_q = self.B * self.MQ_LEN
        q = torch.randn(total_q, self.num_heads, self.head_dim, dtype=DTYPE, device=DEVICE)
        k = torch.randn(self.num_pages, self.page_size, self.num_kv_heads, self.head_dim, dtype=DTYPE, device=DEVICE)
        v = torch.randn(self.num_pages, self.page_size, self.num_kv_heads, self.head_dim, dtype=DTYPE, device=DEVICE)
        cu = torch.arange(self.B + 1, dtype=torch.int32, device=DEVICE) * self.MQ_LEN
        pt = torch.zeros(self.B, self.max_pages_per_seq, dtype=torch.int32, device=DEVICE)
        for b in range(self.B):
            pt[b, :kv_lens[b]] = torch.arange(kv_lens[b], dtype=torch.int32, device=DEVICE) + b * 50
        sk = torch.tensor(kv_lens, dtype=torch.int32, device=DEVICE)

        # Unmasked (causal=False, no score_mod)
        out_unmasked, _ = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu, cu_seqlens_k=None,
            max_seqlen_q=self.MQ_LEN, max_seqlen_k=max(kv_lens),
            seqused_k=sk, page_table=pt,
            softmax_scale=self.head_dim ** -0.5, causal=False,
        )

        # Masked
        score_mod = create_tree_score_mod(self.max_kv_stride)
        context_lens = torch.tensor(kv_lens, dtype=torch.int32)
        cache_hits = torch.tensor([1, 1])
        mask_bias = build_tree_mask_bias(
            context_lens, step=0, K=self.K, MQ_LEN=self.MQ_LEN,
            fan_out_list=self.fan_out_list, fan_out_list_miss=self.fan_out_list_miss,
            cache_hits=cache_hits, max_kv_stride=self.max_kv_stride, device=DEVICE,
        )
        out_masked, _ = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu, cu_seqlens_k=None,
            max_seqlen_q=self.MQ_LEN, max_seqlen_k=max(kv_lens),
            seqused_k=sk, page_table=pt,
            softmax_scale=self.head_dim ** -0.5, causal=False,
            score_mod=score_mod, aux_tensors=[mask_bias],
        )

        assert not torch.isnan(out_masked).any(), "Masked output has NaN"
        assert not torch.allclose(out_masked, out_unmasked, atol=1e-2), \
            "Masked and unmasked should produce different outputs"
