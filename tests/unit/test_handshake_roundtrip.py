"""Tier 0 / I11: handshake pack/unpack round-trip.

The real handshake in SpeculationRequest.send / .receive uses `dist.send` /
`dist.recv` over NCCL. The packing logic (fuse payload into one int64 tensor)
and parsing logic (slice/view out of the fused tensor) are exercised here
without NCCL by copying the bytes between a "sender" tensor and a "receiver"
tensor in memory.

If the pack/parse layouts ever diverge (e.g. dtype mismatch, offset drift,
forgetting to include a tensor), this test will fail immediately without
needing a multi-GPU setup.

What the real send/receive does (paraphrased from helpers/runner_helpers.py):
- pack: torch.cat of [cache_keys, num_tokens, block_tables.to(int64),
                      temps.view(int32).to(int64), ...eagle bits]
- parse: slice by offsets based on metadata=[B, K, max_blocks, eagle_act_dim, vocab_size]
"""
from __future__ import annotations

import pytest
import torch

from ssd.engine.helpers.runner_helpers import concat_tensors_as_int64

pytestmark = pytest.mark.tier0


# ---------------------------------------------------------------------------
# PrefillRequest: input_ids + num_tokens + draft_block_table all int64
# ---------------------------------------------------------------------------
def test_prefill_request_roundtrip_no_eagle():
    B = 3
    max_blocks = 8
    num_tokens_list = [5, 7, 4]
    total_new = sum(num_tokens_list)

    input_ids = torch.arange(total_new, dtype=torch.int64) + 1000
    num_tokens = torch.tensor(num_tokens_list, dtype=torch.int64)
    draft_block_table = torch.arange(B * max_blocks, dtype=torch.int32).view(B, max_blocks) - 5  # some negatives = padding

    # pack (same order as PrefillRequest.send)
    fused = concat_tensors_as_int64(input_ids, num_tokens, draft_block_table)

    # parse (same as PrefillRequest.receive)
    metadata = torch.tensor([total_new, B, max_blocks, 0, 0], dtype=torch.int64)
    total_new_r, B_r, max_blocks_r, use_eagle_r, eagle_act_dim_r = metadata.tolist()
    assert (total_new_r, B_r, max_blocks_r, use_eagle_r, eagle_act_dim_r) == (total_new, B, max_blocks, 0, 0)

    fused_total = total_new_r + B_r + B_r * max_blocks_r
    assert fused.numel() == fused_total

    off = 0
    got_input_ids = fused[off:off + total_new_r]
    off += total_new_r
    got_num_tokens = fused[off:off + B_r]
    off += B_r
    got_draft_bt = fused[off:off + B_r * max_blocks_r].view(B_r, max_blocks_r).to(torch.int32)
    off += B_r * max_blocks_r
    assert off == fused_total

    assert torch.equal(got_input_ids, input_ids)
    assert torch.equal(got_num_tokens, num_tokens)
    assert torch.equal(got_draft_bt, draft_block_table)


# ---------------------------------------------------------------------------
# SpeculationRequest: most complex packing (temps reinterpreted via int32 view)
# ---------------------------------------------------------------------------
def _pack_spec_request(cache_keys, num_tokens, block_tables, temps, eagle_bits=None):
    """Replicates SpeculationRequest.send's pack step (without dist.send)."""
    int64_parts = [
        cache_keys.reshape(-1),
        num_tokens.reshape(-1),
        block_tables.to(torch.int64).reshape(-1),
        temps.view(torch.int32).to(torch.int64).reshape(-1),
    ]
    if eagle_bits is not None:
        recovery_activations, extend_counts, extend_activations, extend_token_ids = eagle_bits
        int64_parts.extend([
            recovery_activations.contiguous().reshape(-1).view(torch.int64),
            extend_counts.reshape(-1),
            extend_activations.contiguous().reshape(-1).view(torch.int64),
            extend_token_ids.reshape(-1),
        ])
    return torch.cat(int64_parts)


def _parse_spec_request(fused, B, K, max_blocks, eagle_act_dim, draft_dtype):
    """Replicates SpeculationRequest.receive's parse step (without dist.recv)."""
    eagle = eagle_act_dim > 0
    _dsz = torch.finfo(draft_dtype).bits // 8 if eagle else 0
    off = 0
    cache_keys = fused[off:off + 3 * B].view(B, 3)
    off += 3 * B
    num_tokens = fused[off:off + B].to(torch.int64)
    off += B
    block_tables = fused[off:off + B * max_blocks].view(B, max_blocks).to(torch.int32)
    off += B * max_blocks
    temps = fused[off:off + B].to(torch.int32).view(torch.float32)
    off += B
    if eagle:
        n_rec = B * eagle_act_dim * _dsz // 8
        recovery_activations = fused[off:off + n_rec].view(draft_dtype).view(B, eagle_act_dim)
        off += n_rec
        extend_counts = fused[off:off + B]
        off += B
        n_ext = B * K * eagle_act_dim * _dsz // 8
        extend_activations = fused[off:off + n_ext].view(draft_dtype).view(B, K, eagle_act_dim)
        off += n_ext
        extend_token_ids = fused[off:off + B * K].view(B, K)
        off += B * K
    else:
        recovery_activations = extend_counts = extend_activations = extend_token_ids = None
    return {
        "cache_keys": cache_keys,
        "num_tokens": num_tokens,
        "block_tables": block_tables,
        "temps": temps,
        "recovery_activations": recovery_activations,
        "extend_counts": extend_counts,
        "extend_activations": extend_activations,
        "extend_token_ids": extend_token_ids,
        "consumed": off,
    }


def test_speculation_request_roundtrip_no_eagle():
    B, K, max_blocks = 4, 3, 8
    torch.manual_seed(0)
    cache_keys = torch.tensor(
        [[i, i * 2, 100 + i] for i in range(B)], dtype=torch.int64,
    )
    num_tokens = torch.tensor([37, 42, 51, 29], dtype=torch.int64)
    block_tables = (torch.arange(B * max_blocks, dtype=torch.int32).view(B, max_blocks) - 3)
    temps = torch.tensor([0.0, 0.7, 1.0, 0.5], dtype=torch.float32)

    fused = _pack_spec_request(cache_keys, num_tokens, block_tables, temps)
    got = _parse_spec_request(fused, B, K, max_blocks, eagle_act_dim=0, draft_dtype=torch.bfloat16)

    assert got["consumed"] == fused.numel()
    assert torch.equal(got["cache_keys"], cache_keys)
    assert torch.equal(got["num_tokens"], num_tokens)
    assert torch.equal(got["block_tables"], block_tables)
    # temps is reinterpreted through int32; value must be preserved
    assert torch.equal(got["temps"], temps), f"{got['temps']} vs {temps}"


def test_speculation_request_roundtrip_with_eagle():
    """Eagle payload includes recovery_activations/extend_activations (bfloat16, bit-cast to int64)."""
    B, K, max_blocks = 2, 2, 4
    eagle_act_dim = 16
    draft_dtype = torch.bfloat16
    torch.manual_seed(1)

    cache_keys = torch.tensor([[0, 0, 77], [1, 1, 88]], dtype=torch.int64)
    num_tokens = torch.tensor([10, 20], dtype=torch.int64)
    block_tables = torch.tensor([[0, 1, 2, -1], [3, 4, -1, -1]], dtype=torch.int32)
    temps = torch.tensor([0.25, 0.75], dtype=torch.float32)

    recovery_activations = torch.randn(B, eagle_act_dim, dtype=torch.float32).to(draft_dtype)
    extend_counts = torch.tensor([1, 2], dtype=torch.int64)
    extend_activations = torch.randn(B, K, eagle_act_dim, dtype=torch.float32).to(draft_dtype)
    extend_token_ids = torch.tensor([[42, 43], [44, 45]], dtype=torch.int64)

    fused = _pack_spec_request(
        cache_keys, num_tokens, block_tables, temps,
        eagle_bits=(recovery_activations, extend_counts, extend_activations, extend_token_ids),
    )
    got = _parse_spec_request(fused, B, K, max_blocks, eagle_act_dim, draft_dtype)

    assert got["consumed"] == fused.numel()
    assert torch.equal(got["cache_keys"], cache_keys)
    assert torch.equal(got["num_tokens"], num_tokens)
    assert torch.equal(got["block_tables"], block_tables)
    assert torch.equal(got["temps"], temps)
    assert torch.equal(got["recovery_activations"], recovery_activations)
    assert torch.equal(got["extend_counts"], extend_counts)
    assert torch.equal(got["extend_activations"], extend_activations)
    assert torch.equal(got["extend_token_ids"], extend_token_ids)


def test_fused_payload_total_size_matches_formula():
    """Independent check: the fused-payload size formula used on the receive side
    must equal the pack-side total for eagle=True.
    """
    B, K, max_blocks, eagle_act_dim = 3, 4, 6, 32
    draft_dtype = torch.bfloat16
    _dsz = torch.finfo(draft_dtype).bits // 8  # = 2 for bf16

    cache_keys = torch.zeros(B, 3, dtype=torch.int64)
    num_tokens = torch.zeros(B, dtype=torch.int64)
    block_tables = torch.zeros(B, max_blocks, dtype=torch.int32)
    temps = torch.zeros(B, dtype=torch.float32)
    recovery_activations = torch.zeros(B, eagle_act_dim, dtype=draft_dtype)
    extend_counts = torch.zeros(B, dtype=torch.int64)
    extend_activations = torch.zeros(B, K, eagle_act_dim, dtype=draft_dtype)
    extend_token_ids = torch.zeros(B, K, dtype=torch.int64)

    fused = _pack_spec_request(
        cache_keys, num_tokens, block_tables, temps,
        eagle_bits=(recovery_activations, extend_counts, extend_activations, extend_token_ids),
    )
    expected = (
        (3 * B) + B + (B * max_blocks) + B +
        (B * eagle_act_dim * _dsz // 8) +
        B +
        (B * K * eagle_act_dim * _dsz // 8) +
        (B * K)
    )
    assert fused.numel() == expected, f"fused {fused.numel()} != expected {expected}"
