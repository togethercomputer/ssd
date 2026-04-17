"""Tier 0: lock down build_tree_mask_bias semantics before optimizing it.

The current implementation (ssd/layers/tree_mask.py) builds the dense bias
tensor in numpy on CPU, per-step, and transfers it to the target device.
We plan to rewrite it as a pure-torch function so it can run on GPU with no
.tolist()/H2D and (eventually) produce all K steps at once. These tests pin
the observable contract so the rewrite is bit-exact.

The tests run on CPU (torch.device("cpu")) so they don't require CUDA.
"""
from __future__ import annotations

import itertools

import pytest
import torch

from ssd.layers.tree_mask import _MASK_VAL, build_tree_mask_bias

pytestmark = pytest.mark.tier0


# ---------------------------------------------------------------------------
# A simple pure-python reference so the test can assert the full contract
# (shape, dtype, unmasked=0.0 / masked=_MASK_VAL at every (q, kv) entry)
# independent of the implementation under test.
# ---------------------------------------------------------------------------
def _reference_bias(
    context_lens, step, K, MQ_LEN, fan_out_list, fan_out_list_miss, cache_hits, max_kv_stride,
):
    B = int(context_lens.shape[0])
    cl = context_lens.tolist()
    ch = cache_hits[:B].tolist()

    # Glue patterns: repeat_interleave(tril(K+1), fan_out) -> [MQ_LEN, K+1]
    tril = torch.tril(torch.ones(K + 1, K + 1, dtype=torch.bool))
    glue_hit = tril.repeat_interleave(torch.tensor(fan_out_list, dtype=torch.int64), dim=0)
    glue_miss = tril.repeat_interleave(torch.tensor(fan_out_list_miss, dtype=torch.int64), dim=0)
    assert glue_hit.shape == (MQ_LEN, K + 1)

    ttl_added = (step + 1) * MQ_LEN + (K + 1)
    out = torch.full((B * MQ_LEN, max_kv_stride), _MASK_VAL, dtype=torch.float32)

    for b in range(B):
        cols_b = int(cl[b])
        prefix = cols_b - ttl_added
        row_off = b * MQ_LEN

        if prefix > 0:
            out[row_off : row_off + MQ_LEN, :prefix] = 0.0

        glue = glue_hit if int(ch[b]) == 1 else glue_miss
        gstart = prefix
        # unmasked where glue==True
        glue_slice = out[row_off : row_off + MQ_LEN, gstart : gstart + K + 1]
        glue_slice[glue] = 0.0

        diag_start = prefix + K + 1
        for blk in range(step + 1):
            for r in range(MQ_LEN):
                c = diag_start + blk * MQ_LEN + r
                if c < max_kv_stride:
                    out[row_off + r, c] = 0.0

    return out.reshape(-1)


# ---------------------------------------------------------------------------
# Parameter matrix. Keeps the search small but covers:
#   - uniform vs non-uniform fan_out_list
#   - hit-only, miss-only, mixed
#   - multiple step values (diag block count grows)
#   - multiple batch sizes incl. B=1 fast path
#   - prefix_len=0 edge case (context_len exactly equals ttl_added)
# ---------------------------------------------------------------------------
CONFIGS = [
    # (K, fan_out_list, fan_out_list_miss)
    (2, [2, 2, 2], [2, 2, 2]),            # uniform
    (2, [1, 3, 3], [7, 0, 0]),            # jit hit vs big miss
    (3, [2, 2, 2, 2], [8, 0, 0, 0]),      # K=3
    (1, [1, 4], [1, 4]),                  # small K
]

BATCHES = [1, 2, 5]
STEPS = [0, 1, 2]
PREFIX_OFFSETS = [0, 3, 17]  # added to ttl_added to exercise prefix_len > 0


@pytest.mark.parametrize("K,fan_out_list,fan_out_list_miss", CONFIGS)
@pytest.mark.parametrize("B", BATCHES)
@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("prefix_off", PREFIX_OFFSETS)
def test_matches_reference(K, fan_out_list, fan_out_list_miss, B, step, prefix_off):
    MQ_LEN = sum(fan_out_list)
    assert MQ_LEN == sum(fan_out_list_miss)
    ttl_added = (step + 1) * MQ_LEN + (K + 1)
    max_kv_stride = ttl_added + max(PREFIX_OFFSETS) + 8

    torch.manual_seed(B * 100 + step * 10 + prefix_off)
    # Varying per-seq context_lens so different rows exercise different prefix_len.
    context_lens = torch.tensor(
        [ttl_added + prefix_off + i for i in range(B)], dtype=torch.int32
    )
    # Alternating cache_hits hit patterns.
    cache_hits = torch.tensor([b % 2 for b in range(B)], dtype=torch.float32)

    got = build_tree_mask_bias(
        context_lens, step=step, K=K, MQ_LEN=MQ_LEN,
        fan_out_list=fan_out_list, fan_out_list_miss=fan_out_list_miss,
        cache_hits=cache_hits, max_kv_stride=max_kv_stride, device=torch.device("cpu"),
    )
    ref = _reference_bias(
        context_lens, step, K, MQ_LEN, fan_out_list, fan_out_list_miss, cache_hits, max_kv_stride,
    )

    assert got.shape == ref.shape, f"shape {got.shape} vs ref {ref.shape}"
    assert got.dtype == torch.float32
    assert torch.equal(got, ref), (
        f"mismatch: K={K} B={B} step={step} prefix_off={prefix_off}"
    )


# ---------------------------------------------------------------------------
# Explicit contract checks — don't just compare to a second reference, check
# the structural invariants directly so a bug in BOTH the impl and the ref
# (same-shape copy-paste) is still caught.
# ---------------------------------------------------------------------------
def test_shape_and_dtype():
    B, K, MQ_LEN, max_kv = 3, 2, 6, 64
    bias = build_tree_mask_bias(
        torch.tensor([30, 35, 40], dtype=torch.int32), step=0, K=K, MQ_LEN=MQ_LEN,
        fan_out_list=[2, 2, 2], fan_out_list_miss=[2, 2, 2],
        cache_hits=torch.tensor([1, 0, 1], dtype=torch.float32),
        max_kv_stride=max_kv, device=torch.device("cpu"),
    )
    assert bias.shape == (B * MQ_LEN * max_kv,)
    assert bias.dtype == torch.float32


def test_masked_values_are_MASK_VAL_exactly():
    """Unmasked entries must be exactly 0.0 and masked entries exactly _MASK_VAL.
    No intermediate values are allowed — this is what score_mod adds to scores."""
    bias = build_tree_mask_bias(
        torch.tensor([30], dtype=torch.int32), step=0, K=2, MQ_LEN=6,
        fan_out_list=[2, 2, 2], fan_out_list_miss=[2, 2, 2],
        cache_hits=torch.tensor([1], dtype=torch.float32),
        max_kv_stride=50, device=torch.device("cpu"),
    )
    unique = torch.unique(bias)
    assert torch.equal(unique.sort().values, torch.tensor([_MASK_VAL, 0.0])), (
        f"bias contains values other than {{_MASK_VAL, 0.0}}: {unique.tolist()}"
    )


def test_prefix_all_unmasked():
    """For context_len > ttl_added, all prefix columns (cols < prefix_len) must be 0.0."""
    K, MQ_LEN, step = 2, 6, 0
    ttl_added = (step + 1) * MQ_LEN + (K + 1)
    prefix_len = 10
    ctx = ttl_added + prefix_len
    max_kv = 64
    bias = build_tree_mask_bias(
        torch.tensor([ctx], dtype=torch.int32), step=step, K=K, MQ_LEN=MQ_LEN,
        fan_out_list=[2, 2, 2], fan_out_list_miss=[2, 2, 2],
        cache_hits=torch.tensor([1], dtype=torch.float32),
        max_kv_stride=max_kv, device=torch.device("cpu"),
    ).view(MQ_LEN, max_kv)
    assert torch.all(bias[:, :prefix_len] == 0.0), "prefix region must be all 0.0"


def test_diag_blocks_are_identity():
    """Each diagonal block in KV space (step+1 blocks of size MQ_LEN each) must be identity."""
    K, MQ_LEN = 1, 5  # fan_out_list=[1,4]
    fol = [1, 4]
    for step in (0, 1, 2):
        ttl_added = (step + 1) * MQ_LEN + (K + 1)
        prefix_len = 3
        ctx = ttl_added + prefix_len
        max_kv = ctx + 4
        bias = build_tree_mask_bias(
            torch.tensor([ctx], dtype=torch.int32), step=step, K=K, MQ_LEN=MQ_LEN,
            fan_out_list=fol, fan_out_list_miss=fol,
            cache_hits=torch.tensor([1], dtype=torch.float32),
            max_kv_stride=max_kv, device=torch.device("cpu"),
        ).view(MQ_LEN, max_kv)
        diag_start = prefix_len + K + 1
        for blk in range(step + 1):
            sub = bias[:, diag_start + blk * MQ_LEN : diag_start + (blk + 1) * MQ_LEN]
            expected = torch.where(
                torch.eye(MQ_LEN, dtype=torch.bool),
                torch.tensor(0.0), torch.tensor(_MASK_VAL),
            )
            assert torch.equal(sub, expected), (
                f"diag block {blk} (step={step}) is not identity-shaped"
            )


def test_hit_vs_miss_uses_different_glue_when_fan_outs_differ():
    """The glue block (cols [prefix_len, prefix_len+K+1)) must differ for hit vs miss
    rows when fan_out_list != fan_out_list_miss."""
    K, MQ_LEN, step = 2, 7, 0
    fol_hit = [1, 3, 3]
    fol_miss = [7, 0, 0]
    assert sum(fol_hit) == MQ_LEN == sum(fol_miss)
    ttl_added = (step + 1) * MQ_LEN + (K + 1)
    ctx = ttl_added  # prefix_len=0, so the mask starts with the glue block
    max_kv = ctx + 4
    bias = build_tree_mask_bias(
        torch.tensor([ctx, ctx], dtype=torch.int32), step=step, K=K, MQ_LEN=MQ_LEN,
        fan_out_list=fol_hit, fan_out_list_miss=fol_miss,
        cache_hits=torch.tensor([1, 0], dtype=torch.float32),
        max_kv_stride=max_kv, device=torch.device("cpu"),
    ).view(2 * MQ_LEN, max_kv)
    glue_hit = bias[:MQ_LEN, :K + 1]
    glue_miss = bias[MQ_LEN:, :K + 1]
    assert not torch.equal(glue_hit, glue_miss), (
        "hit and miss rows must use different glue when fan_out_list != fan_out_list_miss"
    )


# ---------------------------------------------------------------------------
# CORE OPTIMIZATION INVARIANT: the step=K-1 mask built at the widened context_lens
# equals the step=s mask on every (q, kv) position FA4 actually reads
# (kv_idx < context_lens_s).
#
# This is the correctness claim behind "build the mask once per decode_tree at
# step=K-1 and reuse for all steps": every earlier step only reads a prefix of
# the KV range, and the earlier step's mask agrees with the K-1 mask on that
# prefix (prefix_len is constant across steps; diag blocks beyond (s+1) sit at
# KV offsets beyond context_lens_s).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K", [2, 3, 4])
@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("fan_out", [2, 4])
@pytest.mark.parametrize("prefix_off", [0, 5, 17])
def test_step_k1_mask_contains_all_earlier_steps(K, B, fan_out, prefix_off):
    """At step=K-1 with the widened context_lens, mask[b, q, :context_lens_s] must
    bitwise equal the step=s mask at every s in [0, K-1].
    """
    fan_out_list = [fan_out] * (K + 1)
    fan_out_list_miss = [sum(fan_out_list)] + [0] * K
    MQ_LEN = sum(fan_out_list)

    # Per-step s: context_lens_s = context_lens_0 + s * MQ_LEN, ttl_added_s = (s+1)*MQ_LEN + K+1.
    ttl_added_0 = 1 * MQ_LEN + (K + 1)
    context_lens_0 = torch.tensor(
        [ttl_added_0 + prefix_off + b for b in range(B)], dtype=torch.int32
    )
    # Alternating hit/miss
    cache_hits = torch.tensor([b % 2 for b in range(B)], dtype=torch.float32)

    # Max-step mask (context widened by (K-1)*MQ_LEN).
    max_context_lens = context_lens_0 + (K - 1) * MQ_LEN
    max_kv = int(max_context_lens.max().item()) + 4
    mask_k1 = build_tree_mask_bias(
        max_context_lens, step=K - 1, K=K, MQ_LEN=MQ_LEN,
        fan_out_list=fan_out_list, fan_out_list_miss=fan_out_list_miss,
        cache_hits=cache_hits, max_kv_stride=max_kv, device=torch.device("cpu"),
    ).view(B, MQ_LEN, max_kv)

    for s in range(K):
        context_lens_s = context_lens_0 + s * MQ_LEN
        # Build the per-step mask the way the engine would today.
        mask_s = build_tree_mask_bias(
            context_lens_s, step=s, K=K, MQ_LEN=MQ_LEN,
            fan_out_list=fan_out_list, fan_out_list_miss=fan_out_list_miss,
            cache_hits=cache_hits, max_kv_stride=max_kv, device=torch.device("cpu"),
        ).view(B, MQ_LEN, max_kv)

        # FA4 only reads kv_idx < context_lens_s[b]. Check agreement there.
        for b in range(B):
            cl_b = int(context_lens_s[b].item())
            # The mask_k1 row at this batch index, restricted to the first cl_b cols,
            # must match the mask_s row restricted to the first cl_b cols.
            a = mask_k1[b, :, :cl_b]
            c = mask_s[b, :, :cl_b]
            assert torch.equal(a, c), (
                f"K-1 mask diverges from step={s} mask at b={b} within readable KV range "
                f"(K={K} MQ_LEN={MQ_LEN} prefix_off={prefix_off}): "
                f"{(a != c).sum().item()} positions differ"
            )


def test_batch1_and_batch5_rows_match():
    """Batch position must not affect a row's mask content — a single-row call
    with context_len=X must produce the same bytes as the matching row of a
    B=5 call with context_lens[0]=X (after stride slicing).

    Guards against an optimization that accidentally couples rows across batch.
    """
    K, MQ_LEN, step = 2, 6, 1
    ttl_added = (step + 1) * MQ_LEN + (K + 1)
    ctx_primary = ttl_added + 7
    max_kv = 64

    # B=1 reference for ctx_primary
    solo = build_tree_mask_bias(
        torch.tensor([ctx_primary], dtype=torch.int32),
        step=step, K=K, MQ_LEN=MQ_LEN,
        fan_out_list=[2, 2, 2], fan_out_list_miss=[2, 2, 2],
        cache_hits=torch.tensor([1], dtype=torch.float32),
        max_kv_stride=max_kv, device=torch.device("cpu"),
    ).view(MQ_LEN, max_kv)

    # Same seq placed at batch index 2 in B=5 (other seqs have different ctxs).
    ctxs = [ttl_added + 1, ttl_added + 2, ctx_primary, ttl_added + 3, ttl_added + 4]
    hits = [0, 1, 1, 0, 1]
    batch = build_tree_mask_bias(
        torch.tensor(ctxs, dtype=torch.int32),
        step=step, K=K, MQ_LEN=MQ_LEN,
        fan_out_list=[2, 2, 2], fan_out_list_miss=[2, 2, 2],
        cache_hits=torch.tensor(hits, dtype=torch.float32),
        max_kv_stride=max_kv, device=torch.device("cpu"),
    ).view(len(ctxs) * MQ_LEN, max_kv)
    row_b2 = batch[2 * MQ_LEN : 3 * MQ_LEN, :]
    assert torch.equal(solo, row_b2), "per-row mask must be batch-position independent"
