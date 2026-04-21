"""Tier 0 / I9: mask helpers equivalence and structure.

The engine picks a different code path based on batch size:
- B <= 8: get_custom_mask_cached (precomputes components into a global cache)
- B > 8:  get_custom_mask_vectorized (ragged concat; avoids per-batch loop)

For every combination of (K, F, fan_out_list, fan_out_list_miss, cache_hits,
context_lens, step), both paths must produce the same flat bool tensor. These
tests also validate the structural contract (shape, causal layout).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ssd.engine.helpers import mask_helpers
from ssd.engine.helpers.mask_helpers import (
    get_custom_mask_cached,
    get_custom_mask_vectorized,
    get_mask_iter_i,
)

pytestmark = pytest.mark.tier0


def _cfg(fan_out_list, fan_out_list_miss, max_model_len=4096):
    return SimpleNamespace(
        fan_out_list=fan_out_list,
        fan_out_list_miss=fan_out_list_miss,
        max_model_len=max_model_len,
    )


def _reset_caches():
    """Mask helpers use module-level global caches — reset between tests to avoid cross-test contamination."""
    mask_helpers._mask_cache = {
        "glue_and_rec_mask": None,
        "diag_components": None,
        "ones_tensor": None,
        "cached_params": None,
    }
    mask_helpers._vec_cache = {}


# ---------------------------------------------------------------------------
# Cached vs vectorized equivalence
# ---------------------------------------------------------------------------
CONFIGS = [
    # (K, F, fan_out_list, fan_out_list_miss)
    (2, 3, [1, 3, 3], [1, 3, 3]),
    (2, 3, [1, 3, 3], [7, 0, 0]),
    (3, 2, [2, 2, 2, 2], [8, 0, 0, 0]),
    (1, 4, [1, 4], [1, 4]),
]


@pytest.mark.parametrize("K,F,fan_out_list,fan_out_list_miss", CONFIGS)
@pytest.mark.parametrize("B", [1, 3, 8, 9, 16])
@pytest.mark.parametrize("step", [0, 1])
def test_cached_equals_vectorized(K, F, fan_out_list, fan_out_list_miss, B, step):
    _reset_caches()
    device = torch.device("cpu")
    MQ_LEN = sum(fan_out_list)
    glue_added = K + 1
    tree_decode_added = (step + 1) * MQ_LEN
    ttl_added = glue_added + tree_decode_added
    # Context lens must satisfy prefix_len = context_len - ttl_added >= 0.
    torch.manual_seed(B * 10 + step)
    context_lens_cpu = torch.tensor(
        [ttl_added + 3 + i * 2 for i in range(B)], dtype=torch.int64, device=device,
    )
    cache_hits = torch.tensor([i % 2 for i in range(B)], dtype=torch.int64, device=device)

    cfg = _cfg(fan_out_list, fan_out_list_miss)

    mask_cached = get_custom_mask_cached(
        cfg, context_lens_cpu, step, K, F, B, device,
        fan_out_list=fan_out_list, fan_out_list_miss=fan_out_list_miss, cache_hits=cache_hits,
    )
    mask_vec = get_custom_mask_vectorized(
        cfg, context_lens_cpu, step, K, B, device, cache_hits,
    )
    assert mask_cached.shape == mask_vec.shape, f"shapes differ: {mask_cached.shape} vs {mask_vec.shape}"
    assert mask_cached.dtype == torch.bool
    assert mask_vec.dtype == torch.bool
    # Flat content must match bit-for-bit.
    assert torch.equal(mask_cached, mask_vec), (
        f"cached and vectorized masks differ for K={K},F={F},B={B},step={step},"
        f" fan_out_list={fan_out_list}, fan_out_list_miss={fan_out_list_miss}"
    )


# ---------------------------------------------------------------------------
# Structural contract: shape
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K,F,fan_out_list,fan_out_list_miss", CONFIGS)
@pytest.mark.parametrize("B", [1, 4, 12])
def test_mask_total_length_matches_expected(K, F, fan_out_list, fan_out_list_miss, B):
    _reset_caches()
    device = torch.device("cpu")
    MQ_LEN = sum(fan_out_list)
    step = 0
    ttl_added = (step + 1) * MQ_LEN + (K + 1)
    torch.manual_seed(42)
    context_lens = torch.tensor(
        [ttl_added + 5 + i for i in range(B)], dtype=torch.int64, device=device,
    )
    cache_hits = torch.zeros(B, dtype=torch.int64, device=device)
    cfg = _cfg(fan_out_list, fan_out_list_miss)

    mask = get_custom_mask_cached(
        cfg, context_lens, step, K, F, B, device,
        fan_out_list=fan_out_list, fan_out_list_miss=fan_out_list_miss, cache_hits=cache_hits,
    )
    # Expected length: sum_b MQ_LEN * context_len[b]
    expected_len = int((MQ_LEN * context_lens).sum().item())
    assert mask.numel() == expected_len, (
        f"mask length {mask.numel()} != expected {expected_len}"
    )


# ---------------------------------------------------------------------------
# Structural contract: cache-hit rows use fan_out_list, cache-miss rows use fan_out_list_miss
# ---------------------------------------------------------------------------
def test_hit_vs_miss_row_uses_correct_glue():
    """When fan_out_list != fan_out_list_miss, the glue block must differ by row."""
    _reset_caches()
    device = torch.device("cpu")
    K = 2
    F = 3
    fan_out_list = [1, 3, 3]            # hit-path fan-out
    fan_out_list_miss = [7, 0, 0]       # miss-path fan-out
    MQ_LEN = sum(fan_out_list)
    assert MQ_LEN == sum(fan_out_list_miss)
    step = 0
    ttl_added = (step + 1) * MQ_LEN + (K + 1)
    B = 2  # one hit, one miss
    context_lens = torch.tensor([ttl_added, ttl_added], dtype=torch.int64, device=device)
    cache_hits = torch.tensor([1, 0], dtype=torch.int64, device=device)
    cfg = _cfg(fan_out_list, fan_out_list_miss)

    mask = get_custom_mask_cached(
        cfg, context_lens, step, K, F, B, device,
        fan_out_list=fan_out_list, fan_out_list_miss=fan_out_list_miss, cache_hits=cache_hits,
    )
    # prefix_len = 0 here, so the only content is [glue | diag].
    # glue block for a row has shape (MQ_LEN, K+1).
    per_row_cols = K + 1 + (step + 1) * MQ_LEN
    mask2d_hit = mask[:MQ_LEN * per_row_cols].view(MQ_LEN, per_row_cols)
    mask2d_miss = mask[MQ_LEN * per_row_cols:].view(MQ_LEN, per_row_cols)

    glue_hit = mask2d_hit[:, :K + 1]
    glue_miss = mask2d_miss[:, :K + 1]
    # The two glue blocks must NOT be equal because fan_out_list differs from miss.
    assert not torch.equal(glue_hit, glue_miss), (
        "glue blocks for hit and miss rows unexpectedly equal"
    )


# ---------------------------------------------------------------------------
# Reference check: with uniform fan_out_list and step=0, the mask layout must
# match a hand-built reference via get_mask_iter_i.
# ---------------------------------------------------------------------------
def test_mask_matches_reference_iter_i():
    """For uniform fan_out_list=[F]*(K+1), step=0, the per-row mask equals the
    output of get_mask_iter_i(i=0, prefix_len, K, F) followed by flatten."""
    _reset_caches()
    device = torch.device("cpu")
    K, F = 2, 3
    fan_out_list = [F] * (K + 1)  # uniform
    cfg = _cfg(fan_out_list, fan_out_list)
    MQ_LEN = F * (K + 1)
    step = 0
    ttl_added = (step + 1) * MQ_LEN + (K + 1)
    B = 2
    context_lens = torch.tensor([ttl_added + 5, ttl_added + 5], dtype=torch.int64, device=device)
    cache_hits = torch.ones(B, dtype=torch.int64, device=device)

    mask_flat = get_custom_mask_cached(
        cfg, context_lens, step, K, F, B, device,
        fan_out_list=fan_out_list, fan_out_list_miss=fan_out_list, cache_hits=cache_hits,
    )

    # Reference: get_mask_iter_i returns [MQ_LEN, prefix_len + K+1 + (i+1)*MQ_LEN]
    # (uniform F), matches our per-row layout exactly.
    cols_per_row = int(context_lens[0].item())
    prefix_len = cols_per_row - ttl_added
    ref_row = get_mask_iter_i(i=0, prefix_len=prefix_len, K=K, F=F).to(torch.bool)
    assert ref_row.shape == (MQ_LEN, cols_per_row)

    got = mask_flat.view(B, MQ_LEN, cols_per_row)
    for b in range(B):
        assert torch.equal(got[b], ref_row), f"row {b} does not match reference"


# ---------------------------------------------------------------------------
# Structural contract: prefix is all-ones, diagonal section is identity-stacked
# ---------------------------------------------------------------------------
def test_prefix_is_all_ones_and_diag_is_identity():
    _reset_caches()
    device = torch.device("cpu")
    K, F = 1, 4
    fan_out_list = [1, 4]
    cfg = _cfg(fan_out_list, fan_out_list)
    MQ_LEN = sum(fan_out_list)  # 5
    step = 2
    prefix_len = 6
    ttl_added = (step + 1) * MQ_LEN + (K + 1)
    context_len = prefix_len + ttl_added
    B = 1
    context_lens = torch.tensor([context_len], dtype=torch.int64, device=device)
    cache_hits = torch.ones(B, dtype=torch.int64, device=device)

    flat = get_custom_mask_cached(
        cfg, context_lens, step, K, F, B, device,
        fan_out_list=fan_out_list, fan_out_list_miss=fan_out_list, cache_hits=cache_hits,
    )
    m = flat.view(MQ_LEN, context_len)
    # Prefix region is all True
    assert torch.all(m[:, :prefix_len])
    # Each diagonal sub-block is an identity
    diag_start = prefix_len + (K + 1)
    eye = torch.eye(MQ_LEN, dtype=torch.bool)
    for s in range(step + 1):
        sub = m[:, diag_start + s * MQ_LEN: diag_start + (s + 1) * MQ_LEN]
        assert torch.equal(sub, eye), f"diagonal sub-block at step {s} not identity"
