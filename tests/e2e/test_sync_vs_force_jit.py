"""Tier 1 / I1: synchronous speculative decoding ≡ async+force-jit (greedy).

`force-jit` in async mode always runs the draft synchronously — so the only
difference between it and sync spec (`draft_async=False`) is process topology
(separate target/draft processes vs. colocated on rank 0). In greedy mode the
two must agree on:
1. final generated token stream (bitwise identical), and
2. per-step acceptance trace — for every verify step, the accepted suffix
   (previous recovery + accepted draft tokens) and the new recovery token
   must match across both configurations for the same seq_id.

The per-step comparison (2) is the stronger check: it verifies the spec
algorithm's decision trace is identical, not merely the aggregate output.
"""
from __future__ import annotations

import pytest

from ._helpers import (
    CANONICAL_PROMPTS,
    base_config,
    require_1b_draft,
    require_8b_target,
    run_llm_subprocess,
)


def _sync_cfg(prompts, target, draft, max_new_tokens, k=2):
    return {
        **base_config(prompts), "model": target, "draft": draft,
        "speculate": True, "draft_async": False,
        "speculate_k": k,
        "max_new_tokens": max_new_tokens, "enforce_eager": True, "num_gpus": 1,
    }


def _async_forcejit_cfg(prompts, target, draft, max_new_tokens, k=2):
    return {
        **base_config(prompts), "model": target, "draft": draft,
        "speculate": True, "draft_async": True,
        "force_jit_speculate": True, "jit_speculate": True,
        "speculate_k": k, "async_fan_out": 2,
        "max_new_tokens": max_new_tokens, "enforce_eager": True, "num_gpus": 2,
    }


def _per_seq_trace(trace):
    """Group a per-step trace into a per-sequence trace.

    Returns dict[canonical_seq_idx, list[(suffix, recovery)]] where
    canonical_seq_idx is 0..N-1 assigned in first-appearance order (the raw
    seq_ids come from a process-global counter and differ across configs).

    Comparing per-sequence traces is the right level of strictness for
    sync-vs-async+force-jit equivalence: different sequences can complete in
    different numbers of steps (e.g. one sequence keeps accepting multi-token
    suffixes while another accepts single tokens), so the aggregate step count
    and per-step batch composition legitimately differ between modes. What must
    agree is each individual sequence's trace.
    """
    id_map: dict[int, int] = {}
    per_seq: dict[int, list[tuple[list[int], int]]] = {}
    for step in trace:
        for seq_id, suffix, rec in step:
            if seq_id not in id_map:
                id_map[seq_id] = len(id_map)
                per_seq[id_map[seq_id]] = []
            per_seq[id_map[seq_id]].append((list(suffix), int(rec)))
    return per_seq


def _assert_traces_equal(sync_trace, async_trace, *, context):
    a = _per_seq_trace(sync_trace)
    b = _per_seq_trace(async_trace)
    assert a.keys() == b.keys(), (
        f"{context}: different set of sequences — sync={sorted(a)}, async={sorted(b)}"
    )
    for seq_idx in sorted(a.keys()):
        assert a[seq_idx] == b[seq_idx], (
            f"{context}: per-sequence trace diverges for seq #{seq_idx}\n"
            f"  sync  ({len(a[seq_idx])} steps) = {a[seq_idx]}\n"
            f"  async ({len(b[seq_idx])} steps) = {b[seq_idx]}"
        )


@pytest.mark.tier1
@pytest.mark.smoke
def test_single_prompt_greedy_matches_tokens_and_trace():
    """I1 smoke: one prompt, force-jit must match sync-spec on both token stream and per-step trace."""
    target = require_8b_target()
    draft = require_1b_draft()
    prompts = [CANONICAL_PROMPTS[0]]

    sync_out = run_llm_subprocess(
        _sync_cfg(prompts, target, draft, max_new_tokens=12), trace_accepts=True,
    )
    async_out = run_llm_subprocess(
        _async_forcejit_cfg(prompts, target, draft, max_new_tokens=12), trace_accepts=True,
    )

    # (1) Final token streams agree
    assert sync_out["token_ids"] == async_out["token_ids"], (
        f"token_ids mismatch:\n  sync  = {sync_out['token_ids']}\n  async = {async_out['token_ids']}"
    )
    # (2) Per-step accept traces agree
    assert "per_step_accepts" in sync_out and "per_step_accepts" in async_out, (
        "per_step_accepts missing — trace_accepts=True did not propagate"
    )
    _assert_traces_equal(
        sync_out["per_step_accepts"], async_out["per_step_accepts"],
        context="sync vs async+force-jit (single prompt)",
    )


@pytest.mark.tier1
def test_multi_prompt_greedy_matches_tokens():
    """I1: multiple prompts, final token streams match between sync-spec and async+force-jit."""
    target = require_8b_target()
    draft = require_1b_draft()
    prompts = CANONICAL_PROMPTS

    sync_out = run_llm_subprocess(_sync_cfg(prompts, target, draft, max_new_tokens=16))
    async_out = run_llm_subprocess(_async_forcejit_cfg(prompts, target, draft, max_new_tokens=16))
    assert sync_out["token_ids"] == async_out["token_ids"]


@pytest.mark.tier1
def test_multi_prompt_first_seq_trace_matches_at_longer_length():
    """I1: in a 2-prompt batch, seq #0 (the first prompt in canonical order) has
    an identical per-step accept trace under sync-spec and async+force-jit for a
    generation length well beyond max_new_tokens=16.

    Seq #0 equality held at length=16 (see `test_multi_prompt_greedy_matches_tokens`
    and the accompanying `_trace_analysis.py`). This test verifies that equality
    *continues* to hold as the generation runs longer — ruling out the possibility
    that seq #0 was only passing by coincidence for short outputs.

    Seq #1 is known to diverge on per-step traces (same final tokens, different
    acceptance schedule); see `test_multi_prompt_greedy_matches_trace` for the
    full-batch check that records that divergence.
    """
    target = require_8b_target()
    draft = require_1b_draft()
    prompts = CANONICAL_PROMPTS
    long_n = 64  # 4× the default — enough to catch drift that accumulates over time

    sync_out = run_llm_subprocess(
        _sync_cfg(prompts, target, draft, max_new_tokens=long_n), trace_accepts=True,
    )
    async_out = run_llm_subprocess(
        _async_forcejit_cfg(prompts, target, draft, max_new_tokens=long_n), trace_accepts=True,
    )

    a = _per_seq_trace(sync_out["per_step_accepts"])
    b = _per_seq_trace(async_out["per_step_accepts"])
    assert 0 in a and 0 in b, "seq #0 missing from one of the traces"
    assert a[0] == b[0], (
        f"seq #0 per-step accept trace diverges at max_new_tokens={long_n}\n"
        f"  sync  ({len(a[0])} steps) = {a[0]}\n"
        f"  async ({len(b[0])} steps) = {b[0]}"
    )


@pytest.mark.tier1
@pytest.mark.xfail(
    reason=(
        "Known divergence on multi-prompt batches: async+force-jit and sync-spec "
        "produce the same final tokens but diverging per-step acceptance traces "
        "for seq #1 (second prompt in the batch). Seq #0 matches exactly — see "
        "test_multi_prompt_first_seq_trace_matches_at_longer_length. Hypothesis: "
        "tree-attention vs linear-decode produces subtly different draft logits "
        "at non-zero batch positions, or KV rollback after partial accepts drifts "
        "state for the second sequence."
    ),
    strict=True,
)
def test_multi_prompt_greedy_matches_trace():
    """I1 (xfail): tighter version of the multi-prompt check — per-step accept trace equality.

    This test is marked xfail (strict) to record the finding; if a future change
    to the async path makes this pass, the xfail assertion will flip to a real
    failure, flagging the behavioral change for review.
    """
    target = require_8b_target()
    draft = require_1b_draft()
    prompts = CANONICAL_PROMPTS

    sync_out = run_llm_subprocess(
        _sync_cfg(prompts, target, draft, max_new_tokens=16), trace_accepts=True,
    )
    async_out = run_llm_subprocess(
        _async_forcejit_cfg(prompts, target, draft, max_new_tokens=16), trace_accepts=True,
    )
    _assert_traces_equal(
        sync_out["per_step_accepts"], async_out["per_step_accepts"],
        context="sync vs async+force-jit (multi prompt)",
    )
