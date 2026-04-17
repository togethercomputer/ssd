"""Tier 1: fused K-step tree-decode graph ≡ per-step tree-decode graph (greedy).

The fused path captures all K draft forwards into a single CUDA graph instead
of running the Python for-loop per call (see `docs/decode_tree_fused_graph.md`).
In greedy mode it must be observationally identical to the per-step path:

1. Final token stream must match byte-for-byte.
2. Per-step acceptance trace (suffix + recovery per verify step) must match
   for every sequence.

The fused path only supports argmax (greedy). Tests assert temp=0 throughout.
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


def _cfg(prompts, target, draft, max_new_tokens, k=2, fused=False):
    """async+force-jit is the simplest greedy async config to exercise the tree-decode path.
    `fused` toggles `config.fused_tree_decode_graph`."""
    return {
        **base_config(prompts), "model": target, "draft": draft,
        "speculate": True, "draft_async": True,
        "force_jit_speculate": True, "jit_speculate": True,
        "speculate_k": k, "async_fan_out": 2,
        "max_new_tokens": max_new_tokens, "enforce_eager": False, "num_gpus": 2,
        "fused_tree_decode_graph": fused,
    }


def _per_seq_trace(trace):
    """Group per-step (seq_id, suffix, recovery) records by sequence (first-appearance order)."""
    id_map: dict[int, int] = {}
    per_seq: dict[int, list[tuple[list[int], int]]] = {}
    for step in trace:
        for seq_id, suffix, rec in step:
            if seq_id not in id_map:
                id_map[seq_id] = len(id_map)
                per_seq[id_map[seq_id]] = []
            per_seq[id_map[seq_id]].append((list(suffix), int(rec)))
    return per_seq


@pytest.mark.tier1
@pytest.mark.smoke
def test_fused_matches_per_step_single_prompt():
    """Smoke: one prompt, fused graph must produce identical tokens + trace vs per-step."""
    target = require_8b_target()
    draft = require_1b_draft()
    prompts = [CANONICAL_PROMPTS[0]]

    out_base = run_llm_subprocess(
        _cfg(prompts, target, draft, max_new_tokens=16, k=2, fused=False),
        trace_accepts=True,
    )
    out_fused = run_llm_subprocess(
        _cfg(prompts, target, draft, max_new_tokens=16, k=2, fused=True),
        trace_accepts=True,
    )

    assert out_base["token_ids"] == out_fused["token_ids"], (
        f"token_ids mismatch:\n  base  = {out_base['token_ids']}\n  fused = {out_fused['token_ids']}"
    )

    assert "per_step_accepts" in out_base and "per_step_accepts" in out_fused, (
        "per_step_accepts missing — trace_accepts=True did not propagate"
    )
    a = _per_seq_trace(out_base["per_step_accepts"])
    b = _per_seq_trace(out_fused["per_step_accepts"])
    assert a.keys() == b.keys(), f"seq-id set differs: base={sorted(a)} fused={sorted(b)}"
    for sid in sorted(a):
        assert a[sid] == b[sid], (
            f"per-step accept trace diverges for seq #{sid}\n"
            f"  base  ({len(a[sid])} steps) = {a[sid]}\n"
            f"  fused ({len(b[sid])} steps) = {b[sid]}"
        )


@pytest.mark.tier1
def test_fused_matches_per_step_multi_prompt():
    """Multi-prompt greedy: final token streams must match between fused and per-step."""
    target = require_8b_target()
    draft = require_1b_draft()
    prompts = CANONICAL_PROMPTS

    out_base = run_llm_subprocess(_cfg(prompts, target, draft, max_new_tokens=16, k=2, fused=False))
    out_fused = run_llm_subprocess(_cfg(prompts, target, draft, max_new_tokens=16, k=2, fused=True))
    assert out_base["token_ids"] == out_fused["token_ids"]


@pytest.mark.tier1
def test_fused_matches_per_step_larger_k():
    """k=4 (MQ_LEN=10, 4 fused forwards per call) — wider tree than the smoke test."""
    target = require_8b_target()
    draft = require_1b_draft()
    prompts = [CANONICAL_PROMPTS[0]]

    out_base = run_llm_subprocess(_cfg(prompts, target, draft, max_new_tokens=24, k=4, fused=False))
    out_fused = run_llm_subprocess(_cfg(prompts, target, draft, max_new_tokens=24, k=4, fused=True))
    assert out_base["token_ids"] == out_fused["token_ids"]
