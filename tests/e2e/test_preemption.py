"""Tier 1 / I6: preemption round-trip preserves greedy output.

When KV-cache blocks are scarce, the scheduler preempts running sequences
(deallocates their blocks, moves them back to waiting, then re-prefills). The
final generated tokens must equal those of an un-preempted run.

We force preemption by configuring `num_kvcache_blocks` to a tight value with
`max_num_seqs > 1`, so the second sequence cannot fit without preempting the
first. Compare to a run with plenty of blocks (no preemption).
"""
from __future__ import annotations

import pytest

from ._helpers import (
    CANONICAL_PROMPTS,
    base_config,
    require_8b_target,
    run_llm_subprocess,
)


@pytest.mark.tier1
def test_preemption_matches_unpreempted_output():
    target = require_8b_target()
    prompts = CANONICAL_PROMPTS

    # Both runs use the same prompts and sampling; only num_kvcache_blocks differs.
    common = {
        **base_config(prompts),
        "model": target,
        "max_new_tokens": 16,
        "max_num_seqs": 2,
        "num_gpus": 1,
        "enforce_eager": True,
        "kvcache_block_size": 256,
    }
    unpreempted = run_llm_subprocess({**common, "num_kvcache_blocks": 512})
    # With block_size=256 and max_model_len=2048, each seq can need up to 8 blocks.
    # Setting num_kvcache_blocks=10 with two sequences and prompts of ~16 tokens forces
    # preemption when a second sequence's blocks can't be appended.
    preempted = run_llm_subprocess({**common, "num_kvcache_blocks": 10})

    assert unpreempted["token_ids"] == preempted["token_ids"], (
        f"preempted run diverged from unpreempted (same greedy prompts):\n"
        f"  unpreempted = {unpreempted['token_ids']}\n"
        f"  preempted   = {preempted['token_ids']}"
    )
