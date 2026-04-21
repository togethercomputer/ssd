"""Tier 1 / I5: shared-prefix prefix caching.

When two prompts share a prefix, the block manager must reuse blocks for the
shared region. Operationally: running two identical prompts in one batch must
produce the same output for both, and prefill should account for the shared
blocks (e.g. fewer newly allocated blocks than for a non-sharing batch).

We check the output-equivalence condition as the primary signal, since
prefix-caching bugs typically manifest as one sequence getting the other's
cached logits and diverging in output.
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
def test_duplicate_prompt_yields_identical_outputs():
    target = require_8b_target()
    # A long-ish prompt to ensure at least one full block is shared.
    p = "The following is a detailed explanation of the theory of relativity, which was proposed by Albert Einstein in the early twentieth century. It states that"
    cfg = {
        **base_config([p, p]),
        "model": target,
        "max_new_tokens": 12,
        "max_num_seqs": 2,
        "num_gpus": 1,
        "enforce_eager": True,
    }
    out = run_llm_subprocess(cfg)
    assert out["token_ids"][0] == out["token_ids"][1], (
        f"duplicate prompts produced different outputs (prefix-cache bug?):\n"
        f"  [0] = {out['token_ids'][0]}\n"
        f"  [1] = {out['token_ids'][1]}"
    )
