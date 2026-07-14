"""Tier 1 / I4: greedy output of a prompt is independent of batch position.

Running a prompt alone (batch=1) must produce the same greedy tokens as
running the same prompt at any position in a batch of prompts, since greedy
decoding has no cross-sequence dependencies.
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
def test_prompt_output_independent_of_batch_position():
    target = require_8b_target()
    p = CANONICAL_PROMPTS[0]
    other = CANONICAL_PROMPTS[1]

    solo_cfg = {**base_config([p]), "model": target, "max_new_tokens": 12, "num_gpus": 1, "enforce_eager": True, "max_num_seqs": 1}
    batched_cfg = {**base_config([p, other]), "model": target, "max_new_tokens": 12, "num_gpus": 1, "enforce_eager": True, "max_num_seqs": 2}

    solo = run_llm_subprocess(solo_cfg)
    batched = run_llm_subprocess(batched_cfg)

    # Output order matches input order (see llm_engine.generate).
    assert solo["token_ids"][0] == batched["token_ids"][0], (
        f"prompt output changed with batch position:\n"
        f"  solo[0]    = {solo['token_ids'][0]}\n"
        f"  batched[0] = {batched['token_ids'][0]}"
    )
