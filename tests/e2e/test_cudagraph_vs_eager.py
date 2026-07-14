"""Tier 1 / I3: CUDA-graph decode ≡ eager decode (greedy).

Target-only decode with enforce_eager=True must produce the same tokens as
with CUDA graphs enabled. Tests catch bugs introduced during graph capture
(e.g. missed variable updates, padding errors).
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
def test_cudagraph_vs_eager_target_only():
    target = require_8b_target()
    prompts = CANONICAL_PROMPTS

    common = {**base_config(prompts), "model": target, "max_new_tokens": 16, "num_gpus": 1}

    eager_cfg = {**common, "enforce_eager": True}
    graph_cfg = {**common, "enforce_eager": False}

    eager = run_llm_subprocess(eager_cfg)
    graph = run_llm_subprocess(graph_cfg)

    assert eager["token_ids"] == graph["token_ids"], (
        f"cudagraph vs eager mismatch (target-only greedy):\n"
        f"  eager = {eager['token_ids']}\n"
        f"  graph = {graph['token_ids']}\n"
    )
