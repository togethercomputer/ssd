"""Tier 1 / I2: in greedy mode, force-jit ≡ jit ≡ fast.

In greedy sampling the target's argmax solely determines the output; what the
draft proposes only changes *speed* and *acceptance rate*. So all three async
backup strategies must produce the same final token stream for the same prompts
with temperature=0.

Note: `fast` mode returns all-zero speculations on cache misses, which means
the target will reject every speculated token on a miss and sample the recovery
directly. That still yields the same greedy tokens, just one at a time.
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


def _async_cfg(prompts, *, target, draft, backup: str):
    """Build an async-spec config with the given backup strategy."""
    cfg = {
        **base_config(prompts),
        "model": target, "draft": draft,
        "speculate": True, "draft_async": True,
        "speculate_k": 2, "async_fan_out": 2,
        "enforce_eager": True,
        "num_gpus": 2,
        "max_new_tokens": 12,
    }
    if backup == "force-jit":
        cfg["force_jit_speculate"] = True
        cfg["jit_speculate"] = True
    elif backup == "jit":
        cfg["force_jit_speculate"] = False
        cfg["jit_speculate"] = True
    elif backup == "fast":
        cfg["force_jit_speculate"] = False
        cfg["jit_speculate"] = False
    else:
        raise ValueError(backup)
    return cfg


@pytest.mark.tier1
def test_force_jit_jit_fast_match_greedy():
    target = require_8b_target()
    draft = require_1b_draft()
    prompts = [CANONICAL_PROMPTS[0]]

    results = {
        b: run_llm_subprocess(_async_cfg(prompts, target=target, draft=draft, backup=b))
        for b in ("force-jit", "jit", "fast")
    }

    fj = results["force-jit"]["token_ids"]
    jt = results["jit"]["token_ids"]
    ft = results["fast"]["token_ids"]

    assert fj == jt, f"force-jit ≠ jit\n  force-jit={fj}\n  jit={jt}"
    assert fj == ft, f"force-jit ≠ fast\n  force-jit={fj}\n  fast={ft}"
