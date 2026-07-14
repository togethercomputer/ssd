"""Shared pytest config for the SSD testbed.

Markers:
- tier0: no GPU / no model weights. Always runnable.
- tier1: single GPU, real 8B weights. Requires CUDA and the 8B model snapshot.
- smoke: a tiny subset of tier1 suitable for per-commit CI.
- tier2..5: reserved for future tiers (HF ref, cross-repo, 70B, perf).

Run examples (see tests/README.md for more):
    pytest tests/unit -m tier0
    pytest tests/e2e -m tier1
    pytest tests -m "tier0 or smoke"
"""
from __future__ import annotations

import pytest


def pytest_configure(config):
    for marker in ("tier0", "tier1", "tier2", "tier3", "tier4", "tier5", "smoke"):
        config.addinivalue_line("markers", f"{marker}: see tests/ssd_test_plan_cc.md")


def _cuda_count() -> int:
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU-dependent tiers when insufficient GPUs are available."""
    n = _cuda_count()
    skip_no_gpu = pytest.mark.skip(reason="requires >=1 CUDA device")
    skip_lt4_gpu = pytest.mark.skip(reason="requires >=4 CUDA devices")
    for item in items:
        if "tier1" in item.keywords or "tier2" in item.keywords or "tier3" in item.keywords:
            if n < 1:
                item.add_marker(skip_no_gpu)
        if "tier4" in item.keywords and n < 4:
            item.add_marker(skip_lt4_gpu)
