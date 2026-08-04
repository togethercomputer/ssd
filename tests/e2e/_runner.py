"""Subprocess runner used by Tier 1 tests.

Runs a single LLM configuration and prints a JSON line `RUNNER_RESULT: {...}`
containing output token ids and metrics. This lives behind a subprocess boundary
because `LLMEngine.exit()` calls os._exit(0) on teardown, which would kill pytest.

Invoked as:
    python tests/e2e/_runner.py --config-json '{"model": ..., "speculate": true, ...}'

The config JSON supports a superset of LLMEngine kwargs plus:
- prompts:      list[str]  (required)
- max_new_tokens: int       (default 32)
- temperature:  float       (default 0.0)
- seed:         int | None  (default None — no explicit seed)
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _load_config() -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--config-json", required=True)
    args = p.parse_args()
    return json.loads(args.config_json)


def main():
    cfg = _load_config()
    prompts: list[str] = cfg.pop("prompts")
    max_new_tokens: int = cfg.pop("max_new_tokens", 32)
    temperature: float = cfg.pop("temperature", 0.0)
    ignore_eos: bool = cfg.pop("ignore_eos", True)
    seed = cfg.pop("seed", None)

    if seed is not None:
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        import random
        random.seed(seed)
        import torch
        torch.manual_seed(seed)

    # Import AFTER seed setup so any CUDA init happens with a stable seed.
    from ssd import LLM, SamplingParams  # noqa: E402

    llm = LLM(**cfg)
    sp = [SamplingParams(temperature=temperature, max_new_tokens=max_new_tokens, ignore_eos=ignore_eos)] * len(prompts)
    outputs, metrics = llm.generate(prompts, sp, use_tqdm=False)

    # Keep only token ids from outputs — text decoding is the tokenizer's job, tested separately.
    result = {
        "token_ids": [o["token_ids"] for o in outputs],
        "n_seqs": len(outputs),
        # A few scalar metrics (aggregate) that are safe to compare across runs.
        "prefill_total_tokens": metrics.get("prefill_total_tokens", 0),
        "decode_total_tokens": metrics.get("decode_total_tokens", 0),
        "num_cache_hits": int(sum(metrics.get("cache_hits", []))),
        "num_verify_steps": len(metrics.get("accepted_suffix_lens_with_recovery", [])),
    }
    # Opt-in: include the full per-step accept trace (enabled by SSD_TRACE_ACCEPTS=1
    # — the engine populates this key only when the env var is set).
    if "per_step_accepts" in metrics:
        result["per_step_accepts"] = metrics["per_step_accepts"]
    print("RUNNER_RESULT: " + json.dumps(result), flush=True)

    # Tear the engine down EXPLICITLY, then die hard. Relying on interpreter
    # shutdown is fragile: multiprocessing's atexit joins the non-daemon
    # draft/worker children, whose exit in turn depends on NCCL teardown
    # ordering — when that ordering loses, this process never exits and the
    # calling test times out even though generation succeeded.
    try:
        llm.exit(hard=False)
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
