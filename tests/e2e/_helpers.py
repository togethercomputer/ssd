"""Helpers used by Tier 1 E2E tests.

Runs the `_runner.py` subprocess with a given config and returns the parsed
JSON result. Each test invokes this multiple times with different configs and
asserts that the (greedy) token outputs match.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


# Canonical local model snapshots (8B target + 1B standalone draft).
LLAMA_3_1_8B_SNAPSHOT = "/scratch/avner/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
LLAMA_3_2_1B_SNAPSHOT = "/scratch/avner/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
EAGLE3_8B_SNAPSHOT = "/scratch/avner/huggingface/hub/models--yuhuili--EAGLE3-LLaMA3.1-Instruct-8B/snapshots/61aa096484ad9752292507b0cc9973bb423abb35"


def require_8b_target() -> str:
    if not Path(LLAMA_3_1_8B_SNAPSHOT).is_dir():
        import pytest
        pytest.skip(f"Llama-3.1-8B snapshot not found at {LLAMA_3_1_8B_SNAPSHOT}")
    return LLAMA_3_1_8B_SNAPSHOT


def require_1b_draft() -> str:
    if not Path(LLAMA_3_2_1B_SNAPSHOT).is_dir():
        import pytest
        pytest.skip(f"Llama-3.2-1B snapshot not found at {LLAMA_3_2_1B_SNAPSHOT}")
    return LLAMA_3_2_1B_SNAPSHOT


def run_llm_subprocess(config: dict, timeout: int = 600, trace_accepts: bool = False) -> dict:
    """Run the LLM runner in a fresh subprocess with the given config dict.

    Returns the parsed runner result (see `_runner.py`).

    When `trace_accepts=True`, sets SSD_TRACE_ACCEPTS=1 so the engine records
    the per-step accept trace (list of (seq_id, suffix, recovery) per verify
    step), which the runner includes in the result under "per_step_accepts".
    """
    runner = Path(__file__).parent / "_runner.py"
    env = dict(os.environ)
    # Ensure no lingering stale NCCL/shm state leaks into this child process.
    env.setdefault("SSD_BRIEF_LOG", "0")
    env.setdefault("SSD_NCCL_LOG", "0")
    if trace_accepts:
        env["SSD_TRACE_ACCEPTS"] = "1"

    proc = subprocess.run(
        [sys.executable, str(runner), "--config-json", json.dumps(config)],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"runner exited with code {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
        )
    # Find the RUNNER_RESULT line
    for line in proc.stdout.splitlines():
        if line.startswith("RUNNER_RESULT: "):
            return json.loads(line[len("RUNNER_RESULT: "):])
    raise RuntimeError(
        f"runner did not emit RUNNER_RESULT\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}\n"
    )


def base_config(prompts: list[str], *, max_new_tokens: int = 32, target: str | None = None) -> dict:
    """A default base config that tests customize by adding/overriding fields."""
    return {
        "model": target or require_8b_target(),
        "prompts": prompts,
        "temperature": 0.0,
        "max_new_tokens": max_new_tokens,
        "ignore_eos": True,
        "max_model_len": 2048,
        "max_num_seqs": 4,
        "enforce_eager": False,
        "num_gpus": 1,
    }


CANONICAL_PROMPTS = [
    "The capital city of France is",
    "The largest ocean on Earth is",
]
