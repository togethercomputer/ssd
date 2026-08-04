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


# Canonical local model snapshots (8B target + 1B standalone draft) — shared
# with tests/hf/helpers.py so both suites track the same machine layout.
from tests.hf.helpers import (  # noqa: E402
    LLAMA_3_1_8B_SNAPSHOT,
    LLAMA_3_2_1B_SNAPSHOT,
    EAGLE3_8B_SNAPSHOT,
)


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

    # stdout/stderr go to temp FILES, not pipes: with pipes, subprocess.run
    # returns only on pipe EOF, which requires every GRANDCHILD (the engine's
    # draft/worker processes) to close them too — a lingering child turns a
    # successful run into a spurious 600s timeout. With files, wait() returns
    # the moment the runner itself exits. start_new_session gives us a process
    # group so a timeout can reap the whole engine family instead of leaving
    # orphans squatting on GPUs.
    import signal
    import tempfile

    with tempfile.TemporaryFile(mode="w+") as fout, tempfile.TemporaryFile(mode="w+") as ferr:
        proc = subprocess.Popen(
            [sys.executable, str(runner), "--config-json", json.dumps(config)],
            stdout=fout,
            stderr=ferr,
            stdin=subprocess.DEVNULL,
            text=True,
            env=env,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait(timeout=30)
            raise
        finally:
            # Reap any engine children the runner left behind (its exit path
            # os._exit()s and cannot guarantee every grandchild died).
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        fout.seek(0)
        ferr.seek(0)
        stdout = fout.read()
        stderr = ferr.read()

    if returncode != 0:
        raise RuntimeError(
            f"runner exited with code {returncode}\n"
            f"--- stdout ---\n{stdout}\n"
            f"--- stderr ---\n{stderr}\n"
        )
    # Find the RUNNER_RESULT line
    for line in stdout.splitlines():
        if line.startswith("RUNNER_RESULT: "):
            return json.loads(line[len("RUNNER_RESULT: "):])
    raise RuntimeError(
        f"runner did not emit RUNNER_RESULT\n"
        f"--- stdout ---\n{stdout}\n"
        f"--- stderr ---\n{stderr}\n"
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
