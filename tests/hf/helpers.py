"""Helpers used by Tier 1 E2E tests.

Runs the `_runner.py` subprocess with a given config and returns the parsed
JSON result. Each test invokes this multiple times with different configs and
asserts that the (greedy) token outputs match.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import psutil
import requests
import subprocess
import sys
import signal
import time


TGL_BASE_DIR = "/work/avner/git/tgl"

# Canonical local model snapshots (8B target + 1B standalone draft).
LLAMA_3_1_8B_SNAPSHOT = "/data/shared/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
LLAMA_3_2_1B_SNAPSHOT = "/data/shared/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
EAGLE3_8B_SNAPSHOT = "/data/shared/huggingface/hub/models--yuhuili--EAGLE3-LLaMA3.1-Instruct-8B/snapshots/61aa096484ad9752292507b0cc9973bb423abb35"

QWEN3_8B_SNAPSHOT = "/data/shared/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
QWEN3_0_6B_SNAPSHOT = "/data/shared/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

# EAGLE3 draft models (for use with `use_eagle=True`).
EAGLE3_LLAMA_8B_SNAPSHOT = "/data/shared/huggingface/hub/models--lmsys--SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge/snapshots/4a8e38f7dbee5d6dc82369f59a58540855fe09af"
EAGLE3_QWEN3_8B_SNAPSHOT = "/data/shared/huggingface/hub/models--AngelSlim--Qwen3-8B_eagle3/snapshots/9629dfce7a4a10564dd48d3e5485c3976095653c"

PHOENIX_LLAMA_8B_SNAPSHOT = "/data/avner/checkpoints/phoenix-4layer-llama-3p1-8b-lookahead16"

def require_8b_target() -> str:
    assert Path(LLAMA_3_1_8B_SNAPSHOT).is_dir(), f"Llama-3.1-8B snapshot not found at {LLAMA_3_1_8B_SNAPSHOT}"
    return LLAMA_3_1_8B_SNAPSHOT


def require_1b_draft() -> str:
    assert Path(LLAMA_3_2_1B_SNAPSHOT).is_dir(), f"Llama-3.2-1B snapshot not found at {LLAMA_3_2_1B_SNAPSHOT}"
    return LLAMA_3_2_1B_SNAPSHOT


def require_qwen3_8b_target() -> str:
    assert Path(QWEN3_8B_SNAPSHOT).is_dir(), f"Qwen3-8B snapshot not found at {QWEN3_8B_SNAPSHOT}"
    return QWEN3_8B_SNAPSHOT


def require_qwen3_0p6b_draft() -> str:
    assert Path(QWEN3_0_6B_SNAPSHOT).is_dir(), f"Qwen3-0.6B snapshot not found at {QWEN3_0_6B_SNAPSHOT}"
    return QWEN3_0_6B_SNAPSHOT


def require_eagle_llama_8b_draft() -> str:
    assert Path(EAGLE3_LLAMA_8B_SNAPSHOT).is_dir(), f"EAGLE3-LLaMA3.1 snapshot not found at {EAGLE3_LLAMA_8B_SNAPSHOT}"
    return EAGLE3_LLAMA_8B_SNAPSHOT


def require_phoenix_llama_8b_draft() -> str:
    assert Path(PHOENIX_LLAMA_8B_SNAPSHOT).is_dir(), f"Phoenix-4layer-Llama3.1-8B snapshot not found at {PHOENIX_LLAMA_8B_SNAPSHOT}"
    return PHOENIX_LLAMA_8B_SNAPSHOT


def require_eagle_qwen3_8b_draft() -> str:
    assert Path(EAGLE3_QWEN3_8B_SNAPSHOT).is_dir(), f"EAGLE3 Qwen3 snapshot not found at {EAGLE3_QWEN3_8B_SNAPSHOT}"
    return EAGLE3_QWEN3_8B_SNAPSHOT


def _get_speculative_algorithm(speculator_type: str) -> str:
    if speculator_type == "standalone":
        return "ASYNC_STANDALONE"
    elif speculator_type == "sync_standalone":
        return "STANDALONE"
    elif speculator_type == "eagle":
        return "ASYNC_EAGLE3"
    elif speculator_type == "sync_eagle":
        return "EAGLE3"
    elif speculator_type == "phoenix":
        return "ASYNC_PHOENIX"
    else:
        raise ValueError(f"unknown speculator type: {speculator_type}")


def launch_tgl_server(
    speculator_type: str,
    backup: str,
    target: str,
    draft: str,
    lookahead: int,
    fanout: int,
    port: int,
    cross_node: bool = False,
):
    env = os.environ.copy()
    env["NCCL_CUMEM_ENABLE"] = "0"  # match sglang; avoids P2P/IPC vs P2P/CUMEM mismatch on same-node
    cmd = [
        # sys.executable, "-m", "sglang.launch_server",
        "sglang", "serve",
        "--model-path", target,
        "--speculative-algorithm", _get_speculative_algorithm(speculator_type),
        "--speculative-draft-model-path", draft,
        "--tp", "1", "--mem-fraction-static", "0.7",
        "--max-running-requests", "1",
        "--log-level", "warning",
        "--port", str(port),
        "--context-length", "2048",
        "--dtype", "bfloat16",
        "--skip-server-warmup",
        ### THESE ARE FOR DYNAMIC LOOKAHEAD TEST
        # "--speculative-num-steps", str(8),
        # "--speculative-num-draft-tokens", str(8 + 1),
        # "--speculative-num-steps-list", "[3,3,4,5,6,7,8]",
        ### ABOVE ARE FOR DYNAMIC LOOKAHEAD TEST
        "--speculative-num-steps", str(lookahead),
        "--speculative-num-draft-tokens", str(lookahead + 1),
        "--speculative-eagle-topk", "1",
        "--page-size", "64",
        "--speculative-async-communicate-cache-hits",
        "--speculative-async-communicate-logits",
        # "--disable-cuda-graph",
    ]

    if speculator_type in ["standalone", "eagle", "phoenix"]:
        if backup == "force-jit":
            cmd.append("--speculative-async-jit-speculate")
            cmd.append("--speculative-async-force-jit-speculate")
        elif backup == "jit":
            cmd.append("--speculative-async-jit-speculate")

        if cross_node:
            cmd.append("--speculative-async-remote-draft")

    print(f"[tgl] Launching server: {' '.join(cmd)}", flush=True)
    server_process = subprocess.Popen(cmd, start_new_session=True, env=env)
    draft_process = None
    
    if cross_node:
        draft_cmd = [
            "python", f"{TGL_BASE_DIR}/scripts/launch_remote_draft.py",
            "--draft-model-path", draft,
            "--target-host", "localhost",
            "--gpu-id", "1",
            "--speculate-k", str(lookahead),
            "--max-model-len", "4096",
            "--fan-out", str(fanout),
        ]
        if backup == "jit" or backup == "force-jit":
            draft_cmd.append("--jit-speculate")
        if backup == "force-jit":
            draft_cmd.append("--force-jit-speculate")
        if speculator_type == "phoenix":
            draft_cmd.append("--use-phoenix")
            draft_cmd.append("--d-model-target", "4096")
            draft_cmd.append("--tokenizer-path", "/data/shared/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/")
        if speculator_type == "eagle":
            draft_cmd.append("--use-eagle")
            draft_cmd.append("--d-model-target", "4096")
            draft_cmd.append("--tokenizer-path", "/data/shared/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/")
        
        print(f"[tgl] Launching draft: {' '.join(draft_cmd)}", flush=True)
        draft_process = subprocess.Popen(draft_cmd, start_new_session=True, env=env)
    return server_process, draft_process


def wait_for_server(port: int, timeout: int = 300) -> bool:
    deadline = time.time() + timeout
    print(f"[tgl] waiting for server", flush=True)
    while time.time() < deadline:
        try:
            if requests.get(
                f"http://localhost:{port}/health", timeout=2,
            ).status_code == 200:
                print(f"[tgl] server health check passed", flush=True)
                return True
        except Exception:
            pass
        time.sleep(3)
    print(f"[tgl] server health check timed out", flush=True)
    return False



def kill_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        print(f"[tgl] killed server", flush=True)
    except (ProcessLookupError, PermissionError):
        print(f"[tgl] failed to kill server", flush=True)
        pass
    # Close pipes so wait() doesn't block on buffer drainage
    for fd in (proc.stdout, proc.stderr, proc.stdin):
        if fd:
            print(f"[tgl] closing pipe {fd}", flush=True)
            try:
                fd.close()
                print(f"[tgl] closed pipe {fd}", flush=True)
            except Exception:
                print(f"[tgl] failed to close pipe {fd}", flush=True)
                pass
