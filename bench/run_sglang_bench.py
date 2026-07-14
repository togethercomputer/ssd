"""Launch an SGLang server and benchmark it.

Handles server lifecycle: launch, health-check, benchmark, cleanup.
The benchmark client (sglang_eval_client.py) sends requests and logs metrics.

Usage:
    python -O /work/avner/git/ssd/bench/run_sglang_bench.py --llama                       # SD, Llama 70B
    python -O /work/avner/git/ssd/bench/run_sglang_bench.py --qwen                        # SD, Qwen 32B
    python -O /work/avner/git/ssd/bench/run_sglang_bench.py --llama --mode AR             # autoregressive baseline
    python -O /work/avner/git/ssd/bench/run_sglang_bench.py --llama --wandb --name myrun  # log to wandb
    python -O /work/avner/git/ssd/bench/run_sglang_bench.py --llama --mode EAGLE3 --size 8 --dataset humaneval --numseqs 1 --profile --tp 1

Set model paths via env vars (BENCH_LLAMA_70B, etc.) or edit bench_paths.py.
"""
import os
import sys
import time
import signal
import argparse
import subprocess
import requests

sys.path.insert(0, os.path.dirname(__file__))
from bench_paths import MODELS, resolve_snapshot


def main():
    parser = argparse.ArgumentParser(description="Launch SGLang server and benchmark it")
    parser.add_argument("--llama", action="store_true", default=True)
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--size", type=int, default=0)
    parser.add_argument("--mode", choices=["AR", "STANDALONE", "ASYNC_STANDALONE", "EAGLE3", "ASYNC_EAGLE3"], default="STANDALONE",
                        help="ar = autoregressive, sd = speculative decoding (default)")
    parser.add_argument("--backup", choices=["fast", "jit", "force-jit"], default="jit",
                        help="Backup strategy (fast, jit, force-jit)")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--port", type=int, default=40010)
    parser.add_argument("--mem-frac", type=float, default=0.70)
    parser.add_argument("--num-steps", type=int, default=4, help="draft chain depth (k = num_steps + 1)")
    parser.add_argument("--context-length", type=int, default=4096)
    # Pass-through to eval client
    parser.add_argument("--numseqs", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=512)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--dataset", type=str, choices=["all", "humaneval", "alpaca", "c4", "ultrafeedback", "random", "example"], default="all")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--group", type=str, default="ssd")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--chat-template", action="store_true")

    parser.add_argument("--f", type=int, default=4, help="Async fan out value")
    parser.add_argument("--fl", type=int, nargs='+', default=None, help="Fan out list (e.g., --fl 1 3 4 becomes [1, 3, 4])")
    parser.add_argument("--flh", type=int, nargs='+', default=None, help="Fan out list (e.g., --flh 1 3 4 becomes [1, 3, 4])")
    parser.add_argument("--flm", type=int, nargs='+', default=None, help="Fan out list miss (e.g., --flm 1 3 4 becomes [1, 3, 4])")
    parser.add_argument("--communicate-cache-hits", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--acceptance-rate-log", type=str, default=None,
                        help="Path to log acceptance rates (sets ACCEPTANCE_RATE_LOG env var for the server)")
    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()
    if args.qwen:
        args.llama = False

    if args.size == 0:
        args.size = 70 if args.llama else 32

    server_cmd, target = get_server_cmd(args)
    print(f"Mode: {args.mode}, Target: {target}")
    print(f"Server cmd: {' '.join(server_cmd)}")

    # Kill stale sglang processes
    subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"],
                   capture_output=True)
    time.sleep(2)

    env = os.environ.copy()
    if args.acceptance_rate_log:
        env["ACCEPTANCE_RATE_LOG"] = args.acceptance_rate_log
        print(f"ACCEPTANCE_RATE_LOG={args.acceptance_rate_log}")
    if args.profile:
        env["SSD_PROFILE"] = "1"
        print("SSD_PROFILE=1")

    proc = subprocess.Popen(server_cmd, preexec_fn=os.setsid, env=env)
    try:
        print("Waiting for server...")
        if not wait_for_server(args.port):
            print("Server failed to start"); sys.exit(1)
        print("Server ready")

        # Build eval client command
        bench_dir = os.path.dirname(__file__)
        eval_cmd = [
            sys.executable, os.path.join(bench_dir, "sglang_eval_client.py"),
            "--size", str(args.size),
            "--numseqs", str(args.numseqs),
            "--output_len", str(args.output_len),
            "--temp", str(args.temp),
            f"--{args.dataset}",
            "--b", "1",
            "--port", str(args.port),
        ]
        if args.chat_template:
            eval_cmd.append("--chat-template")
        if args.llama:
            eval_cmd.append("--llama")
        else:
            eval_cmd.append("--qwen")
        if is_eagle3(args.mode):
            eval_cmd.append("--eagle")
        if args.wandb:
            eval_cmd += ["--wandb"]
            if args.group:
                eval_cmd += ["--group", args.group]
            if args.name:
                eval_cmd += ["--name", args.name]

        print(f"Eval cmd: {' '.join(eval_cmd)}")
        subprocess.run(eval_cmd, check=True, cwd=bench_dir)
    finally:
        kill_server(proc)
        print("Server stopped")


def is_spec(mode):
    return mode in ["STANDALONE", "ASYNC_STANDALONE", "EAGLE3", "ASYNC_EAGLE3"]


def is_async(mode):
    return mode in ["ASYNC_STANDALONE", "ASYNC_EAGLE3"]


def is_standalone(mode):
    return mode in ["STANDALONE", "ASYNC_STANDALONE"]

def is_eagle3(mode):
    return mode in ["EAGLE3", "ASYNC_EAGLE3"]


def get_server_cmd(args):
    if args.llama:
        draft_name = "llama_1b"
        if args.size == 70:
            if is_eagle3(args.mode):
                target = resolve_snapshot(MODELS["llama_70b_3p1"])
            else:
                target = resolve_snapshot(MODELS["llama_70b"])
            draft_name = "llama_1b" if is_standalone(args.mode) else "eagle3_llama_70b"
        elif args.size == 8:
            target = resolve_snapshot(MODELS["llama_8b"])
            draft_name = "llama_1b" if is_standalone(args.mode) else "eagle3_llama_8b"
        else:
            raise ValueError(f"Unsupported size for llama: {args.size}")

        draft = resolve_snapshot(MODELS[draft_name])
    else:
        target = resolve_snapshot(MODELS["qwen_32b"])
        if is_standalone(args.mode):
            draft = resolve_snapshot(MODELS["qwen_0.6b"])
        elif is_eagle3(args.mode):
            draft = resolve_snapshot(MODELS["eagle3_qwen_32b"])
        else:
            raise ValueError(f"Unsupported mode for qwen: {args.mode}")

    cmd = [
        "sglang", "serve",
        "--model-path", target,
        "--tp", str(args.tp),
        "--mem-fraction-static", str(args.mem_frac),
        "--max-running-requests", "1",
        # "--disable-radix-cache",
        "--log-level", "warning",
        "--port", str(args.port),
        "--context-length", str(args.context_length),
        "--dtype", "bfloat16",
    ]

    if is_spec(args.mode):
        # Speculative decoding with standalone draft model.
        # Default: k=5 (num_steps=4, num_draft_tokens=5).
        cmd += [
            "--speculative-algorithm", args.mode,
            "--speculative-draft-model-path", draft,
            "--speculative-num-steps", str(args.num_steps),
            "--speculative-eagle-topk", "1",
            "--speculative-num-draft-tokens", str(args.num_steps + 1),
        ]
        if is_async(args.mode):
            cmd += [
                "--speculative-async-fan-out", str(args.f),
            ]
            if args.fl:
                cmd += [
                    "--speculative-async-fan-out-list", ",".join(map(str, args.fl)),
                ]
            if args.flh:
                cmd += [
                    "--speculative-async-fan-out-list-hit", ",".join(map(str, args.flh)),
                ]
            if args.flm:
                cmd += [
                    "--speculative-async-fan-out-list-miss", ",".join(map(str, args.flm)),
                ]
            if args.backup in ["jit", "force-jit"]:
                cmd += [
                    "--speculative-async-jit-speculate",
                ]
            if args.backup == "force-jit":
                cmd += [
                    "--speculative-async-force-jit-speculate",
                ]
            if args.communicate_cache_hits:
                cmd += [
                    "--speculative-async-communicate-cache-hits",
                ]
            if args.verbose:
                cmd += [
                    "--speculative-async-verbose",
                ]

    # mode == "ar": no speculative flags, just serve the target model.
    return cmd, target


def wait_for_server(port, timeout=900, interval=5):
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(interval)
    return False


def kill_server(proc):
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


if __name__ == "__main__":
    main()
