"""Benchmark DraftRunner across batch sizes, lookaheads, and fanouts.

We spawn a DraftRunner in a child process using the existing `async_nccl_port`
cross-node handshake. The main process plays the role of "target": it forms a
2-rank NCCL group with the draft, performs the handshake, then drives the
draft by sending well-formed PrefillRequest / SpeculationRequest tensors and
timing the responses. No real target model is loaded.

Because (K = speculate_k) and (F = async_fan_out) are baked into the
DraftRunner at construction (graph capture, prealloc buffers), each (K, F)
combo spawns a fresh DraftRunner subprocess. Inside one DraftRunner we sweep
all batch sizes — graph capture covers a list of BSes up to max_num_seqs.

Defaults match bench/small_test.py: Llama 3.2 1B draft, Llama 3.3 70B target
(only target hf_config is read for vocab/hidden_size). EAGLE / Phoenix are
supported via --eagle / --phoenix.

Usage:
  python bench/bench_draft_runner.py --batch-sizes 1 4 16 --lookaheads 3 5 7 --fanouts 1 3 5
  python bench/bench_draft_runner.py --eagle --batch-sizes 1 4 16
"""

import argparse
import os
import socket
import sys
import time
from contextlib import closing
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import TCPStore

from ssd.config import Config
from ssd.engine.draft_runner import DraftRunner
from ssd.engine.helpers.runner_helpers import (
    COMMAND,
    PrefillRequest,
    SpeculationRequest,
    SpeculationResponse,
    send_tensor,
    receive_tensor,
)
from ssd.utils.dist_utils import init_custom_process_group


LLAMA_1B = "/scratch/avner/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
LLAMA_70B = "/scratch/avner/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"
EAGLE_PATH = "/scratch/avner/huggingface/hub/models--lmsys--SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge/snapshots/63ebaa6585f96b89685adad8fdfa0da53be6a8fd"
PHOENIX_PATH = "/scratch/avner/huggingface/hub/models--togethercomputer--phoenix-Llama-3p2-1B-Instruct-tgt-Llama-3p3-70b-instruct-UNTRAINED/snapshots/3af59d71514388e14d8685f2b684f74e3e311717"


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_target_config(args, port: int, K: int, F: int) -> Config:
    """Build a target-style Config; DraftRunner.create_draft_config flips it
    into the draft-side view (model=draft path, d_model_target set, etc.)."""
    return Config(
        model=args.model,
        draft=args.draft,
        speculate=True,
        speculate_k=K,
        draft_async=True,
        async_fan_out=F,
        async_nccl_port=port,
        async_nccl_host="127.0.0.1",
        num_gpus=2,                          # draft_rank = num_gpus - 1 = 1
        max_model_len=args.max_model_len,
        max_num_seqs=max(args.batch_sizes),  # ⇒ CUDA-graph BS list covers our sweep
        kvcache_block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        use_eagle=args.eagle,
        use_phoenix=args.phoenix,
        jit_speculate=True,
        force_jit_speculate=True,            # deterministic per-iter cost
        communicate_logits=False,
        communicate_cache_hits=False,
        verbose=False,
        enforce_eager=args.enforce_eager,
    )


def _draft_entrypoint(draft_cfg: Config, rank: int):
    """Child process: construct DraftRunner. Its __init__ enters draft_loop()
    and only returns on DRAFT_EXIT."""
    DraftRunner(draft_cfg, rank=rank)


def _make_async_pg_as_target(port: int, device: torch.device, timeout_min: int = 60):
    """Mirror of the target-side handshake in model_runner.py
    (`setup_and_warmup_model_and_cudagraphs`, the `async_nccl_port` branch),
    but on the master side: world_size=2, is_master=True, rank=0."""
    timeout = timedelta(minutes=timeout_min)
    store = TCPStore(
        "127.0.0.1",
        port=port,
        world_size=2,
        is_master=True,
        timeout=timeout,
    )
    with torch.cuda.device(device):
        pg = init_custom_process_group(
            backend="nccl",
            store=store,
            world_size=2,
            rank=0,
            group_name="async_spec",
            timeout=timeout,
        )
    return pg


def _do_handshake(pg, draft_rank: int, device: torch.device) -> int:
    """Send 0 (= "use your own kv pool"), receive draft's num_kvcache_blocks."""
    kv_buf = torch.zeros(1, dtype=torch.int64, device=device)
    send_tensor(kv_buf, pg, draft_rank, name="target kv_cache_size")
    ready = torch.empty(1, dtype=torch.int64, device=device)
    receive_tensor(ready, pg, draft_rank, name="num_kvcache_blocks")
    return int(ready.item())


def _alloc_block_tables(B: int, num_tokens: int, K: int, max_blocks: int,
                        block_size: int, num_kv_blocks: int,
                        device: torch.device) -> torch.Tensor:
    """Hand each seq a contiguous run of distinct block ids.

    We need to cover positions [0 .. num_tokens-1+K] = num_tokens+K positions.
    Block_size=1 -> blocks_per_seq = num_tokens+K. We give a small pad too.
    """
    needed = (num_tokens + K + block_size - 1) // block_size + 2
    assert needed <= max_blocks, f"block_table too narrow: need {needed}, have {max_blocks}"
    assert B * needed <= num_kv_blocks, (
        f"not enough KV blocks: B={B} * needed={needed} = {B*needed} > "
        f"num_kvcache_blocks={num_kv_blocks}. Lower --prompt-len, --batch-sizes, "
        f"or raise --gpu-memory-utilization."
    )
    bt = torch.full((B, max_blocks), -1, dtype=torch.int32, device=device)
    base = torch.arange(needed, dtype=torch.int32, device=device)
    for i in range(B):
        bt[i, :needed] = base + i * needed
    return bt


def _do_prefill(pg, draft_rank: int, device: torch.device,
                draft_cfg: Config, B: int, prompt_len: int,
                max_blocks: int, num_kv_blocks: int,
                eagle_act_dim: int, K_for_padding: int) -> torch.Tensor:
    block_size = draft_cfg.kvcache_block_size
    block_tables = _alloc_block_tables(
        B, prompt_len, K_for_padding, max_blocks, block_size, num_kv_blocks, device,
    )
    vocab = draft_cfg.draft_hf_config.vocab_size
    input_ids_flat = torch.randint(
        0, vocab, (B * prompt_len,), dtype=torch.int64, device=device,
    )
    num_tokens = torch.full((B,), prompt_len, dtype=torch.int64, device=device)
    eagle_acts = None
    if draft_cfg.use_eagle_or_phoenix:
        eagle_acts = torch.randn(
            B * prompt_len, eagle_act_dim,
            dtype=draft_cfg.draft_hf_config.torch_dtype, device=device,
        )
    req = PrefillRequest.prepare(
        input_ids=input_ids_flat,
        num_tokens=num_tokens,
        draft_block_table=block_tables,
        eagle_acts=eagle_acts,
        max_blocks=max_blocks,
        device=device,
    )
    req.send(pg, draft_rank)
    return block_tables


def _build_spec_request(B: int, K: int, max_blocks: int, prompt_len: int,
                        block_tables: torch.Tensor, vocab_size: int,
                        eagle: bool, eagle_act_dim: int,
                        draft_dtype: torch.dtype, device: torch.device,
                        ) -> SpeculationRequest:
    """One pre-allocated SpeculationRequest, reused per timed iter."""
    req = SpeculationRequest.prepare(
        batch_size=B,
        lookahead=K,
        max_blocks=max_blocks,
        vocab_size=vocab_size,
        draft_dtype=draft_dtype,
        device=device,
        eagle=eagle,
        eagle_act_dim=eagle_act_dim,
    )
    # cache_keys: (seq_id, last_spec_accepted_len-1, recovery_token_id).
    # With force_jit_speculate=True the cache lookup is bypassed, so these
    # only need to be in-range; the sentinel -1 for accepted_len-1 mimics
    # the very first spec iter in the real engine.
    seq_ids = torch.arange(B, dtype=torch.int64, device=device)
    req.cache_keys[:, 0] = seq_ids
    req.cache_keys[:, 1] = -1
    req.cache_keys[:, 2] = (seq_ids + 1) % vocab_size  # any valid token id

    # The sequence has prompt_len real tokens plus the recovery token already
    # appended by the target (see speculator_async.speculate). The draft uses
    # num_tokens-1 as the position of the recovery token, then advances by 1
    # per JIT step.
    req.num_tokens.fill_(prompt_len + 1)
    req.block_tables[:, :] = block_tables[:B, :max_blocks]
    req.temps.fill_(0.0)
    if eagle:
        # All-zero extends → recovery activation lives at slot 0.
        req.extend_counts.zero_()
        req.extend_activations.normal_()
        req.extend_token_ids.fill_(0)
    return req


def _exit_draft(pg, draft_rank: int, device: torch.device):
    cmd = torch.tensor([COMMAND.DRAFT_EXIT], dtype=torch.int64, device=device)
    send_tensor(cmd, pg, draft_rank, name="exit cmd")


def _run_one_kf(args, K: int, F: int, results: list):
    port = _free_port()
    target_cfg = _build_target_config(args, port, K, F)
    draft_cfg = DraftRunner.create_draft_config(target_cfg)

    # Spawn the DraftRunner.
    draft_rank = 1
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_draft_entrypoint, args=(draft_cfg, draft_rank))
    proc.start()

    target_device = torch.device(f"cuda:{args.target_gpu}")
    torch.cuda.set_device(target_device)

    pg = _make_async_pg_as_target(port, target_device)
    print(f"[bench] [K={K} F={F}] NCCL group formed; handshaking...", flush=True)
    num_kv_blocks = _do_handshake(pg, draft_rank, target_device)
    print(f"[bench] [K={K} F={F}] draft num_kvcache_blocks={num_kv_blocks}",
          flush=True)

    eagle = draft_cfg.use_eagle_or_phoenix
    if draft_cfg.use_eagle:
        eagle_act_dim = 3 * draft_cfg.d_model_target
    elif draft_cfg.use_phoenix:
        eagle_act_dim = draft_cfg.d_model_target
    else:
        eagle_act_dim = 0
    draft_dtype = draft_cfg.draft_hf_config.torch_dtype
    vocab_size = target_cfg.hf_config.vocab_size  # draft now returns target-vocab logits
    block_size = draft_cfg.kvcache_block_size
    max_blocks = (draft_cfg.max_model_len + block_size - 1) // block_size

    try:
        for B in args.batch_sizes:
            block_tables = _do_prefill(
                pg, draft_rank, target_device, draft_cfg,
                B=B, prompt_len=args.prompt_len,
                max_blocks=max_blocks, num_kv_blocks=num_kv_blocks,
                eagle_act_dim=eagle_act_dim, K_for_padding=K,
            )

            req = _build_spec_request(
                B=B, K=K, max_blocks=max_blocks, prompt_len=args.prompt_len,
                block_tables=block_tables, vocab_size=vocab_size,
                eagle=eagle, eagle_act_dim=eagle_act_dim,
                draft_dtype=draft_dtype, device=target_device,
            )
            resp = SpeculationResponse.prepare(
                batch_size=B,
                lookahead=K,
                device=target_device,
                draft_dtype=draft_dtype,
                vocab_size=vocab_size,
                communicate_logits=False,
                communicate_cache_hits=False,
            )

            # Warmup
            for _ in range(args.warmup_iters):
                req.send(pg, draft_rank)
                resp.receive(pg, draft_rank, batch_size=B)

            # Timed: includes target-side send + NCCL round-trip + draft compute.
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.num_iters):
                req.send(pg, draft_rank)
                resp.receive(pg, draft_rank, batch_size=B)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0 / args.num_iters

            results.append((K, F, B, ms))
            print(f"[bench] K={K:>2} F={F:>2} B={B:>4}: {ms:8.3f} ms/iter",
                  flush=True)
    finally:
        try:
            _exit_draft(pg, draft_rank, target_device)
        except Exception as e:
            print(f"[bench] exit signal failed: {e}", flush=True)
        proc.join(timeout=120)
        if proc.is_alive():
            print("[bench] draft did not exit cleanly; terminating", flush=True)
            proc.terminate()
            proc.join(timeout=10)
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=LLAMA_70B,
                        help="Target model path (for vocab/hidden_size).")
    parser.add_argument("--draft", type=str, default=LLAMA_1B,
                        help="Draft model path.")
    parser.add_argument("--eagle", action="store_true",
                        help="Use EAGLE3 draft (overrides --draft to the EAGLE path).")
    parser.add_argument("--phoenix", action="store_true",
                        help="Use Phoenix draft (overrides --draft to the Phoenix path).")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--lookaheads", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--fanouts", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Skip CUDA-graph capture (faster startup, slower per-iter).")
    parser.add_argument("--target-gpu", type=int, default=0,
                        help="GPU device for this (simulator-target) process.")
    parser.add_argument("--csv-out", type=str, default=None,
                        help="Optional path to dump CSV results.")
    args = parser.parse_args()

    assert not (args.eagle and args.phoenix), "Pick at most one of --eagle / --phoenix"
    if args.eagle:
        args.draft = EAGLE_PATH
        args.model = LLAMA_70B
    elif args.phoenix:
        args.draft = PHOENIX_PATH
        args.model = LLAMA_70B

    for p in (args.model, args.draft):
        assert os.path.isdir(p), f"Not a directory: {p}"

    assert torch.cuda.device_count() >= 2, (
        "Need at least 2 visible CUDA devices (target on cuda:0, draft on cuda:1)."
    )

    print(f"[bench] target={args.model}", flush=True)
    print(f"[bench] draft={args.draft}", flush=True)
    print(f"[bench] eagle={args.eagle} phoenix={args.phoenix}", flush=True)
    print(f"[bench] sweep: K={args.lookaheads} F={args.fanouts} B={args.batch_sizes}",
          flush=True)

    results = []  # (K, F, B, ms_per_iter)
    for K in args.lookaheads:
        for F in args.fanouts:
            _run_one_kf(args, K, F, results)

    print("\n=== Results ===")
    print(f"{'K':>4} {'F':>4} {'B':>6} {'ms/iter':>12}")
    for K, F, B, ms in results:
        print(f"{K:>4} {F:>4} {B:>6} {ms:>12.3f}")

    if args.csv_out:
        with open(args.csv_out, "w") as f:
            f.write("K,F,B,ms_per_iter\n")
            for K, F, B, ms in results:
                f.write(f"{K},{F},{B},{ms:.4f}\n")
        print(f"[bench] wrote {args.csv_out}", flush=True)


if __name__ == "__main__":
    main()
