"""Microbenchmark for build_tree_mask_bias and the fan_idx flattening pattern.

Runs at representative shapes for the sglang 8B eagle3 async-fast config:
  K=15, MQ_LEN=64 (fan_out=4 uniform), max_kv_stride≈4096, B=1.

Usage (from ssd-cc repo root with .venv active):
    python bench/bench_tree_mask.py
"""
from __future__ import annotations

import argparse
import time

import torch

from ssd.layers.tree_mask import _MASK_VAL, build_tree_mask_bias


# ------------------------------------------------------------------
# Reference vectorized fan_idx flatten (no Python list-comp).
# ------------------------------------------------------------------
def fan_idx_loop(fan_hit, fan_miss, cache_hits):
    return torch.cat([fan_hit if bool(h) else fan_miss for h in cache_hits.tolist()])


def fan_idx_vec(fan_hit, fan_miss, cache_hits):
    B = cache_hits.shape[0]
    hits_bool = cache_hits.to(torch.bool).view(B, 1)
    return torch.where(
        hits_bool, fan_hit.view(1, -1), fan_miss.view(1, -1)
    ).reshape(-1)


# ------------------------------------------------------------------
# Prototype GPU build_tree_mask_bias. We time this only for comparison;
# the real implementation lives in ssd/layers/tree_mask.py (the next step
# in this project will move this prototype into that module once it
# matches the CPU reference on the existing tier0 tests).
# ------------------------------------------------------------------
def build_tree_mask_bias_gpu(
    context_lens: torch.Tensor,
    step: int,
    K: int,
    MQ_LEN: int,
    fan_out_list,
    fan_out_list_miss,
    cache_hits: torch.Tensor,
    max_kv_stride: int,
) -> torch.Tensor:
    device = context_lens.device
    B = int(context_lens.shape[0])

    # Glue patterns, precomputed on device.
    tril = torch.tril(torch.ones(K + 1, K + 1, dtype=torch.bool, device=device))
    glue_hit = tril.repeat_interleave(
        torch.tensor(fan_out_list, dtype=torch.int64, device=device), dim=0
    )  # [MQ_LEN, K+1]
    glue_miss = tril.repeat_interleave(
        torch.tensor(fan_out_list_miss, dtype=torch.int64, device=device), dim=0
    )

    ttl_added = (step + 1) * MQ_LEN + (K + 1)
    ctx = context_lens.to(torch.int64)  # [B]
    prefix_len = ctx - ttl_added        # [B]

    kv_idx = torch.arange(max_kv_stride, device=device)  # [max_kv]
    # Prefix mask per batch: True where kv_idx < prefix_len[b]
    prefix_mask = kv_idx.view(1, 1, -1) < prefix_len.view(B, 1, 1)  # [B,1,max_kv]

    # Glue region: columns [prefix_len, prefix_len+K+1).
    # For each (b,q,kv), compute col_in_glue = kv_idx - prefix_len[b].
    col_in_glue = kv_idx.view(1, 1, -1) - prefix_len.view(B, 1, 1)  # [B,1,max_kv]
    in_glue = (col_in_glue >= 0) & (col_in_glue < (K + 1))          # [B,1,max_kv]
    # Glue pattern per row: select hit/miss per batch.
    hits = cache_hits[:B].to(torch.bool).view(B, 1, 1, 1)
    # glue_hit/glue_miss: [MQ_LEN, K+1] -> index by col_in_glue.clamp(0, K).
    col_clamped = col_in_glue.clamp(min=0, max=K)                    # [B,1,max_kv]
    # Build [B, MQ_LEN, max_kv] glue-attend bool by gathering.
    # glue_pattern[b, q, kv] = (hit ? glue_hit : glue_miss)[q, col_clamped[b,0,kv]]
    # Expand col_clamped to [B, MQ_LEN, max_kv].
    col_idx = col_clamped.expand(B, MQ_LEN, max_kv_stride)
    glue_h = glue_hit.view(1, MQ_LEN, K + 1).expand(B, MQ_LEN, K + 1)
    glue_m = glue_miss.view(1, MQ_LEN, K + 1).expand(B, MQ_LEN, K + 1)
    glue_sel = torch.where(
        hits.view(B, 1, 1).expand(B, MQ_LEN, K + 1), glue_h, glue_m
    )  # [B, MQ_LEN, K+1]
    glue_attend = torch.gather(glue_sel, 2, col_idx.clamp(0, K))     # [B, MQ_LEN, max_kv]
    glue_attend = glue_attend & in_glue                              # zero outside glue cols

    # Diag region: for blk in [0, step], col = prefix + K+1 + blk*MQ_LEN + row
    # An entry (b, row=q, kv) is on-diagonal iff there exists blk in [0,step] s.t.
    #   kv - (prefix[b] + K+1) - q == blk*MQ_LEN  and  0 <= blk <= step.
    q_idx = torch.arange(MQ_LEN, device=device).view(1, MQ_LEN, 1)     # [1,MQ_LEN,1]
    diag_col0 = prefix_len.view(B, 1, 1) + (K + 1) + q_idx             # [B,MQ_LEN,1]
    rel = kv_idx.view(1, 1, -1) - diag_col0                            # [B,MQ_LEN,max_kv]
    # on-diag iff rel >= 0, rel % MQ_LEN == 0, rel // MQ_LEN in [0, step]
    diag_attend = (rel >= 0) & (rel % MQ_LEN == 0) & ((rel // MQ_LEN) <= step)

    attend = prefix_mask.expand(B, MQ_LEN, -1) | glue_attend | diag_attend  # [B,MQ_LEN,max_kv]

    bias = torch.where(
        attend, torch.zeros((), device=device, dtype=torch.float32),
        torch.tensor(_MASK_VAL, device=device, dtype=torch.float32),
    )
    return bias.reshape(-1)


# ------------------------------------------------------------------
# Timing helper.
# ------------------------------------------------------------------
def bench(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000.0  # ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=15)
    p.add_argument("--fan-out", type=int, default=4)
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--max-kv", type=int, default=4096)
    args = p.parse_args()

    K = args.K
    fan_out_list = [args.fan_out] * (K + 1)
    fan_out_list_miss = [sum(fan_out_list)] + [0] * K
    MQ_LEN = sum(fan_out_list)
    B = args.B
    max_kv = args.max_kv

    cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if cuda else "cpu")
    print(f"device={dev}  K={K}  MQ_LEN={MQ_LEN}  B={B}  max_kv={max_kv}")

    ttl_added = (0 + 1) * MQ_LEN + (K + 1)
    ctx = torch.tensor([ttl_added + 50 + b for b in range(B)], dtype=torch.int32, device=dev)
    hits = torch.ones(B, dtype=torch.float32, device=dev)

    # --- build_tree_mask_bias: CPU-numpy baseline vs GPU prototype ---
    def run_baseline():
        return build_tree_mask_bias(
            ctx.cpu(), step=0, K=K, MQ_LEN=MQ_LEN,
            fan_out_list=fan_out_list, fan_out_list_miss=fan_out_list_miss,
            cache_hits=hits.cpu(), max_kv_stride=max_kv, device=dev,
        )

    def run_gpu_proto():
        return build_tree_mask_bias_gpu(
            ctx, step=0, K=K, MQ_LEN=MQ_LEN,
            fan_out_list=fan_out_list, fan_out_list_miss=fan_out_list_miss,
            cache_hits=hits, max_kv_stride=max_kv,
        )

    # Correctness spot check
    ref = run_baseline().to(dev)
    proto = run_gpu_proto()
    assert ref.shape == proto.shape, (ref.shape, proto.shape)
    match = torch.equal(ref, proto)
    print(f"prototype matches baseline: {match}")
    if not match:
        diff = (ref != proto).nonzero()[:5]
        print("first mismatches:", diff.tolist())

    t_base = bench(run_baseline)
    t_proto = bench(run_gpu_proto)
    print(f"build_tree_mask_bias baseline (cpu→gpu):  {t_base:.3f} ms")
    print(f"build_tree_mask_bias prototype (on gpu):  {t_proto:.3f} ms  ({t_base/t_proto:.2f}× faster)")

    # --- fan_idx flatten ---
    fan_hit = torch.arange(K + 1, device=dev, dtype=torch.int64).repeat_interleave(
        torch.tensor(fan_out_list, device=dev)
    )
    fan_miss = torch.arange(K + 1, device=dev, dtype=torch.int64).repeat_interleave(
        torch.tensor(fan_out_list_miss, device=dev)
    )
    cache_hits = torch.tensor([b % 2 for b in range(B)], device=dev, dtype=torch.int64)

    def run_loop():
        return fan_idx_loop(fan_hit, fan_miss, cache_hits)

    def run_vec():
        return fan_idx_vec(fan_hit, fan_miss, cache_hits)

    match2 = torch.equal(run_loop(), run_vec())
    print(f"fan_idx vectorized matches loop: {match2}")
    t_loop = bench(run_loop)
    t_vec = bench(run_vec)
    print(f"fan_idx loop:       {t_loop*1000:.1f} µs")
    print(f"fan_idx vectorized: {t_vec*1000:.1f} µs  ({t_loop/t_vec:.2f}× faster)")


if __name__ == "__main__":
    main()
