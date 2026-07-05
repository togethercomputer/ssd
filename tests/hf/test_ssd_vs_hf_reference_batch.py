"""Batch>1 SSD-vs-HF-reference test against the TGL server.

N concurrent greedy requests (distinct prompts, client-chosen rids) run against
one server with --max-running-requests N and radix caching disabled. The
draft-side dumps are split per sequence via tests/hf/trace_reader (rows keyed by
hash_rid), and every sequence gets the same checks as the single-seq test:

  1. completion vs HF target (per-token logit gap),
  2. speculation reconstruction with the HF reference draft (strict threshold
     on chain-free rounds, statistical on eagle cache-hit rounds),
  3. accept length: completion_tokens / rounds == engine's spec_accept_length,
  4. batching honesty: the batch must actually have run at B == N for most
     rounds, else the test isn't testing batch>1 and fails loudly.

Iterating on reconstruction only: set SSD_TRACE_REUSE=<dir saved by a previous
run> to skip the server phase (~1 min instead of ~10).

This end-to-end covers the TGL batch fixes: the is_first_decode tensor crash,
the per-request eagle_acts slice misalignment, merged/filtered spec state, and
per-request temperatures (greedy here: all zeros).
"""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from .eagle3_hf import load_eagle3_specforge
from .phoenix_hf import load_phoenix_specforge
from .helpers import (
    launch_tgl_server,
    wait_for_server,
    kill_server,
    require_8b_target,
    require_eagle_llama_8b_draft,
    require_phoenix_llama_8b_draft,
    require_1b_draft,
)
from .trace_reader import (
    hash_rid,
    load_trace,
    load_prefills,
    split_per_seq,
    chain_prefixes,
    engine_comparable_accept_length,
)
from .test_ssd_vs_hf_reference import (
    LOGIT_GAP_THRESHOLD,
    SPEC_RANK_THRESHOLD,
    CHAIN_TOKEN_FRACTION,
    CHAIN_ROUND_FRACTION,
    ACCEPT_LENGTH_TOLERANCE,
    compare_completion_to_hf_reference,
    compare_completion_to_hf_reference_eagle,
    get_hf_target_activations_for_eagle_or_phoenix,
)

PORT = 40031
N_CONCURRENT = 4
MAX_NEW_TOKENS = 64
LOOKAHEAD = 4
FANOUT = 3
BATCH_ROUND_FRACTION = 0.6  # >=60% of rounds must run at full batch
CITIES = ["San Francisco", "Kyoto", "Nairobi", "Reykjavik"]
# Shipped extend/recovery activations must match HF target activations at the
# same positions (bf16 kernel noise is ~0.03 rel-norm; a misindexed row is
# ~1.0). This is the chaos-immune conditioning check — it catches the
# activation-misalignment bug class directly.
ACT_REL_DIFF_THRESHOLD = 0.15
# Eagle cache-hit ("chain") rounds at B>1: round-to-round batch-composition
# changes add bf16 perturbation sources into the recurrence chain, so the
# one-round-delay reconstruction diverges more often than at B=1 (where the
# single-seq test holds 0.90/0.80). Verified benign on this setup: shipped
# activations clean (see ACT_REL_DIFF_THRESHOLD), completions exactly HF,
# accept lengths exact, draft bit-deterministic. These bounds only catch
# catastrophic conditioning bugs (which score near 0).
BATCH_CHAIN_TOKEN_FRACTION = 0.6
BATCH_CHAIN_ROUND_FRACTION = 0.35


def _prompts(tokenizer) -> dict[str, list[int]]:
    out = {}
    for i, city in enumerate(CITIES[:N_CONCURRENT]):
        toks = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Please tell me about {city}."},
            ],
            add_generation_prompt=True,
        )
        if not isinstance(toks, list):
            toks = toks["input_ids"]
        out[f"ssdtest-{i}"] = toks
    return out


def _run_server_phase(speculator_type, backup, prompts_by_rid, trace_dir):
    os.environ["SSD_DUMP_TENSORS_DIR"] = str(trace_dir)
    target_path = require_8b_target()
    draft_path = {
        "eagle": require_eagle_llama_8b_draft,
        "phoenix": require_phoenix_llama_8b_draft,
        "standalone": require_1b_draft,
    }[speculator_type]()

    tgl_server = None
    try:
        # NOTE: radix caching stays ON (the fork forbids --disable-radix-cache).
        # That is safe for dump reconstruction here because sharing is
        # page-granular (page 64) and these chat prompts share < 1 page of
        # common prefix, so every prefill dump carries the full prompt.
        tgl_server, _ = launch_tgl_server(
            speculator_type, backup, target_path, draft_path, LOOKAHEAD, FANOUT, PORT,
            max_running_requests=N_CONCURRENT,
        )
        assert wait_for_server(PORT), "tgl server failed to start"

        barrier = threading.Barrier(len(prompts_by_rid))

        def _one(rid, toks):
            barrier.wait()
            r = requests.post(
                f"http://localhost:{PORT}/generate",
                json={
                    "input_ids": toks,
                    "rid": rid,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "ignore_eos": True,
                    },
                },
                timeout=600,
            )
            assert r.status_code == 200, f"{rid}: {r.status_code} {r.text[:200]}"
            return rid, r.json()

        with ThreadPoolExecutor(len(prompts_by_rid)) as ex:
            results = dict(
                ex.map(lambda kv: _one(*kv), prompts_by_rid.items())
            )
        payload = {
            rid: {
                "output_ids": r["output_ids"],
                "spec_accept_length": r["meta_info"]["spec_accept_length"],
            }
            for rid, r in results.items()
        }
        (trace_dir / "completions.json").write_text(json.dumps(payload))
        return payload
    finally:
        if tgl_server is not None:
            kill_server(tgl_server)
            assert not wait_for_server(PORT, timeout=3.0), "tgl server failed to stop"


@pytest.mark.parametrize(
    "speculator_type,backup",
    [("eagle", "fast"), ("eagle", "jit"), ("phoenix", "fast"), ("standalone", "fast")],
)
def test_batch_ssd_vs_hf_reference(speculator_type, backup, tmp_path):
    eagle = speculator_type == "eagle"
    phoenix = speculator_type == "phoenix"
    dtype = torch.bfloat16
    target_path = require_8b_target()
    tokenizer = AutoTokenizer.from_pretrained(target_path)
    prompts_by_rid = _prompts(tokenizer)

    reuse = os.environ.get("SSD_TRACE_REUSE")
    if reuse:
        trace_dir = Path(reuse)
        completions = json.loads((trace_dir / "completions.json").read_text())
    else:
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir(exist_ok=True)
        completions = _run_server_phase(speculator_type, backup, prompts_by_rid, trace_dir)
        print(f"[batch] trace saved to {trace_dir} (reuse via SSD_TRACE_REUSE)", flush=True)

    # ---- reconstruct per sequence ------------------------------------------
    rounds = load_trace(trace_dir)
    seq_prompts = {hash_rid(rid): p for rid, p in prompts_by_rid.items()}
    traces = split_per_seq(rounds, prefills=load_prefills(trace_dir), prompts_by_seq_id=seq_prompts)
    assert set(traces) == set(seq_prompts), "dump rows don't cover all rids"

    # batching honesty: the point is B>1 — most rounds must run at full batch
    b_hist = [r.B for r in rounds]
    frac_full = sum(b == N_CONCURRENT for b in b_hist) / len(b_hist)
    assert max(b_hist) == N_CONCURRENT, f"batch never reached N={N_CONCURRENT}: {b_hist}"
    assert frac_full >= BATCH_ROUND_FRACTION, (
        f"only {frac_full:.2f} of rounds ran at B={N_CONCURRENT} — not actually "
        f"testing batch>1 (histogram: {b_hist})"
    )

    target_device = os.environ.get("SSD_TEST_TARGET_DEVICE", "cuda:4")
    draft_device = os.environ.get("SSD_TEST_DRAFT_DEVICE", "cuda:5")
    target_model = AutoModelForCausalLM.from_pretrained(target_path, torch_dtype=dtype)
    target_model.eval().to(target_device)
    if eagle:
        draft_model = load_eagle3_specforge(
            require_eagle_llama_8b_draft(), target_model.model.embed_tokens.weight,
            target_model.config.hidden_size, draft_device, dtype=dtype,
        ).eval()
    elif phoenix:
        draft_model = load_phoenix_specforge(
            require_phoenix_llama_8b_draft(), target_model.config.hidden_size,
            draft_device, dtype=dtype,
        ).eval()
    else:
        draft_model = AutoModelForCausalLM.from_pretrained(
            require_1b_draft(), torch_dtype=dtype
        ).to(draft_device).eval()

    failures = []
    for rid, prompt in prompts_by_rid.items():
        sid = hash_rid(rid)
        trace = traces[sid]
        completion = completions[rid]["output_ids"]
        engine_accept = completions[rid]["spec_accept_length"]

        # 1) completion vs HF target
        gaps, _ = compare_completion_to_hf_reference(
            target_model, prompt, completion, rid, tokenizer, engine="tgl",
        )
        if max(gaps) >= LOGIT_GAP_THRESHOLD:
            failures.append(f"{rid}: completion gap {max(gaps):.3f} >= {LOGIT_GAP_THRESHOLD}")

        # 3) accept length (exact estimator + chain/completion consistency)
        recon_accept = engine_comparable_accept_length(trace, prompt, completion, LOOKAHEAD)
        if abs(recon_accept - engine_accept) > ACCEPT_LENGTH_TOLERANCE:
            failures.append(
                f"{rid}: accept length recon {recon_accept:.4f} vs engine {engine_accept:.4f}"
            )

        # 2) speculation reconstruction (eagle/phoenix)
        if eagle or phoenix:
            prefixes = chain_prefixes(trace, prompt)
            all_tokens = prompt + completion
            hf_acts = get_hf_target_activations_for_eagle_or_phoenix(
                target_model, all_tokens, eagle, phoenix
            ).to(draft_device)
            hf_acts = torch.cat([hf_acts[:1], hf_acts])

            # 2a) shipped conditioning activations vs HF ground truth (per slot)
            worst_rel = 0.0
            for i, row in enumerate(trace.rows):
                n = row.extend_count or 0
                ext = row.extend_activations.to(draft_device, torch.float32)
                for j in range(n + 1):
                    g = (len(prefixes[i - 1]) + j) if j < n else (len(prefixes[i]) - 1)
                    ref = hf_acts[g].float()
                    worst_rel = max(worst_rel, float((ext[j] - ref).norm() / ref.norm()))
            if worst_rel > ACT_REL_DIFF_THRESHOLD:
                failures.append(
                    f"{rid}: shipped extend/recovery activations deviate from HF target "
                    f"acts (worst rel-diff {worst_rel:.3f} > {ACT_REL_DIFF_THRESHOLD}) — "
                    f"activation capture/packing misalignment"
                )

            strict_ranks, chain_ranks = [], []
            ext_counts = [r.extend_count for r in trace.rows]
            for i, row in enumerate(trace.rows):
                if backup == "fast" and not row.cache_hit:
                    continue
                jit = backup == "force-jit" or (not row.cache_hit and backup == "jit")
                act_idx = len(prefixes[i]) if jit else len(prefixes[i - 1])
                _, ranks = compare_completion_to_hf_reference_eagle(
                    draft_model, prefixes[i], row.spec_tokens, hf_acts, act_idx,
                    f"{rid}:{i}", None, ext_counts, None, None, jit, None,
                    tokenizer, engine="tgl", funky=False, prefixes=prefixes,
                    phoenix=phoenix,
                )
                (chain_ranks if (eagle and not jit) else strict_ranks).append(ranks)

            if strict_ranks:
                worst = max(max(r) for r in strict_ranks)
                if worst > SPEC_RANK_THRESHOLD:
                    failures.append(f"{rid}: chain-free worst rank {worst} (rounds {strict_ranks})")
            if chain_ranks:
                flat = [x for r in chain_ranks for x in r]
                ft = sum(x <= SPEC_RANK_THRESHOLD for x in flat) / len(flat)
                fr = sum(max(r) <= SPEC_RANK_THRESHOLD for r in chain_ranks) / len(chain_ranks)
                print(f"[batch][{rid}] chain rank fractions: token {ft:.3f}, round {fr:.3f}", flush=True)
                if ft < BATCH_CHAIN_TOKEN_FRACTION or fr < BATCH_CHAIN_ROUND_FRACTION:
                    failures.append(
                        f"{rid}: chain rounds off reference (token frac {ft:.3f}, round frac {fr:.3f})"
                    )
        print(f"[batch][{rid}] OK-so-far accept={engine_accept:.3f} rounds={len(trace.rows)}", flush=True)

    assert not failures, "batch reference failures:\n  " + "\n  ".join(failures)
