"""Scripted-target harness for driving a real DraftRunner (no server, 2 GPUs).

The main process plays the target: it spawns a DraftRunner child process,
forms the 2-rank NCCL group via the async handshake, and drives the draft with
PrefillRequest / SpeculationRequest messages, returning SpeculationResponses.

Adapted from bench/bench_draft_runner.py (kept separate so a perf script never
becomes a test dependency). Faithful to the TGL scheduler's draft config
(python/sglang/private/managers/scheduler.py:_init_async_spec_worker).
"""
from __future__ import annotations

import os
import socket
from contextlib import closing
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import TCPStore

from ssd.config import Config
from ssd.engine.helpers.runner_helpers import (
    COMMAND,
    PrefillRequest,
    SpeculationRequest,
    SpeculationResponse,
    send_tensor,
    receive_tensor,
)
from ssd.utils.dist_utils import init_custom_process_group


def free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_target_config(
    *,
    target_path: str,
    draft_path: str,
    K: int,
    fanout: int,
    mode: str,  # "standalone" | "eagle" | "phoenix"
    backup: str,  # "fast" | "jit" | "force-jit"
    max_num_seqs: int = 4,
    max_model_len: int = 2048,
    block_size: int = 64,
    enforce_eager: bool = False,
    port: int | None = None,
    gpu_memory_utilization: float = 0.5,
) -> Config:
    return Config(
        model=target_path,
        draft=draft_path,
        speculate=True,
        speculate_k=K,
        draft_async=True,
        async_fan_out=fanout,
        async_nccl_port=port or free_port(),
        async_nccl_host="127.0.0.1",
        num_gpus=2,  # draft rank 1
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        kvcache_block_size=block_size,
        gpu_memory_utilization=gpu_memory_utilization,
        tokenizer_path=target_path if mode in ("eagle", "phoenix") else None,
        use_eagle=(mode == "eagle"),
        use_phoenix=(mode == "phoenix"),
        jit_speculate=backup in ("jit", "force-jit"),
        force_jit_speculate=backup == "force-jit",
        communicate_logits=True,
        communicate_cache_hits=True,
        verbose=False,
        enforce_eager=enforce_eager,
        dtype=torch.bfloat16,
    )


def _draft_entrypoint(draft_cfg: Config, rank: int):
    # never let the child re-dump or expect profile ACKs
    os.environ.pop("SSD_DUMP_TENSORS_DIR", None)
    os.environ.pop("SSD_PROFILE", None)
    from ssd.engine.draft_runner import DraftRunner  # import inside the spawned child

    DraftRunner(draft_cfg, rank=rank)


class DraftSession:
    """Owns the DraftRunner child + NCCL group. Target side runs on `device`."""

    def __init__(self, cfg: Config, target_device: str = "cuda:0", timeout_min: int = 30):
        os.environ.pop("SSD_PROFILE", None)
        from ssd.engine.draft_runner import DraftRunner

        self.cfg = cfg
        self.draft_cfg = DraftRunner.create_draft_config(cfg)
        self.device = torch.device(target_device)
        self.draft_rank = 1
        self.K = cfg.speculate_k

        ctx = mp.get_context("spawn")
        self.proc = ctx.Process(
            target=_draft_entrypoint, args=(self.draft_cfg, 1), daemon=True
        )
        self.proc.start()

        timeout = timedelta(minutes=timeout_min)
        store = TCPStore(
            "127.0.0.1", port=cfg.async_nccl_port, world_size=2, is_master=True, timeout=timeout,
        )
        with torch.cuda.device(self.device):
            self.pg = init_custom_process_group(
                backend="nccl", store=store, world_size=2, rank=0,
                group_name="async_spec", timeout=timeout,
            )
        # handshake: 0 = "draft sizes its own KV pool"; returns its block count
        kv_buf = torch.zeros(1, dtype=torch.int64, device=self.device)
        send_tensor(kv_buf, self.pg, self.draft_rank, name="target kv_cache_size")
        ready = torch.empty(1, dtype=torch.int64, device=self.device)
        receive_tensor(ready, self.pg, self.draft_rank, name="num_kvcache_blocks")
        self.num_kv_blocks = int(ready.item())

        self.vocab_size = cfg.hf_config.vocab_size
        self.eagle_act_dim = 0
        if cfg.use_eagle:
            self.eagle_act_dim = 3 * cfg.hf_config.hidden_size
        elif cfg.use_phoenix:
            self.eagle_act_dim = cfg.hf_config.hidden_size
        self.draft_dtype = torch.bfloat16
        self._resp = None

    # ---- message helpers -------------------------------------------------
    def send_prefill(self, input_ids, num_tokens, block_tables, eagle_acts, max_blocks):
        req = PrefillRequest.prepare(
            input_ids=input_ids.to(self.device),
            num_tokens=num_tokens.to(self.device),
            draft_block_table=block_tables.to(torch.int32).to(self.device),
            eagle_acts=None if eagle_acts is None else eagle_acts.to(self.draft_dtype).to(self.device),
            max_blocks=max_blocks,
            device=self.device,
        )
        req.send(self.pg, self.draft_rank)

    def spec_round(self, *, cache_keys, num_tokens, block_tables, temps=None,
                   extend_counts=None, extend_activations=None, extend_token_ids=None):
        B = cache_keys.shape[0]
        max_blocks = block_tables.shape[1]
        req = SpeculationRequest.prepare(
            batch_size=B, lookahead=self.K, max_blocks=max_blocks,
            vocab_size=self.vocab_size, draft_dtype=self.draft_dtype,
            device=self.device, eagle=self.eagle_act_dim > 0,
            eagle_act_dim=self.eagle_act_dim,
        )
        req.cache_keys.copy_(cache_keys.to(self.device))
        req.num_tokens.copy_(num_tokens.to(self.device))
        req.block_tables.copy_(block_tables.to(torch.int32).to(self.device))
        req.temps.copy_(
            torch.zeros(B, dtype=torch.float32) if temps is None else temps.float()
        )
        if self.eagle_act_dim > 0:
            req.extend_counts.copy_(extend_counts.to(self.device))
            req.extend_activations.copy_(extend_activations.to(self.draft_dtype).to(self.device))
            req.extend_token_ids.copy_(extend_token_ids.to(self.device))
        req.send(self.pg, self.draft_rank)

        # NOTE: SpeculationResponse defines a classmethod receive() that is
        # shadowed by the later instance method of the same name (latent bug in
        # runner_helpers.py) — use prepare() + instance .receive() like
        # speculator_async does.
        if self._resp is None:
            self._resp = SpeculationResponse.prepare(
                lookahead=self.K, device=self.device, draft_dtype=self.draft_dtype,
                batch_size=B, vocab_size=self.vocab_size,
                communicate_logits=True, communicate_cache_hits=True,
            )
        self._resp.receive(self.pg, self.draft_rank, batch_size=B)
        return self._resp

    def close(self, graceful: bool = True):
        if graceful and self.proc.is_alive():
            try:
                cmd = torch.tensor([COMMAND.DRAFT_EXIT], dtype=torch.int64, device=self.device)
                send_tensor(cmd, self.pg, self.draft_rank, name="exit cmd")
            except Exception:
                pass
        self.proc.join(timeout=30 if graceful else 5)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=10)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(timeout=5)
        try:
            dist.destroy_process_group(self.pg)
        except Exception:
            pass
