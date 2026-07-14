from datetime import datetime
from dataclasses import dataclass
import os
import enum
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from ssd.engine.sequence import Sequence
from ssd.utils.misc import compress_neg_ones_and_zeros

NCCL_LOG = os.environ.get("SSD_NCCL_LOG", "0") == "1"
BRIEF_LOG = os.environ.get("SSD_BRIEF_LOG", "0") == "1"
RUN_NAME = os.environ.get("SSD_RUN_NAME", "")

def _ts():
    return datetime.now().strftime('%H:%M:%S.%f')[:-3]

def _dump_ts():
    if RUN_NAME:
        return RUN_NAME
    else:
        return datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')  # [:-4]

def list_to_str(lst: list[float] | list[list[float]], num_decimals: int = 4) -> str:
    assert len(lst) > 0
    if isinstance(lst[0], float):
         return str([round(v, 4) for v in lst])
    else:
        assert isinstance(lst[0], list)
        return str([[round(v, 4) for v in row] for row in lst])


@enum.unique
class COMMAND(enum.IntEnum):
    PREFILL = 0
    SPECULATION = 1
    DRAFT_EXIT = 2


@dataclass
class PrefillRequest:
    cmd: torch.Tensor | None
    metadata: torch.Tensor
    input_ids: torch.Tensor
    num_tokens: torch.Tensor
    draft_block_table: torch.Tensor
    eagle_acts: torch.Tensor

    @classmethod
    def prepare(
        cls,
        input_ids: torch.Tensor,  # flat tensor of input ids
        num_tokens: torch.Tensor,  # tensor of num tokens per sequence
        draft_block_table: torch.Tensor,
        eagle_acts: torch.Tensor,
        max_blocks: int,
        device: torch.device,
        cmd_buffer: torch.Tensor = None,
        metadata_buffer: torch.Tensor = None,
        tokenizer: AutoTokenizer = None,
    ):
        if eagle_acts is not None:
            assert eagle_acts.shape[0] == input_ids.shape[0], (
                f"Eagle activations length {eagle_acts.shape[0]} != input_ids_flat length {input_ids.shape[0]}"
            )

        metadata = [
            input_ids.shape[0],
            num_tokens.shape[0],
            max_blocks,
            1 if eagle_acts is not None else 0,
            eagle_acts.shape[1] if eagle_acts is not None else 0,
        ]
        if metadata_buffer is None:
            metadata_buffer = torch.tensor(metadata, dtype=torch.int64, device=device)
        else:
            metadata_buffer[:] = metadata

        if cmd_buffer is None:
            cmd_buffer = torch.tensor([COMMAND.PREFILL], dtype=torch.int64, device=device)
        else:
            cmd_buffer[0] = COMMAND.PREFILL

        prefill_request = cls(
            cmd=cmd_buffer,
            metadata=metadata_buffer,
            input_ids=input_ids,
            num_tokens=num_tokens,
            draft_block_table=draft_block_table,
            eagle_acts=eagle_acts,
        )
        prefill_request.tokenizer = tokenizer
        return prefill_request

    def send(self, async_pg: dist.ProcessGroup, draft_rank: int):
        if NCCL_LOG:
            sep = '=' * 80
            print(f"[{_ts()}] \n{sep}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SEND_PREFILL] cmd={self.cmd.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SEND_PREFILL] metadata={self.metadata.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SEND_PREFILL] input_ids shape={self.input_ids.shape}, values={self.input_ids.tolist()}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SEND_PREFILL] input_ids decoded='{_decode_ids(self.input_ids, self.tokenizer)}'", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SEND_PREFILL] num_tokens={self.num_tokens.tolist()}", flush=True)
            draft_block_table_values_str = compress_neg_ones_and_zeros(f"{self.draft_block_table.tolist()}")
            print(f"[{_ts()}] [NCCL_LOG SEND_PREFILL] draft_block_table shape={self.draft_block_table.shape}, values={draft_block_table_values_str}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG SEND_PREFILL] eagle_acts={'None' if self.eagle_acts is None else f'shape={self.eagle_acts.shape}'}", flush=True)
            print(f"[{_ts()}] {sep}\n", flush=True)
        send_tensor(self.cmd, async_pg, draft_rank, name="cmd", prefix="TARGET:PrefillRequest.send")
        send_tensor(self.metadata, async_pg, draft_rank, name="metadata", prefix="TARGET:PrefillRequest.send")
        fused_payload = concat_tensors_as_int64(self.input_ids, self.num_tokens, self.draft_block_table)
        send_tensor(fused_payload, async_pg, draft_rank, name="fused payload", prefix="TARGET:PrefillRequest.send")
        if self.eagle_acts is not None:
            send_tensor(self.eagle_acts, async_pg, draft_rank, name="eagle acts", prefix="TARGET:PrefillRequest.send")

    @classmethod
    def receive(
        cls,
        async_pg: dist.ProcessGroup,
        target_rank: int,
        device: torch.device,
        metadata_buffer: torch.Tensor=None,
        eagle_act_dtype: torch.dtype=torch.bfloat16,
        tokenizer: AutoTokenizer = None,
    ):
        # 1) Receive metadata then individual tensors
        # First receive prefill metadata to learn sizes
        if metadata_buffer is None:
            metadata_buffer = torch.empty(5, dtype=torch.int64, device=device)

        metadata = receive_tensor(metadata_buffer, async_pg, target_rank, name="metadata", prefix="DRAFT:PrefillRequest.receive")
        total_new_tokens, batch_size, max_blocks, use_eagle, eagle_act_dim = metadata.tolist()

        # 2) receive fused int64 payload (input_ids + num_tokens + draft_block_table)
        fused_total = total_new_tokens + batch_size + batch_size * max_blocks
        fused = torch.empty(fused_total, dtype=torch.int64, device=device)
        fused = receive_tensor(fused, async_pg, target_rank, name="fused payload", prefix="DRAFT:PrefillRequest.receive")
        off = 0
        input_ids = fused[off:off + total_new_tokens]
        off += total_new_tokens
        num_tokens = fused[off:off + batch_size]
        off += batch_size
        draft_block_table = fused[off:off + batch_size * max_blocks].view(batch_size, max_blocks).to(torch.int32)
        off += batch_size * max_blocks
        assert off == fused_total

        eagle_acts = None
        if use_eagle:
            eagle_acts = torch.empty(
                total_new_tokens, eagle_act_dim, dtype=eagle_act_dtype, device=device,
            )
            eagle_acts = receive_tensor(eagle_acts, async_pg, target_rank, name="eagle acts", prefix="DRAFT:PrefillRequest.receive")

        if BRIEF_LOG:
            print(f"[{_ts()}] [PrefillRequest.receive] metadata={metadata.tolist()}", flush=True)
            print(f"[{_ts()}] [PrefillRequest.receive] num_tokens={num_tokens.tolist()}", flush=True)
            decoded_input_ids = _decode_ids(input_ids, tokenizer)
            print(f"[{_ts()}] [PrefillRequest.receive] input_ids shape={input_ids.shape}, values={input_ids.tolist()}, decoded='{decoded_input_ids}'", flush=True)
            if eagle_acts is not None:
                print(f"[{_ts()}] [PrefillRequest.receive] eagle_acts shape={eagle_acts.shape}, eagle_acts[:3, :3]={list_to_str(eagle_acts[:3, :3].tolist())}", flush=True)

        received_request = cls(
            cmd=None,
            metadata=metadata,
            input_ids=input_ids,
            num_tokens=num_tokens,
            draft_block_table=draft_block_table,
            eagle_acts=eagle_acts,
        )
        received_request.dump()
        return received_request

    def dump(self):
        dump_dir = os.environ.get("SSD_DUMP_TENSORS_DIR", "")
        if dump_dir:
            torch.save({
                'metadata': self.metadata.cpu(),
                'input_ids': self.input_ids.cpu(),
                'num_tokens': self.num_tokens.cpu(),
                'draft_block_table': self.draft_block_table.cpu(),
                'eagle_acts': self.eagle_acts.cpu() if self.eagle_acts is not None else None,
            }, f"{dump_dir}/prefill_request_{_dump_ts()}.pt")


@dataclass
class SpeculationRequest:
    cmd: torch.Tensor | None
    metadata: torch.Tensor
    cache_keys: torch.Tensor
    num_tokens: torch.Tensor
    block_tables: torch.Tensor
    temps: torch.Tensor  # .view(torch.int32).to(torch.int64)
    # extend_activations holds the concatenation of the per-request extend activations
    # followed by the recovery activation (the target activation for the last accepted
    # token, just before the recovery token). For request i, the recovery activation
    # lives at extend_activations[i, extend_counts[i]] (extend_counts excludes recovery).
    extend_activations: torch.Tensor | None
    extend_counts: torch.Tensor | None
    extend_token_ids: torch.Tensor | None

    @classmethod
    def prepare(
        cls,
        batch_size: int,
        lookahead: int,
        max_blocks: int,
        vocab_size: int,
        draft_dtype: torch.dtype,
        device: torch.device,
        eagle: bool = False,
        eagle_act_dim: int = 0,
        tokenizer: AutoTokenizer = None,
    ):
        speculation_request = cls(*([None] * 9))
        speculation_request.batch_size = batch_size
        speculation_request.lookahead = lookahead
        speculation_request.max_blocks = max_blocks
        speculation_request.vocab_size = vocab_size
        speculation_request.draft_dtype = draft_dtype
        speculation_request.eagle = eagle
        speculation_request.eagle_act_dim = eagle_act_dim
        speculation_request.device = device
        speculation_request.tokenizer = tokenizer
        speculation_request._alloc_buffers()
        return speculation_request

    def _alloc_buffers(self):
        B, K = self.batch_size, self.lookahead
        self.cmd = torch.tensor([COMMAND.SPECULATION], dtype=torch.int64, device=self.device)
        self.metadata = torch.tensor([B, K, self.max_blocks, self.eagle_act_dim, self.vocab_size], dtype=torch.int64, device=self.device)
        self.cache_keys = torch.empty(B, 3, dtype=torch.int64, device=self.device)
        self.num_tokens = torch.empty(B, dtype=torch.int64, device=self.device)
        self.temps = torch.zeros(B, dtype=torch.float32, device=self.device)
        if self.max_blocks > 0:
            self.block_tables = torch.full((B, self.max_blocks), -1, dtype=torch.int32, device=self.device)
        else:
            self.block_tables = None
        if self.eagle:
            # K extend slots + 1 recovery slot (recovery sits at index extend_counts[i]).
            self.extend_activations = torch.empty(B, K + 1, self.eagle_act_dim, dtype=self.draft_dtype, device=self.device)
            self.extend_counts = torch.zeros(B, dtype=torch.int64, device=self.device)
            self.extend_token_ids = torch.empty(B, K + 1, dtype=torch.int64, device=self.device)
        else:
            self.extend_activations = None
            self.extend_counts = None
            self.extend_token_ids = None

    def maybe_update_buffers(self, batch_size: int, max_blocks: int = -1):
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            if max_blocks > 0:
                self.max_blocks = max_blocks
            self._alloc_buffers()

    def send(self, async_pg: dist.ProcessGroup, draft_rank: int):
        send_tensor(self.cmd, async_pg, draft_rank, name="cmd", prefix="TARGET:SpeculationRequest.send")
        send_tensor(self.metadata, async_pg, draft_rank, name="metadata", prefix="TARGET:SpeculationRequest.send")
        # Fuse all payload fields (including EAGLE) into a single NCCL send
        int64_parts = [
            self.cache_keys.reshape(-1),
            self.num_tokens.reshape(-1),
            self.block_tables.to(torch.int64).reshape(-1),
            self.temps.view(torch.int32).to(torch.int64).reshape(-1),
        ]
        if self.eagle:
            int64_parts.extend([
                self.extend_counts.reshape(-1),
                self.extend_activations.contiguous().reshape(-1).view(torch.int64),
                self.extend_token_ids.reshape(-1),
            ])
        fused_payload = torch.cat(int64_parts)
        send_tensor(fused_payload, async_pg, draft_rank, name="fused payload", prefix="TARGET:SpeculationRequest.send")

    @classmethod
    def receive(
        cls,
        async_pg: dist.ProcessGroup,
        target_rank: int,
        device: torch.device,
        draft_dtype: torch.dtype,
        tokenizer: AutoTokenizer = None,
        verbose: bool = False,
    ):
        meta = torch.empty(5, dtype=torch.int64, device=device)
        meta = receive_tensor(meta, async_pg, target_rank, name="metadata", prefix="DRAFT:SpeculationRequest.receive")
        B, K, max_blocks, eagle_act_dim, vocab_size = meta.tolist()
        if NCCL_LOG:
            print(f"[{_ts()}] [NCCL_LOG DRAFT_RECV_SPEC] SPECULATION REQUEST META RECEIVED, B={B}, K={K}, max_blocks={max_blocks}", flush=True)

        eagle = eagle_act_dim > 0
        speculation_request = cls.prepare(
            batch_size=B,
            lookahead=K,
            max_blocks=max_blocks,
            vocab_size=vocab_size,
            draft_dtype=draft_dtype,
            device=device,
            eagle=eagle,
            eagle_act_dim=eagle_act_dim,
            tokenizer=tokenizer,
        )

        # Receive all payload (including EAGLE tensors) in one fused int64 burst
        _dsz = torch.finfo(draft_dtype).bits // 8 if eagle else 0  # draft dtype element size
        fused_total = (3 * B) + B + (B * max_blocks) + B  # cache_keys + num_tokens + block_tables + temps
        if eagle:
            fused_total += B                                          # extend_counts
            fused_total += B * (K + 1) * eagle_act_dim * _dsz // 8    # extend_activations (K extends + recovery)
            fused_total += B * (K + 1)                                # extend_token_ids (K extends + recovery)
        fused_req = torch.empty(fused_total, dtype=torch.int64, device=device)
        fused_req = receive_tensor(fused_req, async_pg, target_rank, name="fused payload", prefix="DRAFT:SpeculationRequest.receive")
        off = 0
        speculation_request.cache_keys = fused_req[off:off + (3 * B)].view(B, 3)
        off += 3 * B
        speculation_request.num_tokens = fused_req[off:off + B].to(torch.int64)
        off += B
        speculation_request.block_tables = fused_req[off:off + B * max_blocks].view(B, max_blocks).to(torch.int32)
        off += B * max_blocks
        temps_as_int64 = fused_req[off:off + B]
        off += B
        speculation_request.temps = temps_as_int64.to(torch.int32).view(torch.float32)
        if eagle:
            speculation_request.extend_counts = fused_req[off:off + B]
            off += B
            n_ext = B * (K + 1) * eagle_act_dim * _dsz // 8
            speculation_request.extend_activations = fused_req[off:off + n_ext].view(draft_dtype).view(B, K + 1, eagle_act_dim)
            off += n_ext
            speculation_request.extend_token_ids = fused_req[off:off + B * (K + 1)].view(B, K + 1)
            off += B * (K + 1)
        assert off == fused_total

        cache_keys, draft_block_tables, temperatures, num_tokens = (
            speculation_request.cache_keys, speculation_request.block_tables, speculation_request.temps, speculation_request.num_tokens
        )
        if NCCL_LOG:
            sep = '=' * 80
            print(f"[{_ts()}] \n{sep}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG DRAFT_RECV_SPEC] meta=[B={B}, K={K}]", flush=True)
            print(f"[{_ts()}] [NCCL_LOG DRAFT_RECV_SPEC] cache_keys shape={cache_keys.shape}", flush=True)
            for i in range(B):
                seq_id, accept_len, verified_id = cache_keys[i].tolist()
                if tokenizer is not None:
                    verified_text = f" (f'{tokenizer.decode([int(verified_id)])}')"
                else:
                    verified_text = ""
                print(f"[{_ts()}]   req[{i}]: seq_id={seq_id}, accept_len={accept_len}, verified_id={int(verified_id)}{verified_text}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG DRAFT_RECV_SPEC] num_tokens={num_tokens.tolist()}", flush=True)
            draft_block_table_values_str = compress_neg_ones_and_zeros(f"{draft_block_tables.tolist()}")
            print(f"[{_ts()}] [NCCL_LOG DRAFT_RECV_SPEC] draft_block_tables shape={draft_block_tables.shape}, values={draft_block_table_values_str}", flush=True)
            print(f"[{_ts()}] [NCCL_LOG DRAFT_RECV_SPEC] temperatures={temperatures.tolist()}", flush=True)
            print(f"[{_ts()}] {sep}\n", flush=True)

        if eagle and verbose:
            extend_counts = speculation_request.extend_counts
            extend_eagle_acts = speculation_request.extend_activations
            extend_token_ids = speculation_request.extend_token_ids
            print(f"[{_ts()}] [CACHE REQUEST] extend_counts.shape={extend_counts.shape}, {extend_counts.tolist()}", flush=True)
            print(f"[{_ts()}] [CACHE REQUEST] extend_eagle_acts.shape={extend_eagle_acts.shape} (last slot per request is recovery activation)", flush=True)
            print(f"[{_ts()}] [CACHE REQUEST] extend_token_ids.shape={extend_token_ids.shape}, {extend_token_ids.tolist()}", flush=True)
            recovery_tokens_target = cache_keys[:, 2].clone()
            print(f"[{_ts()}] \n{'='*80}", flush=True)
            print(f"[{_ts()}] [CACHE REQUEST] Batch size: {B}, Spec depth: {K}", flush=True)
            for i in range(B):
                seq_id = cache_keys[i, 0].item()
                keep_idx = cache_keys[i, 1].item()
                rec_token_target = recovery_tokens_target[i].item()
                if tokenizer is not None:
                    rec_token_text = f" (f'{tokenizer.decode([rec_token_target])}')"
                else:
                    rec_token_text = ""
                n_ext = extend_counts[i].item()
                print(f"[{_ts()}]   Seq {seq_id}: keep_idx={keep_idx}, recovery_token={rec_token_target}{rec_token_text}, n_ext={n_ext}", flush=True)
            print(f"[{_ts()}] {'='*80}\n", flush=True)

        if BRIEF_LOG:
            cache_keys = speculation_request.cache_keys
            num_tokens = speculation_request.num_tokens
            # block_tables = speculation_request.block_tables
            # temps = speculation_request.temps
            extend_activations = speculation_request.extend_activations
            extend_counts = speculation_request.extend_counts
            extend_token_ids = speculation_request.extend_token_ids
            print(f"[{_ts()}] [SpeculationRequest.receive] {B=}, {K=}, {max_blocks=}, {eagle_act_dim=}", flush=True)
            for i in range(B):
                seq_id, accept_len, verified_id = cache_keys[i].tolist()
                verified_text = _decode_ids(verified_id, tokenizer)
                # print(f"[{_ts()}]      req[{i}]: seq_id={seq_id}, accept_len={accept_len}, verified_id={int(verified_id)} ({verified_text})", flush=True)
                print(f"[{_ts()}]      req[{i}]: ACCEPT_LENGTH={accept_len}, VERIFIED_TEXT={verified_text}", flush=True)
                if eagle:
                    print(f"[{_ts()}]      req[{i}]: extend_activations shape={extend_activations.shape}, values[i, :, :3]={list_to_str(extend_activations[i, :, :3].tolist())}", flush=True)
                    num_extend = extend_counts[i].item()
                    print(f"[{_ts()}]      req[{i}]: extend_counts shape={extend_counts.shape}, values[i]={num_extend}", flush=True)
                    decoded_extend_token_ids = _decode_ids(extend_token_ids[i, :num_extend], tokenizer)
                    print(f"[{_ts()}]      req[{i}]: extend_token_ids shape={extend_token_ids.shape}, values={extend_token_ids[i].tolist()}, decoded[:, :{num_extend}]='{decoded_extend_token_ids}'", flush=True)

        speculation_request.dump()
        return speculation_request

    def dump(self):
        dump_dir = os.environ.get("SSD_DUMP_TENSORS_DIR", "")
        if dump_dir:
            torch.save({
                'metadata': self.metadata.cpu(),
                'cache_keys': self.cache_keys.cpu(),
                'num_tokens': self.num_tokens.cpu(),
                'block_tables': self.block_tables.cpu() if self.block_tables is not None else None,
                'temps': self.temps.cpu(),
                'extend_activations': self.extend_activations.cpu() if self.extend_activations is not None else None,
                'extend_counts': self.extend_counts.cpu() if self.extend_counts is not None else None,
                'extend_token_ids': self.extend_token_ids.cpu() if self.extend_token_ids is not None else None,
            }, f"{dump_dir}/speculation_request_{_dump_ts()}.pt")


@dataclass
class SpeculationResponse:
    speculations: torch.Tensor
    logits_q: torch.Tensor | None
    cache_hits: torch.Tensor | None

    @classmethod
    def prepare(
        cls,
        lookahead: int,
        device: torch.device,
        draft_dtype: torch.dtype = torch.bfloat16,
        batch_size: int = 1,
        vocab_size: int = -1,
        communicate_logits: bool = False,
        communicate_cache_hits: bool = False,
        tokenizer: AutoTokenizer = None,
    ):
        response = cls(
            speculations=None,
            logits_q=None,
            cache_hits=None,
        )
        response.batch_size = batch_size
        response.lookahead = lookahead
        response.draft_dtype = draft_dtype
        response.device = device
        response.vocab_size = vocab_size
        response.communicate_logits = communicate_logits
        response.communicate_cache_hits = communicate_cache_hits
        response.tokenizer = tokenizer
        if response.communicate_logits:
            assert response.vocab_size > 0, "vocab_size must be set when communicate_logits is True"
        response._alloc_buffers()
        return response

    def _alloc_buffers(self):
        self.speculations = torch.empty(self.batch_size, self.lookahead, dtype=torch.int64, device=self.device)
        if getattr(self, 'communicate_logits', False):
            self.logits_q = torch.empty(self.batch_size, self.lookahead, self.vocab_size, dtype=self.draft_dtype, device=self.device)
        if getattr(self, 'communicate_cache_hits', False):
            self.cache_hits = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)

    def maybe_update_buffers(self, batch_size: int = -1):
        if batch_size > 0 and batch_size != self.batch_size:
            self.batch_size = batch_size
            self._alloc_buffers()

    def send(self, async_pg: dist.ProcessGroup, target_rank: int, tokenizer: AutoTokenizer = None):
        send_tensor(self.speculations, async_pg, target_rank, name="speculations", prefix="DRAFT:SpeculationResponse.send")

        if BRIEF_LOG:
            decoded_speculations = _decode_ids(self.speculations, tokenizer)
            print(f"[{_ts()}] [SpeculationResponse.send] SPECULATION: '{decoded_speculations}'", flush=True)
            print(f"[{_ts()}] {'='*80}\n", flush=True)

        if self.logits_q is not None:
            assert getattr(self, 'communicate_logits', True), "logits_q is not None but communicate_logits is False"
            send_tensor(self.logits_q, async_pg, target_rank, name="logits", prefix="DRAFT:SpeculationResponse.send")
        if self.cache_hits is not None:
            assert getattr(self, 'communicate_cache_hits', True), "cache_hits is not None but communicate_cache_hits is False"
            send_tensor(self.cache_hits, async_pg, target_rank, name="cache hits", prefix="DRAFT:SpeculationResponse.send")

        self.dump()

    def dump(self):
        dump_dir = os.environ.get("SSD_DUMP_TENSORS_DIR", "")
        if dump_dir:
            torch.save({
                'speculations': self.speculations.cpu(),
                'logits': self.logits_q.cpu() if self.logits_q is not None else None,
                'cache_hits': self.cache_hits.cpu() if self.cache_hits is not None else None,
            }, f"{dump_dir}/speculation_response_{_dump_ts()}.pt")

    @classmethod
    def receive(
        cls,
        async_pg: dist.ProcessGroup,
        draft_rank: int,
        batch_size: int,
        lookahead: int,
        device: torch.device,
        draft_dtype: torch.dtype = torch.bfloat16,
        receive_logits: bool = False,
        receive_cache_hits: bool = False,
        vocab_size: int = -1,
        tokenizer: AutoTokenizer = None,
    ):
        speculation_response = cls.prepare(
            batch_size=batch_size,
            lookahead=lookahead,
            device=device,
            draft_dtype=draft_dtype,
            communicate_logits=receive_logits,
            communicate_cache_hits=receive_cache_hits,
            vocab_size=vocab_size,
            tokenizer=tokenizer,
        )
        speculation_response.receive(async_pg, draft_rank, batch_size=batch_size)
        return speculation_response

    def receive(self, async_pg: dist.ProcessGroup, draft_rank: int, batch_size: int=-1):
        self.maybe_update_buffers(batch_size=batch_size)
        self.speculations = receive_tensor(self.speculations, async_pg, draft_rank, name="speculations", prefix="TARGET:SpeculationResponse.receive")
        if self.communicate_logits:
            self.logits_q = receive_tensor(self.logits_q, async_pg, draft_rank, name="logits", prefix="TARGET:SpeculationResponse.receive")
        if self.communicate_cache_hits:
            self.cache_hits = receive_tensor(self.cache_hits, async_pg, draft_rank, name="cache hits", prefix="TARGET:SpeculationResponse.receive")


def _decode_ids(ids_tensor, tokenizer: AutoTokenizer = None):
    if tokenizer is None:
        return "<no tokenizer>"
    if isinstance(ids_tensor, int):
        ids = [ids_tensor]
    else:
        ids = ids_tensor.cpu().tolist()
        if isinstance(ids, int):
            ids = [ids]
    return tokenizer.decode(ids)


def concat_tensors_as_int64(*tensors: torch.Tensor) -> torch.Tensor:
    """Concatenate tensors into a single flat int64 payload."""
    parts = []
    for t in tensors:
        if t is None:
            continue
        if t.dtype != torch.int64:
            t = t.to(torch.int64)
        parts.append(t.reshape(-1))
    if not parts:
        return torch.empty(0, dtype=torch.int64)
    return torch.cat(parts, dim=0)


def receive_tensor(
    tensor: torch.Tensor,
    async_pg: dist.ProcessGroup,
    draft_runner_rank: int,
    name: str = "",
    prefix: str = "",
    print_shape: bool = True,
    print_values: bool = False,
) -> torch.Tensor:
    prefix = f"[{prefix:>35}]" if prefix else ""
    if NCCL_LOG:
        tensor_str = f"{name:>30}" if name else ""
        if print_shape:
            tensor_str += (", " if tensor_str else "") + f"shape={tensor.shape}"
        print(f"[{_ts()}][NCCL:START_RECEIVE_TENSOR]{prefix} {tensor_str}", flush=True)
    
    dist.recv(tensor, src=draft_runner_rank, group=async_pg)

    if NCCL_LOG:
        if print_values:
            tensor_str += (", " if tensor_str else "") + f"values={tensor.tolist()}"
        print(f"[{_ts()}][NCCL:  END_RECEIVE_TENSOR]{prefix} {tensor_str}", flush=True)

    return tensor


def send_tensor(
    tensor: torch.Tensor,
    async_pg: dist.ProcessGroup,
    draft_runner_rank: int,
    name: str = "",
    prefix: str = "",
    print_shape: bool = True,
    print_values: bool = False,
) -> None:
    prefix = f"[{prefix:>35}]" if prefix else ""
    if NCCL_LOG:
        tensor_str = f"{name:>30}" if name else ""
        if print_shape:
            tensor_str += (", " if tensor_str else "") + f"shape={tensor.shape}"
        print(f"[{_ts()}][NCCL:   START_SEND_TENSOR]{prefix} {tensor_str}", flush=True)

    dist.send(tensor, dst=draft_runner_rank, group=async_pg)

    if NCCL_LOG:
        if print_values:
            tensor_str += (", " if tensor_str else "") + f"values={tensor.tolist()}"
        print(f"[{_ts()}][NCCL:     END_SEND_TENSOR]{prefix} {tensor_str}", flush=True)


def prepare_decode_tensors_from_seqs(
    seqs: list[Sequence],
    block_size: int,
    is_draft: bool,
    verify: bool = False,
    k: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []

    if not verify:  # normal decoding or draft fwd in speculation
        assert k == -1, "k should be -1 for normal decoding or draft fwd in speculation"
        for seq in seqs:
            block_table = seq.draft_block_table if is_draft else seq.block_table
            assert len(seq) // block_size <= len(block_table), "in sync spec draft decode, not enough blocks allocated"
            num_cached_tokens = seq.num_cached_tokens if not is_draft else seq.num_draft_cached_tokens
            assert num_cached_tokens == len(seq) - 1, "num_cached_tokens should be equal to len(seq) - 1 in pure sq decode path"
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))

            pos = seq.num_tokens - 1
            block_idx = pos // block_size
            pos_in_block = pos % block_size
            slot_mapping.append(block_table[block_idx] * block_size + pos_in_block)
    else:  # verify and glue decode prep both go here
        assert not is_draft, "verify path only supported for target model" # we prep tensors to send to draft for glue on the target 
        assert k > 0, "k should be > 0 for target fwd in verify"

        for seq_idx, seq in enumerate(seqs):
            # can hardcode block_table here for target since this is only target codepath 
            assert (seq.num_tokens - 1) // block_size <= len(seq.block_table), "in sync spec target verify, not enough blocks allocated"
            
            pos0 = seq.num_tokens - (k+1)
            input_ids.extend(seq[pos0:])
            positions.extend(list(range(pos0, pos0 + k + 1)))
            assert seq.num_cached_tokens == pos0, f"num_cached_tokens={seq.num_cached_tokens} != pos0={pos0} (num_tokens={seq.num_tokens}, k={k})"
            context_lens.append(len(seq))  

            for j in range(k + 1):
                pos = pos0 + j
                block_idx = pos // block_size
                block_id = seq.block_table[block_idx]
                pos_in_block = pos % block_size
                slot_mapping.append(
                    block_id * block_size + pos_in_block)


    input_ids = torch.tensor(
        input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(
        positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    slot_mapping = torch.tensor(
        slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    context_lens = torch.tensor(
        context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    return input_ids, positions, slot_mapping, context_lens

def prepare_block_tables_from_seqs(
    seqs: list[Sequence],
    is_draft: bool = False
) -> torch.Tensor:
        if is_draft:
            max_len = max(len(seq.draft_block_table) for seq in seqs)
            block_tables = [seq.draft_block_table + [-1] * (max_len - len(seq.draft_block_table)) for seq in seqs]
        else:
            max_len = max(len(seq.block_table) for seq in seqs)
            block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

def prepare_prefill_tensors_from_seqs(
    seqs: list[Sequence],
    block_size: int,
    is_draft: bool = False,
    skip_first_token: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert skip_first_token in (0, 1)
    input_ids = []
    positions = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []
    
    for seq in seqs:
        seqlen = len(seq)
        if is_draft:
            num_cached_tokens = seq.num_draft_cached_tokens
            block_table = seq.draft_block_table
        else:
            num_cached_tokens = seq.num_cached_tokens
            block_table = seq.block_table

        start = num_cached_tokens + (skip_first_token if is_draft else 0)
        input_ids.extend(seq[start:])
        pos_offset = -skip_first_token if is_draft else 0
        positions.extend(list(range(start + pos_offset, seqlen + pos_offset)))
        seqlen_q = seqlen - start
        seqlen_k = seqlen + pos_offset
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)

        if not block_table:  # first prefill
            continue

        # new: emit exactly one slot for each *new* token
        #    map each token index -> (block_id * block_size + offset)
        for pos in range(start + pos_offset, seq.num_tokens + pos_offset):
            block_i = pos // block_size
            offset = pos % block_size
            slot = block_table[block_i] * block_size + offset
            slot_mapping.append(slot)

    input_ids = torch.tensor(
        input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(
        positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    cu_seqlens_q = torch.tensor(
        cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    cu_seqlens_k = torch.tensor(
        cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    slot_mapping = torch.tensor(
        slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    
    return input_ids, positions, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping
