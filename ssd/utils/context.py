from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    is_jit: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, is_jit=False):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, is_jit, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def context_to_dict(ctx: Context) -> dict:
    """Serialize a Context to a plain dict (picklable for SHM broadcast)."""
    return {
        'is_prefill': ctx.is_prefill,
        'is_jit': ctx.is_jit,
        'cu_seqlens_q': ctx.cu_seqlens_q,
        'cu_seqlens_k': ctx.cu_seqlens_k,
        'max_seqlen_q': ctx.max_seqlen_q,
        'max_seqlen_k': ctx.max_seqlen_k,
        'slot_mapping': ctx.slot_mapping,
        'context_lens': ctx.context_lens,
        'block_tables': ctx.block_tables,
    }

def set_context_from_dict(d: dict):
    """Restore context from a dict produced by context_to_dict."""
    set_context(
        is_prefill=d['is_prefill'],
        cu_seqlens_q=d.get('cu_seqlens_q'),
        cu_seqlens_k=d.get('cu_seqlens_k'),
        max_seqlen_q=d.get('max_seqlen_q', 0),
        max_seqlen_k=d.get('max_seqlen_k', 0),
        slot_mapping=d.get('slot_mapping'),
        context_lens=d.get('context_lens'),
        block_tables=d.get('block_tables'),
        is_jit=d.get('is_jit', False),
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
