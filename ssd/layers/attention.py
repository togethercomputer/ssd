import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn.cute.interface import flash_attn_varlen_func as fa4_varlen_func
from ssd.layers.tree_mask import create_tree_score_mod
from ssd.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot.to(tl.int64) * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        draft: bool = False,
        speculate: bool = False,
        draft_async: bool = False,
        use_eagle: bool = False,
        F: int = 1,
        K: int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.draft = draft
        self.speculate = speculate
        self.draft_async = draft_async
        self.use_eagle = use_eagle
        self.F = F # async_fan_out
        self.K = K # speculate_k
        self.max_seqlen_k = 0  # set during KV cache allocation to config.max_model_len
        self.tree_score_mod = None  # set during KV cache allocation

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        k_cache, v_cache = self.k_cache, self.v_cache

        context = get_context()
        if self.k_cache.numel() and self.v_cache.numel():
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache

            k, v = k.view(-1, self.num_kv_heads, self.head_dim), v.view(-1, self.num_kv_heads, self.head_dim)
            o, _ = fa4_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True)
        else:
            # verify/glue decode: multi-query with cu_seqlens_q (K+1 or variable per seq)
            verify_or_glue = (
                self.speculate and context.cu_seqlens_q is not None
            )
            decode = not verify_or_glue
            tree_decode = (
                decode and self.speculate and self.draft and self.draft_async
                and not context.is_jit
            )

            if verify_or_glue:
                assert context.context_lens is not None
                o, _ = fa4_varlen_func(q, k_cache, v_cache,
                                        cu_seqlens_q=context.cu_seqlens_q,
                                        cu_seqlens_k=None,
                                        max_seqlen_q=context.max_seqlen_q,
                                        max_seqlen_k=self.max_seqlen_k,
                                        seqused_k=context.context_lens,
                                        page_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True,
                                        )

            elif tree_decode:
                score_mod_kwargs = {}
                if self.tree_score_mod is not None and context.tree_mask_bias is not None:
                    score_mod_kwargs["score_mod"] = self.tree_score_mod
                    score_mod_kwargs["aux_tensors"] = [context.tree_mask_bias]
                o, _ = fa4_varlen_func(
                    q,
                    self.k_cache,
                    self.v_cache,
                    cu_seqlens_q=context.tree_cu_seqlens_q,
                    cu_seqlens_k=None,
                    max_seqlen_q=self.F * (self.K + 1),
                    max_seqlen_k=self.max_seqlen_k,
                    seqused_k=context.context_lens,
                    page_table=context.block_tables,
                    softmax_scale=self.scale,
                    causal=False,
                    **score_mod_kwargs,
                )
            else: # single query decode
                batch_size = context.context_lens.shape[0]
                cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=q.device)
                o, _ = fa4_varlen_func(q, k_cache, v_cache,
                                            cu_seqlens_q=cu_seqlens_q,
                                            cu_seqlens_k=None,
                                            max_seqlen_q=1,
                                            max_seqlen_k=self.max_seqlen_k,
                                            seqused_k=context.context_lens,
                                            page_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True,
                                            )

        o = o.view(-1, self.num_heads * self.head_dim)
        return o
