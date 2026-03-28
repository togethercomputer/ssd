"""Tree decode mask for FA4 via score_mod + aux_tensors.

The tree mask is stored as a dense float32 bias tensor of shape
(max_total_q, max_kv_stride), flattened to 1D. Unmasked positions have
value 0.0; masked positions have a large negative value (-1e6).

score_mod adds the bias to each attention score, effectively masking out
positions where the bias is -1e6.
"""

import torch
import numpy as np
import cutlass
import cutlass.cute as cute

# Large negative value used to mask attention scores.
_MASK_VAL = -1.0e6


def create_tree_score_mod(max_kv_stride: int):
    """Return a @cute.jit score_mod that reads a mask bias from aux_tensors[0].

    The aux_tensor is a 1D float32 tensor indexed by:
        (offset_q + q_idx) * max_kv_stride + kv_idx

    where offset_q comes from seqlen_info for varlen sequences.
    """

    @cute.jit
    def tree_score_mod(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        mask_bias = aux_tensors[0]
        dtype = mask_bias.element_type
        global_q = seqlen_info.offset_q + q_idx
        flat_idx = global_q * max_kv_stride + kv_idx
        idx_frag = cute.make_rmem_tensor(1, cutlass.Int32)
        idx_frag.store(flat_idx)
        val_frag = cute.make_rmem_tensor(1, dtype)
        val_frag[0] = mask_bias[idx_frag[0]]
        bias = (val_frag.load()).to(cutlass.Float32)
        return tSrS_ssa + bias

    return tree_score_mod


def build_tree_mask_bias(
    context_lens: torch.Tensor,
    step: int,
    K: int,
    MQ_LEN: int,
    fan_out_list: list[int],
    fan_out_list_miss: list[int],
    cache_hits: torch.Tensor,
    max_kv_stride: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the dense mask bias tensor for one tree decode step.

    Returns a 1D float32 tensor of shape (B * MQ_LEN * max_kv_stride,)
    with 0.0 for attend and _MASK_VAL for masked positions.
    """
    B = context_lens.shape[0]
    context_lens_list = context_lens.tolist()
    cache_hits_list = cache_hits[:B].tolist()

    # Pre-compute glue patterns
    tril = np.tril(np.ones((K + 1, K + 1), dtype=np.float32))
    fol = np.array(fan_out_list)
    fol_miss = np.array(fan_out_list_miss)
    glue_hit = np.repeat(tril, fol, axis=0)   # (MQ_LEN, K+1)
    glue_miss = np.repeat(tril, fol_miss, axis=0)

    ttl_added = (step + 1) * MQ_LEN + (K + 1)
    rows = np.arange(MQ_LEN)

    # Build mask as numpy, then convert
    bias = np.full((B * MQ_LEN, max_kv_stride), _MASK_VAL, dtype=np.float32)

    for b in range(B):
        cols_b = int(context_lens_list[b])
        prefix_len_b = cols_b - ttl_added
        row_offset = b * MQ_LEN

        # Prefix: attend to all
        if prefix_len_b > 0:
            bias[row_offset:row_offset + MQ_LEN, :prefix_len_b] = 0.0

        # Glue pattern
        glue = glue_hit if int(cache_hits_list[b]) == 1 else glue_miss
        glue_start = prefix_len_b
        glue_bias = np.where(glue > 0, 0.0, _MASK_VAL).astype(np.float32)
        bias[row_offset:row_offset + MQ_LEN, glue_start:glue_start + K + 1] = glue_bias

        # Diagonal blocks
        diag_start = prefix_len_b + K + 1
        for blk in range(step + 1):
            col_indices = diag_start + blk * MQ_LEN + rows
            valid = col_indices < max_kv_stride
            bias[row_offset + rows[valid], col_indices[valid]] = 0.0

    return torch.from_numpy(bias.reshape(-1)).to(device, non_blocking=True)
