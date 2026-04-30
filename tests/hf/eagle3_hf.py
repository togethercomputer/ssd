from __future__ import annotations

import argparse
import glob
import os

import torch
import torch.nn.functional as F
from torch import nn
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm


EAGLE_LAYERS_LLAMA_8B = [2, 16, 29]   # set in ssd/config.py for L=32
D_MODEL_TARGET_LLAMA_8B = 4096


# ---------------------------------------------------------------------------
# Minimal from-scratch Eagle3 model. SpecForge keys land here cleanly.
# ---------------------------------------------------------------------------
class Eagle3Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nh = cfg.num_attention_heads
        self.nkh = cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // self.nh)
        self.scale = self.hd ** -0.5
        # qkv input dim is 2*hidden (concat of embeds and target_hidden, post-norm).
        in_dim = 2 * cfg.hidden_size
        self.q_proj = nn.Linear(in_dim, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(in_dim, self.nkh * self.hd, bias=False)
        self.v_proj = nn.Linear(in_dim, self.nkh * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)
        self.rope_theta = getattr(cfg, "rope_theta", 10000.0)
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(self, positions, x):
        # x: [T, H, D]; positions: [T]. Matches HF Llama's interleaved-pair RoPE.
        pos_f = positions.float()
        freqs = torch.outer(pos_f, self.inv_freq.to(pos_f.device))  # [T, D/2]
        cos = freqs.cos().unsqueeze(1)   # [T, 1, D/2]
        sin = freqs.sin().unsqueeze(1)
        # HF Llama's default RoPE: split the last dim into HALVES (not even/odd).
        d = x.shape[-1]
        half = d // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.to(x.dtype)

    def forward(self, positions, h):
        # h: [T, 2*hidden] (after concat+norms); positions: [T].
        q = self.q_proj(h).view(-1, self.nh, self.hd)
        k = self.k_proj(h).view(-1, self.nkh, self.hd)
        v = self.v_proj(h).view(-1, self.nkh, self.hd)
        q = self._rope(positions, q)
        k = self._rope(positions, k)
        # Stash post-rotary K and V for per-step dumps (diagnostic only).
        self.last_k = k.detach().contiguous()
        self.last_v = v.detach().contiguous()
        # SDPA: [B=1, H, T, D]
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            is_causal=True, scale=self.scale, enable_gqa=True,
        )
        o = o.squeeze(0).transpose(0, 1).contiguous().view(-1, self.nh * self.hd)
        return self.o_proj(o)


class Eagle3DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = Eagle3Attention(cfg)
        self.mlp = LlamaMLP(cfg)
        self.input_layernorm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.hidden_norm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, positions, embeds, target_h_proj):
        # Matches upstream sglang/llama_eagle3.py exactly.
        residual = target_h_proj
        embeds_n = self.input_layernorm(embeds)
        hidden_n = self.hidden_norm(target_h_proj)
        combined = torch.cat([embeds_n, hidden_n], dim=-1)
        attn_out = self.self_attn(positions, combined)
        # Fused add+norm equivalent: return (mlp(norm(attn+res)), attn+res).
        new_res = attn_out + residual
        normed = self.post_attention_layernorm(new_res)
        mlp_out = self.mlp(normed)
        return mlp_out + new_res  # the "prenorm" sum used for final_norm


class Eagle3Model(nn.Module):
    def __init__(self, cfg, d_model_target, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.fc = nn.Linear(3 * d_model_target, cfg.hidden_size, bias=False)
        self.midlayer = Eagle3DecoderLayer(cfg)
        self.norm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.draft_vocab_size, bias=False)
        self.register_buffer(
            "d2t", torch.zeros(cfg.draft_vocab_size, dtype=torch.long), persistent=False,
        )

    def forward(self, input_ids, target_hidden):
        # input_ids: [T]; target_hidden: [T, 3*D_target].
        embeds = self.embed_tokens(input_ids)
        target_h_proj = self.fc(target_hidden.to(self.fc.weight.dtype))
        positions = torch.arange(input_ids.shape[0], device=input_ids.device)
        prenorm = self.midlayer(positions, embeds, target_h_proj)
        final = self.norm(prenorm)
        return F.linear(final, self.lm_head.weight)   # [T, draft_vocab]

    def forward_with_cond(self, input_ids, positions, cond):
        """Like forward() but takes a pre-projected conditioning stream
        (shape [T, hidden_size]) so callers can mix target-hidden and
        draft-hidden conditioning per-position. Returns prenorm (pre-
        final_norm hidden states)."""
        embeds = self.embed_tokens(input_ids)
        return self.midlayer(positions, embeds, cond)

    def draft_tok_to_target(self, draft_idx: int) -> int:
        return int(draft_idx) + int(self.d2t[draft_idx].item())


def load_eagle3_specforge(
    path: str, target_embed: torch.Tensor, d_model_target: int, device: str = "cuda", dtype=torch.bfloat16,
) -> Eagle3Model:
    if not os.path.exists(os.path.join(path, "config.json")):
        hits = glob.glob(os.path.join(path, "snapshots", "*", "config.json"))
        assert hits, f"no config.json under {path}"
        path = os.path.dirname(hits[0])

    cfg = LlamaConfig.from_pretrained(path)
    model = Eagle3Model(cfg, d_model_target, device=device).to(dtype)

    sd = load_file(glob.glob(os.path.join(path, "*.safetensors"))[0])
    with torch.no_grad():
        model.d2t.copy_(sd["d2t"].long())
        model.fc.weight.copy_(sd["fc.weight"])
        model.norm.weight.copy_(sd["norm.weight"])
        model.lm_head.weight.copy_(sd["lm_head.weight"])
        ml = model.midlayer
        ml.self_attn.q_proj.weight.copy_(sd["midlayer.self_attn.q_proj.weight"])
        ml.self_attn.k_proj.weight.copy_(sd["midlayer.self_attn.k_proj.weight"])
        ml.self_attn.v_proj.weight.copy_(sd["midlayer.self_attn.v_proj.weight"])
        ml.self_attn.o_proj.weight.copy_(sd["midlayer.self_attn.o_proj.weight"])
        ml.mlp.gate_proj.weight.copy_(sd["midlayer.mlp.gate_proj.weight"])
        ml.mlp.up_proj.weight.copy_(sd["midlayer.mlp.up_proj.weight"])
        ml.mlp.down_proj.weight.copy_(sd["midlayer.mlp.down_proj.weight"])
        ml.input_layernorm.weight.copy_(sd["midlayer.input_layernorm.weight"])
        ml.hidden_norm.weight.copy_(sd["midlayer.hidden_norm.weight"])
        ml.post_attention_layernorm.weight.copy_(sd["midlayer.post_attention_layernorm.weight"])
        # embed_tokens is shared with the target.
        model.embed_tokens.weight.copy_(target_embed.to(dtype))
    return model.to(device, dtype=dtype)
