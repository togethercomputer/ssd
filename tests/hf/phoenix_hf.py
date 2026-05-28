from __future__ import annotations

import glob
import os

import torch
import torch.nn.functional as F
from torch import nn
from safetensors.torch import load_file
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm


# ---------------------------------------------------------------------------
# Minimal from-scratch Phoenix model. Mirrors
# ssd/models/phoenix_draft_llama3.PhoenixLlamaForCausalLM at the math level:
#   concat(embeds, target_hidden) -> eh_proj (bias) -> N llama decoder layers
#   -> norm -> lm_head. Same RoPE convention as eagle3_hf (the SSD runtime
#   drops rope_scaling for llama3, so we do too).
# ---------------------------------------------------------------------------
class PhoenixAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nh = cfg.num_attention_heads
        self.nkh = cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // self.nh)
        self.scale = self.hd ** -0.5
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkh * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkh * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)
        self.rope_theta = getattr(cfg, "rope_theta", 10000.0)
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(self, positions, x):
        # x: [T, H, D]; positions: [T]. Matches HF Llama's halves-split RoPE.
        pos_f = positions.float()
        freqs = torch.outer(pos_f, self.inv_freq.to(pos_f.device))  # [T, D/2]
        cos = freqs.cos().unsqueeze(1)
        sin = freqs.sin().unsqueeze(1)
        d = x.shape[-1]
        half = d // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.to(x.dtype)

    def forward(self, positions, h):
        q = self.q_proj(h).view(-1, self.nh, self.hd)
        k = self.k_proj(h).view(-1, self.nkh, self.hd)
        v = self.v_proj(h).view(-1, self.nkh, self.hd)
        q = self._rope(positions, q)
        k = self._rope(positions, k)
        # Stash post-rotary K and V for per-step dumps (diagnostic only).
        self.last_k = k.detach().contiguous()
        self.last_v = v.detach().contiguous()
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            is_causal=True, scale=self.scale, enable_gqa=True,
        )
        o = o.squeeze(0).transpose(0, 1).contiguous().view(-1, self.nh * self.hd)
        return self.o_proj(o)


class PhoenixDecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = PhoenixAttention(cfg)
        self.mlp = LlamaMLP(cfg)
        self.input_layernorm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, positions, h):
        residual = h
        h = self.input_layernorm(h)
        h = self.self_attn(positions, h)
        h = h + residual
        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        return h + residual


class PhoenixModel(nn.Module):
    def __init__(self, cfg, d_model_target, device: str = "cuda"):
        super().__init__()
        self.config = cfg
        self.device = device
        self.d_model_target = d_model_target
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        # eh_proj has a bias term (unlike eagle3's fc).
        self.eh_proj = nn.Linear(d_model_target + cfg.hidden_size, cfg.hidden_size, bias=True)
        self.layers = nn.ModuleList([
            PhoenixDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids, target_hidden):
        # input_ids: [T]; target_hidden: [T, d_model_target].
        positions = torch.arange(input_ids.shape[0], device=input_ids.device)
        prenorm = self.forward_with_cond(input_ids, positions, target_hidden)
        final = self.norm(prenorm)
        return F.linear(final, self.lm_head.weight)  # [T, vocab]

    def forward_with_cond(self, input_ids, positions, cond):
        """Like forward() but returns prenorm (pre-final_norm) hidden states
        so callers can mix target-hidden and draft-hidden conditioning per-
        position. cond: [T, d_model_target] — the raw target (or recurrent
        draft) hidden stream, NOT pre-projected. Returns [T, hidden_size]."""
        embeds = self.embed_tokens(input_ids)
        h = torch.cat([embeds, cond.to(self.eh_proj.weight.dtype)], dim=-1)
        h = self.eh_proj(h)
        for layer in self.layers:
            h = layer(positions, h)
        return h


def load_phoenix_specforge(
    path: str, d_model_target: int, device: str = "cuda", dtype=torch.bfloat16,
) -> PhoenixModel:
    if not os.path.exists(os.path.join(path, "config.json")):
        hits = glob.glob(os.path.join(path, "snapshots", "*", "config.json"))
        assert hits, f"no config.json under {path}"
        path = os.path.dirname(hits[0])

    cfg = LlamaConfig.from_pretrained(path)
    model = PhoenixModel(cfg, d_model_target, device=device).to(dtype)

    sd = load_file(glob.glob(os.path.join(path, "*.safetensors"))[0])
    with torch.no_grad():
        model.eh_proj.weight.copy_(sd["eh_proj.weight"])
        model.eh_proj.bias.copy_(sd["eh_proj.bias"])
        model.norm.weight.copy_(sd["model.norm.weight"])
        model.lm_head.weight.copy_(sd["lm_head.weight"])
        model.embed_tokens.weight.copy_(sd["model.embed_tokens.weight"])
        for i, layer in enumerate(model.layers):
            prefix = f"model.layers.{i}"
            layer.self_attn.q_proj.weight.copy_(sd[f"{prefix}.self_attn.q_proj.weight"])
            layer.self_attn.k_proj.weight.copy_(sd[f"{prefix}.self_attn.k_proj.weight"])
            layer.self_attn.v_proj.weight.copy_(sd[f"{prefix}.self_attn.v_proj.weight"])
            layer.self_attn.o_proj.weight.copy_(sd[f"{prefix}.self_attn.o_proj.weight"])
            layer.mlp.gate_proj.weight.copy_(sd[f"{prefix}.mlp.gate_proj.weight"])
            layer.mlp.up_proj.weight.copy_(sd[f"{prefix}.mlp.up_proj.weight"])
            layer.mlp.down_proj.weight.copy_(sd[f"{prefix}.mlp.down_proj.weight"])
            layer.input_layernorm.weight.copy_(sd[f"{prefix}.input_layernorm.weight"])
            layer.post_attention_layernorm.weight.copy_(sd[f"{prefix}.post_attention_layernorm.weight"])
    return model.to(device, dtype=dtype)
