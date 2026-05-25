"""MatNet encoder + POMO decoder for asymmetric VRPTW.

Reference: Kwon et al. 2021 (MatNet, NeurIPS) for the asymmetric-edge
encoder; Kwon et al. 2020 (POMO) for the multi-start decoder.

Designed for an 8 GB GPU (RTX 3070 Ti laptop): at N=200, K=32 parallel
rollouts, dim=128, 4 encoder layers, batch=16 → ~3-4 GB peak per the
research brief.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class POMOConfig:
    node_feat_dim: int = 6        # x, y, demand, ready, due, service
    edge_feat_dim: int = 2        # T[i,j], T[j,i]
    embed_dim: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 4
    ff_dim: int = 512
    decoder_glimpse_heads: int = 8
    decoder_clip: float = 10.0
    dropout: float = 0.0


class _MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_o = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        B, Lq, _ = q.shape
        Lk = kv.shape[1]
        Q = self.W_q(q).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(kv).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(kv).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, Lq, self.dim)
        return self.W_o(out)


class MatNetEncoderLayer(nn.Module):
    """One MatNet encoder block.  Dual-graph attention: node embeddings
    are mixed with the asymmetric edge-feature matrix via per-head mixers."""

    def __init__(self, cfg: POMOConfig):
        super().__init__()
        self.cfg = cfg
        self.attn = _MultiHeadAttention(cfg.embed_dim, cfg.n_heads, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.ff_dim),
            nn.ReLU(),
            nn.Linear(cfg.ff_dim, cfg.embed_dim),
        )
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        # Edge-feature mixer: maps (n_heads, edge_feat_dim) -> (n_heads,)
        # per (i, j) pair; multiplied into attention scores as a bias.
        self.edge_mixer = nn.Sequential(
            nn.Linear(cfg.edge_feat_dim, cfg.n_heads),
            nn.ReLU(),
            nn.Linear(cfg.n_heads, cfg.n_heads),
        )

    def forward(self, x: torch.Tensor, edge_feats: torch.Tensor) -> torch.Tensor:
        # edge_feats: (B, N, N, edge_feat_dim).  bias: (B, n_heads, N, N).
        bias = self.edge_mixer(edge_feats).permute(0, 3, 1, 2)
        B, N, _ = x.shape
        # Q,K,V from node embeddings (using the standard MHA), then add bias to
        # scores BEFORE softmax.  For simplicity we add bias by computing attn
        # ourselves rather than the MHA above.
        Q = self.attn.W_q(x).view(B, N, self.cfg.n_heads, -1).transpose(1, 2)
        K = self.attn.W_k(x).view(B, N, self.cfg.n_heads, -1).transpose(1, 2)
        V = self.attn.W_v(x).view(B, N, self.cfg.n_heads, -1).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))
        scores = scores + bias
        attn = F.softmax(scores, dim=-1)
        h = torch.matmul(attn, V).transpose(1, 2).reshape(B, N, self.cfg.embed_dim)
        h = self.attn.W_o(h)
        x = self.norm1(x + h)
        x = self.norm2(x + self.ff(x))
        return x


class MatNetEncoder(nn.Module):
    def __init__(self, cfg: POMOConfig):
        super().__init__()
        self.cfg = cfg
        self.node_embed = nn.Linear(cfg.node_feat_dim, cfg.embed_dim)
        self.layers = nn.ModuleList(
            [MatNetEncoderLayer(cfg) for _ in range(cfg.n_encoder_layers)]
        )

    def forward(self, node_feats: torch.Tensor, edge_feats: torch.Tensor) -> torch.Tensor:
        x = self.node_embed(node_feats)
        for layer in self.layers:
            x = layer(x, edge_feats)
        return x


class POMODecoder(nn.Module):
    """Single-step decoder: context (cur_node + global) glimpses + softmax."""

    def __init__(self, cfg: POMOConfig):
        super().__init__()
        self.cfg = cfg
        # Context: [graph_embed, cur_node_embed, remaining_cap_scalar, current_time_scalar]
        self.context_proj = nn.Linear(cfg.embed_dim * 2 + 2, cfg.embed_dim)
        self.glimpse = _MultiHeadAttention(cfg.embed_dim, cfg.decoder_glimpse_heads)
        self.compat = nn.Linear(cfg.embed_dim, cfg.embed_dim)

    def forward(self, node_embeds: torch.Tensor, cur_idx: torch.Tensor,
                state_scalars: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """
        node_embeds  : (B, N, d)
        cur_idx      : (B,) long  — current node index per rollout
        state_scalars: (B, 2)     — [remaining_cap, current_time] normalized
        action_mask  : (B, N) bool — True = masked (invalid)
        Returns logits (B, N).
        """
        B, N, d = node_embeds.shape
        graph = node_embeds.mean(dim=1)
        cur_embed = node_embeds[torch.arange(B), cur_idx]
        ctx = torch.cat([graph, cur_embed, state_scalars], dim=-1)
        q = self.context_proj(ctx).unsqueeze(1)
        h = self.glimpse(q, node_embeds, mask=action_mask.unsqueeze(1)).squeeze(1)
        # Compat logits — single head.
        keys = self.compat(node_embeds)
        scores = torch.bmm(h.unsqueeze(1), keys.transpose(1, 2)).squeeze(1) / math.sqrt(d)
        scores = self.cfg.decoder_clip * torch.tanh(scores)
        scores = scores.masked_fill(action_mask, float("-inf"))
        return scores


class POMOModel(nn.Module):
    """Top-level wrapper: encoder + decoder + step-by-step rollout helper."""

    def __init__(self, cfg: POMOConfig | None = None):
        super().__init__()
        self.cfg = cfg or POMOConfig()
        self.encoder = MatNetEncoder(self.cfg)
        self.decoder = POMODecoder(self.cfg)

    def encode(self, node_feats: torch.Tensor, edge_feats: torch.Tensor) -> torch.Tensor:
        return self.encoder(node_feats, edge_feats)

    def step_logits(self, node_embeds: torch.Tensor, cur_idx: torch.Tensor,
                    state_scalars: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        return self.decoder(node_embeds, cur_idx, state_scalars, action_mask)
