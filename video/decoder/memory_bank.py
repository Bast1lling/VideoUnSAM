"""Cutie-inspired KV memory bank for the click-prompt decoder, built from scratch
(no vendored Cutie code) — reuses segment_anything's cross-attention block as the
read core. See [[propagation-ot-vs-alternatives]] / cached-swinging-island plan.

  MemoryEncoder: img_emb + mask -> (key, value) tokens, pooled to a 32x32 grid.
  MemoryBank:    one permanent anchor (frame 0) + a ring buffer of recent frames.
  MemoryReader:  cross-attention from the current frame's tokens to the bank,
                 fused back via a zero-initialised residual gate so the model
                 starts identical to the memory-free decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling.transformer import Attention

_EMB = 256
_GRID = 64
_MEM_GRID = 32


class MemoryEncoder(nn.Module):
    """[1,256,64,64] image embedding + [1,1,64,64] mask -> (key, value), each [1,1024,256]."""

    def __init__(self, emb_dim: int = _EMB, mem_grid: int = _MEM_GRID):
        super().__init__()
        self.mem_grid = mem_grid
        self.key_proj = nn.Conv2d(emb_dim, emb_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(emb_dim + 1, emb_dim, kernel_size=1)

    def forward(self, img_emb: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled_emb = F.adaptive_avg_pool2d(img_emb, (self.mem_grid, self.mem_grid))
        pooled_mask = F.adaptive_avg_pool2d(mask, (self.mem_grid, self.mem_grid))
        key = self.key_proj(pooled_emb)
        value = self.value_proj(torch.cat([pooled_emb, pooled_mask], dim=1))
        key = key.flatten(2).transpose(1, 2)      # [1, mem_grid^2, emb_dim]
        value = value.flatten(2).transpose(1, 2)
        return key, value


class MemoryBank:
    """Anchor slot (frame 0, never evicted) + ring buffer of the last `max_recent` frames."""

    def __init__(self, max_recent: int = 3):
        self.max_recent = max_recent
        self.anchor: tuple[torch.Tensor, torch.Tensor] | None = None
        self.recent: list[tuple[torch.Tensor, torch.Tensor]] = []

    def write_anchor(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.anchor = (key, value)
        self.recent = []

    def write_recent(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.recent.append((key, value))
        if len(self.recent) > self.max_recent:
            self.recent.pop(0)

    def is_empty(self) -> bool:
        return self.anchor is None

    def read(self) -> tuple[torch.Tensor, torch.Tensor]:
        keys = [self.anchor[0]] + [k for k, _ in self.recent]
        values = [self.anchor[1]] + [v for _, v in self.recent]
        return torch.cat(keys, dim=1), torch.cat(values, dim=1)

    def reset(self) -> None:
        self.anchor = None
        self.recent = []


class MemoryReader(nn.Module):
    """Cross-attention from the current frame's image embedding to the memory bank,
    fused back via a small residual gate (near-identity at init, but nonzero so
    predictions are immediately track-specific and the gate gets a non-cancelling
    per-track gradient -- see [[memory-bank-gate-stuck]])."""

    def __init__(self, emb_dim: int = _EMB, num_heads: int = 8, downsample_rate: int = 2,
                 gate_init: float = 0.05):
        super().__init__()
        self.attn = Attention(embedding_dim=emb_dim, num_heads=num_heads, downsample_rate=downsample_rate)
        self.gate = nn.Parameter(torch.full((1,), gate_init))

    def forward(self, img_emb: torch.Tensor, memory_bank: MemoryBank) -> torch.Tensor:
        if memory_bank.is_empty():
            return img_emb
        B, C, H, W = img_emb.shape
        q = img_emb.flatten(2).transpose(1, 2)    # [1, H*W, C]
        k, v = memory_bank.read()
        ctx = self.attn(q, k, v)                  # [1, H*W, C]
        ctx = ctx.transpose(1, 2).reshape(B, C, H, W)
        return img_emb + self.gate * ctx
