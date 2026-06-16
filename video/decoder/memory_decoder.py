"""MemoryDecoder: UnsupervisedDecoder (v3 click-prompt decoder, [[unsupervised-decoder-trainer]])
+ KV memory bank (memory_bank.py).

When `memory_bank` is empty (or None), this is exactly v3: adapter -> prompt_encoder
-> mask_decoder. When non-empty, the adapter's image embedding is first fused with
cross-attended memory context via MemoryReader (zero-init gated, so a freshly
warm-started MemoryDecoder reproduces v3 until the gate is trained).

`forward` also returns `img_emb` (post-fusion) so callers can write it (with a
mask) into the memory bank via `encode_memory`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from video.decoder.train_sam_decoder import UnsupervisedDecoder  # noqa: E402
from video.decoder.memory_bank import MemoryEncoder, MemoryBank, MemoryReader  # noqa: E402


class MemoryDecoder(nn.Module):
    def __init__(self, max_recent: int = 3):
        super().__init__()
        self.decoder = UnsupervisedDecoder()
        self.memory_encoder = MemoryEncoder()
        self.memory_reader = MemoryReader()
        self.max_recent = max_recent

    def load_decoder_weights(self, ckpt_path: str) -> None:
        sd = torch.load(ckpt_path, map_location="cuda")["model"]
        self.decoder.load_state_dict(sd)

    def new_memory_bank(self) -> MemoryBank:
        return MemoryBank(max_recent=self.max_recent)

    def encode_memory(self, img_emb: torch.Tensor, mask: torch.Tensor):
        return self.memory_encoder(img_emb, mask)

    def forward(self, dino_feats_chw, memory_bank: MemoryBank | None = None,
                boxes_xyxy=None, mask_prompts=None, points=None):
        img_emb = self.decoder.adapter(dino_feats_chw)               # [1, 256, 64, 64]
        if memory_bank is not None and not memory_bank.is_empty():
            img_emb = self.memory_reader(img_emb, memory_bank)
        sparse, dense = self.decoder.prompt_encoder(points=points, boxes=boxes_xyxy, masks=mask_prompts)
        low_res, _ = self.decoder.mask_decoder(
            image_embeddings=img_emb,
            image_pe=self.decoder.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        return low_res, img_emb                                      # [N,1,256,256], [1,256,64,64]
