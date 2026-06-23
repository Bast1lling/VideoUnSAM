"""
Memory-conditioned video extension of IMaskDINODecoder / IMaskDINOHead.

Motivation
----------
The first video variant (``video_mask_dino.py``) only let the *decoder queries*
exchange information across frames via a per-query ``TemporalSelfAttention``.
The image *features* of each frame are still encoded completely independently,
and the predicted mask of the prompt frame is never fed forward.  A query in a
non-prompt frame therefore has to re-localise a changed object purely from a
refined reference box plus a tiny "is-this-the-prompt-frame" marker — a weak
propagation signal.

This variant adds the mechanism that actually makes promptable video
segmentation work (SAM 2's core idea), adapted to the multi-prompt MaskDINO
decoder so that **all pretrained UnSAM weights are reused unchanged**:

  * a small **memory bank** is built from the prompt frame, and
  * the queries that generate each *other* frame's mask **cross-attend** to it,
  * where the memory is **conditioned on the prompt frame's predicted mask**
    (its object is distilled into foreground / background appearance prototypes).

Concretely the memory bank for prompt *p* of video *b* contains, per decoder
layer:

  * ``M`` **object-pointer** tokens  – the prompt frame's own query states for
    prompt *p* at this layer (identity / "what was clicked").
  * one **foreground** prototype     – mask-pooled appearance of *p* inside its
    predicted mask in the prompt frame (mask-conditioning).
  * one **background** prototype     – mask-pooled appearance outside it.

Architecture overview
---------------------
* Each frame is encoded independently by the backbone + pixel decoder (unchanged).
* The transformer decoder runs, per layer::

      spatial cross-attention      (identical to the image model, per frame)
      ↓
      mask-conditioned memory read (NEW – query frames attend to the prompt
                                    frame's mask-conditioned memory bank)
      ↓
      iterative box refinement     (unchanged)
      ↓
      prediction heads             (unchanged; also yields the prompt-frame mask
                                    that conditions the *next* layer's memory)

* The memory read is **depth-causal**: layer *i* conditions on the prompt
  frame's mask predicted at layer *i-1* (at layer 0 only the object pointers are
  used).  The prompt frame itself is never modified by the memory read, so its
  prediction stays identical to the image model's — it is the trusted reference.

Identity at initialisation
--------------------------
Every memory layer is **pre-norm with a zero-initialised attention output
projection**, so at load time the residual contribution is exactly zero and the
whole video model reproduces the image model frame-by-frame.  The memory
behaviour is learned during finetuning.  (This is stronger than the temporal
variant, whose attention was active at init and so did *not* reproduce the image
model.)

Batch / prompt conventions
--------------------------
Identical to ``video_mask_dino.py``:

  * batch dim is **video-first**: index ``b*T + t`` is frame *t* of video *b*.
  * ``targets`` holds B entries (one per video); the shared point/box prompt is
    replicated to all T frames inside ``forward``.
  * decoder queries are laid out **prompt-major**: ``pad_size = P*M`` with prompt
    *p* occupying rows ``[p*M : (p+1)*M]`` (``M = num_mask_tokens``).
  * ``pred_masks`` output is ``(B*T, P*M, H, W)`` with video-first ordering.

Weight loading & finetuning
---------------------------
All new parameters live under ``memory_layers.*`` and load with
``strict=False``; only those keys are missing from an image checkpoint.  Use
``freeze_non_memory(model)`` to train just the memory layers first, then
optionally unfreeze the heads / decoder.
"""

from __future__ import annotations

import torch
from torch import nn, Tensor
from typing import Optional

from semantic_sam.body.decoder.utils.utils import (
    gen_sineembed_for_position,
    inverse_sigmoid,
)

import sys
sys.path.append("./promptable_segmentation")

from model.body.decoder.interactive_mask_dino import (
    IMaskDINODecoder,
)
from model.body.general_head import IMaskDINOHead
from model.utils import configurable

# Resolution the video model was trained at.  Callers should resize all input
# frames to this square size before passing them through the backbone.
INFERENCE_SIZE: int = 512


# ---------------------------------------------------------------------------
# Mask-conditioned memory attention layer
# ---------------------------------------------------------------------------

class MaskConditionedMemoryAttention(nn.Module):
    """
    Per-prompt cross-attention from a query frame's mask tokens to a small
    memory bank built from the prompt (memory) frame.

    For prompt *p* of video *b*, the memory bank is::

        [ object-pointer_0 … object-pointer_{M-1},  fg_prototype,  bg_prototype ]

    where the object pointers are the prompt frame's own query states for prompt
    *p* at the current layer, and the fg/bg prototypes are mask-pooled
    appearance vectors distilled from the prompt frame's ``mask_features`` using
    the mask it predicted at the previous layer (``pf_soft_mask``).

    The M mask-token queries of prompt *p* in every *other* frame attend to that
    bank; the prompt frame is left unchanged.  The block is **pre-norm with a
    zero-init output projection**, so at initialisation it is the identity and
    the model reproduces the image model exactly.

    Token type tags (0 = pointer, 1 = fg, 2 = bg) let attention distinguish the
    three memory roles.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Pre-norm applied to attention inputs (queries and memory) so the
        # residual stream passes through untouched at init.
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Map mask-pooled fg / bg appearance vectors into memory tokens.
        self.fg_proj = nn.Linear(d_model, d_model)
        self.bg_proj = nn.Linear(d_model, d_model)
        # 0 = object pointer, 1 = foreground prototype, 2 = background prototype.
        self.type_embed = nn.Embedding(3, d_model)

        # Zero-init the attention output projection → identity residual at load
        # time, so a freshly-built video model == the image model frame-by-frame.
        nn.init.zeros_(self.cross.out_proj.weight)
        nn.init.zeros_(self.cross.out_proj.bias)

    def forward(
        self,
        output: Tensor,                       # (P*M, B*T, d)  prompt-major / video-first
        B: int,
        T: int,
        P: int,
        M: int,
        pf: Tensor,                           # (B,) long – prompt frame per video
        mask_features: Tensor,                # (B*T, d, h, w)
        pf_soft_mask: "Optional[Tensor]",     # (B, P, h, w) in [0,1] or None (layer 0)
    ) -> Tensor:
        """Return ``output`` with query-frame tokens residually updated."""
        d = self.d_model
        device = output.device
        # (P*M, B*T, d) → (P, M, B, T, d)
        O = output.reshape(P, M, B, T, d)
        bidx = torch.arange(B, device=device)

        # ---- object-pointer tokens: prompt frame's query states for prompt p --
        OP = O[:, :, bidx, pf, :]                              # (P, M, B, d)
        ptr = self.norm(OP).permute(1, 0, 2, 3).reshape(M, P * B, d)
        ptr = ptr + self.type_embed.weight[0]                 # (M, P*B, d)
        mem_tokens = [ptr]

        # ---- mask-conditioned fg / bg appearance prototypes (from layer i-1) --
        if pf_soft_mask is not None:
            h, w = mask_features.shape[-2:]
            MF = mask_features.reshape(B, T, d, h, w)
            MF_pf = MF[bidx, pf]                               # (B, d, h, w)
            m = pf_soft_mask.clamp(0.0, 1.0)                   # (B, P, h, w)
            msum = m.flatten(2).sum(-1).clamp(min=1.0)         # (B, P)
            fg = torch.einsum("bphw,bdhw->bpd", m, MF_pf) / msum[..., None]
            inv = 1.0 - m
            isum = inv.flatten(2).sum(-1).clamp(min=1.0)
            bg = torch.einsum("bphw,bdhw->bpd", inv, MF_pf) / isum[..., None]
            fg = self.norm(self.fg_proj(fg)).permute(1, 0, 2).reshape(1, P * B, d)
            bg = self.norm(self.bg_proj(bg)).permute(1, 0, 2).reshape(1, P * B, d)
            fg = fg + self.type_embed.weight[1]
            bg = bg + self.type_embed.weight[2]
            mem_tokens += [fg, bg]

        mem = torch.cat(mem_tokens, dim=0)                    # (M+2 or M, P*B, d)

        # ---- each query frame's tokens attend to the memory bank --------------
        cols = []
        for t in range(T):
            q_in = O[:, :, :, t, :]                           # (P, M, B, d)
            q = self.norm(q_in).permute(1, 0, 2, 3).reshape(M, P * B, d)
            a, _ = self.cross(q, mem, mem)                    # (M, P*B, d)
            a = a.reshape(M, P, B, d).permute(1, 0, 2, 3)     # (P, M, B, d)
            # Only update query frames; the prompt frame stays as the reference.
            qmask = (t != pf).to(a.dtype).view(1, 1, B, 1)
            cols.append(q_in + self.dropout(a) * qmask)
        out = torch.stack(cols, dim=3)                        # (P, M, B, T, d)
        return out.reshape(P * M, B * T, d)


# ---------------------------------------------------------------------------
# Memory-conditioned video decoder
# ---------------------------------------------------------------------------

class VideoMemoryIMaskDINODecoder(IMaskDINODecoder):
    """
    IMaskDINODecoder with one ``MaskConditionedMemoryAttention`` interleaved
    after every spatial decoder layer.

    **Weight loading** – all pretrained keys load without remapping.  New
    parameters live exclusively under ``memory_layers.*``.

    **Single-frame mode** – when ``T == 1`` the memory layers are bypassed and
    the model behaves identically to the image model.
    """

    @configurable
    def __init__(
        self,
        lang_encoder: nn.Module,
        in_channels: int,
        mask_classification: bool = True,
        *,
        # ---- all parent kwargs (mirrored from IMaskDINODecoder) ----
        num_classes: int,
        hidden_dim: int,
        dim_proj: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        mask_dim: int,
        enforce_input_project: bool,
        two_stage: bool,
        dn: str,
        noise_scale: float,
        dn_num: int,
        initialize_box_type: bool,
        initial_pred: bool,
        learn_tgt: bool,
        total_num_feature_levels: int = 4,
        dropout: float = 0.0,
        activation: str = "relu",
        nhead: int = 8,
        dec_n_points: int = 4,
        return_intermediate_dec: bool = True,
        query_dim: int = 4,
        dec_layer_share: bool = False,
        semantic_ce_loss: bool = False,
        num_mask_tokens: int = 3,
        # ---- video-specific ----
        num_frames: int = 2,
        memory_nhead: int = 8,
        memory_dropout: float = 0.0,
    ):
        super().__init__(
            lang_encoder=lang_encoder,
            in_channels=in_channels,
            mask_classification=mask_classification,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dim_proj=dim_proj,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            mask_dim=mask_dim,
            enforce_input_project=enforce_input_project,
            two_stage=two_stage,
            dn=dn,
            noise_scale=noise_scale,
            dn_num=dn_num,
            initialize_box_type=initialize_box_type,
            initial_pred=initial_pred,
            learn_tgt=learn_tgt,
            total_num_feature_levels=total_num_feature_levels,
            dropout=dropout,
            activation=activation,
            nhead=nhead,
            dec_n_points=dec_n_points,
            return_intermediate_dec=return_intermediate_dec,
            query_dim=query_dim,
            dec_layer_share=dec_layer_share,
            semantic_ce_loss=semantic_ce_loss,
            num_mask_tokens=num_mask_tokens,
        )

        # One memory layer per spatial decoder layer.
        # State-dict key:  "memory_layers.{i}.*"
        num_spatial_layers = len(self.decoder.layers)
        self.memory_layers = nn.ModuleList([
            MaskConditionedMemoryAttention(self.hidden_dim, memory_nhead, memory_dropout)
            for _ in range(num_spatial_layers)
        ])
        self.num_frames = num_frames

    @classmethod
    def from_config(cls, cfg, in_channels, lang_encoder, mask_classification, extra):
        ret = IMaskDINODecoder.from_config(
            cfg, in_channels, lang_encoder, mask_classification, extra
        )
        dec_cfg = cfg["MODEL"]["DECODER"]
        ret["num_frames"] = dec_cfg.get("NUM_FRAMES", 2)
        ret["memory_nhead"] = dec_cfg.get("MEMORY_NHEAD", dec_cfg.get("TEMPORAL_NHEAD", 8))
        ret["memory_dropout"] = dec_cfg.get("MEMORY_DROPOUT", 0.0)
        return ret

    # ------------------------------------------------------------------
    # Video forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x,
        mask_features,
        masks,
        targets=None,
        target_queries=None,
        target_vlp=None,
        task: str = "seg",
        extra: dict = {},
        num_frames: Optional[int] = None,
        prompt_frame_idx: Optional[Tensor] = None,
    ):
        """
        Args:
            x:             list of multi-scale feature maps, each ``(B*T, C, H, W)``.
            mask_features: ``(B*T, mask_dim, H, W)`` from the pixel decoder.
            masks:         spatial padding masks, or None.
            targets:       list of B dicts (one per *video*), containing shared
                           point/box prompts replicated to all T frames.
            num_frames:    override ``self.num_frames`` for this call.
            prompt_frame_idx: optional ``(B,)`` long tensor giving, per video, the
                           frame the prompt was sampled from (default: frame 0).
                           This frame supplies the memory; the others read from it.

        Returns:
            ``(out_dict, mask_dict)`` – same structure as the image model.
            ``out_dict["pred_masks"]`` has shape ``(B*T, P*M, H, W)`` with
            video-first batch ordering.
        """
        T = num_frames if num_frames is not None else self.num_frames
        BT = x[0].shape[0]
        B = BT // T

        prediction_switch = extra
        self.prediction_switch = prediction_switch
        assert len(x) == self.num_feature_levels
        do_seg = True

        # ----------------------------------------------------------------
        # Build flattened spatial memory from B*T frame features (per frame;
        # identical to the image model, B*T acts as the batch size).
        # ----------------------------------------------------------------
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [
                torch.zeros(
                    (src.size(0), src.size(2), src.size(3)),
                    device=src.device,
                    dtype=torch.bool,
                )
                for src in x
            ]

        src_flatten, mask_flatten, spatial_shapes = [], [], []
        for i in range(self.num_feature_levels):
            idx = self.num_feature_levels - 1 - i
            src_flatten.append(
                self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2)
            )
            mask_flatten.append(masks[i].flatten(1))
            spatial_shapes.append(x[idx].shape[-2:])
        src_flatten = torch.cat(src_flatten, 1)    # (B*T, hw_sum, d)
        mask_flatten = torch.cat(mask_flatten, 1)  # (B*T, hw_sum)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in masks], 1
        )  # (B*T, nlevel, 2)

        # ----------------------------------------------------------------
        # Query preparation – for B videos (shared prompts across frames).
        # ----------------------------------------------------------------
        tgt_mask = None
        mask_dict = None
        if self.dn != "no":
            assert targets is not None
            if task == "demo":
                input_query_label, input_query_bbox, tgt_mask, mask_dict = \
                    self.prepare_for_dn_mo_infer(targets, None, None, B)
            else:
                input_query_label, input_query_bbox, tgt_mask, mask_dict = \
                    self.prepare_for_dn_mo(targets, None, None, B)
            tgt = input_query_label            # (B, pad_size, d)
            refpoint_embed = input_query_bbox  # (B, pad_size, 4)
            if tgt is None:
                tgt = torch.zeros(B, self.num_queries, self.hidden_dim).cuda()
                refpoint_embed = torch.zeros(B, self.num_queries, 4).cuda()

        # Replicate queries to all T frames with video-first ordering:
        #   (B, pad_size, d) → (B, T, pad_size, d) → (B*T, pad_size, d)
        nq = tgt.shape[1]
        tgt = tgt.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, nq, self.hidden_dim)
        refpoint_embed = (
            refpoint_embed.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, nq, 4)
        )

        # Decoder convention: leading dim is nq, second is batch.
        output = tgt.transpose(0, 1)               # (pad_size, B*T, d)
        refpoint_embed_t = refpoint_embed.transpose(0, 1)
        memory = src_flatten.transpose(0, 1)       # (hw_sum, B*T, d)

        # Prompt-major query layout: pad_size = P * M.
        M = self.num_all_tokens
        P = nq // M
        if prompt_frame_idx is None:
            pf = torch.zeros(B, dtype=torch.long, device=output.device)
        else:
            pf = prompt_frame_idx.to(output.device).long().view(B)

        # ----------------------------------------------------------------
        # Decoder loop: spatial cross-attn → mask-conditioned memory read.
        # ----------------------------------------------------------------
        dec = self.decoder
        reference_points = refpoint_embed_t.sigmoid()
        ref_points = [reference_points]

        predictions_class, predictions_class_part = [], []
        predictions_mask, predictions_iou_score, new_hs = [], [], []

        # Prompt-frame per-prompt soft mask predicted at the *previous* layer;
        # None at layer 0 (memory then uses object pointers only).
        pf_soft_mask: Optional[Tensor] = None

        for layer_id, (layer, memory_layer) in enumerate(
            zip(dec.layers, self.memory_layers)
        ):
            reference_points_input = (
                reference_points[:, :, None]
                * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            )  # (nq, B*T, nlevel, 4)
            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :],
                dim=output.shape[-1] // 2,
            )
            raw_query_pos = dec.ref_point_head(query_sine_embed)
            pos_scale = dec.query_scale(output) if dec.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos

            # ---- Spatial cross-attention (per frame, same as image model) ----
            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=None,
                tgt_reference_points=reference_points_input,
                memory=memory,
                memory_key_padding_mask=mask_flatten,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=None,
                self_attn_mask=tgt_mask,
                cross_attn_mask=None,
                task_switch=dec.task_switch,
                extra=extra,
            )

            # ---- Mask-conditioned memory read (skipped for single-frame) -----
            if T > 1:
                output = memory_layer(
                    output, B, T, P, M, pf, mask_features, pf_soft_mask
                )

            # ---- Iterative box refinement (unchanged) ----
            if dec.bbox_embed is not None:
                delta_unsig = dec.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + inverse_sigmoid(reference_points)
                new_reference_points = outputs_unsig.sigmoid()
                reference_points = new_reference_points.detach()
                ref_points.append(new_reference_points)

            # ---- Prediction heads ----
            # Always compute masks (every layer, train and eval) so the prompt
            # frame's mask is available to condition the next layer's memory.
            normed = dec.norm(output)  # (nq, B*T, d)
            outputs_class, outputs_mask, iou_score, dec_out_mask = \
                self.interactive_forward_prediction_heads(
                    normed, mask_features, do_seg,
                )
            outputs_class_whole, outputs_class_part = outputs_class
            predictions_class.append(outputs_class_whole)
            predictions_class_part.append(outputs_class_part)
            predictions_mask.append(outputs_mask)
            if iou_score is not None:
                predictions_iou_score.append(iou_score)
                new_hs.append(dec_out_mask)  # (B*T, P*M, d)

            # ---- Stash the prompt frame's soft mask for the next layer -------
            if outputs_mask is not None:
                om = outputs_mask.detach()                         # (B*T, P*M, h, w)
                h, w = om.shape[-2:]
                om = om.reshape(B, T, P, M, h, w)
                bidx = torch.arange(B, device=om.device)
                pf_soft_mask = om[bidx, pf].sigmoid().mean(2)       # (B, P, h, w)

        hs = new_hs if new_hs else [dec.norm(output).transpose(0, 1)]

        # References need to be (B*T, nq, 4) for pred_box.
        ref_points_bt = [r.transpose(0, 1) for r in ref_points]
        out_boxes = self.pred_box(ref_points_bt, hs)
        out_boxes[-1] = out_boxes[-1] + 0.0 * (
            self.label_enc.weight.sum()
            + self.pb_embedding.weight.sum()
            + self.mask_tokens.weight.sum()
            + self.lang_mapper.sum()
        )

        if mask_dict is not None:
            if do_seg:
                predictions_mask = list(predictions_mask)
        elif self.training:
            for i in range(self.mask_embed.num_layers):
                predictions_class[-1] = predictions_class[-1] + 0.0 * (
                    self.mask_embed.layers[i].weight[0][0]
                    + self.mask_embed.layers[i].bias[0]
                )
            predictions_class[-1] = (
                predictions_class[-1] + 0.0 * mask_features[0][0][0][0]
            )

        out = {
            "pred_logits": predictions_class[-1],
            "pred_logits_part": predictions_class_part[-1],
            "pred_masks": None if not do_seg else predictions_mask[-1],
            "pred_boxes": out_boxes[-1],
            "pred_ious": predictions_iou_score[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class if self.mask_classification else None,
                predictions_mask,
                out_boxes,
                predictions_iou_score,
                predictions_class_part,
            ),
        }
        return out, mask_dict


# ---------------------------------------------------------------------------
# Memory-conditioned video head (wraps pixel decoder + video decoder)
# ---------------------------------------------------------------------------

class VideoMemoryIMaskDINOHead(IMaskDINOHead):
    """
    IMaskDINOHead that uses VideoMemoryIMaskDINODecoder as the transformer
    predictor.  The pixel decoder (MaskDINOEncoder) is reused unchanged; it runs
    independently on each frame (B*T frames treated as a larger batch).
    """

    @classmethod
    def from_config(cls, cfg, input_shape, lang_encoder, extra):
        from promptable_segmentation.model.body.encoder import build_encoder

        enc_cfg = cfg["MODEL"]["ENCODER"]
        transformer_predictor_in_channels = enc_cfg["CONVS_DIM"]
        return {
            "input_shape": {
                k: v
                for k, v in input_shape.items()
                if k in enc_cfg["IN_FEATURES"]
            },
            "ignore_value": enc_cfg["IGNORE_VALUE"],
            "num_classes": enc_cfg.get("NUM_CLASSES", None),
            "pixel_decoder": build_encoder(cfg, input_shape),
            "loss_weight": enc_cfg["LOSS_WEIGHT"],
            "transformer_predictor": VideoMemoryIMaskDINODecoder(
                cfg,
                transformer_predictor_in_channels,
                lang_encoder,
                mask_classification=True,
                extra=extra,
            ),
        }

    def layers(
        self,
        features,
        mask=None,
        targets=None,
        target_queries=None,
        target_vlp=None,
        prediction_switch=None,
        task="seg",
        extra={},
        num_frames: Optional[int] = None,
        prompt_frame_idx: Optional[Tensor] = None,
    ):
        mask_features, _, multi_scale_features = \
            self.pixel_decoder.forward_features(features, mask)
        return self.predictor(
            multi_scale_features,
            mask_features,
            mask,
            targets=targets,
            target_queries=target_queries,
            target_vlp=target_vlp,
            task=task,
            extra=extra,
            num_frames=num_frames,
            prompt_frame_idx=prompt_frame_idx,
        )

    def forward(
        self,
        features,
        mask=None,
        targets=None,
        target_queries=None,
        target_vlp=None,
        task="seg",
        extra={},
        num_frames: Optional[int] = None,
        prompt_frame_idx: Optional[Tensor] = None,
    ):
        return self.layers(
            features,
            mask,
            targets=targets,
            target_queries=target_queries,
            target_vlp=target_vlp,
            task=task,
            extra=extra,
            num_frames=num_frames,
            prompt_frame_idx=prompt_frame_idx,
        )


# ---------------------------------------------------------------------------
# Weight-loading and freezing utilities
# ---------------------------------------------------------------------------

# Parameter-name prefixes that are *new* in the video variants and therefore
# expected to be missing from an image-model checkpoint.
_NEW_PREFIXES = ("memory_layers", "temporal_layers")


def freeze_non_memory(model: nn.Module) -> None:
    """
    Freeze every parameter except ``memory_layers.*``.

    Recommended starting point for finetuning: the pretrained spatial layers,
    pixel decoder and heads stay fixed while only the new memory layers train.
    After validating convergence, unfreeze additional components as needed.
    """
    for name, param in model.named_parameters():
        param.requires_grad = "memory_layers" in name


def load_image_weights(
    video_model: nn.Module,
    ckpt_path: str,
    map_location: str = "cpu",
) -> "tuple[list[str], list[str]]":
    """
    Load a pretrained image-model checkpoint into the memory video model.

    Handles two checkpoint layouts automatically:
    - ``VideoMemoryIMaskDINOHead`` saved standalone → keys like ``pixel_decoder.*``
    - Full ``GeneralizedMaskDINO`` checkpoint → keys like
      ``model.sem_seg_head.pixel_decoder.*``; the prefix is stripped.

    All spatial layer keys match exactly.  Keys that do not appear in the
    checkpoint (``memory_layers.*``) are left at their (identity) init.

    Returns ``(missing_keys, unexpected_keys)`` from ``load_state_dict``.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state_dict = ckpt.get("model", ckpt)

    _HEAD_PREFIX = "model.sem_seg_head."
    if any(k.startswith(_HEAD_PREFIX) for k in state_dict):
        state_dict = {
            k[len(_HEAD_PREFIX):]: v
            for k, v in state_dict.items()
            if k.startswith(_HEAD_PREFIX)
        }

    missing, unexpected = video_model.load_state_dict(state_dict, strict=False)
    other_missing = [
        k for k in missing if not any(p in k for p in _NEW_PREFIXES)
    ]
    if other_missing:
        import warnings
        warnings.warn(
            f"load_image_weights: {len(other_missing)} non-memory keys missing "
            f"(first 5: {other_missing[:5]})"
        )
    return missing, unexpected


if __name__ == "__main__":
    import os

    _THIS = os.path.dirname(os.path.abspath(__file__))
    _REPO = os.path.dirname(os.path.dirname(_THIS))    # repo root
    sys.path.insert(0, _REPO)
    sys.path.insert(0, os.path.join(_REPO, "promptable_segmentation"))

    # ---- user-settable variables ----
    CONFIG   = os.path.join(_THIS, "..", "configs", "video_sam_swinT.yaml")
    WEIGHTS  = os.path.join(_THIS, "..", "ckpts", "baseline.pth")
    B        = 1      # videos per batch
    T        = 3      # frames per video
    H, W     = INFERENCE_SIZE, INFERENCE_SIZE
    N_CLICKS = 2      # point prompts per video (shared across all T frames)
    DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

    from promptable_segmentation.utils.arguments import load_opt_from_config_file
    from promptable_segmentation.model.backbone import build_backbone
    from semantic_sam.language import build_language_encoder

    opt = load_opt_from_config_file(CONFIG)
    # Skip the swin auto-loader (its 'ckpts/baseline.pth' path is repo-root
    # relative); head weights are loaded from the absolute WEIGHTS path below.
    # A random backbone is fine for the shape / identity-at-init check.
    opt["MODEL"]["BACKBONE"]["LOAD_PRETRAINED"] = False

    print("Building backbone …")
    backbone = build_backbone(opt)
    backbone.to(DEVICE)
    backbone.eval()   # disable SwinT stochastic depth so identical frames match

    print("Building language encoder …")
    lang_encoder = build_language_encoder(opt)
    if lang_encoder is not None:
        lang_encoder.to(DEVICE)
        lang_encoder.eval()

    print("Building VideoMemoryIMaskDINOHead …")
    extra = {"task_switch": {"bbox": True, "mask": True}}
    head = VideoMemoryIMaskDINOHead(
        opt, backbone.output_shape(), lang_encoder, extra
    ).to(DEVICE)

    if os.path.exists(WEIGHTS):
        print(f"Loading weights from {WEIGHTS} …")
        missing, _ = load_image_weights(head, WEIGHTS, map_location=DEVICE)
        n_mem   = sum(1 for k in missing if "memory_layers" in k)
        n_other = sum(1 for k in missing if "memory_layers" not in k)
        print(f"  memory_layers keys (identity init): {n_mem}")
        if n_other:
            other = [k for k in missing if "memory_layers" not in k]
            print(f"  WARNING – {n_other} non-memory keys missing (first 3: {other[:3]})")
    else:
        print(f"Checkpoint not found at {WEIGHTS} — proceeding with random weights.")

    freeze_non_memory(head)
    total     = sum(p.numel() for p in head.parameters())
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"\nParameter budget:")
    print(f"  total     : {total:,}")
    print(f"  trainable : {trainable:,}  ({100. * trainable / total:.2f}%  — memory layers only)")

    # ---- dummy input: T copies of the SAME frame (identity-at-init check) ----
    pixel_mean = torch.tensor([123.675, 116.280, 103.530], device=DEVICE).view(1, 3, 1, 1)
    pixel_std  = torch.tensor([58.395,  57.120,  57.375 ], device=DEVICE).view(1, 3, 1, 1)
    one_frame  = (torch.rand(1, 3, H, W, device=DEVICE) * 255.0 - pixel_mean) / pixel_std
    frames     = one_frame.repeat(B * T, 1, 1, 1)

    with torch.no_grad():
        features = backbone(frames)

    rng = torch.Generator(device=DEVICE).manual_seed(0)
    clicks = torch.rand(B, N_CLICKS, 4, generator=rng, device=DEVICE) * 0.5 + 0.25
    clicks[:, :, 2:] = 0.01
    targets = [
        {
            "points": clicks[b],
            "pb":     torch.zeros(N_CLICKS, dtype=torch.long, device=DEVICE),
        }
        for b in range(B)
    ]

    prediction_switch = {"part": False, "whole": False, "seg": True, "det": True}
    print(f"\nVideo forward: B={B}, T={T}, N_CLICKS={N_CLICKS} …")
    head.eval()
    with torch.no_grad():
        out, _ = head(
            features, targets=targets, task="demo",
            extra=prediction_switch, num_frames=T,
        )

    pm = out["pred_masks"]
    print("\nOutput shapes  (B*T batch, video-first ordering):")
    print(f"  pred_masks  : {tuple(pm.shape)}")
    print(f"  pred_ious   : {tuple(out['pred_ious'].shape)}")
    print(f"  pred_boxes  : {tuple(out['pred_boxes'].shape)}")

    # Identity-at-init check: identical input frames + zero memory residual →
    # every frame must yield identical predictions.
    pm_v = pm.reshape(B, T, *pm.shape[1:])
    max_dev = (pm_v - pm_v[:, :1]).abs().max().item()
    print(f"\nIdentity-at-init max frame deviation: {max_dev:.3e}  "
          f"(≈0 expected — memory residual is zero at load)")
    print("\nDone.")
