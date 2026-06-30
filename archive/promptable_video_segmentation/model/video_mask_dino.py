"""
Video extension of IMaskDINODecoder / IMaskDINOHead.

Architecture overview
---------------------
* Each frame is encoded **independently** by the backbone and MaskDINOEncoder
  (pixel decoder).  No changes needed there.
* The transformer decoder runs a loop of:
      spatial cross-attention  (identical to the image model, per frame)
      ↓
      temporal self-attention  (NEW – queries exchange information across T frames)
  one temporal layer is interleaved after every spatial decoder layer.
* All pretrained spatial-layer weight keys are preserved exactly, so a
  checkpoint from the image model loads with ``strict=False`` and only
  ``temporal_layers.*`` parameters need to be initialised from scratch.

Batch convention
----------------
Input tensors use **video-first ordering** along the batch dimension::

    index  b*T + t  →  frame t of video b

Example: B=2 videos, T=4 frames → indices [0..3] are video 0, [4..7] video 1.

This lets temporal attention reshape ``(nq, B*T, d)`` → ``(T, nq*B, d)`` with
a simple ``.view()`` call (no scatter/gather needed).

Prompt convention
-----------------
``targets`` contains B entries (one per *video*).  Each dict holds the shared
point/box prompt that is replicated to all T frames inside ``forward``.
``pred_masks`` output has shape ``(B*T, N_q, H, W)`` with video-first ordering;
callers can reshape to ``(B, T, N_q, H, W)`` as needed.

Finetuning
----------
To finetune only the new temporal layers (recommended first step)::

    freeze_non_temporal(model)

To also finetune prediction heads::

    freeze_non_temporal(model)
    for name, p in model.named_parameters():
        if any(k in name for k in ('mask_embed', 'iou_prediction_head',
                                   'class_embed', 'bbox_embed')):
            p.requires_grad = True
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
# Temporal attention layer
# ---------------------------------------------------------------------------

class TemporalSelfAttention(nn.Module):
    """
    Per-query self-attention across T video frames.

    Each query token attends to its own state in every other frame, letting the
    decoder build a temporally-consistent representation before predicting masks.

    Prompt-frame marker
    --------------------
    The prompt is shared across frames but is only *anchored* (sampled) in one
    reference frame; in the other frames the same normalised coordinate is
    off-object.  Without any signal, the temporal attention is permutation-
    invariant and cannot tell which frame owns the prompt.  We add a tiny
    ``Embedding(2, d_model)`` ("this frame holds the prompt" vs "it does not")
    injected into the attention **query/key only** (not the value, so content is
    not corrupted).  This lets the model route the prompt's meaning from the
    owning frame to the others; combined with the decoder's iterative box
    refinement, non-prompt frames can re-localise the object.

    The embedding is **zero-initialised**, so at load time this layer behaves
    exactly as before (no positional signal) and the marker is learned during
    finetuning.  Relative frame *order* is still not encoded — only prompt
    ownership — which suits short, order-agnostic synthetic clips.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # 0 = "not the prompt frame", 1 = "this frame holds the prompt".
        self.prompt_frame_embed = nn.Embedding(2, d_model)
        nn.init.zeros_(self.prompt_frame_embed.weight)

    def forward(
        self,
        x: Tensor,
        B: int,
        T: int,
        prompt_frame_idx: "Optional[Tensor]" = None,
    ) -> Tensor:
        """
        Args:
            x:  ``(nq, B*T, d_model)`` – video-first ordering along batch dim.
            B:  number of videos in the batch.
            T:  frames per video.
            prompt_frame_idx: optional ``(B,)`` long tensor; for each video, the
                frame index the prompt was sampled from.  Defaults to frame 0.
        Returns:
            ``(nq, B*T, d_model)`` – residual-updated tensor.
        """
        nq, BT, d = x.shape
        # Reshape: (nq, B*T, d) → (nq, B, T, d) → (T, nq*B, d)
        # Each row in the temporal sequence is one frame; the "batch" for the
        # attention op combines all queries and all video elements.
        x_t = x.view(nq, B, T, d).permute(2, 0, 1, 3).reshape(T, nq * B, d)

        # Prompt-frame positional marker on q/k only.
        if prompt_frame_idx is None:
            pf = torch.zeros(B, dtype=torch.long, device=x.device)
        else:
            pf = prompt_frame_idx.to(device=x.device).long().view(B)
        frame_ids = torch.arange(T, device=x.device).unsqueeze(1)   # (T, 1)
        is_prompt = (frame_ids == pf.unsqueeze(0)).long()           # (T, B)
        pos = self.prompt_frame_embed(is_prompt)                    # (T, B, d)
        pos = pos.unsqueeze(1).expand(T, nq, B, d).reshape(T, nq * B, d)

        q = k = x_t + pos
        x2, _ = self.attn(q, k, x_t)
        x_t = x_t + self.dropout(x2)
        x_t = self.norm(x_t)
        # (T, nq*B, d) → (nq, B, T, d) → (nq, B*T, d)
        return x_t.reshape(T, nq, B, d).permute(1, 2, 0, 3).reshape(nq, BT, d)


# ---------------------------------------------------------------------------
# Video decoder
# ---------------------------------------------------------------------------

class VideoIMaskDINODecoder(IMaskDINODecoder):
    """
    IMaskDINODecoder with one TemporalSelfAttention layer interleaved after
    every spatial decoder layer.

    **Weight loading** – all pretrained keys load without remapping.  New
    parameters live exclusively under ``temporal_layers.*``.

    **Single-frame mode** – when ``num_frames=1`` (or ``T=1`` at call time)
    the temporal layers are bypassed entirely, so the model behaves identically
    to the image model.
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
        temporal_nhead: int = 8,
        temporal_dropout: float = 0.0,
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

        # One temporal layer per spatial decoder layer.
        # State-dict key:  "temporal_layers.{i}.{attn/norm/dropout}.*"
        num_spatial_layers = len(self.decoder.layers)
        self.temporal_layers = nn.ModuleList([
            TemporalSelfAttention(self.hidden_dim, temporal_nhead, temporal_dropout)
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
        ret["temporal_nhead"] = dec_cfg.get("TEMPORAL_NHEAD", 8)
        ret["temporal_dropout"] = dec_cfg.get("TEMPORAL_DROPOUT", 0.0)
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
                           point/box prompts that are replicated to all T frames.
            num_frames:    override ``self.num_frames`` for this call.
            prompt_frame_idx: optional ``(B,)`` long tensor giving, per video, the
                           frame the prompt was sampled from (default: frame 0).
                           Fed to the temporal layers' prompt-frame marker so the
                           model knows which frame owns the prompt.

        Returns:
            ``(out_dict, mask_dict)`` – same structure as the image model.
            ``out_dict["pred_masks"]`` has shape ``(B*T, N_q, H, W)`` with
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
        # Build flattened spatial memory from B*T frame features.
        # This is identical to the image model; B*T acts as the batch size.
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
        #   (B*T, pad_size, d) → (pad_size, B*T, d)
        output = tgt.transpose(0, 1)
        refpoint_embed_t = refpoint_embed.transpose(0, 1)
        memory = src_flatten.transpose(0, 1)   # (hw_sum, B*T, d)

        # ----------------------------------------------------------------
        # Decoder loop: spatial cross-attention → temporal self-attention.
        # The spatial layers are identical to the image model.
        # ----------------------------------------------------------------
        dec = self.decoder
        reference_points = refpoint_embed_t.sigmoid()
        ref_points = [reference_points]

        predictions_class, predictions_class_part = [], []
        predictions_mask, predictions_iou_score, new_hs = [], [], []

        for layer_id, (layer, temporal_layer) in enumerate(
            zip(dec.layers, self.temporal_layers)
        ):
            # Reference-point positional encoding (unchanged from TransformerDecoder)
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

            # ---- Temporal self-attention (new; skipped for single-frame) ----
            if T > 1:
                output = temporal_layer(output, B, T, prompt_frame_idx)

            # ---- Iterative box refinement (unchanged) ----
            if dec.bbox_embed is not None:
                delta_unsig = dec.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + inverse_sigmoid(reference_points)
                new_reference_points = outputs_unsig.sigmoid()
                reference_points = new_reference_points.detach()
                ref_points.append(new_reference_points)

            # ---- Prediction heads ----
            # Pass normed output as (nq, B*T, d); the head handles the transpose.
            # mask_features is (B*T, mask_dim, H, W) → one mask set per frame.
            normed = dec.norm(output)  # (nq, B*T, d)
            last_layer = layer_id == dec.num_layers - 1
            outputs_class, outputs_mask, iou_score, dec_out_mask = \
                self.interactive_forward_prediction_heads(
                    normed, mask_features,
                    (self.training or last_layer) and do_seg,
                )
            outputs_class_whole, outputs_class_part = outputs_class
            predictions_class.append(outputs_class_whole)
            predictions_class_part.append(outputs_class_part)
            predictions_mask.append(outputs_mask)
            if iou_score is not None:
                predictions_iou_score.append(iou_score)
                new_hs.append(dec_out_mask)  # (B*T, N_q', d)

        hs = new_hs if new_hs else [dec.norm(output).transpose(0, 1)]

        # References need to be (B*T, nq, 4) for pred_box
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
# Video head (wraps pixel decoder + video decoder)
# ---------------------------------------------------------------------------

class VideoIMaskDINOHead(IMaskDINOHead):
    """
    IMaskDINOHead that uses VideoIMaskDINODecoder as the transformer predictor.

    The pixel decoder (MaskDINOEncoder) is reused unchanged; it runs
    independently on each frame (B*T frames treated as a larger batch).
    """

    @classmethod
    def from_config(cls, cfg, input_shape, lang_encoder, extra):
        from promptable_segmentation.model.body.encoder import build_encoder

        enc_cfg = cfg["MODEL"]["ENCODER"]
        dec_cfg = cfg["MODEL"]["DECODER"]
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
            "transformer_predictor": VideoIMaskDINODecoder(
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

def freeze_non_temporal(model: nn.Module) -> None:
    """
    Freeze every parameter except ``temporal_layers.*``.

    This is the recommended starting point for finetuning: the pretrained
    spatial layers stay fixed while only the new temporal layers are trained.
    After validating that temporal training converges, unfreeze additional
    components (e.g. prediction heads) as needed.
    """
    for name, param in model.named_parameters():
        param.requires_grad = "temporal_layers" in name


def load_image_weights(
    video_model: nn.Module,
    ckpt_path: str,
    map_location: str = "cpu",
) -> tuple[list[str], list[str]]:
    """
    Load a pretrained image-model checkpoint into a video model.

    Handles two checkpoint layouts automatically:
    - ``VideoIMaskDINOHead`` saved standalone → keys like ``pixel_decoder.*``
    - Full ``GeneralizedMaskDINO`` checkpoint → keys like
      ``model.sem_seg_head.pixel_decoder.*``; the prefix is stripped.

    All spatial layer keys match exactly.  Keys that do not appear in the
    checkpoint (``temporal_layers.*``) are left at their random init.

    Returns:
        ``(missing_keys, unexpected_keys)`` from ``load_state_dict``.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state_dict = ckpt.get("model", ckpt)

    # If checkpoint is a full model, extract just the head weights.
    _HEAD_PREFIX = "model.sem_seg_head."
    if any(k.startswith(_HEAD_PREFIX) for k in state_dict):
        state_dict = {
            k[len(_HEAD_PREFIX):]: v
            for k, v in state_dict.items()
            if k.startswith(_HEAD_PREFIX)
        }

    missing, unexpected = video_model.load_state_dict(state_dict, strict=False)
    other_missing = [k for k in missing if "temporal_layers" not in k]
    if other_missing:
        import warnings
        warnings.warn(
            f"load_image_weights: {len(other_missing)} non-temporal keys missing "
            f"(first 5: {other_missing[:5]})"
        )
    return missing, unexpected


if __name__ == "__main__":
    import sys
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

    # ---- imports that depend on sys.path ----
    from promptable_segmentation.utils.arguments import load_opt_from_config_file
    from promptable_segmentation.model.backbone import build_backbone
    from semantic_sam.language import build_language_encoder

    # ---- load config ----
    opt = load_opt_from_config_file(CONFIG)

    # ---- build model components ----
    print("Building backbone …")
    backbone = build_backbone(opt)
    backbone.to(DEVICE)   # SwinTransformer.train() doesn't return self, so no chaining

    print("Building language encoder …")
    lang_encoder = build_language_encoder(opt)   # None for TEXT.ARCH: noencoder
    if lang_encoder is not None:
        lang_encoder.to(DEVICE)
        lang_encoder.eval()

    print("Building VideoIMaskDINOHead …")
    task_switch = {"bbox": True, "mask": True}
    extra = {"task_switch": task_switch}
    head = VideoIMaskDINOHead(
        opt, backbone.output_shape(), lang_encoder, extra
    ).to(DEVICE)

    # ---- load pretrained image-model weights ----
    if os.path.exists(WEIGHTS):
        print(f"Loading weights from {WEIGHTS} …")
        missing, _ = load_image_weights(head, WEIGHTS, map_location=DEVICE)
        n_temporal = sum(1 for k in missing if "temporal_layers" in k)
        n_other    = sum(1 for k in missing if "temporal_layers" not in k)
        print(f"  temporal_layers keys (random init): {n_temporal}")
        if n_other:
            other = [k for k in missing if "temporal_layers" not in k]
            print(f"  WARNING – {n_other} non-temporal keys missing (first 3: {other[:3]})")
    else:
        print(f"Checkpoint not found at {WEIGHTS} — proceeding with random weights.")

    # ---- freeze spatial layers; only temporal_layers will be trained ----
    freeze_non_temporal(head)

    total     = sum(p.numel() for p in head.parameters())
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"\nParameter budget:")
    print(f"  total     : {total:,}")
    print(f"  trainable : {trainable:,}  ({100. * trainable / total:.2f}%  — temporal layers only)")

    # ---- dummy input: B*T frames stacked in batch dimension (video-first) ----
    pixel_mean = torch.tensor([123.675, 116.280, 103.530], device=DEVICE).view(1, 3, 1, 1)
    pixel_std  = torch.tensor([58.395,  57.120,  57.375 ], device=DEVICE).view(1, 3, 1, 1)
    frames_raw = torch.rand(B * T, 3, H, W, device=DEVICE) * 255.0
    frames     = (frames_raw - pixel_mean) / pixel_std

    # ---- backbone: process all B*T frames as a single large batch ----
    print(f"\nBackbone forward on B*T={B*T} frames …")
    with torch.no_grad():
        features = backbone(frames)   # {res2/3/4/5: (B*T, C, H', W')}
    print("  " + ", ".join(f"{k}: {tuple(v.shape)}" for k, v in features.items()))

    # ---- dummy targets: N_CLICKS shared point prompts per video ----
    # Format: normalised cxcywh, same as in prepare_targets_interactive.
    # Using a near-zero wh makes the box behave as a point click.
    rng = torch.Generator(device=DEVICE).manual_seed(0)
    clicks = torch.rand(B, N_CLICKS, 4, generator=rng, device=DEVICE) * 0.5 + 0.25
    clicks[:, :, 2:] = 0.01   # tiny box → point click
    targets = [
        {
            "points": clicks[b],                                              # (N_CLICKS, 4)
            "pb":     torch.zeros(N_CLICKS, dtype=torch.long, device=DEVICE), # 0 = point
        }
        for b in range(B)
    ]

    # ---- forward pass ----
    prediction_switch = {"part": False, "whole": False, "seg": True, "det": True}
    print(f"\nVideo forward: B={B}, T={T}, N_CLICKS={N_CLICKS} …")
    head.eval()
    with torch.no_grad():
        out, _ = head(
            features,
            targets=targets,
            task="demo",
            extra=prediction_switch,
            num_frames=T,
        )

    print("\nOutput shapes  (B*T batch, video-first ordering):")
    if out["pred_masks"] is not None:
        pm = out["pred_masks"]
        print(f"  pred_masks  : {tuple(pm.shape)}")
        print(f"               → .view({B}, {T}, {pm.shape[1]}, {pm.shape[2]}, {pm.shape[3]}) "
              f"for per-video per-frame masks")
    print(f"  pred_ious   : {tuple(out['pred_ious'].shape)}")
    print(f"  pred_boxes  : {tuple(out['pred_boxes'].shape)}")
    print("\nDone.")
