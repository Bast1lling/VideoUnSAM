"""VideoUnSAM — interactive Gradio demo.

User selects a DAVIS clip (or picks a custom one), clicks on the object in
frame 0, and the pipeline runs CuVLER+conquer → Sinkhorn OT → periodic reseed
→ optional Dense CRF refinement, outputting a side-by-side video.

Run:
    python demo.py
    python demo.py --share   # public link
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import subprocess

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "divide_and_conquer"))

from video.features.dinov3_dense import DenseDINOv3
from video.loaders import davis
from video.divide.cuvler_divide import CuVLERDivider
from video.divide.conquer import load_backbone, run_conquer
from video.propagation.sinkhorn_ot import propagate_patch, _mask_to_patch_indicator
from video.refine.dense_crf import crf_refine, crf_confidence

RESEED_INTERVAL = 10
RESEED_THRESH = 0.3
RESEED_CROP_CONTEXT = 1.5  # expand OT mask bbox by this factor for focused re-seed proposals
OT_THRESH = 0.5
OT_BLUR = 0.05
OT_COLOR_WEIGHT = 0.2      # LAB color cost blended into Sinkhorn cost matrix
CRF_CONF = 0.65
PROBE_FUSE_WEIGHT = 0.5    # heat = (1-w)*OT + w*probe ; appearance probe vs OT propagation
PROBE_STEPS = 100          # gradient steps to train the frame-0 instance probe
PROBE_LR = 0.01

_PREVIEW_CACHE: dict[str, str] = {}  # clip -> path to preview mp4


def _build_preview(clip: str) -> str:
    """Compile raw DAVIS frames into a preview MP4 (cached after first call)."""
    if clip in _PREVIEW_CACHE:
        return _PREVIEW_CACHE[clip]

    n = davis.num_frames(clip)
    frame0 = davis.load_frame(clip, 0)
    H, W = frame0.shape[:2]

    tmp = tempfile.NamedTemporaryFile(suffix="_preview_raw.mp4", delete=False)
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, 10, (W, H))
    for i in range(n):
        frame = davis.load_frame(clip, i)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    out_path = tmp.name.replace("_preview_raw.mp4", "_preview.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp.name, "-vcodec", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "23", out_path],
        capture_output=True,
    )
    result = out_path if Path(out_path).exists() else tmp.name
    _PREVIEW_CACHE[clip] = result
    return result


DAVIS_VAL_20 = [
    "blackswan", "bmx-trees", "breakdance", "camel", "car-roundabout",
    "car-shadow", "cows", "dance-twirl", "dog", "drift-chicane",
    "drift-straight", "goat", "horsejump-high", "kite-surf", "libby",
    "motocross-jump", "paragliding-launch", "parkour", "scooter-black", "soapbox",
]
ALL_CLIPS = davis.list_clips()

print("Loading models…")
_dino = DenseDINOv3()
_divider = CuVLERDivider(score_thresh=0.35)
_conquer_bb = load_backbone()
print("Models ready.")


# ── helpers ──────────────────────────────────────────────────────────────────

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union + 1e-6)


def _mask_to_patch(mask: np.ndarray, gh: int = 64, gw: int = 64) -> torch.Tensor:
    try:
        return _mask_to_patch_indicator(mask, gh, gw).cuda()
    except ValueError:
        return torch.full((gh * gw,), 1.0 / (gh * gw), device="cuda")


def _iou_pick(props: list, ref: np.ndarray, click_xy: tuple | None) -> np.ndarray | None:
    """Pick the proposal containing the click with best IoU vs ref; fallback to best-area."""
    if not props:
        return None
    if click_xy is not None:
        cx, cy = click_xy
        containing = [m for m in props if m[int(cy), int(cx)] > 0]
        if containing:
            return max(containing, key=lambda m: _iou(m, ref))
    return max(props, key=lambda m: _iou(m, ref))


def _crop_proposals(frame: np.ndarray, ref_mask: np.ndarray, context: float) -> list:
    """Run CuVLER on a crop around ref_mask's bbox, return proposals in full-frame coords."""
    H, W = frame.shape[:2]
    ys, xs = np.where(ref_mask > 0)
    if len(ys) == 0:
        return []
    mcx = (xs.min() + xs.max()) / 2
    mcy = (ys.min() + ys.max()) / 2
    hx = max((xs.max() - xs.min()) / 2 * context, 32)
    hy = max((ys.max() - ys.min()) / 2 * context, 32)
    x0 = max(0, int(mcx - hx))
    y0 = max(0, int(mcy - hy))
    x1 = min(W, int(mcx + hx))
    y1 = min(H, int(mcy + hy))
    if (x1 - x0) < 32 or (y1 - y0) < 32:
        return []
    crop_masks = _divider.predict(frame[y0:y1, x0:x1])
    full = []
    for m in crop_masks:
        fm = np.zeros((H, W), dtype=np.uint8)
        fm[y0:y1, x0:x1] = m
        full.append(fm)
    return full


def _color_cost(frame_a: np.ndarray, frame_b: np.ndarray,
                gh_a: int, gw_a: int, gh_b: int, gw_b: int) -> torch.Tensor:
    """Per-patch LAB color distance [N_a, N_b], normalised to [0, 1]."""
    fa = cv2.cvtColor(frame_a, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    fb = cv2.cvtColor(frame_b, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    fa_t = torch.from_numpy(fa.transpose(2, 0, 1)).unsqueeze(0)
    fb_t = torch.from_numpy(fb.transpose(2, 0, 1)).unsqueeze(0)
    fa_p = F.adaptive_avg_pool2d(fa_t, (gh_a, gw_a))[0].permute(1, 2, 0).reshape(-1, 3)
    fb_p = F.adaptive_avg_pool2d(fb_t, (gh_b, gw_b))[0].permute(1, 2, 0).reshape(-1, 3)
    dist = (fa_p[:, None, :] - fb_p[None, :, :]).norm(dim=-1)
    return OT_COLOR_WEIGHT * dist / (dist.max() + 1e-8)


def _patch_labels(mask: np.ndarray, gh: int, gw: int) -> torch.Tensor:
    """Binary per-patch label [gh*gw] — patch is positive if >50% covered by mask."""
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    pooled = F.adaptive_avg_pool2d(m, (gh, gw))[0, 0]
    return (pooled.flatten() > 0.5).float().cuda()


def _train_probe(feats_flat: torch.Tensor, y: torch.Tensor,
                 steps: int = PROBE_STEPS, lr: float = PROBE_LR) -> nn.Module:
    """Test-time adaptation: train a 1-layer linear probe to recognise the clicked
    instance. Supervision is the unsupervised seed mask (no labels). Frozen DINOv3
    features in, instance-discriminative logit out — learns to separate this object
    from look-alike distractors (a second dancer, similar background texture) that
    cosine-similarity OT cannot tell apart.
    """
    D = feats_flat.shape[1]
    probe = nn.Linear(D, 1).cuda()
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    n_pos = y.sum().clamp_min(1)
    n_neg = (1 - y).sum().clamp_min(1)
    lossf = nn.BCEWithLogitsLoss(pos_weight=(n_neg / n_pos).detach())
    feats_flat = feats_flat.detach()
    for _ in range(steps):
        opt.zero_grad()
        lossf(probe(feats_flat).squeeze(-1), y).backward()
        opt.step()
    probe.eval()
    return probe


@torch.no_grad()
def _probe_score(probe: nn.Module, feats_norm: torch.Tensor) -> torch.Tensor:
    """Per-patch instance probability [N] in [0,1] for a [gh, gw, D] feature grid."""
    flat = feats_norm.reshape(-1, feats_norm.shape[-1])
    return torch.sigmoid(probe(flat).squeeze(-1))


def _overlay(frame: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.45) -> np.ndarray:
    out = frame.copy()
    m = mask.astype(bool)
    for c, v in enumerate(color):
        out[:, :, c] = np.where(m, (1 - alpha) * frame[:, :, c] + alpha * v, frame[:, :, c])
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


# ── core pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(clip: str, click_x: float, click_y: float,
                 refine: bool, feat_size: int = 1024, use_probe: bool = True,
                 progress=gr.Progress()) -> str:
    """Run the full pipeline and return path to output video."""
    n = davis.num_frames(clip)
    frame0 = davis.load_frame(clip, 0)
    H, W = frame0.shape[:2]

    click_xy = (click_x * W, click_y * H)

    img_sized = cv2.resize(frame0, (feat_size, feat_size))
    with torch.no_grad():
        feats_prev = _dino.extract(img_sized, normalize=False)["feats"].cuda().float()
    feats_prev_norm = F.normalize(feats_prev, dim=-1)
    gh_feat, gw_feat = feats_prev_norm.shape[:2]  # 64 at 1024px, 128 at 2048px

    progress(0.05, desc="Seeding frame 0 with CuVLER + conquer…")
    proposals = _divider.predict(frame0)
    proposals = run_conquer(_conquer_bb, frame0, proposals)
    # Frame 0: pick seed containing click.
    # If the largest candidate covers > 40% of the frame it's a merged blob
    # (e.g. all 3 dancers detected as one region) — prefer smallest in that case.
    cx0, cy0 = click_xy
    containing0 = [m for m in proposals if m[int(cy0), int(cx0)] > 0]
    if containing0:
        largest0 = max(containing0, key=lambda m: m.sum())
        if largest0.sum() > H * W * 0.40:
            valid0 = [m for m in containing0 if m.sum() >= H * W * 0.005]
            seed = min(valid0, key=lambda m: m.sum()) if valid0 else largest0
        else:
            seed = largest0
    else:
        seed = max(proposals, key=lambda m: m.sum()) if proposals else None
    if seed is None:
        seed = np.zeros((H, W), dtype=np.uint8)

    cur_mask = seed.astype(np.uint8)
    patch = _mask_to_patch(cur_mask, gh=gh_feat, gw=gw_feat)
    frame_prev = frame0

    # Test-time adaptation: train a frame-0 instance probe on the seed mask.
    probe = None
    if use_probe and cur_mask.sum() > 0:
        progress(0.04, desc="Training instance probe (test-time adaptation)…")
        y0 = _patch_labels(cur_mask, gh_feat, gw_feat)
        probe = _train_probe(feats_prev_norm.reshape(-1, feats_prev_norm.shape[-1]), y0)

    frames_out = []

    # frame 0
    soft0 = cur_mask.astype(np.float32)
    display0 = crf_refine(frame0, soft0, dino_feats=_dino.extract(img_sized, normalize=False)["feats"].cpu()) \
        if refine and crf_confidence(soft0) >= CRF_CONF else cur_mask
    frames_out.append(_overlay(frame0, display0, (0, 120, 255)))

    for fidx in range(1, n):
        progress(0.05 + 0.90 * fidx / n, desc=f"Frame {fidx}/{n-1}")
        frame = davis.load_frame(clip, fidx)
        img_sized = cv2.resize(frame, (feat_size, feat_size))

        with torch.no_grad():
            feats_cur_raw = _dino.extract(img_sized, normalize=False)["feats"]
            feats_cur = feats_cur_raw.cuda().float()
        feats_cur_norm = F.normalize(feats_cur, dim=-1)
        gh_cur, gw_cur = feats_cur_norm.shape[:2]

        color_cost = _color_cost(frame_prev, frame, gh_feat, gw_feat, gh_cur, gw_cur)
        heat = propagate_patch(feats_prev_norm, feats_cur_norm, patch, blur=OT_BLUR,
                               cost_addend=color_cost)
        heat = heat / (heat.max() + 1e-8)  # [N_b] OT heat, peak-normalised

        # Fuse with the appearance probe: OT carries spatial/temporal coherence,
        # the memoryless probe re-acquires the instance after occlusion and rejects
        # look-alike distractors. The fused (probe-corrected) mask feeds back into
        # the chain, so a transient overlap no longer causes permanent identity loss.
        if probe is not None:
            score = _probe_score(probe, feats_cur_norm)  # [N_b] in [0,1]
            heat = (1.0 - PROBE_FUSE_WEIGHT) * heat + PROBE_FUSE_WEIGHT * score

        heat_up = F.interpolate(
            heat.reshape(1, 1, gh_cur, gw_cur), size=(H, W), mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
        soft_up = heat_up / (heat_up.max() + 1e-8)
        ot_mask = (soft_up > OT_THRESH).astype(np.uint8)
        cur_mask = ot_mask
        patch = _mask_to_patch(ot_mask, gh=gh_feat, gw=gw_feat)

        if RESEED_INTERVAL > 0 and fidx % RESEED_INTERVAL == 0:
            props = _divider.predict(frame)
            props += _crop_proposals(frame, ot_mask, RESEED_CROP_CONTEXT)
            props = run_conquer(_conquer_bb, frame, props)
            candidate = _iou_pick(props, ot_mask, click_xy)
            if candidate is not None and _iou(ot_mask, candidate) >= RESEED_THRESH:
                cur_mask = candidate
                patch = _mask_to_patch(cur_mask, gh=gh_feat, gw=gw_feat)

        if refine and crf_confidence(soft_up) >= CRF_CONF:
            display = crf_refine(frame, soft_up, dino_feats=feats_cur_raw.cpu())
        else:
            display = cur_mask

        frames_out.append(_overlay(frame, display, (0, 120, 255)))
        feats_prev_norm = feats_cur_norm
        frame_prev = frame

    progress(0.97, desc="Encoding video…")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    h, w = frames_out[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, 10, (w, h))
    for f in frames_out:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

    # re-encode to H.264 for browser playback
    out_path = tmp.name.replace(".mp4", "_h264.mp4")
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp.name, "-vcodec", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "20", out_path],
        capture_output=True,
    )
    progress(1.0, desc="Done.")
    return out_path if Path(out_path).exists() else tmp.name


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def load_frame0(clip: str):
    frame = davis.load_frame(clip, 0)
    n = davis.num_frames(clip)
    preview = _build_preview(clip)
    return frame, f"{clip} — {n} frames", preview


def on_click(clip: str, refine: bool, evt: gr.SelectData):
    frame = davis.load_frame(clip, 0)
    H, W = frame.shape[:2]
    cx_norm = evt.index[0] / W
    cy_norm = evt.index[1] / H
    # draw crosshair on preview
    preview = frame.copy()
    cv2.drawMarker(preview, (evt.index[0], evt.index[1]), (255, 50, 50),
                   cv2.MARKER_CROSS, 30, 3)
    return preview, cx_norm, cy_norm


def on_run(clip: str, cx_norm: float, cy_norm: float, refine: bool, quality: str,
           use_probe: bool, progress=gr.Progress()):
    if cx_norm is None:
        return None, "Click on the object in frame 0 first."
    feat_size = 2048 if quality == "High (2048px — sharper boundaries, ~8× slower)" else 1024
    video_path = run_pipeline(clip, cx_norm, cy_norm, refine, feat_size=feat_size,
                              use_probe=use_probe, progress=progress)
    return video_path, "Done."


with gr.Blocks(title="VideoUnSAM") as demo:
    gr.Markdown(
        "## VideoUnSAM — Unsupervised Video Object Segmentation\n"
        "Select a clip, click the object in frame 0, then press **Run**.\n"
        "No labels. No fine-tuning. Fully unsupervised."
    )

    with gr.Row():
        with gr.Column(scale=1):
            clip_dd = gr.Dropdown(
                choices=ALL_CLIPS,
                value="blackswan",
                label="DAVIS clip",
            )
            refine_cb = gr.Checkbox(value=True, label="Dense CRF refinement (unsupervised boundary sharpening)")
            probe_cb = gr.Checkbox(value=True, label="Instance probe (test-time adaptation — re-acquires after occlusion, rejects look-alikes)")
            quality_radio = gr.Radio(
                choices=["Standard (1024px — fast)", "High (2048px — sharper boundaries, ~8× slower)"],
                value="Standard (1024px — fast)",
                label="Feature resolution",
            )
            info_txt = gr.Textbox(label="Clip info", interactive=False)
            run_btn = gr.Button("Run", variant="primary")

        with gr.Column(scale=2):
            preview_video = gr.Video(label="Clip preview", autoplay=True, loop=True, interactive=False)
            frame0_img = gr.Image(label="Frame 0 — click the object", interactive=True)
            cx_state = gr.State(value=None)
            cy_state = gr.State(value=None)

    with gr.Row():
        out_video = gr.Video(label="Segmentation result", autoplay=True)
        status_txt = gr.Textbox(label="Status", interactive=False)

    # on clip change: load frame 0 + preview video
    clip_dd.change(fn=load_frame0, inputs=clip_dd, outputs=[frame0_img, info_txt, preview_video])
    demo.load(fn=load_frame0, inputs=clip_dd, outputs=[frame0_img, info_txt, preview_video])

    # on image click: mark crosshair and store normalised coords
    frame0_img.select(fn=on_click, inputs=[clip_dd, refine_cb], outputs=[frame0_img, cx_state, cy_state])

    # run pipeline
    run_btn.click(
        fn=on_run,
        inputs=[clip_dd, cx_state, cy_state, refine_cb, quality_radio, probe_cb],
        outputs=[out_video, status_txt],
    )

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    demo.launch(share=args.share, server_port=args.port)
