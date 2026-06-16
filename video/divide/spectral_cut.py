"""Training-free spectral-cut divide stage on DINOv3 dense features.

Drop-in replacement for the frozen ResNet-50 CutLER detector used in
divide_conquerV3.py. Instead of a learned ImageNet detector, the divide masks
are obtained by a normalized-cut on the *same* DINOv3 features the conquer stage
already uses — unifying the pipeline on a single backbone.

Two modes:
  maskcut  : iterative Fiedler bipartition + mask-out (TokenCut/MaskCut). Each
             round extracts the connected component around the most-salient
             seed patch, then removes it and repeats. Stops adaptively when the
             next foreground is too small / background-like — the number of
             masks is data-driven, not the fixed N of CutLER's MaskCut.
  spectral : multi-eigenvector spectral clustering. K is chosen by the largest
             eigengap of the normalized Laplacian, then k-means on the leading
             eigenvectors assigns patches to segments (Deep-Spectral-Methods
             flavour).

KNOWN LIMITATION (measured — see eval_divide_ar.py and _maskcut's caveat): on
textured natural scenes raw spectral saliency selects background texture as
readily as objects, so this is NOT yet a drop-in replacement for the trained
CutLER detector. It needs an objectness prior to pick the foreground side.

Both return full-frame binary masks [H, W] uint8 — exactly the format that
divide_conquerV3.run_conquer() expects from `predictor(image).pred_masks`.

To drop into divide_conquerV3.main(), replace:

    predictor = DefaultPredictor(cfg)
    ...
    predictions = predictor(image)               # image is BGR (cv2.imread)
    divide_masks_tensor = predictions["instances"].get("pred_masks")
    divide_masks = [divide_masks_tensor[i].cpu().numpy() for i in range(...)]

with:

    divide = SpectralDivide(DenseDINOv3())
    ...
    divide_masks = divide.predict(image[:, :, ::-1])   # predict() wants RGB
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

try:  # cv2 only needed for upsampling / connected components
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# --------------------------------------------------------------------------- #
# Graph construction + spectral primitives
# --------------------------------------------------------------------------- #
def _affinity(
    feats_flat: torch.Tensor,   # [N, D] L2-normalised
    tau: float,
    grid_hw: tuple[int, int],
    spatial_weight: float,
    spatial_sigma: float,
    attn: torch.Tensor | None = None,   # [N] objectness in [0, inf)
    attn_floor: float = 0.3,
) -> torch.Tensor:
    """Symmetric non-negative patch affinity from cosine similarity.

    A = relu(cos), optionally hard-thresholded at tau and blended with a
    Gaussian spatial-proximity kernel to discourage cuts that teleport across
    the frame (the same locality prior the OT propagator is missing).
    """
    A = torch.clamp(feats_flat @ feats_flat.T, min=0.0)        # [N, N] in [0, 1]
    if tau > 0.0:
        A = torch.where(A >= tau, A, torch.zeros_like(A))
    if spatial_weight > 0.0:
        gh, gw = grid_hw
        ys, xs = torch.meshgrid(
            torch.arange(gh, device=A.device, dtype=A.dtype) / max(gh - 1, 1),
            torch.arange(gw, device=A.device, dtype=A.dtype) / max(gw - 1, 1),
            indexing="ij",
        )
        xy = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=1)  # [N, 2]
        d2 = torch.cdist(xy, xy) ** 2
        S = torch.exp(-d2 / (2.0 * spatial_sigma ** 2))
        A = (1.0 - spatial_weight) * A + spatial_weight * S
    if attn is not None:
        # Objectness gate: shrink edges touching low-attention (background)
        # patches so the cut prefers to isolate foreground. Floor keeps the
        # background subgraph connected.
        w = attn.flatten().clamp_min(0)
        w = w / (w.max() + 1e-8)
        gate = torch.sqrt(w[:, None] * w[None, :])
        A = A * ((1.0 - attn_floor) * gate + attn_floor)
    A = A.clone()
    A.fill_diagonal_(0.0)
    return A + 1e-5  # keep the graph connected


def _laplacian_eigs(A: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smallest-k eigenpairs of the symmetric normalised Laplacian L = I - D^-1/2 A D^-1/2.

    Returns (eigvals[:k] ascending, eigvecs[:, :k], d^-1/2) — the last is used to
    map a symmetric-Laplacian eigenvector back to the generalised NCut indicator.
    """
    d = A.sum(1)
    d_inv_sqrt = torch.rsqrt(d.clamp_min(1e-12))
    L = torch.eye(A.shape[0], device=A.device, dtype=A.dtype) \
        - d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    L = 0.5 * (L + L.T)                       # symmetrise against fp drift
    evals, evecs = torch.linalg.eigh(L)       # ascending
    return evals[:k], evecs[:, :k], d_inv_sqrt


# --------------------------------------------------------------------------- #
# Mode 1: MaskCut — iterative Fiedler bipartition + mask-out (TokenCut [40])
# --------------------------------------------------------------------------- #
#
# CAVEAT (measured on DAVIS, see eval_divide_ar.py): the seed heuristic assumes
# the most salient patch (max |Fiedler|) is the foreground object. On textured
# natural scenes (water, grass) the leading eigenvector's extreme is often
# *background texture*, so the first cut can grab the background instead of the
# object — the swan is cleanly present in eigvecs 2-3 but is not the dominant
# mode. This is the objectness gap that the *trained* CutLER detector fills and
# a training-free cut does not. An external objectness prior (e.g. the DINO
# [CLS]-attention foreground map) is needed to pick the right side/eigenvector.
def _grid_border(gh: int, gw: int, device) -> torch.Tensor:
    b = torch.zeros(gh, gw, dtype=torch.bool, device=device)
    b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = True
    return b.reshape(-1)


def _maskcut(
    A: torch.Tensor, gh: int, gw: int,
    *, max_iter: int, min_patch_frac: float, max_fg_frac: float,
    attn: torch.Tensor | None = None, min_attn_frac: float = 0.3,
) -> list[torch.Tensor]:
    """Returns a list of patch-index tensors, one per discovered foreground.

    Each iteration takes the Fiedler vector of the current (masked) graph to
    define two sides, then keeps the connected component around a seed patch.
    Seed selection:
      - with attn (objectness): foreground = the side with higher mean
        attention; seed = highest-attention patch in it. Stops when the most
        attended *remaining* patch falls below min_attn_frac of the global max
        (all objects consumed) — adaptive, objectness-driven.
      - without attn: TokenCut fallback — seed = max |Fiedler| (lands on the
        most spectrally-salient patch, which on textured scenes is often
        background; see module caveat).
    """
    N = gh * gw
    min_patches = max(1, int(min_patch_frac * N))
    border = _grid_border(gh, gw, A.device)
    A_work = A.clone()
    avail = None
    if attn is not None:
        avail = attn.flatten().clamp_min(0).to(A.device).clone()
        global_max = float(avail.max()) + 1e-9
    out: list[torch.Tensor] = []
    for _ in range(max_iter):
        if float(A_work.sum(1).max()) < 1e-4:
            break
        _, evecs, d_inv_sqrt = _laplacian_eigs(A_work, 2)
        centered = (evecs[:, 1] * d_inv_sqrt)
        centered = centered - centered.mean()

        if avail is not None:
            if float(avail.max()) < min_attn_frac * global_max:
                break  # nothing object-like left
            pos = centered > 0
            mp = pos.float()
            a_pos = (avail * mp).sum() / mp.sum().clamp_min(1)
            a_neg = (avail * (1 - mp)).sum() / (1 - mp).sum().clamp_min(1)
            side = pos if a_pos >= a_neg else ~pos
            seed = int((avail * side.float()).argmax())
        else:
            seed = int(centered.abs().argmax())
            side = (centered > 0) if centered[seed] > 0 else (centered < 0)

        grid = side.reshape(gh, gw).cpu().numpy().astype(np.uint8)
        n_cc, lab = cv2.connectedComponents(grid)
        comp = torch.from_numpy(
            (lab == lab[seed // gw, seed % gw]).reshape(-1)
        ).to(A.device)

        af = float(comp.float().mean())
        b_frac = float((comp & border).sum()) / float(border.sum().clamp_min(1))
        if comp.sum() >= min_patches and af <= max_fg_frac and b_frac < 0.6:
            out.append(torch.nonzero(comp, as_tuple=False).squeeze(1))
        A_work[comp, :] = 0.0
        A_work[:, comp] = 0.0
        if avail is not None:
            avail[comp] = 0.0
        elif comp.sum() < min_patches:
            break
    return out


# --------------------------------------------------------------------------- #
# Mode 2: multi-eigenvector spectral clustering, K from the eigengap
# --------------------------------------------------------------------------- #
def _kmeans(X: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):                                   # k-means++ seeding
        d2 = np.min(np.stack([((X - c) ** 2).sum(1) for c in centers]), axis=0)
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
        centers.append(X[rng.choice(n, p=probs)])
    centers = np.stack(centers)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        labels = dist.argmin(1)
        new = np.stack([
            X[labels == j].mean(0) if np.any(labels == j) else centers[j]
            for j in range(k)
        ])
        if np.allclose(new, centers):
            break
        centers = new
    return labels


def _spectral_groups(A: torch.Tensor, k_max: int, seed: int) -> list[torch.Tensor]:
    k_max = min(k_max, A.shape[0] - 1)
    evals, evecs, _ = _laplacian_eigs(A, k_max + 1)
    gaps = (evals[1:k_max + 1] - evals[:k_max]).cpu().numpy()
    k = int(np.argmax(gaps[1:]) + 2) if gaps.shape[0] > 1 else 2   # skip trivial gap, k>=2
    emb = F.normalize(evecs[:, 1:k], dim=1).cpu().numpy()
    labels = _kmeans(emb, k, seed=seed)
    return [torch.from_numpy(np.nonzero(labels == j)[0]).to(A.device)
            for j in range(k) if np.any(labels == j)]


# --------------------------------------------------------------------------- #
# Patches -> full-frame masks
# --------------------------------------------------------------------------- #
def _border_touch(mask: np.ndarray) -> float:
    edges = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return float(edges.mean())


def _groups_to_masks(
    groups: list[torch.Tensor],
    gh: int, gw: int, patch: int, H: int, W: int,
    *,
    min_area: float, max_area: float, border_drop: float, split_cc: bool,
) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for g in groups:
        grid = np.zeros(gh * gw, dtype=np.uint8)
        grid[g.cpu().numpy()] = 1
        grid = grid.reshape(gh, gw)
        up = cv2.resize(grid, (gw * patch, gh * patch), interpolation=cv2.INTER_NEAREST)
        up = up[:H, :W]

        components = []
        if split_cc:
            n_cc, lab = cv2.connectedComponents(up)
            components = [(lab == c).astype(np.uint8) for c in range(1, n_cc)]
        else:
            components = [up]

        for comp in components:
            af = comp.mean()
            if af < min_area or af > max_area:
                continue
            if af > 0.25 and _border_touch(comp) > border_drop:   # likely background
                continue
            masks.append(comp)
    return masks


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a) + np.count_nonzero(b) - inter
    return inter / union if union else 0.0


def _dedup(masks: list[np.ndarray], iou_thr: float) -> list[np.ndarray]:
    kept: list[np.ndarray] = []
    for m in sorted(masks, key=lambda x: np.count_nonzero(x), reverse=True):
        if all(_iou(m, k) <= iou_thr for k in kept):
            kept.append(m)
    return kept


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
class SpectralDivide:
    """DINOv3 spectral-cut divide stage. See module docstring."""

    def __init__(
        self,
        extractor,                       # DenseDINOv3 (or anything with .extract())
        mode: str = "maskcut",           # "maskcut" | "spectral"
        device: str = "cuda",
        # graph
        tau: float = 0.2,
        spatial_weight: float = 0.1,
        spatial_sigma: float = 0.2,
        # objectness (CLS-attention) prior — needs extractor(with_attention=True)
        use_attention: bool = True,
        attn_floor: float = 0.3,
        min_attn_frac: float = 0.3,
        # maskcut mode
        max_iter: int = 8,
        min_patch_frac: float = 0.01,
        max_fg_frac: float = 0.6,
        # spectral mode
        k_max: int = 8,
        seed: int = 0,
        # mask post-proc
        min_area: float = 0.001,
        max_area: float = 0.85,
        border_drop: float = 0.5,
        split_cc: bool = True,
        dedup_iou: float = 0.7,
    ):
        if cv2 is None:
            raise ImportError("SpectralDivide requires opencv (cv2).")
        if mode not in ("maskcut", "spectral"):
            raise ValueError(f"unknown mode {mode!r}")
        self.extractor = extractor
        self.mode = mode
        self.device = device
        self.tau = tau
        self.spatial_weight = spatial_weight
        self.spatial_sigma = spatial_sigma
        self.use_attention = use_attention
        self.attn_floor = attn_floor
        self.min_attn_frac = min_attn_frac
        self._warned_no_attn = False
        self.max_iter = max_iter
        self.min_patch_frac = min_patch_frac
        self.max_fg_frac = max_fg_frac
        self.k_max = k_max
        self.seed = seed
        self.min_area = min_area
        self.max_area = max_area
        self.border_drop = border_drop
        self.split_cc = split_cc
        self.dedup_iou = dedup_iou

    @torch.no_grad()
    def predict(self, frame_rgb: np.ndarray) -> list[np.ndarray]:
        """RGB uint8 [H, W, 3] -> list of binary masks [H, W] uint8 (full frame)."""
        out = self.extractor.extract(frame_rgb, normalize=True)
        gh, gw = out["grid_h"], out["grid_w"]
        H, W = out["H"], out["W"]
        feats = out["feats"].reshape(gh * gw, -1).to(self.device).float()

        attn = None
        if self.use_attention:
            attn = out.get("attn")
            if attn is None and not self._warned_no_attn:
                print("[SpectralDivide] use_attention=True but the extractor returned no "
                      "attention map — build DenseDINOv3(with_attention=True). Falling back "
                      "to the spectral-saliency seed.")
                self._warned_no_attn = True
            elif attn is not None:
                attn = attn.reshape(-1).to(self.device).float()

        A = _affinity(feats, self.tau, (gh, gw), self.spatial_weight, self.spatial_sigma,
                      attn=attn, attn_floor=self.attn_floor)

        if self.mode == "maskcut":
            groups = _maskcut(A, gh, gw, max_iter=self.max_iter,
                              min_patch_frac=self.min_patch_frac,
                              max_fg_frac=self.max_fg_frac,
                              attn=attn, min_attn_frac=self.min_attn_frac)
        else:
            groups = _spectral_groups(A, self.k_max, self.seed)

        masks = _groups_to_masks(
            groups, gh, gw, self.extractor.patch_size, H, W,
            min_area=self.min_area, max_area=self.max_area,
            border_drop=self.border_drop, split_cc=self.split_cc,
        )
        return _dedup(masks, self.dedup_iou)


if __name__ == "__main__":
    # Smoke test on one DAVIS frame.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from video.features.dinov3_dense import DenseDINOv3
    from video.loaders import davis

    clip = sys.argv[1] if len(sys.argv) > 1 else "blackswan"
    frame = davis.load_frame(clip, 0)
    div = SpectralDivide(DenseDINOv3(with_attention=True),
                         mode=sys.argv[2] if len(sys.argv) > 2 else "maskcut")
    masks = div.predict(frame)
    print(f"{clip}: {len(masks)} divide masks "
          f"(areas: {sorted(round(float(m.mean()), 3) for m in masks)})")
