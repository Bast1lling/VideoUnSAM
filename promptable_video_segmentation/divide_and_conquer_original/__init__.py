# promptable_video_segmentation/divide_and_conquer_original
#
# Faithful reproduction of the ORIGINAL divide_and_conquer/divide_conquer.py
# pipeline (CutLER + DINOv1 ViT-B/8 cosine spectral merge) with the original
# hyperparameters, packaged like the divide_and_conquer / divide_and_conquer2
# variants.
from .dico import (
    DiCoModel,
    overlay_mask,
    visualize_masks,
    visualize_divide_selected,
    visualize_conquer_crop,
    visualize_conquer_single,
    visualize_all,
    feature_pca_rgb,
    make_synthetic_video,
    palette_color,
    conquer_color,
    PALETTE,
    CONQUER_PALETTE,
    DEFAULT_THETAS,
    DEFAULT_LOCAL_SIZE,
    DEFAULT_KEPT_THRESH,
    DEFAULT_NMS_IOU,
    DEFAULT_NMS_STEP,
    DEFAULT_CONFIDENCE,
)
from .predictor import DEFAULT_CUTLER_CONFIG, DEFAULT_CUTLER_WEIGHTS
from .backbone import (
    DEFAULT_BACKBONE_URL,
    DEFAULT_BACKBONE_ARCH,
    DEFAULT_VIT_FEAT,
    DEFAULT_PATCH_SIZE,
    DEFAULT_FEAT_DIM,
)
from .iterative_merging import iterative_merge

__all__ = [
    "DiCoModel",
    "overlay_mask",
    "visualize_masks",
    "visualize_divide_selected",
    "visualize_conquer_crop",
    "visualize_conquer_single",
    "visualize_all",
    "feature_pca_rgb",
    "make_synthetic_video",
    "palette_color",
    "conquer_color",
    "PALETTE",
    "CONQUER_PALETTE",
    "DEFAULT_THETAS",
    "DEFAULT_LOCAL_SIZE",
    "DEFAULT_KEPT_THRESH",
    "DEFAULT_NMS_IOU",
    "DEFAULT_NMS_STEP",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_CUTLER_CONFIG",
    "DEFAULT_CUTLER_WEIGHTS",
    "DEFAULT_BACKBONE_URL",
    "DEFAULT_BACKBONE_ARCH",
    "DEFAULT_VIT_FEAT",
    "DEFAULT_PATCH_SIZE",
    "DEFAULT_FEAT_DIM",
    "iterative_merge",
]
