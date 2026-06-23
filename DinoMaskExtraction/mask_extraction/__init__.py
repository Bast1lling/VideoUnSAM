"""
Mask extraction from DINOv3 feature grids.

Public surface
--------------
* Algorithms: :class:`MaskExtractor` (base), :func:`register_extractor`,
  :func:`get_extractor`, ``EXTRACTORS`` (name -> class). Importing this package
  registers all bundled algorithms (currently ``greedy_graph``).
* Similarity: :func:`get_similarity`, :func:`pairwise`, ``SIMILARITIES``.
* Visualisation: :func:`color_labels`, :func:`overlay_masks`, :func:`hstack`.

Example
-------
>>> from mask_extraction import get_extractor
>>> seg = get_extractor("greedy_graph", similarity="cosine", quantile=0.6)
>>> labels = seg(feature_grid)          # (H, W, C) -> (H, W) mask ids
"""
from .base import (
    EXTRACTORS,
    MaskExtractor,
    UnionFind,
    get_extractor,
    neighbour_edges,
    prototype_similarity_map,
    register_extractor,
    relabel,
)
from .similarity import (
    DIRECTION_ONLY,
    MAGNITUDE_AWARE,
    SIMILARITIES,
    get_similarity,
    pairwise,
    register_similarity,
)
from .preprocess import pca_reduce, reduce_features, spatial_smooth
from .postprocess import refine_crf
from .visualize import (
    color_labels,
    conquer_color,
    hstack,
    label_triple,
    overlay_hierarchy,
    overlay_labels_crop,
    overlay_mask,
    overlay_masks,
    palette_color,
)

# import algorithm modules for their registration side effects
from . import greedy_graph      # noqa: F401  (registers "greedy_graph")
from . import greedy_multistage  # noqa: F401  (registers "greedy_multistage")
from . import greedy_multistage2  # noqa: F401  (registers "greedy_multistage2")
from . import frontier_grow     # noqa: F401  (registers "frontier_grow")
from . import sequential_grow   # noqa: F401  (registers "sequential_grow")
from . import maskcut           # noqa: F401  (registers "maskcut")

__all__ = [
    "EXTRACTORS", "MaskExtractor", "UnionFind", "get_extractor",
    "neighbour_edges", "prototype_similarity_map", "register_extractor", "relabel",
    "SIMILARITIES", "DIRECTION_ONLY", "MAGNITUDE_AWARE",
    "get_similarity", "pairwise", "register_similarity",
    "pca_reduce", "reduce_features", "spatial_smooth", "refine_crf",
    "color_labels", "hstack", "overlay_masks", "overlay_mask",
    "overlay_hierarchy", "overlay_labels_crop", "label_triple",
    "palette_color", "conquer_color",
]
