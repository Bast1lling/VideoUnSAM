"""Foundational DINOv3 feature experiments.

A scratch-pad of small, self-contained probes over DINOv3 patch features,
each with a Gradio view. Unlike ``mask_extraction/`` (algorithmic masks) and
``unsup_seg/`` (a trained U-Net), nothing here is a full pipeline — these are
quick "what does the feature field itself look like?" experiments.

Experiment 1 — **feature-edge detection + graph segmentation** ([edges.py](edges.py)):
a Sobel-like edge map built from the cosine (or euclidean) distance between each
patch and its 4- or 8-neighbours (visualised as a greyscale boundary map), plus
a neighbour-graph cut that turns those same distances into connected-component
masks (cut every connection above a threshold, colour what stays connected).

Run::

    uv run python -m feature_experiments.app
"""
