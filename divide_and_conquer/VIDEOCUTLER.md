
# VideoCutLER Pipeline Overview

VideoCutLER trains an unsupervised video instance segmentation model using only ImageNet-1K images — no human annotations, no real videos, no motion estimation. The pipeline has three stages.

---

## Stage 1 — MaskCut: Pseudo-Mask Generation

**Location:** `../maskcut/` (sibling directory, outside `videocutler/`)

MaskCut runs once as a preprocessing step and produces a COCO-format JSON of pseudo-instance masks for all ImageNet images. This JSON is then consumed by Stage 3 as if it were ground-truth supervision.

| File | Role |
|------|------|
| `maskcut/maskcut.py` | Core algorithm: applies normalized graph cuts (NCut) iteratively on DINO patch-feature similarity matrices to segment multiple objects per image |
| `maskcut/dino.py` | DINO ViT backbone used to extract patch-level features |
| `maskcut/crf.py` | Dense CRF post-processing to sharpen mask boundaries |
| `maskcut/predict.py` | Single-machine inference script |
| `maskcut/maskcut_with_submitit.py` | Distributed inference script (SLURM) |
| `maskcut/merge_jsons.py` | Merges per-shard JSON outputs into one annotation file |

The final output is a single JSON file placed under `DETECTRON2_DATASETS/imagenet/annotations/`. A pre-generated version is available for download (see `README.md`).

---

## Stage 2 — ImageCut2Video: Synthetic Video Generation

**Location:** `mask2former_video/data_video/`

There is no standalone script for this stage. ImageCut2Video is implemented entirely inside the data loading pipeline and runs on-the-fly during training. Given a single ImageNet image with its MaskCut pseudo-masks, it applies independent random spatial augmentations to the image `N` times (default `N=2`) to produce a synthetic multi-frame clip. Each frame gets a different crop/scale/flip, simulating inter-frame motion, while the instance IDs remain consistent across frames — forming pseudo mask trajectories.

| File | Role |
|------|------|
| `mask2former_video/data_video/dataset_mapper.py` | `CocoClipDatasetMapper` (line 304): the ImageCut2Video implementation. Takes one image, samples `sampling_frame_num` independently augmented versions, and packages them as a video with tracked instances |
| `mask2former_video/data_video/augmentation.py` | `build_augmentation()`: defines the per-frame random transforms — random crop, resize (shortest-edge), horizontal flip, and optionally brightness/contrast/saturation/rotation |
| `mask2former_video/data_video/build.py` | Builds the PyTorch DataLoader that wraps `CocoClipDatasetMapper` |
| `mask2former_video/data_video/datasets/builtin.py` | Registers the ImageNet dataset with Detectron2's `DatasetCatalog` |

`CocoClipDatasetMapper` is selected during ImageNet training; `YTVISDatasetMapper` (in the same file, line 146) is used at evaluation time on real YouTubeVIS videos.

---

## Stage 3 — Model Training: Unsupervised Video Instance Segmentation

**Location:** `mask2former_video/` and project root

A Video Mask2Former model is trained on the synthetic clips from Stage 2, treating the MaskCut pseudo-masks as ground truth. At inference time the model generalises to real videos.

### Entry points

| File | Role |
|------|------|
| `train_net_video.py` | Main training and evaluation script |
| `configs/imagenet_video/video_mask2former_R50_cls_agnostic.yaml` | Primary training config (ResNet-50 backbone, class-agnostic) |
| `configs/imagenet_video/videocutler_eval_ytvis2019.yaml` | Evaluation config for YouTubeVIS-2019 |
| `configs/imagenet_video/videocutler_eval_ytvis2021.yaml` | Evaluation config for YouTubeVIS-2021 |

### Model architecture

| File | Role |
|------|------|
| `mask2former_video/video_maskformer_model.py` | Top-level `VideoMaskFormer` model: backbone → pixel decoder → transformer decoder → per-frame mask predictions |
| `mask2former_video/modeling/transformer_decoder/video_mask2former_transformer_decoder.py` | Video transformer decoder: propagates object queries across frames to produce temporally-consistent instance masks |
| `mask2former_video/modeling/transformer_decoder/position_encoding.py` | Positional encodings for the transformer |

### Training objective

| File | Role |
|------|------|
| `mask2former_video/modeling/criterion.py` | Loss computation: mask BCE + dice loss + classification loss, summed over frames |
| `mask2former_video/modeling/matcher.py` | Hungarian matcher: assigns predicted instances to pseudo-GT masks per-frame |

### Evaluation

| File | Role |
|------|------|
| `eval_ytvis.py` | Post-processes model outputs and computes AP on YouTubeVIS |
| `mask2former_video/data_video/ytvis_eval.py` | YouTubeVIS evaluator wrapper |
| `mask2former_video/data_video/datasets/ytvis.py` | YouTubeVIS dataset loader |

---

## Data Flow Summary

```
ImageNet images
      │
      ▼ Stage 1 (maskcut/maskcut.py)
Pseudo-mask JSON  ──────────────────────────────────────────────────────┐
                                                                         │
      ┌──────────────────────────────────────────────────────────────────┘
      │
      ▼ Stage 2 (CocoClipDatasetMapper, on-the-fly during training)
Synthetic video clips  (N augmented frames of one image + tracked masks)
      │
      ▼ Stage 3 (train_net_video.py + VideoMaskFormer)
Trained UVIS model
      │
      ▼ Zero-shot evaluation (eval_ytvis.py)
AP on YouTubeVIS-2019 / 2021
```
