# Preparing Dataset
## Preparing Training Dataset

To generate psuedo-masks for your datasets, please follow our `divide_and_conquer' stage, which generates pseudo-masks for all images in your dataset with unsupervised learning methods. Please run the script `divide_and_conquer/generate_pseudo_masks.sh' — it handles the entire pseudo-mask generation process and will save all the masks produced by our unsupervised labeling pipeline. You can choose to only generate pseudo-masks for a subset of your TRAIN_DATSETS via specifying the start-id and end-id.

Note that this script expects the following data structure:

```
UNLABELED_IMAGES/
  FOLDER1/
    Image1
    .......
    Imagen
  FOLDER2/
    Image1
    .......
    Imagen
  .......
```

We transfer our training pseudo labels to tsv format for faster dataloading. Please refer to [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM/blob/main/DATASET.md) for a detailed script to transform json format data to tsv format data.

Please set the environment variable
```shell
export TRAIN_DATASETS=path_to_your_tsv_directory
```
The structure and file names of the tsv directory should follow
```
$TRAIN_DATASETS/
  SAM-1.tsv
  SAM-1.lineidx
  SAM-2.tsv
  SAM-2.lineidx
  ......
  SAM-N.tsv
  SAM-N.lineidx
```

## Preparing Evaluation Dataset (Whole-Image-Segmentation)
By default we support 7 evaluation datasets for whole-image-segmentation evaluation: SA-1B, COCO, LVIS, ADE20K, EntitySeg, PartImagenet, and PACO.
Please set the root directory environment variable first
```shell
export DETECTRON2_DATASETS=path_to_your_root_evaluation_directory
```
and change the test dataset name in config file
```
cfg.DATASETS.TEST: ("unsam_{sa1b, ade20k, entity, paco, partimagenet, coco, lvis}_val")
```
The structure and file names of the tsv directory should follow
```
$DETECTRON2_DATASETS/
  sa1b/
    images/
    annotations/
      sa1b_val.json
  ade/
    images/
    annotations/
      ade_val.json
  entity/
    images/
    annotations/
      entityseg_val.json
  paco/
    images/
    annotations/
      paco_val.json
  partimagenet/
    images/
    annotations/
      partimagenet_val.json
  coco/
    val2017/
    annotations/
      instances_val2017.json
  lvis/
    images/
    annotations/
      lvis_v1_val.json
```
Since ade20k, entity, partimagenet don't have standard image id format, please name each image as {image_id}.jpg. You can adjust them in whole_image_segmentation/data/build.py/get_test_detection_datasets

## Preparing Evaluation Dataset (Promptable-Segmentation)
We support COCO evaluation for promptable segmentation. Please set the root directory environment variable first
```shell
export DETECTRON2_DATASETS=path_to_your_root_evaluation_directory
```
and refer to [MaskDINO](https://github.com/IDEA-Research/MaskDINO/blob/main/README.md) to prepare files under the evaluation directory
