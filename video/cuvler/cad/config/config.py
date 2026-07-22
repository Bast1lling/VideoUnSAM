# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Shahaf Arica from https://github.com/facebookresearch/CutLER/blob/main/cutler/config/cutler_config.py

def add_cuvler_config(cfg):
    cfg.DATALOADER.COPY_PASTE = False
    cfg.DATALOADER.COPY_PASTE_RATE = 0.0
    cfg.DATALOADER.COPY_PASTE_MIN_RATIO = 0.5
    cfg.DATALOADER.COPY_PASTE_MAX_RATIO = 1.0
    cfg.DATALOADER.COPY_PASTE_RANDOM_NUM = True
    cfg.DATALOADER.VISUALIZE_COPY_PASTE = False
    cfg.DATALOADER.COPY_PASTE_PROPORTIONS = (0.2, 0.4)

    cfg.MODEL.ROI_HEADS.USE_DROPLOSS = False
    cfg.MODEL.ROI_HEADS.DROPLOSS_IOU_THRESH = 0.0
    cfg.MODEL.ROI_HEADS.COSINE_SCALE = 20.0
    cfg.MODEL.BACKBONE.FREEZE = False
    cfg.MODEL.ROI_HEADS.FREEZE_FEAT = False
    cfg.MODEL.PROPOSAL_GENERATOR.FREEZE = False

    cfg.MODEL.ROI_HEADS.USE_SOFT_TARGETS = False

    # Federated / sigmoid-CE box-head keys. These were part of detectron2's stock
    # default config in older versions but are absent in some builds; register them
    # here with detectron2's defaults so cfg access in FastRCNNOutputLayers.from_config
    # works regardless of the installed detectron2. All inference-neutral (fed loss and
    # sigmoid CE are training-only and off by default -> standard softmax scoring).
    if not hasattr(cfg.MODEL.ROI_BOX_HEAD, "USE_FED_LOSS"):
        cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False
    if not hasattr(cfg.MODEL.ROI_BOX_HEAD, "USE_SIGMOID_CE"):
        cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
    if not hasattr(cfg.MODEL.ROI_BOX_HEAD, "FED_LOSS_NUM_CLASSES"):
        cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = 50
    if not hasattr(cfg.MODEL.ROI_BOX_HEAD, "FED_LOSS_FREQ_WEIGHT_POWER"):
        cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER = 0.5

    cfg.SOLVER.BASE_LR_MULTIPLIER = 1
    cfg.SOLVER.BASE_LR_MULTIPLIER_NAMES = []

    """
    Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.
    """

    cfg.TEST.NO_SEGM = False
