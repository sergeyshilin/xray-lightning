#!/usr/bin/env python

import hydra
import pytorch_lightning as pl

from omegaconf import DictConfig

from model import XrayDetection
from utils import load_obj, flatten_omegaconf, set_seed


@hydra.main(config_path="config.yaml")
def train(cfg: DictConfig) -> None:
    """
    Run model training.

    Parameters
    ----------
    cfg : DictConfig
        Project configuration object
    """
    model = load_obj(cfg.model.backbone.class_name)
    model = model(**cfg.model.backbone.params)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    head = load_obj(cfg.model.head.class_name)

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = head(
        in_features, cfg.model.head.params.num_classes
    )

    set_seed(cfg.training.seed)
    hparams = flatten_omegaconf(cfg)
    xray_detection = XrayDetection(hparams=hparams, cfg=cfg, model=model)

    callbacks = xray_detection.get_callbacks()
    loggers = xray_detection.get_loggers()

    trainer = pl.Trainer(
        logger=loggers,
        early_stop_callback=callbacks["early_stopping"],
        checkpoint_callback=callbacks["model_checkpoint"],
        **cfg.trainer,
    )
    trainer.fit(xray_detection)


if __name__ == "__main__":
    train()
