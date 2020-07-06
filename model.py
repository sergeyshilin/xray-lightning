from typing import Dict, List

import pytorch_lightning as pl
import torch

from omegaconf import DictConfig
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

from dataset import get_training_dataset
from logger import get_logger
from utils import collate_fn, load_obj
from utils.coco_eval import CocoEvaluator
from utils.coco_utils import get_coco_api_from_dataset, get_iou_types


class XrayDetection(pl.LightningModule):
    """Xray object detection pytorch module."""

    def __init__(self, hparams: DictConfig, cfg: DictConfig, model: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.hparams = hparams
        self.model = model

    def configure_optimizers(self):
        """TODO Add missing docstring."""
        if "decoder_lr" in self.cfg.optimizer.params.keys():
            params = [
                {
                    "params": self.model.decoder.parameters(),
                    "lr": self.cfg.optimizer.params.lr,
                },
                {
                    "params": self.model.encoder.parameters(),
                    "lr": self.cfg.optimizer.params.decoder_lr,
                },
            ]
            optimizer = load_obj(self.cfg.optimizer.class_name)(params)
        else:
            optimizer = load_obj(self.cfg.optimizer.class_name)(
                self.model.parameters(), **self.cfg.optimizer.params
            )
            scheduler = load_obj(self.cfg.scheduler.class_name)(
                optimizer, **self.cfg.scheduler.params
            )

        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": self.cfg.scheduler.step,
                    "monitor": self.cfg.scheduler.monitor,
                }
            ],
        )

    def forward(self, x, *args, **kwargs):
        """TODO Add missing docstring."""
        return self.model(x)

    def get_callbacks(self) -> Dict[str, Callback]:
        """
        Get a list of pytorch callbacks for this model.

        Returns
        -------
        Dict[str, Callback]
            List of callbacks
        """
        early_stopping = EarlyStopping(
            **self.cfg.callbacks.early_stopping.params
        )
        model_checkpoint = ModelCheckpoint(
            **self.cfg.callbacks.model_checkpoint.params
        )

        return {
            "early_stopping": early_stopping,
            "model_checkpoint": model_checkpoint,
        }

    def get_loggers(self) -> List:
        """TODO Add missing docstring."""
        return [TensorBoardLogger(save_dir=self.cfg.logging.logs_dir)]

    def prepare_data(self):
        """TODO Add missing docstring."""
        get_logger().info("Loading training dataset...")
        datasets = get_training_dataset(self.cfg)
        self.train_dataset = datasets["train"]
        self.valid_dataset = datasets["valid"]

    def train_dataloader(self):
        """TODO Add missing docstring."""
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            shuffle=True,
            collate_fn=collate_fn,
        )

        return train_loader

    def training_step(self, batch, batch_idx):
        """TODO Add missing docstring."""
        images, targets, image_ids = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        # separate losses
        loss_dict = self.model(images, targets)

        # total loss
        loss = sum(loss for loss in loss_dict.values())

        return {"loss": loss, "log": loss_dict, "progress_bar": loss_dict}

    def val_dataloader(self):
        """TODO Add missing docstring."""
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # prepare coco evaluator
        coco = get_coco_api_from_dataset(valid_loader.dataset)
        iou_types = get_iou_types(self.model)
        self.coco_evaluator = CocoEvaluator(coco, iou_types)

        return valid_loader

    def validation_epoch_end(self, outputs):
        """TODO Add missing docstring."""
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()

        # coco main metric
        metric = self.coco_evaluator.coco_eval["bbox"].stats[0]
        metric = torch.as_tensor(metric, dtype=torch.float32)
        tensorboard_logs = {"main_score": metric}

        return {
            "val_loss": metric,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }

    def validation_step(self, batch, batch_idx):
        """TODO Add missing docstring."""
        images, targets, image_ids = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images, targets)

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, outputs)
        }

        self.coco_evaluator.update(res)

        return {}
