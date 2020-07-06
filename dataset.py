"""Xray object detection dataset class."""

from typing import Dict, Tuple

import cv2
import numpy as np
import os
import pandas as pd
import torch

from albumentations.core.composition import Compose
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils import load_obj, read_labels


class XrayDataset(Dataset):
    """Xray object detection dataset class."""

    def __init__(
        self,
        dataframe: pd.DataFrame = None,
        mode: str = "train",
        image_dir: str = "",
        cfg: DictConfig = None,
        transforms: Compose = None,
    ):
        """
        Prepare data for object detection on chest X-ray images.

        Parameters
        ----------
        dataframe : pd.DataFrame, optional
            dataframe with image id and bboxes, by default None
        mode : str, optional
            train/val/test, by default "train"
        image_dir : str, optional
            path to images, by default ""
        cfg : DictConfig, optional
            config with parameters, by default None
        transforms : Compose, optional
            albumentations, by default None
        """
        self.image_dir = image_dir
        self.df = dataframe
        self.mode = mode
        self.cfg = cfg
        self.image_ids = (
            os.listdir(self.image_dir)
            if self.df is None
            else self.df["image_id"].unique()
        )
        self.transforms = transforms

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.tensor, Dict[str, torch.tensor], str]:
        """
        Get dataset item.

        Parameters
        ----------
        idx : int
            Dataset item index

        Returns
        -------
        Tuple[Tensor, Dict[str, Tensor], str]
            (image, target, image_id)
        """
        image_id = self.image_ids[idx]
        image = cv2.imread(f"{self.image_dir}/{image_id}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # normalization.
        # TO DO: refactor preprocessing
        image /= 255.0

        # test dataset must have some values so that transforms work.
        target = {
            "labels": torch.as_tensor([[0]], dtype=torch.float32),
            "boxes": torch.as_tensor([[0, 0, 0, 0]], dtype=torch.float32),
        }

        # for train and valid test create target dict.
        if self.mode != "test":
            image_data = self.df.loc[self.df["image_id"] == image_id]
            boxes = image_data[["x", "y", "x1", "y1"]].values
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            areas = image_data["area"].values
            areas = torch.as_tensor(areas, dtype=torch.float32)

            label_dict = read_labels(
                to_absolute_path(self.cfg.data.labels_path)
            )
            labels = [
                label_dict[label] for label in image_data["label"].values
            ]
            labels = torch.as_tensor(labels, dtype=torch.int64)

            iscrowd = torch.zeros((image_data.shape[0],), dtype=torch.int64)

            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = torch.tensor([idx])
            target["area"] = areas
            target["iscrowd"] = iscrowd

            if self.transforms:
                image_dict = {
                    "image": image,
                    "bboxes": target["boxes"],
                    "labels": labels,
                }
                image_dict = self.transforms(**image_dict)
                image = image_dict["image"]
                target["boxes"] = torch.as_tensor(
                    image_dict["bboxes"], dtype=torch.float32
                )

        else:
            image_dict = {
                "image": image,
                "bboxes": target["boxes"],
                "labels": target["labels"],
            }
            image = self.transforms(**image_dict)["image"]

        return image, target, image_id

    def __len__(self) -> int:
        """
        Get dataset size.

        Returns
        -------
        int
            Dataset size
        """
        return len(self.image_ids)


def get_training_dataset(cfg: DictConfig = None) -> Dict[str, Dataset]:
    """
    Get training and validation datasets.

    Parameters
    ----------
    cfg : DictConfig, optional
        Project configuration, by default None

    Returns
    -------
    Dict[str, Dataset]
        {"train": train_dataset, "valid": valid_dataset}
    """
    images_dir = to_absolute_path(cfg.data.images_folder_path)

    data = pd.read_csv(to_absolute_path(cfg.data.dataset_path))
    data["x1"] = data["x"] + data["w"]
    data["y1"] = data["y"] + data["h"]
    data["area"] = data["w"] * data["h"]

    train_ids, valid_ids = train_test_split(
        data["image_id"].unique(),
        test_size=cfg.data.validation_split,
        random_state=cfg.training.seed,
    )

    # for fast training
    if cfg.training.debug:
        train_ids = train_ids[:10]
        valid_ids = valid_ids[:10]

    train_df = data.loc[data["image_id"].isin(train_ids)]
    valid_df = data.loc[data["image_id"].isin(valid_ids)]

    train_augs_list = [
        load_obj(i["class_name"])(**i["params"])
        for i in cfg["augmentation"]["train"]["augs"]
    ]
    train_bbox_params = OmegaConf.to_container(
        (cfg["augmentation"]["train"]["bbox_params"])
    )
    train_augs = Compose(train_augs_list, bbox_params=train_bbox_params)

    valid_augs_list = [
        load_obj(i["class_name"])(**i["params"])
        for i in cfg["augmentation"]["valid"]["augs"]
    ]
    valid_bbox_params = OmegaConf.to_container(
        (cfg["augmentation"]["valid"]["bbox_params"])
    )
    valid_augs = Compose(valid_augs_list, bbox_params=valid_bbox_params)

    train_dataset = XrayDataset(train_df, "train", images_dir, cfg, train_augs)
    valid_dataset = XrayDataset(valid_df, "valid", images_dir, cfg, valid_augs)

    return {"train": train_dataset, "valid": valid_dataset}


def get_test_dataset(cfg: DictConfig = None) -> Dataset:
    """
    Get test dataset.

    Parameters
    ----------
    cfg : DictConfig, optional
        Project configuration, by default None

    Returns
    -------
    Dataset
        Pytorch dataset
    """
    images_dir = to_absolute_path(cfg.data.images_folder_path)
    data_path = to_absolute_path(cfg.data.dataset_path)

    train_ids = set(pd.read_csv(data_path).image_id.values)
    all_ids = set(os.listdir(to_absolute_path(images_dir)))
    test_ids = all_ids.difference(train_ids)

    test_df = pd.DataFrame(test_ids, columns=["image_id"])

    test_augs_list = [
        load_obj(i["class_name"])(**i["params"])
        for i in cfg["augmentation"]["valid"]["augs"]
    ]
    test_bbox_params = OmegaConf.to_container(
        (cfg["augmentation"]["valid"]["bbox_params"])
    )
    test_augs = Compose(test_augs_list, bbox_params=test_bbox_params)

    test_dataset = XrayDataset(test_df, "test", images_dir, cfg, test_augs)

    return test_dataset
