#!/usr/bin/env python

import hydra
import numpy as np
import pandas as pd
import torch

from dataset import get_test_dataset
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm
from utils import collate_fn


def format_prediction_string(boxes, scores) -> str:
    """
    Convert bboxes list into a string.

    Parameters
    ----------
    boxes : list
        List of predicted bboxes
    scores : list
        List of scores for each bbox

    Returns
    -------
    str
        Bboxes as string
    """
    pred_strings = []
    for s, b in zip(scores, boxes.astype(int)):
        pred_strings.append(
            f"{s:.4f} {b[0]} {b[1]} {b[2] - b[0]} {b[3] - b[1]};"
        )

    return " ".join(pred_strings)


@hydra.main(config_path="config.yaml")
def run(cfg: DictConfig) -> None:
    """
    Run model inference on the entire test dataset.

    Parameters
    ----------
    cfg : DictConfig
        Project configuration object
    """
    device = torch.device("cuda")
    model = torch.load(
        to_absolute_path(cfg.logging.best_model_path), map_location=device
    )
    model.eval()

    detection_threshold = 0.5
    results = []

    test_dataset = get_test_dataset(cfg)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    for images, _, image_ids in tqdm(test_loader):

        images = list(image.to(device) for image in images)
        outputs = model(images)

        for i, image in enumerate(images):

            boxes = outputs[i]["boxes"].data.cpu().numpy()
            scores = outputs[i]["scores"].data.cpu().numpy()
            labels = outputs[i]["labels"].data.cpu().numpy()

            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            labels = labels[scores >= detection_threshold]
            scores = scores[scores >= detection_threshold]

            image_id = image_ids[i]

            result = {
                "image_id": image_id,
                "predictions": format_prediction_string(boxes, scores),
                "labels": ",".join([str(x) for x in labels]),
            }

            results.append(result)

    test_df = pd.DataFrame(
        results, columns=["image_id", "predictions", "labels"]
    )
    test_df.to_csv(to_absolute_path("predictions.csv"), index=False)


if __name__ == "__main__":
    run()
