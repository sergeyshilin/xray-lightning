# xray-lightning
Pytorch Lightning object detection pipeline for Chest X-ray data

## Training

```bash
python train.py data.num_workers=10 trainer.max_epochs=3
```

## Inference

```bash
python run.py logging.best_model_path=outputs/2020-07-06/22-47-46/best_model/model.pth data.batch_size=1 data.num_workers=1

```

This will generate `predictions.csv` that can be used for future analysis.

*Note: To do predictions for random 100 images instead of entire dataset, also add `testing.debug=True` to command line arguments*


## Results

Some of the results can be found in [this notebook](notebooks/Visualize\ predictions.ipynb).
