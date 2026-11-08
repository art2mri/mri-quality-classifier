# MRI Quality Classifier

This research explores a side-task related to research on **segmentation of low-quality** brain MRI images. The objective here is to train deep neural networks that classifies brain MRI scans into two categories: **useful** and **not-useful**.

The labeling criteria are inspired by the [Quality Control of Structural MRI Images Applied using FreeSurfer](https://doi.org/10.3389/fnins.2016.00558) paper. This classifier will serve as a quality control step to support downstream segmentation and analysis.

### How to train

First of all, load these libraries:

```python
import torch
import torch.nn as nn

from src.models.model import get_model
from src.data.dataset import make_loaders
from src.training.trainer import MRIQualityTrainer
from src.preprocessing.transforms import get_transforms
from src.evaluation.inference import load_model, evaluate, predict
```

It is a good practice to define all variables before training; for example, decide whether to train a 2D or 3D model, as this affects both the image transformations and the model architecture, and set configuration values such as the data and results directories, learning rate, device, batch size, number of gradient accumulation steps, and any other relevant hyperparameters.

Use the parameters `spatial_dims` to select the 2D or 3D model architecture and the `mode` parameter to defines the image volume used in the transformations.

```python
transforms = get_transforms(mode='3d')
train_loader, val_loader = make_loaders(
    train_data=train_data,
    val_data=val_data,
    transforms=transforms,
    batch_size=BATCH_SIZE
)

model = get_model(model_name=MODEL_NAME, spatial_dims=3)  # 3D model.
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=5,
    factor=0.5
)
```

After this, you should instantiate the **Trainer** class and starting the training process:

```python
trainer = MRIQualityTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    checkpoint_dir=RESULTS,
    log_dir=LOG_DIR,
    accumulation_steps=GRAD_ACCUM_STEPS,
    scheduler=scheduler,
    grad_clip_norm=1.0,
    device=DEVICE,
    monitor_metric=METRIC,
    seed=SEED
)

history = trainer.fit(
    num_epochs=EPOCHS,
    patience=PATIENCE
)
```

### How to evaluate

To evaluate, you should create a test loader using the same `make_loaders()` function. The differences are the fact you should use the same data on both parameters *train_data* and *val_data*, and use `shuffle_train=False`:

```python
_, test_loader = make_loaders(
    train_data=test_data,
    val_data=test_data,
    transforms=transforms,
    shuffle_train=False
)
```

Now, you should rebuild the architecture and load the trained weights from a Path:

```python
model = load_model(
    checkpoint_path='results/ResNet_best.pt',
    model_name='resnet18',
    spatial_dims=3,
    device='cuda'
)
```

Metrics are calculated on the test set (and needs labels, of course). To predict on new data, use `predict()` function:

```python
# Evaluating on test set.
metrics = evaluate(model, test_loader, device='cuda')
print(metrics['accuracy'], metrics['sensitivity'])

# Predictions on new data.
predictions, scores, _ = predict(model, new_loader, device='cuda')
```
