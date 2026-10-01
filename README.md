# MRI Quality Classifier

This research explores a side-task related to research on **segmentation of low-quality** brain MRI images. The objective is to train deep neural networks that classifies brain MRI scans into two categories: **useful** and **not-useful**.

The labeling criteria are based on the [Quality Control of Structural MRI Images Applied Using FreeSurfer](https://doi.org/10.3389/fnins.2016.00558) paper. This classifier serves as a quality control step to support downstream segmentation and analysis.

### How to train

First of all, load these libraries:

```python
import torch
import torch.nn as nn

from src.models.model import get_model
from src.data.dataset import make_loaders
from src.training.trainer import MRIQualityTrainer
```

To train a neural network for this task, you should prepare the data loaders, get the model, and instantiate the optimizer, criterion and the scheduler. You can do this with the following code:

```python
train_loader, val_loader = make_loaders(
    train_data=train_data,
    val_data=val_data,
    transforms=transforms,
    batch_size=4
)

model = get_model(model_name='densenet121')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=5,
    factor=0.5
)
```

After all, you should instantiate the trainer and start the training process:

```python
trainer = MRIQualityTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    checkpoint_dir='results/fold_0',
    log_dir='results/fold_0/tensorboard',
    scheduler=scheduler,
    device='cuda',
    monitor_metric='auc'
)

history = trainer.fit(num_epochs=50, patience=10, verbose=True)
```

### How to evaluate

To evaluate, you should create a test loader, using the same `make_loaders()` function (the only difference is the fact you should use the same data on both parameters *train_data* and *val_data*):

```python
test_loader = make_loaders(
    train_data=test_data,
    val_data=test_data,
    transforms=transforms,
    shuffle_train=False
)
```

With the test loaders, run the `.predict()` method from the trainer:

```python
predictions, score = trainer.predict(test_loader, verbose=True)
```

`predictions` contains the pred label (0 or 1) and `scores` the probabilities of **not-useful** class, from each sample in the test loader.
