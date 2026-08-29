"""
Utilities for creating train/test splits.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold


def _to_records(
    df: pd.DataFrame,
    images_dir: Path,
    image_col: str,
    label_col: str,
    positive_class: str
) -> list[dict]:
    return [
        {
            'image': str(images_dir / row[image_col]),
            'label': int(row[label_col] == positive_class)
        }
        for _, row in df[[image_col, label_col]].iterrows()
    ]


def _print_summary(name: str, items: list[dict]) -> None:
    labels = pd.Series([item['label'] for item in items])

    print(f"[INFO] {name}: {len(items)} samples.")
    print(f"[INFO] {name}: class distribution:")
    print(labels.value_counts(normalize=True).sort_index().round(4))


def make_group_split(
    df: pd.DataFrame,
    images_dir: str | Path,
    split_type: str = 'train_test',
    group_col: str = 'patient',
    image_col: str = 'id',
    label_col: str = 'label',
    positive_class: str = 'not-useful',
    test_size: float = 0.2,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = False
) -> dict:
    """
    Create a group-aware train/test or k-fold split with full image paths.
    """
    images_dir = Path(images_dir)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    if split_type == 'train_test':
        train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        train = _to_records(
            df=train_df,
            images_dir=images_dir,
            image_col=image_col,
            label_col=label_col,
            positive_class=positive_class
        )
        test = _to_records(
            df=test_df,
            images_dir=images_dir,
            image_col=image_col,
            label_col=label_col,
            positive_class=positive_class
        )

        if verbose:
            _print_summary('Train', train)
            _print_summary('Test', test)

        return {
            'split_type': 'train_test',
            'train': train,
            'test': test
        }

    if split_type == 'kfold':
        trainval_idx, test_idx = next(splitter.split(df, groups=df[group_col]))

        trainval_df = df.iloc[trainval_idx]
        test_df = df.iloc[test_idx]

        test = _to_records(
            df=test_df,
            images_dir=images_dir,
            image_col=image_col,
            label_col=label_col,
            positive_class=positive_class
        )

        folds = []
        fold_splitter = GroupKFold(n_splits=n_splits)

        for fold_idx, (train_idx, val_idx) in enumerate(
            fold_splitter.split(trainval_df, groups=trainval_df[group_col])
        ):
            train = _to_records(
                df=trainval_df.iloc[train_idx],
                images_dir=images_dir,
                image_col=image_col,
                label_col=label_col,
                positive_class=positive_class
            )
            val = _to_records(
                df=trainval_df.iloc[val_idx],
                images_dir=images_dir,
                image_col=image_col,
                label_col=label_col,
                positive_class=positive_class,
            )

            folds.append(
                {
                    'fold': fold_idx,
                    'train': train,
                    'val': val
                }
            )

        if verbose:
            _print_summary("Test", test)

            for fold in folds:
                print(f"[INFO] Fold {fold['fold']}")
                _print_summary("Train", fold["train"])
                _print_summary("Val", fold["val"])

        return {
            'split_type': 'kfold',
            'test': test,
            'folds': folds
        }

    raise ValueError("split_type must be 'train_test' or 'kfold'")
