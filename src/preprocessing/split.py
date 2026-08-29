"""
Utilities for creating train/test splits.
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def make_train_test_split(
    df: pd.DataFrame,
    group_col: str = 'patient',
    image_col: str = 'id',
    label_col: str = 'label',
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = False
) -> dict[str, list]:
    """
    Create a group-aware train/test split and return the samples as
    dictionaries. The split ensures that all samples from the same group stay
    in the same partition, preventing leakage between train and test.
    """
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))

    train_groups = set(df.loc[train_idx, group_col])
    test_groups = set(df.loc[test_idx, group_col])

    message = '[WARNING] Leakeage: same group in train and test.'
    assert train_groups.isdisjoint(test_groups), message

    if verbose:
        print(f"[INFO] Train size: {len(train_idx)}.")
        print(f"[INFO] Test size: {len(test_idx)}.")
        print('[INFO] Creating dicts {image, label}.')

    train_dict = (
        df.iloc[train_idx]
        .reindex(columns=[image_col, label_col])
        .rename(columns={image_col: 'image', label_col: 'label'})
        .to_dict(orient='records')
    )
    test_dict = (
        df.iloc[test_idx]
        .reindex(columns=[image_col, label_col])
        .rename(columns={image_col: 'image', label_col: 'label'})
        .to_dict(orient='records')
    )

    if verbose:
        print("[INFO] Class distribution in train set:")
        print(df.iloc[train_idx]['label'].value_counts(normalize=True).round(4))
        print("[INFO] Class distribution in test set:")
        print(df.iloc[test_idx]['label'].value_counts(normalize=True).round(4))

    return {
        'train': train_dict,
        'test': test_dict
    }
