"""Matching algorithm for sample 'useful' and 'non-useful' images."""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def propensity_score_match(
    df: pd.DataFrame,
    covariates: list[str],
    target_col: str = 'label',
    positive_class: str = 'not-useful',
    ratio: int = 2,
    caliper: float = 0.05,
    random_state: int = 42,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Performs a 1:N propensity score matching with caliper.
    """

    X = pd.get_dummies(data=df[covariates], drop_first=True)

    # Minority class.
    y = (df[target_col] == positive_class).astype(int)

    if verbose:
        print(f"[INFO] not-useful available: {int((y == 1).sum())}.")
        print(f"[INFO] useful available: {int((y == 0).sum())}.")
        print('[INFO] Fitting logistic regression.')

    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=random_state
    )
    model.fit(X, y)

    # Get scores.
    df = df.copy()
    df['propensity_score'] = model.predict_proba(X)[:, 1]

    cases = df[y == 1].copy()
    controls = df[y == 0].copy()

    if verbose:
        print('[INFO] Matching cases and controls.')

    nn = NearestNeighbors(n_neighbors=ratio)
    nn.fit(controls[['propensity_score']])

    distances, indices = nn.kneighbors(cases[['propensity_score']])

    matched_control_indices = set()

    for dists, idxs in zip(distances, indices):
        # Keep only controls less equal than caliper.
        valid = [
            idx
            for dist, idx in zip(dists, idxs)
            if dist <= caliper
        ]

        matched_control_indices.update(valid)

    matched_controls = controls.iloc[list(matched_control_indices)]

    output = (
        pd.concat([cases, matched_controls], ignore_index=True)
        .drop('propensity_score', axis=1)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    if verbose:
        print(f"[INFO] useful selected: {len(matched_controls)}.")
        print(f"[INFO] Total images: {len(output)}.")

    return output
