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
    if verbose:
        print('[INFO] Calculating propensity scores.')

    df = df.copy()
    df['propensity_score'] = model.predict_proba(X)[:, 1]

    cases = df[y == 1].copy()
    controls = df[y == 0].copy()

    if verbose:
        print('[INFO] Fitting KNN.')

    nn = NearestNeighbors(n_neighbors=ratio)
    nn.fit(controls[['propensity_score']].values)

    used_controls = set()
    matched_controls = []

    if verbose:
        print('[INFO] Matching cases and controls.')

    for _, case in cases.iterrows():
        # Calculating distance from not-useful to all useful.
        distances, indices = nn.kneighbors([[case['propensity_score']]])

        selected = 0

        for dist, idx in zip(distances[0], indices[0]):
            if dist > caliper:
                continue

            control_idx = controls.index[idx]

            if control_idx in used_controls:  # useful control already used.
                continue

            used_controls.add(control_idx)
            matched_controls.append(control_idx)
            selected += 1

            if selected == ratio:
                break

    matched_controls_df = controls.loc[matched_controls]

    output = (
        pd.concat([cases, matched_controls_df], ignore_index=True)
        .drop('propensity_score', axis=1)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    if verbose:
        print(f"[INFO] useful selected: {len(matched_controls)}")
        print(f"[INFO] Total images: {len(output)}")

    return output
