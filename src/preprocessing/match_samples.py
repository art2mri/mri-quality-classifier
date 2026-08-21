"""Matching algorithm for sample 'useful' and 'non-useful' images."""

import pandas as pd


def create_matched_dataset(
    df: pd.DataFrame,
    useful_proportion: float,
    age_tolerance: float,
    match_columns: list[str],
    random_state: int = 42,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Selects useful images that are compatible with not-useful images based on
    matched list columns and return the two classes concatenated.

    Args:
        df: DataFrame containing the original data.
        useful_proportion: desired proportion of useful images in the final df.
        age_tolerance: maximum allowed difference between ages.
        match_columns: categorical columns used for matching.
        random_state: seed to make the sampling reproducible.
        verbose: whether to print information about the matching process.
    """

    # Creating useful and not_useful samples.
    not_useful = df.query("label == 'not-useful'").reset_index(drop=True).copy()
    useful = df.query("label == 'useful'").reset_index(drop=True).copy()

    number_not_useful = len(not_useful)
    number_useful = round(
        number_not_useful
        * useful_proportion
        / (1 - useful_proportion)
    )

    if verbose:
        print(f"[INFO] not-useful available: {number_not_useful}")
        print(f"[INFO] useful available: {number_useful}")

    # Creating mask for eligible.
    eligible_mask = useful.apply(
        lambda useful_row: (
            (not_useful[match_columns] == useful_row[match_columns]).all(axis=1)
            & ((not_useful['age'] - useful_row['age']).abs() <= age_tolerance)
        ).any(),
        axis=1
    )

    eligible_useful = useful[eligible_mask]

    number_to_sample = min(number_useful, len(eligible_useful))

    if number_to_sample < number_useful:
        print((
            "[WARNING] Not enough eligible useful samples. "
            "Using all eligible useful samples instead."
        ))

    selected_useful = eligible_useful.sample(
        n=number_to_sample,
        random_state=random_state
    )

    # Concatenating two classes.
    matched_df = (
        pd.concat([not_useful, selected_useful], ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    if verbose:
        print(f"[INFO] useful selected: {number_to_sample}")
        print(f"[INFO] Total images: {len(matched_df)}")

    return matched_df
