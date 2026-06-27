"""
preprocessing.py

Cleans the combined wine dataset and prepares it for modeling:
  1. Drop exact duplicate rows (measured to be ~18% of the combined data;
     dropped BEFORE the train/test split to avoid leakage between sets).
  2. Derive the quality 'grade' label from the raw quality score, using
     thresholds adjusted to fit this dataset's actual distribution
     (the standard 0-10 quartered scheme leaves 'Premium' almost empty).
  3. Encode wine_type as a numeric feature (0/1) so it can be fed into
     scikit-learn / XGBoost / LightGBM models alongside the physicochemical
     features.
  4. Produce a train/test split, stratified on grade to preserve class
     balance across the split.
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import FEATURE_COLUMNS, TARGET_COLUMN

# Grade thresholds, adjusted from the original 0-4 / 5-6 / 7-8 / 9-10 scheme.
# Rationale: under the original scheme, red wine has ZERO samples in 9-10
# and white wine has only 5, making 'Premium' unusable as a category.
# These thresholds are based on the actual quality score distribution
# (see notebooks/eda.ipynb), and were intentionally chosen so every grade
# has enough samples for the model and recommendation engine to learn from.
GRADE_THRESHOLDS = {
    "Poor": (0, 4),       # quality <= 4
    "Average": (5, 5),    # quality == 5
    "Good": (6, 7),       # quality 6-7
    "Premium": (8, 10),   # quality >= 8
}

GRADE_ORDER = ["Poor", "Average", "Good", "Premium"]

WINE_TYPE_MAP = {"red": 0, "white": 1}


def assign_grade(quality: int) -> str:
    """Map a raw quality score (0-10) to a grade label."""
    if quality <= 4:
        return "Poor"
    elif quality == 5:
        return "Average"
    elif quality <= 7:
        return "Good"
    else:
        return "Premium"


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows (same physicochemical profile + quality score
    + wine_type). Measured at ~18% of the combined dataset. These are most
    likely repeated measurements rather than data errors, but dropping them
    avoids the same row appearing in both train and test splits, which would
    silently inflate evaluation metrics.
    """
    before = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    after = len(df_clean)
    print(f"Removed {before - after} duplicate rows ({100 * (before - after) / before:.1f}%). "
          f"{after} rows remain.")
    return df_clean


def encode_wine_type(df: pd.DataFrame) -> pd.DataFrame:
    """Encode wine_type as a numeric column (red=0, white=1) for modeling."""
    df = df.copy()
    df["wine_type_encoded"] = df["wine_type"].map(WINE_TYPE_MAP)
    return df


def add_grade_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived 'grade' label column based on the quality score."""
    df = df.copy()
    df["grade"] = df[TARGET_COLUMN].apply(assign_grade)
    return df


def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline: dedup -> add grade -> encode wine_type.
    Returns a DataFrame ready for splitting and model training.
    """
    df = remove_duplicates(df)
    df = add_grade_column(df)
    df = encode_wine_type(df)
    return df


def get_model_columns():
    """
    Feature columns used for modeling: the 11 physicochemical features
    plus the encoded wine_type. Returned as a fresh list each call so
    callers can't accidentally mutate the shared module-level state.
    """
    return FEATURE_COLUMNS + ["wine_type_encoded"]


def train_test_split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into train/test sets, stratified on 'grade' so that the rare
    Poor/Premium classes are represented proportionally in both sets.
    Stratifying on raw quality score would be too fine-grained and could
    fail outright for scores with very few samples (e.g. quality=9).
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["grade"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Convenience function: runs the full cleaning pipeline and returns
    train/test feature matrices and targets, ready to hand to a model.

    Returns a dict with keys:
        X_train, X_test       -- feature DataFrames
        y_train_reg, y_test_reg   -- raw quality score (for regression)
        y_train_grade, y_test_grade -- grade label (for reference/eval only;
                                       grade is derived from regression
                                       output at inference time, not
                                       predicted by a separate classifier)
        train_df, test_df     -- full cleaned DataFrames, for reference
    """
    clean_df = clean_and_prepare(df)
    train_df, test_df = train_test_split_data(clean_df, test_size, random_state)

    feature_cols = get_model_columns()

    return {
        "X_train": train_df[feature_cols],
        "X_test": test_df[feature_cols],
        "y_train_reg": train_df[TARGET_COLUMN],
        "y_test_reg": test_df[TARGET_COLUMN],
        "y_train_grade": train_df["grade"],
        "y_test_grade": test_df["grade"],
        "train_df": train_df,
        "test_df": test_df,
    }


if __name__ == "__main__":
    from data_loader import load_wine_data

    raw = load_wine_data()
    split = prepare_train_test(raw)

    print()
    print(f"Train set: {split['X_train'].shape}")
    print(f"Test set: {split['X_test'].shape}")
    print()
    print("Train grade distribution:")
    print(split["y_train_grade"].value_counts())
    print()
    print("Test grade distribution:")
    print(split["y_test_grade"].value_counts())