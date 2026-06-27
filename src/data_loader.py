"""
data_loader.py

Loads the raw Wine Quality CSVs (red and white), tags each row with its
wine_type, and combines them into a single DataFrame.

Source: P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis.
"Modeling wine preferences by data mining from physicochemical properties."
Decision Support Systems, 47(4):547-553, 2009.
"""

from pathlib import Path
import pandas as pd

# Default paths, relative to project root
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RED_PATH = DATA_DIR / "winequality-red.csv"
WHITE_PATH = DATA_DIR / "winequality-white.csv"

# Canonical feature column order, matches the original dataset documentation
FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

TARGET_COLUMN = "quality"


def load_raw_csv(path: Path) -> pd.DataFrame:
    """
    Load a single wine quality CSV. The UCI files are semicolon-delimited
    with quoted headers, so sep=';' is required (a plain pd.read_csv with
    default comma separation will silently produce a single garbage column).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Expected wine quality CSV at {path}, but it does not exist. "
            f"Place winequality-red.csv and winequality-white.csv in {DATA_DIR}."
        )
    df = pd.read_csv(path, sep=";")

    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Loaded file {path} is missing expected columns: {missing}")

    return df


def load_wine_data(red_path: Path = RED_PATH, white_path: Path = WHITE_PATH) -> pd.DataFrame:
    """
    Load both red and white wine datasets, tag each with a wine_type column,
    and concatenate into one combined DataFrame.

    Returns
    -------
    pd.DataFrame with columns: FEATURE_COLUMNS + ['wine_type', 'quality']
    """
    red = load_raw_csv(red_path)
    white = load_raw_csv(white_path)

    red = red.copy()
    white = white.copy()
    red["wine_type"] = "red"
    white["wine_type"] = "white"

    combined = pd.concat([red, white], ignore_index=True)

    # Reorder columns for readability: features, then wine_type, then target
    ordered_cols = FEATURE_COLUMNS + ["wine_type", TARGET_COLUMN]
    combined = combined[ordered_cols]

    return combined


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Quick diagnostic summary, useful for sanity-checking after load and
    for surfacing in the analytics dashboard later.
    """
    return {
        "total_samples": len(df),
        "red_samples": int((df["wine_type"] == "red").sum()),
        "white_samples": int((df["wine_type"] == "white").sum()),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "quality_min": int(df["quality"].min()),
        "quality_max": int(df["quality"].max()),
        "quality_mean": round(float(df["quality"].mean()), 2),
    }


if __name__ == "__main__":
    data = load_wine_data()
    summary = get_dataset_summary(data)
    print("Dataset loaded successfully.")
    for key, value in summary.items():
        print(f"  {key}: {value}")