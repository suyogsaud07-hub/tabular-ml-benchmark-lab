from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/raw/bank-full.csv")
TARGET = "y"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def main() -> None:
    df = pd.read_csv(DATA_PATH, sep=";")

    # Exclude the target and duration from the input features.
    # Duration is unknown before the current call begins.
    X = df.drop(columns=[TARGET, "duration"])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("========== DATA SPLIT ==========")
    print(f"Full dataset: {len(df):,} rows")
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Input features: {X.shape[1]}")
    print(f"Target: {TARGET}")
    print(f"Random state: {RANDOM_STATE}")

    print("\n========== TARGET RATES ==========")
    print(f"Full dataset subscribers: {(y == 'yes').mean() * 100:.2f}%")
    print(f"Training subscribers: {(y_train == 'yes').mean() * 100:.2f}%")
    print(f"Test subscribers: {(y_test == 'yes').mean() * 100:.2f}%")

    print("\n========== LEAKAGE CHECK ==========")
    print(f"'duration' excluded: {'duration' not in X.columns}")
    print(f"Target excluded from features: {TARGET not in X.columns}")
    print(
        f"Training and test indices overlap: "
        f"{not X_train.index.intersection(X_test.index).empty}"
    )


if __name__ == "__main__":
    main()
