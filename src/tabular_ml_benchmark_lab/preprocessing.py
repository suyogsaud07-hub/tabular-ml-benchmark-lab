from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/raw/bank-full.csv")
TARGET = "y"
TEST_SIZE = 0.20
RANDOM_STATE = 42
LEAKAGE_COLUMNS = ["duration"]


def load_and_split_data(
    data_path: Path = DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the dataset and create a reproducible stratified split."""
    df = pd.read_csv(data_path, sep=";")

    X = df.drop(columns=[TARGET, *LEAKAGE_COLUMNS])
    y = df[TARGET]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipelines for numerical and categorical data."""
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def main() -> None:
    X_train, X_test, y_train, y_test = load_and_split_data()

    preprocessor = build_preprocessor(X_train)

    # Fit only on training data to prevent data leakage.
    X_train_prepared = preprocessor.fit_transform(X_train)

    # Transform test data without fitting again.
    X_test_prepared = preprocessor.transform(X_test)

    print("========== PREPROCESSING RESULTS ==========")
    print(f"Original training shape: {X_train.shape}")
    print(f"Prepared training shape: {X_train_prepared.shape}")
    print(f"Original test shape: {X_test.shape}")
    print(f"Prepared test shape: {X_test_prepared.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Test labels: {y_test.shape}")

    print("\n========== FEATURE CHECK ==========")
    print(
        f"Numerical columns: {len(X_train.select_dtypes(include=['number']).columns)}"
    )
    print(
        f"Categorical columns: {len(X_train.select_dtypes(exclude=['number']).columns)}"
    )
    print(f"Duration excluded: {'duration' not in X_train.columns}")
    print(f"Target excluded: {TARGET not in X_train.columns}")
    print(f"Encoded feature names: {len(preprocessor.get_feature_names_out())}")

    print("\n========== TARGET DISTRIBUTION ==========")
    print(f"Training subscriber rate: {(y_train == 'yes').mean():.2%}")
    print(f"Test subscriber rate: {(y_test == 'yes').mean():.2%}")


if __name__ == "__main__":
    main()
