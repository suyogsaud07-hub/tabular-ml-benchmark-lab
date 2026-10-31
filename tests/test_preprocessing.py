import numpy as np
import pandas as pd

from tabular_ml_benchmark_lab.preprocessing import build_preprocessor


def sample_training_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 40, 55, 35],
            "balance": [100, 500, -20, 250],
            "job": ["student", "manager", "technician", "student"],
            "education": ["tertiary", "secondary", "primary", "tertiary"],
        }
    )


def test_preprocessor_transforms_numerical_and_categorical_features() -> None:
    X_train = sample_training_data()
    preprocessor = build_preprocessor(X_train)

    prepared = preprocessor.fit_transform(X_train)

    assert prepared.shape[0] == len(X_train)
    assert prepared.shape[1] > X_train.shape[1]
    assert np.isfinite(prepared).all()


def test_preprocessor_handles_unseen_categories() -> None:
    X_train = sample_training_data()
    X_test = pd.DataFrame(
        {
            "age": [29],
            "balance": [300],
            "job": ["astronaut"],
            "education": ["doctoral"],
        }
    )

    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    prepared_test = preprocessor.transform(X_test)

    assert prepared_test.shape[0] == 1
    assert prepared_test.shape[1] == len(preprocessor.get_feature_names_out())
    assert np.isfinite(prepared_test).all()


def test_preprocessor_handles_missing_values() -> None:
    X_train = sample_training_data()
    X_train.loc[0, "age"] = np.nan
    X_train.loc[1, "job"] = np.nan

    preprocessor = build_preprocessor(X_train)
    prepared = preprocessor.fit_transform(X_train)

    assert np.isfinite(prepared).all()


def test_train_test_split_is_reproducible() -> None:
    from tabular_ml_benchmark_lab.preprocessing import load_and_split_data

    first_split = load_and_split_data()
    second_split = load_and_split_data()

    pd.testing.assert_frame_equal(first_split[0], second_split[0])
    pd.testing.assert_frame_equal(first_split[1], second_split[1])
    pd.testing.assert_series_equal(first_split[2], second_split[2])
    pd.testing.assert_series_equal(first_split[3], second_split[3])


def test_train_test_split_has_no_overlapping_rows() -> None:
    from tabular_ml_benchmark_lab.preprocessing import load_and_split_data

    X_train, X_test, _, _ = load_and_split_data()

    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_train_test_split_preserves_target_distribution() -> None:
    from tabular_ml_benchmark_lab.preprocessing import load_and_split_data

    X_train, X_test, y_train, y_test = load_and_split_data()

    assert len(X_train) == 36168
    assert len(X_test) == 9043
    assert abs((y_train == "yes").mean() - (y_test == "yes").mean()) < 0.001


def test_leakage_and_target_columns_are_excluded() -> None:
    from tabular_ml_benchmark_lab.preprocessing import load_and_split_data

    X_train, X_test, _, _ = load_and_split_data()

    for X in (X_train, X_test):
        assert "duration" not in X.columns
        assert "y" not in X.columns
