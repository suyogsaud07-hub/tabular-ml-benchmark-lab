import csv
from collections import Counter
from pathlib import Path

import pytest

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "bank-full.csv"

EXPECTED_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "y",
]


@pytest.fixture(scope="module")
def dataset():
    assert DATA_PATH.is_file(), (
        f"Dataset not found at {DATA_PATH}. Follow data/README.md to obtain it."
    )

    with DATA_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        columns = reader.fieldnames
        rows = list(reader)

    return columns, rows


def test_dataset_has_expected_shape(dataset):
    columns, rows = dataset
    assert columns == EXPECTED_COLUMNS
    assert len(rows) == 45_211


def test_target_labels_and_counts(dataset):
    _, rows = dataset
    counts = Counter(row["y"] for row in rows)
    assert counts == {"no": 39_922, "yes": 5_289}


def test_dataset_has_no_empty_cells(dataset):
    _, rows = dataset
    assert all(
        value is not None and value.strip() != ""
        for row in rows
        for value in row.values()
    )


def test_dataset_has_no_duplicate_rows(dataset):
    columns, rows = dataset
    unique_rows = {tuple(row[column] for column in columns) for row in rows}
    assert len(unique_rows) == len(rows)


def test_duration_is_present_but_must_be_excluded_from_pre_call_model(dataset):
    columns, _ = dataset
    assert "duration" in columns
    readme = DATA_PATH.parents[1] / "README.md"
    documentation = readme.read_text(encoding="utf-8").lower()
    assert "duration" in documentation
    assert "pre-call" in documentation
