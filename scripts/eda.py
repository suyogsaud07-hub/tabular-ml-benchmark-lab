from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/bank-full.csv")

df = pd.read_csv(DATA_PATH, sep=";")

print("\n========== 1. DATASET SHAPE ==========")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

print("\n========== 2. FIRST FIVE ROWS ==========")
print(df.head().to_string(index=False))

print("\n========== 3. COLUMN DATA TYPES ==========")
print(df.dtypes.to_string())

print("\n========== 4. MISSING VALUES ==========")
missing = df.isna().sum()
print(missing[missing > 0].to_string() if missing.any() else "No missing values.")

print("\n========== 5. DUPLICATE ROWS ==========")
print(f"Duplicate rows: {df.duplicated().sum():,}")

print("\n========== 6. TARGET DISTRIBUTION ==========")
target_counts = df["y"].value_counts()
target_percentages = df["y"].value_counts(normalize=True).mul(100)

target_summary = pd.DataFrame({
    "count": target_counts,
    "percentage": target_percentages.round(2),
})
print(target_summary.to_string())

print("\n========== 7. NUMERICAL SUMMARY ==========")
print(df.describe().round(2).to_string())

print("\n========== 8. CATEGORICAL SUMMARY ==========")
print(df[["job", "marital", "education", "contact", "poutcome"]]
      .describe()
      .to_string())

print("\n========== 9. MODELLING WARNING ==========")
print(
    "The duration feature describes call length. "
    "We will exclude it from the primary pre-call prediction model."
)
