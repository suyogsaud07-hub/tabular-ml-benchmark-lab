from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path("data/raw/bank-full.csv")
OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, sep=";")

# Chart 1: target counts
target_counts = df["y"].value_counts().reindex(["no", "yes"])
ax = target_counts.plot(kind="bar", rot=0)
ax.set_title("Bank Term Deposit Subscription")
ax.set_xlabel("Subscription")
ax.set_ylabel("Number of customers")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=150)
plt.close()

# Chart 2: age distribution
df["age"].plot(kind="hist", bins=30, edgecolor="black")
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "age_distribution.png", dpi=150)
plt.close()

# Chart 3: balance distribution, including negative balances
df["balance"].plot(kind="hist", bins=60, edgecolor="black")
plt.title("Account Balance Distribution")
plt.xlabel("Account balance")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "balance_distribution.png", dpi=150)
plt.close()

# Chart 4: subscription rate by job
job_rates = (
    df.assign(subscribed=(df["y"] == "yes").astype(int))
    .groupby("job")["subscribed"]
    .mean()
    .mul(100)
    .sort_values()
)

job_rates.plot(kind="barh")
plt.title("Subscription Rate by Job")
plt.xlabel("Customers subscribed (%)")
plt.ylabel("Job")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "subscription_by_job.png", dpi=150)
plt.close()

print("Charts saved to:", OUTPUT_DIR)
for path in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"{path}: {path.stat().st_size:,} bytes")

print("\nSubscription rate by job (%):")
print(job_rates.round(2).to_string())
