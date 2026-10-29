# Exploratory Data Analysis — Bank Marketing

## 1. Dataset overview

- Rows: 45,211
- Columns: 17, including the target `y`
- Target: whether the customer subscribed to a term deposit
- Missing or empty cells: 0
- Duplicate rows: 0
- Target classes: `no` = 39,922; `yes` = 5,289

## 2. Target imbalance

- No subscription: 88.3%
- Subscription: 11.7%

Accuracy alone may be misleading. A model that predicts `no` for every
customer would achieve approximately 88.3% accuracy but identify no
subscribers. Later evaluation will include precision, recall, F1-score,
and precision-recall AUC.

## 3. Numerical feature observations

- Age: mean approximately 40.9 years; median 39.
- Account balance: mean approximately 1,362; median 448.
- Balance has a wide range, including negative values and large positive
  outliers.
- `pdays` includes -1, which represents customers not previously contacted
  in an earlier campaign in this dataset. It is not a literal negative
  number of days.
- `duration` records the length of the current contact.

## 4. Subscription rates by category

### Job

| Job | Subscription rate (%) |
|---|---:|
| blue-collar | 7.27 |
| entrepreneur | 8.27 |
| housemaid | 8.79 |
| services | 8.88 |
| technician | 11.06 |
| unknown | 11.81 |
| self-employed | 11.84 |
| admin. | 12.20 |
| management | 13.76 |
| unemployed | 15.50 |
| retired | 22.79 |
| student | 28.68 |

### Marital status

| Category | Subscription rate (%) |
|---|---:|
| single | 14.95 |
| divorced | 11.95 |
| married | 10.12 |

### Education

| Category | Subscription rate (%) |
|---|---:|
| tertiary | 15.01 |
| unknown | 13.57 |
| secondary | 10.56 |
| primary | 8.63 |

### Contact method

| Category | Subscription rate (%) |
|---|---:|
| cellular | 14.92 |
| telephone | 13.42 |
| unknown | 4.07 |

### Previous campaign outcome (`poutcome`)

| Category | Subscription rate (%) |
|---|---:|
| success | 64.73 |
| other | 16.68 |
| failure | 12.61 |
| unknown | 9.16 |

These are descriptive associations in this dataset, not evidence of causation.
Rates can also be affected by group sizes and other variables.

## 5. Data quality and modeling decisions

- No empty cells or duplicate rows were detected in the supplied dataset.
- Categorical `unknown` values will be retained as categories initially.
- Negative account balances are not automatically invalid.
- The primary pre-call model must exclude `duration`, because the current
  call's duration is not known before that call begins.
- Any preprocessing that learns from data must be fitted on training data
  only, to avoid data leakage.

## 6. Generated charts

- `figures/target_distribution.png`
- `figures/age_distribution.png`
- `figures/balance_distribution.png`
- `figures/subscription_by_job.png`

## 7. Next steps

1. Split data into training and test sets.
2. Build preprocessing pipelines for numerical and categorical features.
3. Establish a simple baseline.
4. Compare models using appropriate metrics.
5. Evaluate errors and check reproducibility.
