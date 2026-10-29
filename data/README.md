# Bank Marketing Dataset

## Source
- Provider: UCI Machine Learning Repository
- Dataset: Bank Marketing
- Official page: https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing
- Downloaded archive: bank-marketing.zip
- Archive SHA-256: e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4

## Selected file
- File: data/raw/bank-full.csv
- Rows: 45,211
- Columns: 17 (16 input features and target `y`)
- Delimiter: semicolon (;)
- Target: `y` (`yes` or `no`)
- Missing cells found in initial validation: 0
- Duplicate rows found in initial validation: 0

## Research objective
Predict whether a customer subscribes to a bank term deposit.

## Leakage prevention
The `duration` feature represents the length of a phone call. It is not
available before a call starts, so it will be excluded from the primary
pre-call prediction model.

## Citation
Moro, S., Laureano, R., and Cortez, P. (2011).
Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM
Methodology. Proceedings of the European Simulation and Modelling Conference,
pp. 117-121.

Paper: http://hdl.handle.net/1822/14838

## Data handling
Raw downloaded data is kept locally under `data/raw/` and excluded from Git.
This README is tracked so the dataset source and acquisition details remain
documented.
