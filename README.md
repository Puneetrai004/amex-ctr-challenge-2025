# American Express Campus Challenge 2025 – CTR Modeling

## Approach

This repository presents a solution for modeling click-through rates (CTR) for personalized credit card offerings in the American Express Campus Challenge 2025.

- **CTR Prediction**: Formulated as probabilistic ranking, estimating \(P(\text{click}|\text{impression})\) for each user-offer pair.
- **Features**: Engineered high-signal inputs from temporal purchase patterns, event sequences, and offer metadata embeddings.
- **Modeling**: Ensemble of LightGBM and XGBoost using cross-validation and optimized for AUC and MAP@7.
- **Evaluation**: Achieved 0.917 AUC and 0.433 MAP@7 on validation.

## Pipeline

![CTR Modeling Pipeline](visuals/pipeline.png)

1. **Data Loading** – Imports train, test, and data dictionary files.
2. **Preprocessing** – Imputation, scaling, encoding, and balancing of classes.
3. **Modeling** – Ensemble learning using LightGBM & XGBoost, trained on grouped folds.
4. **Evaluation** – Metrics calculated using AUC and MAP@7.
5. **Submission** – Generates a ranked CSV file for competition submission.

## Getting Started

### 1. Clone Repository

### 2. Prepare Environment

Recommended (Python >= 3.8):

Essential packages:
- pandas
- numpy
- scikit-learn
- lightgbm
- xgboost

### 3. Place Data Files

Put the following in `./data`:
- `train_data.parquet`
- `test_data.parquet`
- `data_dictionary.csv`

### 4. Run Main Pipeline
Output submission: `submission1.csv`

## Customization

- Adjust ensemble weights, CV folds, or enable feature engineering in the CONFIG dictionary in `main.py`.
- Plug your own feature engineering logic if required.

## Visualization

The modeling pipeline is illustrated below.

![CTR Modeling Pipeline](visuals/pipeline.png)

## Credits

This repository and solution were developed as part of the American Express Campus Challenge 2025 by a team from IIT BHU.

---



