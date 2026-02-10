# Preprocessing Pipeline

This module converts the raw synthetic dataset into clean, scaled, and split data ready for neural network training. It enforces data validity, applies feature scaling, and creates stratified train/validation/test splits without introducing data leakage.

## Pipeline Steps

1. **Data cleaning**
   - Loads the generated dataset from `data/generated/chairs_dataset.csv`.
   - Checks for missing values and invalid rows (e.g., `backrest_height > 0` when `has_backrest == 0`).
   - Saves the validated dataset to `data/processed/chairs_clean.csv`.

2. **Feature scaling**
   - Loads `data/processed/chairs_clean.csv`.
   - Separates features (X) from the label (y).
   - Applies `StandardScaler` **only** to feature columns.
   - Saves the scaled dataset to `data/processed/chairs_scaled.csv`.
   - Persists the scaler to `config/chair_scaler.pkl` for reuse in later stages.

3. **Data splitting**
   - Loads `data/processed/chairs_scaled.csv`.
   - Performs a stratified split on the label:
     - 70% train
     - 15% validation
     - 15% test
    - Saves outputs to:
       - `data/chairs/train/X_train.csv`, `data/chairs/train/y_train.csv`
       - `data/chairs/validation/X_val.csv`, `data/chairs/validation/y_val.csv`
       - `data/chairs/test/X_test.csv`, `data/chairs/test/y_test.csv`

## Why scaling before split

Scaling is done **before** splitting to ensure a single, consistent transformation is applied across the entire dataset and reused identically in later stages (Etapa 5 and 6). This guarantees reproducibility with a fixed scaler saved to `config/chair_scaler.pkl`.

## Run Order

1. `python src/preprocessing/chair_data_cleaner.py`
2. `python src/preprocessing/chair_feature_scaler.py`
3. `python src/preprocessing/chair_data_splitter.py`
