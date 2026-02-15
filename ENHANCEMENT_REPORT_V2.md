# Enhancement Report: Model Training & Validation

## Overview
We have introduced a new **Training** tab to the application. This tab provides a sophisticated "Downstream Task Evaluation" capability, allowing users to assess the *utility* of the generated data by training machine learning models on it and comparing their performance against models trained on real (simulated) data.

## New Features

### 1. Training Tab (TSTR Evaluation)
- **Concept**: Train on Synthetic, Test on Real (TSTR). This detects if the synthetic data preserves the underlying relationships needed for ML models.
- **Workflow**:
  1.  **Model A (Control)**: Train a Random Forest Regressor on "Real" data.
  2.  **Model B (Experiment)**: Train an identical Random Forest Regressor on "Synthetic" data.
  3.  **Validation**: Evaluate BOTH models on a held-out "Real" test set.
- **Metrics**: 
  - **R² Score**: How well the model explains variance in the real test set.
  - **MAE/RMSE**: Error rates.
- **Visualization**: 
  - Side-by-side metric cards for Model A vs Model B.
  - Interactive Line Chart showing "Actual" vs "Model A Prediction" vs "Model B Prediction" to visually compare model behavior.

### 2. Backend Enhancements (`flask_api.py`)
- **New Endpoint**: `/api/evaluate/training`
- **Logic**:
  - Simulates a Radiosonde task: Predicting **Temperature** from **Altitude**.
  - Uses `scikit-learn` for training and evaluation.
  - Returns comparative metrics and a subset of predictions for visualization.
- **Dependencies**: Added `scikit-learn`, `pandas`, `numpy`.

### 3. Frontend Enhancements (`Dashboard.tsx`)
- **New Navigation**: Added "Training" tab.
- **New Controls**: dedicated panel to trigger the Training comparison.
- **New Visualizations**:
  - Comparison Cards for R², MAE, RMSE.
  - Chart for visual inspection of model fit.

## How to Test
1.  Navigate to the **Training** tab.
2.  Click **Run Training Comparison**.
3.  Wait for the backend to train the models (simulated real-time).
4.  Observe the metrics. If the synthetic data is good, Model B's `R²` should be close to Model A's `R²`.
