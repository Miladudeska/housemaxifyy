#!/usr/bin/env python3
"""
train_pipeline_xgboost.py

Train an XGBoost model wrapped in a scikit-learn Pipeline (imputer -> scaler -> xgb).
This script removes the leaked `price_log` feature, trains the pipeline, evaluates it,
and saves both the pipeline and a small artifacts file with the feature names.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import datetime


def main():
    script_dir = os.path.dirname(__file__)
    data_path = os.path.abspath(os.path.join(script_dir, '..', 'dat', 'kc_house_data_clean.csv'))
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    print("Loaded cleaned data", df.shape)

    # Remove any leaked target transforms
    if 'price_log' in df.columns:
        print("Dropping leaked column: price_log")
        df = df.drop(columns=['price_log'])

    if 'price' not in df.columns:
        raise ValueError("'price' column not found in dataset.")

    X = df.drop(columns=['price'])
    y = df['price']

    # Ensure numeric types where possible (zipcode dummies may be True/False strings)
    X = X.apply(pd.to_numeric, errors='coerce')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Pipeline: imputer -> scaler -> xgb
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='reg:squarederror',
            n_jobs=-1,
        ))
    ])

    # Fit pipeline on numpy arrays (preserves column order separately)
    pipeline.fit(X_train.values, y_train.values)
    print("Pipeline trained")

    # Evaluate
    y_pred = pipeline.predict(X_test.values)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print('\n===== MODEL PERFORMANCE =====')
    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²  : {r2:.4f}")

    # Save model pipeline and artifacts
    model_dir = os.path.abspath(os.path.join(script_dir, '..', 'models'))
    os.makedirs(model_dir, exist_ok=True)

    pipeline_path = os.path.join(model_dir, 'pipeline_xgb.pkl')
    artifacts_path = os.path.join(model_dir, 'artifacts_pipeline.pkl')

    joblib.dump(pipeline, pipeline_path)
    artifacts = {
        'feature_names': list(X.columns),
        'created_at': datetime.datetime.utcnow().isoformat() + 'Z',
        'target': 'price'
    }
    joblib.dump(artifacts, artifacts_path)

    print(f"Saved pipeline -> {pipeline_path}")
    print(f"Saved artifacts -> {artifacts_path}")

    # Feature importances from underlying xgb model
    try:
        model = pipeline.named_steps['model']
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print('\nTop 10 feature importances:')
        print(importances.head(10))
    except Exception as e:
        print('Could not extract feature importances:', e)

    print('\nDone.')


if __name__ == '__main__':
    main()
