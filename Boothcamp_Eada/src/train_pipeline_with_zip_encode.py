#!/usr/bin/env python3
"""
train_pipeline_with_zip_encode.py

Train an XGBoost pipeline where high-cardinality zipcode dummy columns are replaced
with a single `zipcode_freq` feature (frequency encoding). Saves pipeline and artifacts.
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


def reduce_zipcode_dummies(df):
    # Identify zipcode dummy columns (start with 'zipcode_')
    zip_cols = [c for c in df.columns if c.startswith('zipcode_')]
    if not zip_cols:
        return df

    # Compute frequency by reconstructing zipcode string from dummies
    # For each row, find which zipcode_* is True and map to zipcode
    # Simpler: compute frequency per column as mean of that column (since dummies are 0/1)
    freqs = df[zip_cols].mean()

    # For each row, compute zipcode_freq as dot product of dummy row and freqs
    zipcode_freq = df[zip_cols].dot(freqs)

    df = df.copy()
    df['zipcode_freq'] = zipcode_freq
    df = df.drop(columns=zip_cols)
    return df


def main():
    script_dir = os.path.dirname(__file__)
    data_path = os.path.abspath(os.path.join(script_dir, '..', 'dat', 'kc_house_data_clean.csv'))
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    print('Loaded data', df.shape)

    # Drop leaked price transform if present
    if 'price_log' in df.columns:
        df = df.drop(columns=['price_log'])

    if 'price' not in df.columns:
        raise ValueError('price column missing')

    # Replace zipcode dummies with zipcode frequency
    df = reduce_zipcode_dummies(df)

    X = df.drop(columns=['price'])
    y = df['price']

    # Ensure numeric
    X = X.apply(pd.to_numeric, errors='coerce')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    pipeline.fit(X_train.values, y_train.values)
    y_pred = pipeline.predict(X_test.values)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print('\nPerformance: MAE {mae:,.2f} | RMSE {rmse:,.2f} | R2 {r2:.4f}'.format(mae=mae, rmse=rmse, r2=r2))

    model_dir = os.path.abspath(os.path.join(script_dir, '..', 'models'))
    os.makedirs(model_dir, exist_ok=True)

    pipeline_path = os.path.join(model_dir, 'pipeline_xgb.pkl')
    artifacts_path = os.path.join(model_dir, 'artifacts_pipeline.pkl')

    joblib.dump(pipeline, pipeline_path)
    artifacts = {
        'feature_names': list(X.columns),
        'created_at': datetime.datetime.utcnow().isoformat() + 'Z',
        'notes': 'zipcode frequency encoded'
    }
    joblib.dump(artifacts, artifacts_path)

    print('Saved pipeline and artifacts')


if __name__ == '__main__':
    main()
