#!/usr/bin/env python3
"""
train_xgboost.py

Train an XGBoost regression model on the cleaned King County housing dataset.
Saves both the trained model and scaler for future predictions.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib


def main():
    # ===============================================================
    # 1️⃣ LOAD CLEAN DATA
    # ===============================================================
    script_dir = os.path.dirname(__file__)
    data_path = os.path.abspath(os.path.join(script_dir, '..', 'dat', 'kc_house_data_clean.csv'))
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    print("===== DATA LOADED =====")
    print(f"Shape: {df.shape}")
    print(df.head(), "\n")

    # Ensure target column exists
    if 'price' not in df.columns:
        raise ValueError("❌ 'price' column not found in dataset.")

    # Split features & target
    X = df.drop(columns=['price'])
    y = df['price']

    # ===============================================================
    # 2️⃣ TRAIN/TEST SPLIT
    # ===============================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("✅ Data split complete.")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # ===============================================================
    # 3️⃣ SCALE FEATURES
    # ===============================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features standardized.")

    # ===============================================================
    # 4️⃣ TRAIN XGBOOST REGRESSOR
    # ===============================================================
    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror',
        eval_metric='rmse',
        n_jobs=-1
    )
    xgb_model.fit(X_train_scaled, y_train)
    print("✅ XGBoost model trained successfully.")

    # ===============================================================
    # 5️⃣ EVALUATE MODEL
    # ===============================================================
    y_pred = xgb_model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n===== MODEL PERFORMANCE =====")
    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²  : {r2:.4f}")

    # ===============================================================
    # 6️⃣ SAVE MODEL AND SCALER
    # ===============================================================
    model_dir = "../models"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "xgboost_regressor.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    joblib.dump(xgb_model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n💾 Model saved to: {model_path}")
    print(f"💾 Scaler saved to: {scaler_path}")

    # ===============================================================
    # 7️⃣ SAMPLE PREDICTIONS
    # ===============================================================
    print("\n===== SAMPLE PREDICTIONS =====")
    sample = X_test.iloc[:5]
    sample_scaled = scaler.transform(sample)
    preds = xgb_model.predict(sample_scaled)
    for i in range(len(sample)):
        print(f"Actual: ${y_test.iloc[i]:,.0f} | Predicted: ${preds[i]:,.0f}")

    # ===============================================================
    # 8️⃣ FEATURE IMPORTANCES
    # ===============================================================
    importance = pd.Series(xgb_model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)
    print("\n===== TOP 10 FEATURES =====")
    print(importance.head(10))

    print("\n✅ Training completed successfully.")


if __name__ == "__main__":
    main()
