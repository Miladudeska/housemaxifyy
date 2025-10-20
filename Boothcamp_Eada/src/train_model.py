import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib


# Train XGBoost on cleaned dataset: robust, reproducible script
script_dir = os.path.dirname(__file__)
data_path = os.path.abspath(os.path.join(script_dir, '..', 'dat', 'kc_house_data_clean.csv'))
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Cleaned data not found at: {data_path}")

df = pd.read_csv(data_path)
print("===== DATA LOADED =====")
print(f"Shape: {df.shape}")
print(df.head(), "\n")

if 'price' not in df.columns:
    raise ValueError("'price' column not found in dataset.")

# Features and target
X = df.drop(columns=['price']).copy()
y = df['price']

# Convert booleans to ints
bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
if bool_cols:
    X[bool_cols] = X[bool_cols].astype(int)

# Handle non-numeric columns: try coercion to numeric, otherwise one-hot encode
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"Handling non-numeric columns: {non_numeric}")
    for col in non_numeric:
        coerced = pd.to_numeric(X[col], errors='coerce')
        # If coercion yields mostly numbers, keep it; else one-hot encode
        if coerced.notna().sum() > 0 and coerced.isna().sum() < X.shape[0] * 0.5:
            X[col] = coerced
        else:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
    print("✅ Non-numeric columns handled.")

# Final check
remaining_non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if remaining_non_numeric:
    raise ValueError(f"Non-numeric columns remain: {remaining_non_numeric}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data split complete.")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Impute and scale
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
print("✅ Imputed (median) and standardized numeric features.")

# Train XGBoost (eval_metric set in constructor)
xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror',
    eval_metric='rmse',
    n_jobs=-1,
    verbosity=0,
)

xgb_model.fit(X_train_scaled, y_train)
print("✅ XGBoost model trained successfully.")

# Evaluate
y_pred = xgb_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print(f"MAE : {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²  : {r2:.4f}")

# Save model + artifacts
model_dir = os.path.abspath(os.path.join(script_dir, '..', 'models'))
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'xgboost_regressor.pkl')
artifact_path = os.path.join(model_dir, 'artifacts_xgb.pkl')
joblib.dump(xgb_model, model_path)
joblib.dump({'scaler': scaler, 'imputer': imputer, 'feature_names': list(X.columns)}, artifact_path)
print(f"\n💾 Model saved to: {model_path}")
print(f"💾 Artifacts saved to: {artifact_path}")

# Sample predictions
sample = X_test.iloc[:5]
sample_scaled = scaler.transform(imputer.transform(sample))
preds = xgb_model.predict(sample_scaled)
print("\n===== SAMPLE PREDICTIONS =====")
for i in range(len(sample)):
    print(f"Actual: ${y_test.iloc[i]:,.0f} | Predicted: ${preds[i]:,.0f}")

# Feature importance
importance = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n===== TOP 10 FEATURES =====")
print(importance.head(10))
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


script_dir = os.path.dirname(__file__)
data_path = os.path.abspath(os.path.join(script_dir, '..', 'dat', 'kc_house_data_clean.csv'))
from xgboost import XGBRegressor


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


script_dir = os.path.dirname(__file__)
data_path = os.path.abspath(os.path.join(script_dir, '..', 'dat', 'kc_house_data_clean.csv'))

if not os.path.exists(data_path):

print("===== DATA LOADED =====")
print(f"Shape: {df.shape}")
y = df['price']
print(df.head(), "\n")

if 'price' not in df.columns:
    raise ValueError("'price' column not found in dataset.")

# Use log price as the training target (stabilizes variance); keep original price for evaluation
if 'price_log' not in df.columns:
    df['price_log'] = np.log1p(df['price'])

X = df.drop(columns=['price', 'price_log'])
y_log = df['price_log']
y_price = df['price']

# Ensure only numeric features are used (the cleaning script should have done this already)
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    print(f"⚠️ Dropping non-numeric columns before training: {non_numeric_cols}")
    import pandas as pd
    import numpy as np
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from xgboost import XGBRegressor
    import joblib

    # ===============================================================
    # 1️⃣ LOAD CLEAN DATA
    # ===============================================================
    script_dir = os.path.dirname(__file__)
    data_path = os.path.abspath(os.path.join(script_dir, '..', 'dat', 'kc_house_data_clean.csv'))
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned data not found at: {data_path}")

    df = pd.read_csv(data_path)

    print("===== DATA LOADED =====")
    print(f"Shape: {df.shape}")
    print(df.head(), "\n")

    # Ensure target column exists
    if 'price' not in df.columns:
        raise ValueError("'price' column not found in dataset.")

    # Split into features (X) and target (y)
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
    # 3️⃣ SCALE NUMERIC FEATURES
    # ===============================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features scaled (standardized).")

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
        n_jobs=-1
    )

    xgb_model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_test_scaled, y_test)],
        eval_metric="rmse",
        verbose=False
    )
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
    model_dir = os.path.abspath(os.path.join(script_dir, '..', 'models'))
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
    sample = X_test.iloc[:5]
    sample_scaled = scaler.transform(sample)
    preds = xgb_model.predict(sample_scaled)

    print("\n===== SAMPLE PREDICTIONS =====")
    for i in range(len(sample)):
        print(f"Actual: ${y_test.iloc[i]:,.0f} | Predicted: ${preds[i]:,.0f}")

    # ===============================================================
    # 8️⃣ FEATURE IMPORTANCE (optional)
    # ===============================================================
    importance = pd.Series(xgb_model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)
    print("\n===== TOP 10 FEATURES =====")
    print(importance.head(10))
