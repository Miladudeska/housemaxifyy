import os
import pandas as pd
import numpy as np


# Candidate paths (relative to this script and a known absolute path in the workspace)
candidates = [
    "../dat/kc_house_data.csv",
    "../kc_house_data.csv",
    "../king_ country_ houses_aa.csv",
    r"c:\\Users\\ThinkPad\\Desktop\\Boothcamp_Eada\\king_ country_ houses_aa.csv",
]

script_dir = os.path.dirname(__file__)
df = None
for p in candidates:
    abs_path = os.path.abspath(os.path.join(script_dir, p))
    print(f"Trying: {abs_path}")
    if os.path.exists(abs_path):
        try:
            df = pd.read_csv(abs_path)
            print(f"Loaded successfully from: {abs_path}\n")
            break
        except Exception as e:
            print(f"Found file but failed to read with pandas: {e}\n")

if df is None:
    raise FileNotFoundError("No CSV found at candidate paths. Update the path and try again.")

print("===== DATASET LOADED =====")
print(f"Shape: {df.shape}")
print(df.head(), "\n")

# Safe initial deduplication
initial_duplicates = df.duplicated().sum()
if initial_duplicates > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"⚠️ Dropped {initial_duplicates} duplicate rows (initial dedupe).")
else:
    print("✅ No duplicate rows found at load time.")

# Drop identifier
if 'id' in df.columns:
    df = df.drop(columns=['id'])
    print("🗑️ Dropped column: 'id'")

# Ensure price exists
if 'price' not in df.columns:
    raise ValueError("'price' column not found. Cannot continue.")

# Date handling: extract sale_year and sale_month
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['sale_year'] = df['date'].dt.year
    df['sale_month'] = df['date'].dt.month
    df = df.drop(columns=['date'])
    print("✅ Extracted 'sale_year' and 'sale_month' from 'date' and dropped raw date column.")

# Feature engineering
if 'yr_built' in df.columns and 'sale_year' in df.columns:
    df['age'] = df['sale_year'] - df['yr_built']
    df.loc[df['age'] < 0, 'age'] = np.nan  # guard against bad dates
    print("✅ Created feature: 'age' (sale_year - yr_built)")

if 'yr_renovated' in df.columns:
    df['was_renovated'] = (df['yr_renovated'] > 0).astype(int)
    print("✅ Created feature: 'was_renovated' (binary)")

if 'sqft_living' in df.columns and 'price' in df.columns:
    df['price_per_sqft'] = df['price'] / df['sqft_living'].replace({0: np.nan})
    print("✅ Created feature: 'price_per_sqft'")

# Create log-price as an optional target column for modeling if desired
df['price_log'] = np.log1p(df['price'])

# Convert zipcode to string (treat as categorical)
if 'zipcode' in df.columns:
    df['zipcode'] = df['zipcode'].astype(str)
    print("✅ Converted 'zipcode' to string (categorical)")

# Remove obvious constants (if any) before imputation/encoding
constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
if constant_cols:
    df = df.drop(columns=constant_cols)
    print(f"⚠️ Dropped constant columns: {constant_cols}")

# Separate target and features, preserve price column in cleaned file
y = df['price']
X = df.drop(columns=['price'])

# Impute numeric missing values with median
numeric_cols = X.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    num_median = X[numeric_cols].median()
    X[numeric_cols] = X[numeric_cols].fillna(num_median)

# For non-numeric columns, fill missing with a placeholder string
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_cols) > 0:
    X[non_numeric_cols] = X[non_numeric_cols].fillna('missing')

# One-hot encode categorical (non-numeric) columns, but avoid exploding rare levels: use drop_first to reduce col count
if len(non_numeric_cols) > 0:
    X = pd.get_dummies(X, columns=list(non_numeric_cols), drop_first=True)
    print(f"✅ One-hot encoded categorical columns: {list(non_numeric_cols)}")
else:
    print("✅ No categorical columns to encode.")

# Drop any constant columns created by encoding
constant_after = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
if constant_after:
    X = X.drop(columns=constant_after)
    print(f"⚠️ Dropped constant columns after encoding: {constant_after}")

# Clip numeric outliers at 1st and 99th percentiles
numeric_cols = X.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    lower, upper = X[col].quantile([0.01, 0.99])
    X[col] = X[col].clip(lower, upper)
print("✅ Numeric features clipped at the 1st and 99th percentiles.")

# Final dedupe (after feature engineering)
final_duplicates = X.duplicated().sum()
if final_duplicates > 0:
    X = X.drop_duplicates().reset_index(drop=True)
    y = y.reset_index(drop=True).loc[X.index]
    print(f"⚠️ Dropped {final_duplicates} duplicate feature rows (final dedupe).")

# Final check and save
cleaned = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
out_dir = os.path.abspath(os.path.join(script_dir, '..', 'dat'))
os.makedirs(out_dir, exist_ok=True)
clean_path = os.path.join(out_dir, 'kc_house_data_clean.csv')
cleaned.to_csv(clean_path, index=False)

print("\n===== CLEAN DATA SUMMARY =====")
print(f"Final features shape: {X.shape}")
print(f"Final target shape: {y.shape}")
print(cleaned.head())
print(f"\n💾 Cleaned dataset saved to: {clean_path}")
