import os
import pandas as pd
import numpy as np

# Candidate paths (relative to this script and an absolute path found in the workspace)
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
            print(f"\nLoaded successfully from: {abs_path}\n")
            break
        except Exception as e:
            print(f"Found file but failed to read with pandas: {e}\n")

if df is None:
    print("\nNo CSV found at candidate paths. Please update the path and try again.")
    raise SystemExit(1)

print("===== BASIC INFO =====")
print(df.info())
print("\n===== SUMMARY STATISTICS =====")
print(df.describe(include='all'))

print("\n===== MISSING VALUES =====")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "✅ No missing values found")

print("\n===== DUPLICATE ROWS =====")
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
if duplicates > 0:
    print("Consider dropping duplicates with df.drop_duplicates(inplace=True)")

print("\n===== DATA TYPES =====")
print(df.dtypes)

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    print("\nConverted 'date' column to datetime (coercing errors)")

non_numeric = df.select_dtypes(exclude=[np.number, 'datetime']).columns.tolist()
print("\n===== NON-NUMERIC COLUMNS =====")
print(non_numeric if non_numeric else "✅ All features are numeric or datetime")

print("\n===== CONSTANT COLUMNS =====")
constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
print(constant_cols if constant_cols else "✅ No constant columns found")

print("\n===== OUTLIER CHECK (Z-score > 3) =====")
numeric_df = df.select_dtypes(include=[np.number])
if not numeric_df.empty:
    z_scores = ((numeric_df - numeric_df.mean()) / numeric_df.std()).abs()
    outlier_counts = (z_scores > 3).sum()
    oc = outlier_counts[outlier_counts > 0].sort_values(ascending=False)
    print(oc.head(10) if not oc.empty else "✅ No extreme outliers (z>3) found")
else:
    print("No numeric columns to check for outliers")

if 'price' in df.columns:
    print("\n===== CORRELATION WITH PRICE =====")
    print(df.corr(numeric_only=True)['price'].sort_values(ascending=False))

print("\n✅ Data validation complete.")
