import os
import pandas as pd

# Candidate paths relative to this script (covers common locations)
candidates = [
    "../dat/kc_house_data.csv",
    "../kc_house_data.csv",
    "../king_ country_ houses_aa.csv",
    "../king_country_houses_aa.csv",
    "c:/Users/ThinkPad/Desktop/Boothcamp_Eada/king_ country_ houses_aa.csv",
]

script_dir = os.path.dirname(__file__)

for p in candidates:
    abs_path = os.path.abspath(os.path.join(script_dir, p))
    print(f"Trying: {abs_path}")
    if os.path.exists(abs_path):
        try:
            df = pd.read_csv(abs_path)
            print(f"Loaded successfully from: {abs_path}\n")
            print(df.head())
            break
        except Exception as e:
            print(f"Found file but failed to read with pandas: {e}\n")
else:
    # If we get here, none of the candidates matched. Show helpful diagnostics.
    print("\nNo candidate paths existed. Listing parent directory contents for troubleshooting:\n")
    parent = os.path.abspath(os.path.join(script_dir, ".."))
    try:
        for f in os.listdir(parent):
            print(f)
    except Exception as e:
        print(f"Failed to list directory {parent}: {e}")
    print('\nPlease update the path variable to point to your CSV file.')
