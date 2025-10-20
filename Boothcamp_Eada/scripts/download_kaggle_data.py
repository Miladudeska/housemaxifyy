"""
Small helper to download the Kaggle dataset `harlfoxem/housesalesprediction` using the Kaggle API.

Usage:
  1) Install Kaggle CLI: pip install kaggle
  2) Create a Kaggle API token (https://www.kaggle.com/docs/api) and place it at ~/.kaggle/kaggle.json
     On Windows, you can place it at C:\Users\<username>\.kaggle\kaggle.json
  3) Run: python scripts/download_kaggle_data.py --out dat

This script will download the dataset zip, extract it, and move `kc_house_data.csv` (or similar) to the `dat/` folder.
"""

import argparse
import os
import zipfile
import shutil
import subprocess


def run_kaggle_download(dataset='harlfoxem/housesalesprediction', out_dir='dat'):
    os.makedirs(out_dir, exist_ok=True)
    # Use kaggle CLI to download
    cmd = ['kaggle', 'datasets', 'download', '-d', dataset, '-p', out_dir, '--unzip']
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd)

    # After unzip, attempt to locate kc_house_data.csv or similar name
    possible = ['kc_house_data.csv', 'kc_house_data.csv.zip', 'kc_house_data.txt']
    for root, dirs, files in os.walk(out_dir):
        for f in files:
            if f.lower().startswith('kc_house') and f.lower().endswith('.csv'):
                src = os.path.join(root, f)
                dst = os.path.join(out_dir, 'kc_house_data.csv')
                print('Found dataset:', src)
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.move(src, dst)
                return dst

    # If not found, list files
    print('Downloaded files:')
    for f in os.listdir(out_dir):
        print(' -', f)
    raise FileNotFoundError('kc_house_data.csv not found after download; check dataset contents')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='dat', help='Output directory to place dataset')
    args = parser.parse_args()
    path = run_kaggle_download(out_dir=args.out)
    print('Saved dataset to', path)
