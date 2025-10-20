# DreamNest — King County House Pricing

This repository contains code to clean King County housing data, train regression models (linear and XGBoost), and serve a small Flask web app that provides quick home price estimates.

> Note: The dataset itself is intentionally not included in this repo. Use the Kaggle download helper below to fetch the data locally. If you previously had data in the `dat/` folder, it has been moved locally to `local_data_archive/` to avoid committing datasets.

## Repo layout

- `src/` — data cleaning and training scripts
- `dat/` — data folder (ignored by git). Place `kc_house_data.csv` here.
- `models/` — trained artifacts (ignored by git)
- `real_estate_portal/` — Flask app, templates, static assets
- `scripts/` — helper scripts (download, tests)

## Environment setup

Create a Python venv and install dependencies:

```powershell
python -m venv .venv
# 🏠 KC House Price Prediction Web App

A simple Flask-based web application that uses an XGBoost model to predict
house prices for King County, WA (Seattle area).

## 🚀 How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/kc-house-flask-app.git
cd kc-house-flask-app
pip install -r requirements.txt
python app.py
```
2. Place the token at `%USERPROFILE%\.kaggle\kaggle.json` (Windows) with permissions `600`.
