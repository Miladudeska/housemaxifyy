from flask import Flask, render_template, request, redirect, url_for, flash
import datetime
import os
import pandas as pd
import numpy as np
import joblib

# Project root (one level above this package)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev-secret')


def load_model_and_artifacts():
    """Try several common artifact filenames and return (model, artifacts) or (None, None)."""
    model_dir = os.path.join(PROJECT_ROOT, 'models')
    model_paths = [
        os.path.join(model_dir, 'xgboost_kc_house.pkl'),
        os.path.join(model_dir, 'xgboost_regressor.pkl'),
        os.path.join(model_dir, 'linear_regression_model.pkl'),
        os.path.join(model_dir, 'linear_regression_model_logtarget.pkl'),
    ]
    artifact_paths = [
        os.path.join(model_dir, 'artifacts_xgb.pkl'),
        os.path.join(model_dir, 'artifacts_logtarget.pkl'),
        os.path.join(model_dir, 'artifacts.pkl'),
    ]

    # Prefer a serialized pipeline if present
    pipeline_path = os.path.join(model_dir, 'pipeline_xgb.pkl')
    pipeline_artifacts = os.path.join(model_dir, 'artifacts_pipeline.pkl')

    if os.path.exists(pipeline_path):
        try:
            pipeline = joblib.load(pipeline_path)
            artifacts = None
            if os.path.exists(pipeline_artifacts):
                try:
                    artifacts = joblib.load(pipeline_artifacts)
                except Exception:
                    artifacts = None
            return pipeline, artifacts
        except Exception:
            # fall back to previous behavior
            pass

    model = None
    artifacts = None
    for p in model_paths:
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                break
            except Exception:
                model = None
    for a in artifact_paths:
        if os.path.exists(a):
            try:
                artifacts = joblib.load(a)
                break
            except Exception:
                artifacts = None

    return model, artifacts


@app.route('/')
def index():
    return render_template('index.html', now=datetime.datetime.now())


@app.route('/form', methods=['GET'])
def form():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Collect form fields (safe defaults)
    fields = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
        'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built',
        'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15',
        'sale_year', 'sale_month'
    ]

    data = {}
    for f in fields:
        val = request.form.get(f, '')
        data[f] = val

    # Save submission to CSV (append)
    save_path = os.path.join(PROJECT_ROOT, 'real_estate_portal', 'user_data.csv')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    row = data.copy()
    row['submitted_at'] = pd.Timestamp.now()
    # Ensure CSV exists with headers
    if not os.path.exists(save_path):
        pd.DataFrame([row]).to_csv(save_path, index=False)
    else:
        pd.DataFrame([row]).to_csv(save_path, mode='a', header=False, index=False)

    # Try to load model and artifacts
    model, artifacts = load_model_and_artifacts()
    predicted = None
    message = None
    if model is None and artifacts is None:
        message = 'Model or artifacts not found; saved submission to user_data.csv.'
    else:
        # artifacts should contain feature_names when present
        feature_names = None
        if isinstance(artifacts, dict):
            feature_names = artifacts.get('feature_names')

        if feature_names is None:
            message = 'No feature names in artifacts; cannot reliably build input vector.'
        else:
            # Build dataframe using feature_names then call model.predict
            X = pd.DataFrame(columns=feature_names)
            X.loc[0] = 0
            for fname in feature_names:
                # Accept raw numeric fields from form if present
                if fname in data:
                    try:
                        X.at[0, fname] = float(data[fname])
                    except Exception:
                        pass

            try:
                # If the loaded model is a pipeline it will handle imputation/scaling
                arr = X.values.astype(float)
                predicted = None
                # model could be a pipeline or a bare estimator
                predicted_arr = model.predict(arr)
                # predict may return array-like
                if hasattr(predicted_arr, '__len__'):
                    predicted = float(predicted_arr[0])
                else:
                    predicted = float(predicted_arr)
            except Exception as e:
                message = f'Prediction failed: {e}'

    return render_template('form.html', submitted=data, prediction=predicted, message=message)


if __name__ == '__main__':
    # Run diagnostics first (script lives in ../scripts/run_diagnostics.py)
    try:
        # Ensure scripts directory is importable
        import sys
        scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)

        from run_diagnostics import run_diagnostics
        run_diagnostics()
    except Exception as _diag_err:
        print('⚠️  Could not run diagnostics:', _diag_err)

    try:
        print("🚀 Starting Flask server at http://127.0.0.1:5000")
        app.run(debug=True, host='127.0.0.1', port=5000)
    except Exception as e:
        print("❌ Flask server failed to start.")
        print("Error details:", e)
