from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)


def resolve_path(*parts):
    # Resolve path relative to repo root (one level up from this file)
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return os.path.join(base, *parts)


def load_model_and_artifacts():
    model_candidates = [
        resolve_path('models', 'xgboost_kc_house.pkl'),
        resolve_path('models', 'xgboost_regressor.pkl'),
    ]
    artifact_candidates = [
        resolve_path('models', 'artifacts_xgb.pkl'),
        resolve_path('models', 'artifacts.pkl'),
        resolve_path('models', 'artifacts_logtarget.pkl'),
    ]

    model = None
    artifacts = None
    # Prefer a serialized pipeline if present
    pipeline_path = resolve_path('models', 'pipeline_xgb.pkl')
    pipeline_artifacts = resolve_path('models', 'artifacts_pipeline.pkl')
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
            pass
    for p in model_candidates:
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                break
            except Exception:
                model = None

    for a in artifact_candidates:
        if os.path.exists(a):
            try:
                artifacts = joblib.load(a)
                break
            except Exception:
                artifacts = None

    return model, artifacts


def predict_with_artifacts(model, artifacts, inputs):
    # Build feature vector from artifacts['feature_names'] when available
    feature_names = artifacts.get('feature_names') if isinstance(artifacts, dict) else None
    if not feature_names:
        # Fallback: build DataFrame from inputs keys
        X = pd.DataFrame([inputs])
        return float(model.predict(X)[0])

    X = pd.DataFrame(columns=feature_names)
    X.loc[0] = 0
    for fn in feature_names:
        if fn in inputs:
            try:
                X.at[0, fn] = float(inputs[fn])
            except Exception:
                X.at[0, fn] = inputs[fn]

    # Apply imputer and scaler if present
    arr = X.values.astype(float)
    imputer = artifacts.get('imputer')
    scaler = artifacts.get('scaler')
    if imputer is not None:
        arr = imputer.transform(arr)
    if scaler is not None:
        arr = scaler.transform(arr)

    return float(model.predict(arr)[0])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/form', methods=['GET', 'POST'])
def form():
    model, artifacts = load_model_and_artifacts()

    if request.method == 'POST':
        # collect fields (safe cast where possible)
        form_fields = {}
        for k, v in request.form.items():
            form_fields[k] = v

        # minimal property input keys used by older forms
        property_input = {}
        numeric_keys = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors',
                        'waterfront','view','condition','grade','sqft_above',
                        'sqft_basement','yr_built','yr_renovated','lat','long',
                        'sqft_living15','sqft_lot15','sale_year','sale_month']
        for key in numeric_keys:
            if key in form_fields and form_fields[key] not in (None, ''):
                try:
                    property_input[key] = float(form_fields[key])
                except Exception:
                    property_input[key] = form_fields[key]

        # Save submission
        save_path = resolve_path('real_estate_portal', 'user_data.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        row = form_fields.copy()
        row['submitted_at'] = pd.Timestamp.now()
        dfrow = pd.DataFrame([row])
        if os.path.exists(save_path):
            dfrow.to_csv(save_path, mode='a', header=False, index=False)
        else:
            dfrow.to_csv(save_path, index=False)

        predicted = None
        message = None
        if model is None:
            message = 'Model not available — submission saved.'
        else:
            try:
                pred_price = predict_with_artifacts(model, artifacts or {}, property_input)
                predicted = {'low': int(pred_price * 0.9), 'high': int(pred_price * 1.1), 'price': int(pred_price)}
            except Exception as e:
                message = f'Prediction failed: {e}'

        # similar properties: try to read cleaned data
        similar = None
        cleaned_path = resolve_path('dat', 'kc_house_data_clean.csv')
        if os.path.exists(cleaned_path) and 'sqft_living' in property_input:
            try:
                df_clean = pd.read_csv(cleaned_path)
                ref = property_input.get('sqft_living', 2000)
                sim = df_clean[(df_clean['sqft_living'] >= ref * 0.9) & (df_clean['sqft_living'] <= ref * 1.1)]
                similar = sim.sample(min(3, len(sim)))[['sqft_living','bedrooms','bathrooms','price']].to_dict(orient='records')
            except Exception:
                similar = None

        return render_template('form.html', prediction=predicted, submitted=form_fields, message=message, similar=similar)

    return render_template('form.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    model, artifacts = load_model_and_artifacts()
    if model is None:
        return jsonify({'error': 'model not available'}), 503

    data = request.get_json() or {}
    # Extract numeric keys only; the prediction helper expects a dict
    numeric = {}
    for k, v in data.items():
        try:
            numeric[k] = float(v)
        except Exception:
            # ignore non-numeric
            pass

    try:
        price = predict_with_artifacts(model, artifacts or {}, numeric)
        return jsonify({'price': float(price), 'low': float(price * 0.9), 'high': float(price * 1.1)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
