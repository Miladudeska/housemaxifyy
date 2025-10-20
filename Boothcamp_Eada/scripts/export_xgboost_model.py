import os
import joblib

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
PIPELINE_PATH = os.path.join(MODEL_DIR, 'pipeline_xgb.pkl')
OUT_PATH = os.path.join(MODEL_DIR, 'xgboost_kc_house.pkl')

def main():
    if not os.path.exists(PIPELINE_PATH):
        print('Pipeline not found at', PIPELINE_PATH)
        return

    pipeline = joblib.load(PIPELINE_PATH)
    # attempt to get the underlying estimator
    model = None
    if hasattr(pipeline, 'named_steps') and 'model' in pipeline.named_steps:
        model = pipeline.named_steps['model']
    else:
        # if pipeline is itself an estimator
        model = pipeline

    try:
        joblib.dump(model, OUT_PATH)
        print('Saved XGBoost model to', OUT_PATH)
    except Exception as e:
        print('Failed to save model:', e)

if __name__ == '__main__':
    main()
