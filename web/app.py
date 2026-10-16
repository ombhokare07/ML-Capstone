from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import RiskPredictor

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'delivery_risk_model.pkl')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')

predictor = None

def load_predictor():
    global predictor
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        predictor = RiskPredictor(MODEL_PATH, PREPROCESSOR_PATH)
        return True
    return False

def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None

    try:
        with open(METRICS_PATH, 'r', encoding='utf-8') as metrics_file:
            metrics = json.load(metrics_file)

        report = metrics.get('classification_report', {})
        weighted = report.get('weighted avg', {})
        test_samples = int(weighted.get('support', 0))
        if not test_samples:
            test_samples = sum(
                int(values.get('support', 0))
                for label, values in report.items()
                if label in ('0', '1')
            )

        return {
            'accuracy': float(metrics['accuracy']),
            'roc_auc': float(metrics['roc_auc']),
            'f1': float(weighted['f1-score']),
            'test_samples': test_samples,
        }
    except (OSError, ValueError, TypeError, KeyError):
        return None

@app.route('/')
def index():
    model_loaded = load_predictor()
    metrics = load_metrics() if model_loaded else None
    return render_template(
        'index.html',
        model_loaded=model_loaded,
        metrics=metrics,
    )

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        if not load_predictor():
            return jsonify({'error': 'Model not trained yet. Please train the model first.'}), 400
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        result = predictor.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if predictor is None:
        if not load_predictor():
            return jsonify({'error': 'Model not trained yet'}), 400
    
    try:
        data = request.get_json()
        if not data or 'records' not in data:
            return jsonify({'error': 'Expected {"records": [...]}'}), 400
        
        results = predictor.predict_batch(data['records'])
        return jsonify({'predictions': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch_file', methods=['POST'])
def predict_batch_file():
    if predictor is None:
        if not load_predictor():
            return jsonify({'error': 'Model not trained yet'}), 400
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.json'):
            df = pd.read_json(file)
        else:
            return jsonify({'error': 'Unsupported file format. Use CSV or JSON'}), 400
        
        records = df.to_dict('records')
        results = predictor.predict_batch(records)
        
        summary = {
            'total': len(results),
            'high_risk': sum(1 for r in results if r['prediction'] == 1),
            'low_risk': sum(1 for r in results if r['prediction'] == 0)
        }
        
        return jsonify({'predictions': results, 'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    model_exists = os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)
    return jsonify({
        'status': 'ok',
        'model_loaded': predictor is not None,
        'model_exists': model_exists
    })

if __name__ == '__main__':
    load_predictor()
    app.run(debug=True, host='0.0.0.0', port=5000)