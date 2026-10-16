import pandas as pd
import joblib

try:
    from .data_preprocessing import DataPreprocessor
except ImportError:
    from data_preprocessing import DataPreprocessor


class RiskPredictor:
    def __init__(self, model_path, preprocessor_path):
        """Initialize predictor with trained model and preprocessor"""
        self.model = joblib.load(model_path)
        self.preprocessor = DataPreprocessor()
        self.preprocessor.load_preprocessor(preprocessor_path)

    def preprocess_input(self, input_data):
        """Preprocess single or batch input data"""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise ValueError("Input must be dict, list of dicts, or DataFrame")

        return self.preprocessor.transform_features(df, fit=False)

    def predict(self, input_data):
        """Predict delivery risk"""
        processed_data = self.preprocess_input(input_data)
        predictions = self.model.predict(processed_data)
        probabilities = self.model.predict_proba(processed_data)[:, 1]
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'prediction': int(pred),
                'risk_probability': float(prob),
                'risk_level': 'High' if pred == 1 else 'Low'
            })
        
        return results[0] if len(results) == 1 else results

    def predict_batch(self, input_data_list):
        """Predict for multiple records"""
        return self.predict(input_data_list)


def create_sample_input():
    """Create a sample input for testing"""
    return {
        'order_id': 'ORD-001',
        'product_category': 'Electronics',
        'supplier_region': 'Asia',
        'destination_region': 'North America',
        'shipping_mode': 'Air',
        'order_quantity': 100,
        'unit_price': 250.0,
        'order_month': 6,
        'order_day_of_week': 2,
        'supplier_reliability_score': 0.85,
        'historical_delay_rate': 0.15,
        'weather_risk_score': 0.3,
        'port_congestion_level': 0.4,
        'customs_clearance_time': 2.5,
        'distance_km': 12000
    }


if __name__ == "__main__":
    print("RiskPredictor class ready for use")
    print("\nSample input format:")
    sample = create_sample_input()
    for k, v in sample.items():
        print(f"  {k}: {v}")