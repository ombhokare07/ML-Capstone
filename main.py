import pandas as pd
import numpy as np
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from model_training import DeliveryRiskModel
from predict import RiskPredictor


def save_eda_artifacts(df, raw_df, data_path, model_dir):
    """Save compact, reproducible summaries and plots before model training."""
    os.makedirs(model_dir, exist_ok=True)
    target_counts = df['delivery_risk'].value_counts().sort_index()
    numeric_df = df.select_dtypes(include=[np.number]).drop(
        columns=['delivery_risk'], errors='ignore'
    )
    correlations = df.select_dtypes(include=[np.number]).corr()['delivery_risk']
    correlations = correlations.drop(labels=['delivery_risk'], errors='ignore')
    outlier_counts = {}
    for column in numeric_df.columns:
        lower_quartile = numeric_df[column].quantile(0.25)
        upper_quartile = numeric_df[column].quantile(0.75)
        iqr = upper_quartile - lower_quartile
        outside_iqr = (
            (numeric_df[column] < lower_quartile - 1.5 * iqr)
            | (numeric_df[column] > upper_quartile + 1.5 * iqr)
        )
        outlier_counts[str(column)] = int(outside_iqr.sum())

    summary = {
        'dataset': os.path.basename(data_path),
        'rows_after_cleaning': int(len(df)),
        'columns': df.columns.tolist(),
        'duplicate_rows_removed': int(len(raw_df) - len(raw_df.drop_duplicates())),
        'rows_without_target_removed': int(raw_df['delivery_risk'].isna().sum()),
        'missing_values': {
            str(column): int(count) for column, count in df.isna().sum().items()
        },
        'target_distribution': {
            str(label): int(count) for label, count in target_counts.items()
        },
        'target_percentages': {
            str(label): round(float(count / len(df) * 100), 2)
            for label, count in target_counts.items()
        },
        'numeric_summary': numeric_df.describe().round(3).to_dict(),
        'iqr_outlier_counts': outlier_counts,
        'target_correlations': {
            str(column): round(float(value), 4)
            for column, value in correlations.sort_values(
                key=lambda values: values.abs(), ascending=False
            ).items()
        },
    }
    with open(os.path.join(model_dir, 'eda_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='delivery_risk', color='#3f8065')
    plt.title('Delivery Risk Class Distribution')
    plt.xlabel('Delivery risk (0 = Low, 1 = High)')
    plt.ylabel('Number of orders')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'target_distribution.png'), dpi=180)
    plt.close()

    numeric_plot_df = df.select_dtypes(include=[np.number]).drop(
        columns=['order_id'], errors='ignore'
    )
    if numeric_plot_df.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            numeric_plot_df.corr(),
            annot=True,
            fmt='.2f',
            cmap='vlag',
            center=0,
            square=True,
        )
        plt.title('Numeric Feature Correlations')
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'correlation_heatmap.png'), dpi=180)
        plt.close()

    print(f"   Dataset rows: {len(df)}")
    print(f"   Duplicate rows removed: {summary['duplicate_rows_removed']}")
    print(f"   Rows without target removed: {summary['rows_without_target_removed']}")
    print(f"   Missing values by column: {summary['missing_values']}")
    print(f"   Target distribution: {summary['target_distribution']}")
    print(f"   IQR outlier counts (review, not automatically removed): {outlier_counts}")
    if not correlations.empty:
        print(f"   Strongest numeric target relationships: "
              f"{summary['target_correlations']}")


def generate_sample_dataset(n_samples=5000, save_path='data/supply_chain_data.csv'):
    """Generate a synthetic supply chain dataset for testing"""
    np.random.seed(42)
    
    data = {
        'order_id': [f'ORD-{i:05d}' for i in range(n_samples)],
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Machinery', 'Chemicals'], n_samples),
        'supplier_region': np.random.choice(['Asia', 'Europe', 'North America', 'South America', 'Africa'], n_samples),
        'destination_region': np.random.choice(['North America', 'Europe', 'Asia', 'South America', 'Africa'], n_samples),
        'shipping_mode': np.random.choice(['Air', 'Sea', 'Road', 'Rail'], n_samples),
        'order_quantity': np.random.randint(10, 1000, n_samples),
        'unit_price': np.random.uniform(10, 1000, n_samples).round(2),
        'order_month': np.random.randint(1, 13, n_samples),
        'order_day_of_week': np.random.randint(0, 7, n_samples),
        'supplier_reliability_score': np.random.beta(5, 2, n_samples).round(3),
        'historical_delay_rate': np.random.beta(2, 8, n_samples).round(3),
        'weather_risk_score': np.random.beta(2, 5, n_samples).round(3),
        'port_congestion_level': np.random.beta(3, 3, n_samples).round(3),
        'customs_clearance_time': np.random.gamma(2, 1.5, n_samples).round(2),
        'distance_km': np.random.lognormal(8, 1, n_samples).astype(int),
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable based on realistic risk factors
    risk_score = (
        (1 - df['supplier_reliability_score']) * 0.3 +
        df['historical_delay_rate'] * 0.25 +
        df['weather_risk_score'] * 0.15 +
        df['port_congestion_level'] * 0.15 +
        (df['customs_clearance_time'] / 10) * 0.1 +
        (df['distance_km'] / 20000) * 0.05
    )
    
    # Add some noise
    risk_score += np.random.normal(0, 0.1, n_samples)
    risk_score = np.clip(risk_score, 0, 1)
    
    # Threshold for high risk
    df['delivery_risk'] = (risk_score > 0.5).astype(int)
    
    # Ensure balanced classes
    high_risk = df[df['delivery_risk'] == 1]
    low_risk = df[df['delivery_risk'] == 0]
    
    min_count = min(len(high_risk), len(low_risk))
    df_balanced = pd.concat([
        high_risk.sample(n=min_count, random_state=42),
        low_risk.sample(n=min_count, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_balanced.to_csv(save_path, index=False)
    print(f"Generated sample dataset: {save_path} ({len(df_balanced)} samples)")
    print(f"Class distribution:\n{df_balanced['delivery_risk'].value_counts()}")
    
    return df_balanced


def train_pipeline(data_path, model_dir='models', tune_hyperparams=False):
    """Complete training pipeline"""
    print("=" * 50)
    print("SUPPLY CHAIN DELIVERY RISK PREDICTION - TRAINING")
    print("=" * 50)
    
    # Initialize components
    preprocessor = DataPreprocessor()
    model = DeliveryRiskModel()
    
    # Load and inspect data before splitting; fit preprocessing on training data only.
    print("\n1. Loading data...")
    raw_df = preprocessor.load_data(data_path)
    df = preprocessor.clean_data(raw_df)
    print(f"   Loaded {len(raw_df)} rows")
    print(f"   Retained {len(df)} rows after duplicate/target cleanup")
    if set(df[preprocessor.target_column].unique()) != {0, 1}:
        raise ValueError(
            "delivery_risk must contain both binary labels 0 (Low Risk) and 1 (High Risk)."
        )

    print("\n2. Exploring data...")
    save_eda_artifacts(df, raw_df, data_path, model_dir)

    print("\n3. Preparing features...")
    X_raw = df.drop(columns=[preprocessor.target_column])
    y = df[preprocessor.target_column]
    y = y.astype(int)
    print(f"   Raw features: {X_raw.shape[1]}, Samples: {X_raw.shape[0]}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")

    print("\n4. Splitting data...")
    X_train_raw, X_test_raw, y_train, y_test = preprocessor.split_data(X_raw, y)
    X_train = preprocessor.transform_features(X_train_raw, fit=True)
    X_test = preprocessor.transform_features(X_test_raw, fit=False)
    print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"   Encoded features: {X_train.shape[1]}")
    if preprocessor.identifier_columns:
        print(f"   Excluded identifier columns: {preprocessor.identifier_columns}")

    # Hyperparameter tuning
    if tune_hyperparams:
        print("\n5. Hyperparameter tuning...")
        best_params, best_score = model.hyperparameter_tuning(
            X_train_raw, y_train
        )
        print(f"   Best params: {best_params}")
        print(f"   Best CV accuracy: {best_score:.4f}")
        tuning_results = {
            'scoring': 'accuracy',
            'best_cv_accuracy': float(best_score),
            'best_params': model.best_params,
        }
    else:
        print("\n5. Training model with default params...")
        model.train(X_train, y_train)
        tuning_results = None

    baseline_model = DummyClassifier(strategy='most_frequent')
    baseline_model.fit(X_train, y_train)
    baseline_predictions = baseline_model.predict(X_test)
    baseline_metrics = {
        'accuracy': float(accuracy_score(y_test, baseline_predictions)),
        'balanced_accuracy': float(
            balanced_accuracy_score(y_test, baseline_predictions)
        ),
    }
    print(f"   Majority-class baseline accuracy: {baseline_metrics['accuracy']:.4f}")
    
    # Evaluate
    print("\n6. Evaluating model on held-out test data...")
    metrics, y_pred, y_prob = model.evaluate(X_test, y_test)
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"   High-risk F1: {metrics['classification_report']['1']['f1-score']:.4f}")
    print(f"   F1-score (weighted): {metrics['classification_report']['weighted avg']['f1-score']:.4f}")
    metrics['baseline'] = baseline_metrics
    if tuning_results is not None:
        metrics['hyperparameter_tuning'] = tuning_results
    
    # Cross-validation
    print("\n7. Cross-validation on training data...")
    cv_results = model.cross_validate(
        X_train_raw.reset_index(drop=True),
        y_train.reset_index(drop=True),
        preprocessor_factory=DataPreprocessor,
    )
    print(f"   Mean F1: {cv_results['mean_f1']:.4f} (+/- {cv_results['std_f1']:.4f})")
    metrics['cross_validation'] = cv_results
    
    # Save everything
    print("\n8. Saving model and artifacts...")
    os.makedirs(model_dir, exist_ok=True)
    
    model.save_model(os.path.join(model_dir, 'delivery_risk_model.pkl'))
    preprocessor.save_preprocessor(os.path.join(model_dir, 'preprocessor.pkl'))
    model.save_metrics(metrics, os.path.join(model_dir, 'metrics.json'))
    
    # Save feature importance
    if model.feature_importance is not None:
        model.feature_importance.to_csv(
            os.path.join(model_dir, 'feature_importance.csv'), index=False
        )
        model.plot_feature_importance(
            save_path=os.path.join(model_dir, 'feature_importance.png')
        )
    elif os.path.exists(os.path.join(model_dir, 'feature_importance.csv')):
        os.remove(os.path.join(model_dir, 'feature_importance.csv'))
        feature_plot = os.path.join(model_dir, 'feature_importance.png')
        if os.path.exists(feature_plot):
            os.remove(feature_plot)

    # Generate plots
    model.plot_confusion_matrix(np.array(metrics['confusion_matrix']), 
                                 save_path=os.path.join(model_dir, 'confusion_matrix.png'))
    if hasattr(model.model, 'tree_'):
        model.plot_decision_tree(
            feature_names=X_train.columns.tolist(),
            class_names=['Low Risk', 'High Risk'],
            save_path=os.path.join(model_dir, 'decision_tree.png'),
            max_depth=3
        )
    else:
        tree_plot = os.path.join(model_dir, 'decision_tree.png')
        if os.path.exists(tree_plot):
            os.remove(tree_plot)
    
    print(f"\nAll artifacts saved to: {model_dir}/")
    print("\nTraining complete!")
    
    return model, preprocessor, metrics


def predict_pipeline(input_data, model_dir='models'):
    """Run prediction on new data"""
    model_path = os.path.join(model_dir, 'delivery_risk_model.pkl')
    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        raise FileNotFoundError("Model not found. Please run training first.")
    
    predictor = RiskPredictor(model_path, preprocessor_path)
    return predictor.predict(input_data)


def main():
    parser = argparse.ArgumentParser(description='Supply Chain Delivery Risk Prediction')
    parser.add_argument('--mode', choices=['train', 'predict', 'generate'], default='train',
                        help='Mode: train model, predict, or generate sample data')
    parser.add_argument('--data', type=str, default='data/supply_chain_data.csv',
                        help='Path to dataset CSV')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory for model artifacts')
    parser.add_argument('--tune', action='store_true',
                        help='Enable hyperparameter tuning')
    parser.add_argument('--input', type=str,
                        help='JSON file with input data for prediction')
    
    args = parser.parse_args()
    
    if args.mode == 'generate':
        generate_sample_dataset(save_path=args.data)
        
    elif args.mode == 'train':
        train_pipeline(args.data, args.model_dir, args.tune)
        
    elif args.mode == 'predict':
        if not args.input:
            print("Error: --input required for predict mode")
            sys.exit(1)
        
        import json
        with open(args.input, 'r') as f:
            input_data = json.load(f)
        
        results = predict_pipeline(input_data, args.model_dir)
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()