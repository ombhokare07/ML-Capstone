import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json


class FoldPreprocessor(BaseEstimator, TransformerMixin):
    """Fit feature transformations independently inside each CV fold."""

    def fit(self, X, y=None):
        try:
            from .data_preprocessing import DataPreprocessor
        except ImportError:
            from data_preprocessing import DataPreprocessor

        self.preprocessor = DataPreprocessor()
        self.preprocessor.transform_features(X, fit=True)
        return self

    def transform(self, X):
        return self.preprocessor.transform_features(X, fit=False)


class DeliveryRiskModel:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight='balanced'
        )
        self.best_params = None
        self.feature_importance = None

    def train(self, X_train, y_train):
        """Train the decision tree model"""
        self.model.fit(X_train, y_train)
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return self

    def hyperparameter_tuning(self, X_train, y_train, param_grid=None, cv=5):
        """Select a tree, SVC, or boosting model using fold-local preprocessing."""
        if param_grid is None:
            param_grid = [
                {
                    'classifier': [DecisionTreeClassifier(random_state=42)],
                    'classifier__max_depth': [2, 3, 5, None],
                    'classifier__min_samples_split': [2, 10],
                    'classifier__min_samples_leaf': [1, 4],
                    'classifier__criterion': ['gini', 'entropy'],
                    'classifier__class_weight': [None, 'balanced'],
                },
                {
                    'classifier': [SVC(probability=True, random_state=42)],
                    'classifier__C': [0.1, 1, 10],
                    'classifier__kernel': ['rbf'],
                    'classifier__class_weight': [None, 'balanced'],
                },
                {
                    'classifier': [GradientBoostingClassifier(random_state=42)],
                    'classifier__n_estimators': [50, 100],
                    'classifier__learning_rate': [0.05, 0.1],
                    'classifier__max_depth': [1, 2],
                    'classifier__min_samples_leaf': [3],
                },
            ]

        pipeline = Pipeline([
            ('preprocessor', FoldPreprocessor()),
            ('classifier', DecisionTreeClassifier(random_state=42)),
        ])
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=1
        )
        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_
        self.model = best_pipeline.named_steps['classifier']
        self.best_params = {
            'model': type(self.model).__name__,
            **{
                key.split('__', 1)[1]: value
                for key, value in grid_search.best_params_.items()
                if key.startswith('classifier__')
            },
        }
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': best_pipeline.named_steps[
                    'preprocessor'
                ].preprocessor.feature_columns,
                'importance': self.model.feature_importances_,
            }).sort_values('importance', ascending=False)
        else:
            self.feature_importance = None

        return self.best_params, grid_search.best_score_

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return metrics, y_pred, y_prob

    def cross_validate(self, X, y, cv=5, preprocessor_factory=None):
        """Perform stratified cross-validation, fitting preprocessing within each fold."""
        if preprocessor_factory is None:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1')
        else:
            splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            scores = []
            for train_indices, validation_indices in splitter.split(X, y):
                preprocessor = preprocessor_factory()
                X_fold_train = preprocessor.transform_features(
                    X.iloc[train_indices], fit=True
                )
                X_fold_validation = preprocessor.transform_features(
                    X.iloc[validation_indices], fit=False
                )
                fold_model = clone(self.model)
                fold_model.fit(X_fold_train, y.iloc[train_indices])
                predictions = fold_model.predict(X_fold_validation)
                scores.append(f1_score(y.iloc[validation_indices], predictions))
            scores = np.asarray(scores)

        return {
            'mean_f1': scores.mean(),
            'std_f1': scores.std(),
            'scores': scores.tolist()
        }

    def plot_feature_importance(self, top_n=15, save_path=None):
        """Plot feature importance"""
        if self.feature_importance is None:
            return
        
        plt.figure(figsize=(10, 6))
        top_features = self.feature_importance.head(top_n)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_decision_tree(self, feature_names, class_names, save_path=None, max_depth=3):
        """Plot decision tree visualization"""
        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            max_depth=max_depth,
            fontsize=10
        )
        plt.title('Decision Tree Visualization')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_model(self, filepath):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)

    def save_metrics(self, metrics, filepath):
        """Save evaluation metrics"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    print("DeliveryRiskModel class ready for use")