import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.target_column = 'delivery_risk'
        self.categorical_columns = []
        self.numerical_columns = []
        self.identifier_columns = []
        self.category_modes = {}
        self.numeric_medians = {}
        self.feature_columns = None

    def load_data(self, filepath):
        """Load dataset from CSV file."""
        return pd.read_csv(filepath)

    def clean_data(self, df):
        """Remove duplicate rows and rows without a target label."""
        if self.target_column not in df.columns:
            raise ValueError(f"Dataset must include target column '{self.target_column}'.")

        cleaned = df.copy().drop_duplicates()
        return cleaned.dropna(subset=[self.target_column])

    def _identify_identifiers(self, columns):
        return [
            column for column in columns
            if column.lower() == 'id' or column.lower().endswith('_id')
        ]

    def transform_features(self, features, fit=False):
        """Encode categoricals and scale numeric columns using training-only state."""
        df = features.copy()

        if fit:
            self.identifier_columns = self._identify_identifiers(df.columns)
            df = df.drop(columns=self.identifier_columns, errors='ignore')
            self.categorical_columns = df.select_dtypes(
                include=['object', 'category', 'bool']
            ).columns.tolist()
            self.numerical_columns = [
                column for column in df.columns
                if column not in self.categorical_columns
            ]
            self.category_modes = {
                column: df[column].mode(dropna=True).iloc[0]
                if not df[column].mode(dropna=True).empty else 'Unknown'
                for column in self.categorical_columns
            }
            self.numeric_medians = {
                column: float(df[column].median())
                if df[column].notna().any() else 0.0
                for column in self.numerical_columns
            }
        else:
            df = df.drop(columns=self.identifier_columns, errors='ignore')
            for column in self.categorical_columns:
                if column not in df.columns:
                    df[column] = self.category_modes[column]
            for column in self.numerical_columns:
                if column not in df.columns:
                    df[column] = self.numeric_medians[column]
            df = df[self.categorical_columns + self.numerical_columns]

        for column in self.categorical_columns:
            df[column] = df[column].fillna(self.category_modes[column]).astype(str)
        for column in self.numerical_columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
            df[column] = df[column].fillna(self.numeric_medians[column])

        if self.numerical_columns:
            if fit:
                df[self.numerical_columns] = self.scaler.fit_transform(
                    df[self.numerical_columns]
                )
            else:
                df[self.numerical_columns] = self.scaler.transform(
                    df[self.numerical_columns]
                )

        encoded = pd.get_dummies(
            df,
            columns=self.categorical_columns,
            dtype=float,
        )

        if fit:
            self.feature_columns = encoded.columns.tolist()
        elif self.feature_columns is None:
            raise ValueError("Preprocessor has not been fitted.")

        return encoded.reindex(columns=self.feature_columns, fill_value=0.0)

    def prepare_features(self, df, fit=True):
        """Return transformed features and target, if present."""
        cleaned = self.clean_data(df)
        y = cleaned[self.target_column] if self.target_column in cleaned else None
        X = cleaned.drop(columns=[self.target_column], errors='ignore')
        return self.transform_features(X, fit=fit), y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split raw features before fitting preprocessing to prevent leakage."""
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    def save_preprocessor(self, filepath):
        """Save fitted preprocessing state."""
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'target_column': self.target_column,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'identifier_columns': self.identifier_columns,
            'category_modes': self.category_modes,
            'numeric_medians': self.numeric_medians,
            'feature_columns': self.feature_columns,
        }, filepath)

    def load_preprocessor(self, filepath):
        """Load fitted preprocessing state."""
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.target_column = data['target_column']
        self.categorical_columns = data['categorical_columns']
        self.numerical_columns = data['numerical_columns']
        self.identifier_columns = data['identifier_columns']
        self.category_modes = data['category_modes']
        self.numeric_medians = data['numeric_medians']
        self.feature_columns = data['feature_columns']


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    print("DataPreprocessor class ready for use")
