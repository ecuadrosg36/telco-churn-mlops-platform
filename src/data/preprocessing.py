"""
Data preprocessing module for Telco Churn dataset.

Handles data cleaning, categorical encoding, and feature scaling.
Returns a reusable scikit-learn pipeline.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config
from src.logging_utils import setup_logger

logger = setup_logger(__name__)


class DataCleaner(BaseEstimator, TransformerMixin):
    """Clean raw telco data."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data:
        - Remove customerID (not a feature)
        - Convert TotalCharges to numeric
        - Handle missing values
        """
        X = X.copy()
        
        #Remove customerID if present
        if 'customerID' in X.columns:
            X = X.drop('customerID', axis=1)
        
        # Convert TotalCharges to numeric (sometimes stored as string with spaces)
        if 'TotalCharges' in X.columns:
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
            
            # Fill missing TotalCharges with 0 (likely new customers)
            missing_total_charges = X['TotalCharges'].isna().sum()
            if missing_total_charges > 0:
                logger.info(f"Filling {missing_total_charges} missing TotalCharges with 0")
                X['TotalCharges'] = X['TotalCharges'].fillna(0)
        
        # Convert SeniorCitizen to string for consistent encoding
        if 'SeniorCitizen' in X.columns:
            X['SeniorCitizen'] = X['SeniorCitizen'].astype(str)
        
        logger.info(f"Data cleaned: {X.shape}")
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical variables."""
    
    def __init__(self):
        self.config = get_config()
        self.categorical_features = self.config.get('features.categorical_features')
        self.target = self.config.get('features.target')
        self.encoders: Dict[str, LabelEncoder] = {}
        
    def fit(self, X, y=None):
        """Fit label encoders on categorical features."""
        X = X.copy()
        
        for feature in self.categorical_features:
            if feature in X.columns:
                self.encoders[feature] = LabelEncoder()
                self.encoders[feature].fit(X[feature].astype(str))
        
        # Also encode target if present
        if y is not None and self.target:
            self.encoders[self.target] = LabelEncoder()
            self.encoders[self.target].fit(y.astype(str))
        
        logger.info(f"Fitted encoders for {len(self.encoders)} features")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features using fitted encoders."""
        X = X.copy()
        
        for feature, encoder in self.encoders.items():
            if feature in X.columns:
                # Handle unseen categories
                X[feature] = X[feature].astype(str).map(
                    lambda x: x if x in encoder.classes_ else encoder.classes_[0]
                )
                X[feature] = encoder.transform(X[feature])
        
        return X
    
    def transform_target(self, y: pd.Series) -> np.ndarray:
        """Transform target variable."""
        if self.target in self.encoders:
            y_str = y.astype(str)
            # Handle unseen categories
            y_str = y_str.map(
                lambda x: x if x in self.encoders[self.target].classes_ 
                else self.encoders[self.target].classes_[0]
            )
            return self.encoders[self.target].transform(y_str)
        return y.values
    
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform target variable."""
        if self.target in self.encoders:
            return self.encoders[self.target].inverse_transform(y)
        return y


class FeatureScaler(BaseEstimator, TransformerMixin):
    """Scale numerical features."""
    
    def __init__(self):
        self.config = get_config()
        self.numerical_features = self.config.get('features.numerical_features')
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def fit(self, X, y=None):
        """Fit scaler on numerical features."""
        X = X.copy()
        
        # Identify which numerical features are present
        self.feature_names = [f for f in self.numerical_features if f in X.columns]
        
        if self.feature_names:
            self.scaler.fit(X[self.feature_names])
            logger.info(f"Fitted scaler for {len(self.feature_names)} numerical features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        X = X.copy()
        
        if self.feature_names:
            X[self.feature_names] = self.scaler.transform(X[self.feature_names])
        
        return X


def create_preprocessing_pipeline() -> Pipeline:
    """
    Create sklearn preprocessing pipeline.
    
    Returns:
        Pipeline with cleaning, encoding, and scaling steps
    """
    pipeline = Pipeline([
        ('cleaner', DataCleaner()),
        ('encoder', CategoricalEncoder()),
        ('scaler', FeatureScaler()),
    ])
    
    logger.info("Created preprocessing pipeline")
    return pipeline


def preprocess_data(
    df: pd.DataFrame,
    target_col: str = None,
    fitted_pipeline: Pipeline = None
) -> Tuple[pd.DataFrame, pd.Series, Pipeline]:
    """
    Preprocess data using sklearn pipeline.
    
    Args:
        df: Raw dataframe
        target_col: Name of target column (if None, loads from config)
        fitted_pipeline: Pre-fitted pipeline (for inference)
        
    Returns:
        Tuple of (processed features, processed target, fitted pipeline)
    """
    config = get_config()
    if target_col is None:
        target_col = config.get('features.target')
    
    # Separate features and target
    if target_col and target_col in df.columns:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
    else:
        X = df.copy()
        y = None
    
    # Fit or use existing pipeline
    if fitted_pipeline is None:
        pipeline = create_preprocessing_pipeline()
        
        # Fit pipeline
        if y is not None:
            pipeline.fit(X, y)
        else:
            pipeline.fit(X)
    else:
        pipeline = fitted_pipeline
    
    # Transform features
    X_processed = pipeline.transform(X)
    
    # Transform target
    if y is not None:
        encoder = pipeline.named_steps['encoder']
        y_processed = pd.Series(
            encoder.transform_target(y),
            index=y.index,
            name=target_col
        )
    else:
        y_processed = None
    
    logger.info(f"Preprocessing complete: X shape {X_processed.shape}")
    
    return X_processed, y_processed, pipeline


if __name__ == "__main__":
    # Test preprocessing
    import pandas as pd
    
    # Create sample data
    sample_data = {
        'customerID': ['C001', 'C002', 'C003'],
        'gender': ['Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0],
        'Partner': ['Yes', 'No', 'Yes'],
        'Dependents': ['No', 'Yes', 'No'],
        'tenure': [12, 24, 6],
        'MonthlyCharges': [50.0, 75.0, 45.0],
        'TotalCharges': ['600.0', '1800.0', ' '],  # Note: space for missing
        'PhoneService': ['Yes', 'No', 'Yes'],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Electronic check'],
        'Churn': ['No', 'Yes', 'No'],
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df.head())
    
    X_processed, y_processed, pipeline = preprocess_data(df)
    
    print("\nProcessed features:")
    print(X_processed.head())
    print(f"\nShape: {X_processed.shape}")
    
    if y_processed is not None:
        print("\nProcessed target:")
        print(y_processed)
