"""
Feature engineering module for Telco Churn prediction.

Creates domain-specific features to improve model performance.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config
from src.logging_utils import setup_logger

logger = setup_logger(__name__)


class TenureBinner(BaseEstimator, TransformerMixin):
    """Create tenure bins for customer lifetime categorization."""
    
    def __init__(self):
        self.feature_name = 'tenure_bins'
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Bin tenure into categories:
        - new: 0-12 months
        - medium: 13-36 months
        - long_term: 37+ months
        """
        X = X.copy()
        
        if 'tenure'in X.columns:
            X[self.feature_name] = pd.cut(
                X['tenure'],
                bins=[0, 12, 36, np.inf],
                labels=['new', 'medium', 'long_term'],
                include_lowest=True
            ).astype(str)
            
            logger.debug(f"Created {self.feature_name} feature")
        
        return X


class ChargesRatioCalculator(BaseEstimator, TransformerMixin):
    """Calculate ratio of total charges to monthly charges (proxy for loyalty)."""
    
    def __init__(self):
        self.feature_name = 'charges_ratio'
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate TotalCharges / MonthlyCharges ratio.
        High ratio = long-term customer who pays consistently.
        """
        X = X.copy()
        
        if 'TotalCharges' in X.columns and 'MonthlyCharges' in X.columns:
            # Avoid division by zero
            X[self.feature_name] = np.where(
                X['MonthlyCharges'] > 0,
                X['TotalCharges'] / X['MonthlyCharges'],
                0
            )
            
            logger.debug(f"Created {self.feature_name} feature")
        
        return X


class ServiceCountCalculator(BaseEstimator, TransformerMixin):
    """Count number of services customer has subscribed to."""
    
    def __init__(self):
        self.feature_name = 'service_count'
        self.service_features = [
            'PhoneService',
            'MultipleLines',
            'InternetService',
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies'
        ]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Count how many services customer uses.
        'No' or 'No internet service' or 'No phone service' counts as 0.
        'Yes' or actual service name counts as 1.
        """
        X = X.copy()
        
        service_count = 0
        for feature in self.service_features:
            if feature in X.columns:
                # Count as service if value is 'Yes' or a service type (not 'No' or 'No X service')
                has_service = ~X[feature].astype(str).str.contains('No', case=False, na=False)
                service_count += has_service.astype(int)
        
        X[self.feature_name] = service_count
        
        logger.debug(f"Created {self.feature_name} feature (range: {X[self.feature_name].min()}-{X[self.feature_name].max()})")
        
        return X


class LongTermCustomerIndicator(BaseEstimator, TransformerMixin):
    """Binary indicator for long-term customers."""
    
    def __init__(self, threshold: int = 24):
        self.threshold = threshold
        self.feature_name = 'is_long_term_customer'
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary indicator for customers with tenure > threshold months.
        """
        X = X.copy()
        
        if 'tenure' in X.columns:
            X[self.feature_name] = (X['tenure'] > self.threshold).astype(int)
            
            long_term_pct = (X[self.feature_name].sum() / len(X)) * 100
            logger.debug(f"Created {self.feature_name} feature ({long_term_pct:.1f}% long-term)")
        
        return X


class MonthlyChargesBinner(BaseEstimator, TransformerMixin):
    """Bin monthly charges into price tiers."""
    
    def __init__(self):
        self.feature_name = 'monthly_charges_tier'
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Bin monthly charges into price tiers:
        - low: < $35
        - medium: $35-$70
        - high: > $70
        """
        X = X.copy()
        
        if 'MonthlyCharges' in X.columns:
            X[self.feature_name] = pd.cut(
                X['MonthlyCharges'],
                bins=[0, 35, 70, np.inf],
                labels=['low', 'medium', 'high'],
                include_lowest=True
            ).astype(str)
            
            logger.debug(f"Created {self.feature_name} feature")
        
        return X


class ContractValueIndicator(BaseEstimator, TransformerMixin):
    """Binary indicator for long-term contracts (One year or Two year)."""
    
    def __init__(self):
        self.feature_name = 'has_long_term_contract'
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Indicate whether customer has a long-term contract.
        """
        X = X.copy()
        
        if 'Contract' in X.columns:
            X[self.feature_name] = (
                ~X['Contract'].astype(str).str.contains('Month-to-month', case=False, na=False)
            ).astype(int)
            
            logger.debug(f"Created {self.feature_name} feature")
        
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Master feature engineering transformer."""
    
    def __init__(self):
        self.transformers = [
            TenureBinner(),
            ChargesRatioCalculator(),
            ServiceCountCalculator(),
            LongTermCustomerIndicator(threshold=24),
            MonthlyChargesBinner(),
            ContractValueIndicator(),
        ]
        
    def fit(self, X, y=None):
        """Fit all feature transformers."""
        for transformer in self.transformers:
            transformer.fit(X, y)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature transformations."""
        X_transformed = X.copy()
        
        for transformer in self.transformers:
            X_transformed = transformer.transform(X_transformed)
        
        logger.info(f"Feature engineering complete: added {len(self.transformers)} feature sets")
        return X_transformed
    
    def get_feature_names(self) -> List[str]:
        """Get names of engineered features."""
        return [t.feature_name for t in self.transformers]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to dataframe.
    
    Args:
        df: Input dataframe with raw or preprocessed features
        
    Returns:
        DataFrame with engineered features added
    """
    engineer = FeatureEngineer()
    df_engineered = engineer.fit_transform(df)
    
    logger.info(f"Engineered features: {engineer.get_feature_names()}")
    
    return df_engineered


if __name__ == "__main__":
    # Test feature engineering
    import pandas as pd
    
    # Create sample data
    sample_data = {
        'tenure': [1, 15, 40, 6, 50],
        'MonthlyCharges': [25.0, 50.0, 85.0, 30.0, 100.0],
        'TotalCharges': [25.0, 750.0, 3400.0, 180.0, 5000.0],
        'PhoneService': ['Yes', 'Yes', 'Yes', 'No', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'Fiber optic', 'No', 'DSL'],
        'OnlineSecurity': ['No', 'Yes', 'Yes', 'No internet service', 'No'],
        'StreamingTV': ['No', 'Yes', 'Yes', 'No internet service', 'Yes'],
        'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'Two year'],
    }
    
    df = pd.DataFrame(sample_data)
    print("Original features:")
    print(df)
    
    df_engineered = engineer_features(df)
    
    print("\nEngineered features:")
    engineer = FeatureEngineer()
    for feature in engineer.get_feature_names():
        if feature in df_engineered.columns:
            print(f"\n{feature}:")
            print(df_engineered[feature])
