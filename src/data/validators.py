"""
Lightweight data quality validation layer for Telco Churn data.

Validates schema, data types, ranges, and categorical values before training or inference.
"""

import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config
from src.logging_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]


class TelcoDataValidator:
    """Validator for Telco Customer Churn dataset."""
    
    def __init__(self):
        self.config = get_config()
        self.target = self.config.get('features.target')
        self.categorical_features = self.config.get('features.categorical_features')
        self.numerical_features = self.config.get('features.numerical_features')
        
        # Expected schema
        self.expected_columns = (
            ['customerID'] + 
            self.categorical_features + 
            self.numerical_features + 
            [self.target]
        )
        
        # Valid categorical values
        self.valid_categories = {
            'gender': ['Male', 'Female'],
            'SeniorCitizen': [0, 1],
            'Partner': ['Yes', 'No'],
            'Dependents': ['Yes', 'No'],
            'PhoneService': ['Yes', 'No'],
            'MultipleLines': ['Yes', 'No', 'No phone service'],
            'InternetService': ['DSL', 'Fiber optic', 'No'],
            'OnlineSecurity': ['Yes', 'No', 'No internet service'],
            'OnlineBackup': ['Yes', 'No', 'No internet service'],
            'DeviceProtection': ['Yes', 'No', 'No internet service'],
            'TechSupport': ['Yes', 'No', 'No internet service'],
            'StreamingTV': ['Yes', 'No', 'No internet service'],
            'StreamingMovies': ['Yes', 'No', 'No internet service'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaperlessBilling': ['Yes', 'No'],
            'PaymentMethod': [
                'Electronic check', 'Mailed check',
                'Bank transfer (automatic)', 'Credit card (automatic)'
            ],
            'Churn': ['Yes', 'No'],
        }
    
    def validate(self, df: pd.DataFrame, strict: bool = True) -> ValidationResult:
        """
        Validate dataframe against expected schema and constraints.
        
        Args:
            df: DataFrame to validate
            strict: If True, raise exception on validation errors
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        stats = {}
        
        logger.info("Starting data validation...")
        
        # 1. Schema validation
        missing_cols = set(self.expected_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        extra_cols = set(df.columns) - set(self.expected_columns)
        if extra_cols:
            warnings.append(f"Unexpected columns found: {extra_cols}")
        
        # 2. Empty dataframe check
        if df.empty:
            errors.append("Dataset is empty")
            return ValidationResult(False, errors, warnings, stats)
        
        stats['row_count'] = len(df)
        stats['column_count'] = len(df.columns)
        
        # 3. Critical nulls check
        critical_features = ['customerID', 'tenure', 'MonthlyCharges']
        for feature in critical_features:
            if feature in df.columns:
                null_count = df[feature].isna().sum()
                if null_count > 0:
                    errors.append(
                        f"Critical feature '{feature}' contains {null_count} null values"
                    )
        
    # 4. Numerical features validation
        for feature in self.numerical_features:
            if feature not in df.columns:
                continue
                
            # Check data type (convert if needed)
            if df[feature].dtype == 'object':
                warnings.append(f"Numerical feature '{feature}' has object dtype, attempting conversion")
                try:
                    df[feature] = pd.to_numeric(df[feature], errors='coerce')
                except Exception as e:
                    errors.append(f"Failed to convert '{feature}' to numeric: {str(e)}")
                    continue
            
            # Range validation
            if feature == 'tenure':
                invalid = (df[feature] < 0) | (df[feature] > 100)
                if invalid.sum() > 0:
                    errors.append(
                        f"Feature '{feature}' has {invalid.sum()} values outside valid range [0, 100]"
                    )
            
            elif feature in ['MonthlyCharges', 'TotalCharges']:
                invalid = df[feature] < 0
                if invalid.sum() > 0:
                    errors.append(
                        f"Feature '{feature}' has {invalid.sum()} negative values"
                    )
            
            # Null count
            null_pct = (df[feature].isna().sum() / len(df)) * 100
            if null_pct > 5:
                warnings.append(
                    f"Feature '{feature}' has {null_pct:.1f}% null values"
                )
        
        # 5. Categorical features validation
        for feature in self.categorical_features:
            if feature not in df.columns:
                continue
            
            # Check valid categories
            if feature in self.valid_categories:
                unique_vals = set(df[feature].dropna().unique())
                expected_vals = set(self.valid_categories[feature])
                
                invalid_vals = unique_vals - expected_vals
                if invalid_vals:
                    errors.append(
                        f"Feature '{feature}' has invalid categories: {invalid_vals}"
                    )
            
            # Null count
            null_pct = (df[feature].isna().sum() / len(df)) * 100
            if null_pct > 5:
                warnings.append(
                    f"Feature '{feature}' has {null_pct:.1f}% null values"
                )
        
        # 6. Target variable validation
        if self.target in df.columns:
            target_dist = df[self.target].value_counts()
            stats['target_distribution'] = target_dist.to_dict()
            
            # Check for imbalance
            if len(target_dist) > 0:
                imbalance_ratio = target_dist.max() / target_dist.min()
                if imbalance_ratio > 3:
                    warnings.append(
                        f"Target variable is imbalanced (ratio: {imbalance_ratio:.2f})"
                    )
        
        # Determine validation result
        is_valid = len(errors) == 0
        
        # Log results
        if is_valid:
            logger.info(f"✓ Validation passed with {len(warnings)} warnings")
        else:
            logger.error(f"✗ Validation failed with {len(errors)} errors")
            for error in errors:
                logger.error(f"  - {error}")
        
        for warning in warnings:
            logger.warning(f"  - {warning}")
        
        result = ValidationResult(is_valid, errors, warnings, stats)
        
        if strict and not is_valid:
            raise ValueError(f"Data validation failed:\n" + "\n".join(errors))
        
        return result


def validate_dataframe(df: pd.DataFrame, strict: bool = True) -> ValidationResult:
    """
    Convenience function to validate a dataframe.
    
    Args:
        df: DataFrame to validate
        strict: If True, raise exception on validation errors
        
    Returns:
        ValidationResult
    """
    validator = TelcoDataValidator()
    return validator.validate(df, strict=strict)


if __name__ == "__main__":
    # Test validation
    import pandas as pd
    
    # Create sample data
    sample_data = {
        'customerID': ['C001', 'C002'],
        'gender': ['Male', 'Female'],
        'SeniorCitizen': [0, 1],
        'Partner': ['Yes', 'No'],
        'Dependents': ['No', 'Yes'],
        'tenure': [12, 24],
        'MonthlyCharges': [50.0, 75.0],
        'TotalCharges': [600.0, 1800.0],
        'Churn': ['No', 'Yes'],
    }
    
    df = pd.DataFrame(sample_data)
    
    try:
        result = validate_dataframe(df, strict=False)
        print(f"\nValidation Result: {'PASS' if result.is_valid else 'FAIL'}")
        print(f"Errors: {len(result.errors)}")
        print(f"Warnings: {len(result.warnings)}")
    except Exception as e:
        print(f"Validation failed: {e}")
