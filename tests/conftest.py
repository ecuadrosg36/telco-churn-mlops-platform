"""
Pytest configuration and shared fixtures.

Provides test data, mock models, and common test utilities.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import tempfile
import shutil


@pytest.fixture
def sample_customer_data() -> Dict[str, Any]:
    """Sample customer data for testing."""
    return {
        'customerID': 'C001',
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'Yes',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'One year',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 65.6,
        'TotalCharges': 787.2,
        'Churn': 'No'
    }


@pytest.fixture
def sample_dataframe(sample_customer_data) -> pd.DataFrame:
    """Sample dataframe with multiple customers."""
    # Create variations of the sample data
    data = []
    for i in range(10):
        customer = sample_customer_data.copy()
        customer['customerID'] = f'C{i:03d}'
        customer['tenure'] = np.random.randint(1, 72)
        customer['MonthlyCharges'] = np.random.uniform(20, 120)
        customer['TotalCharges'] = customer['tenure'] * customer['MonthlyCharges']
        customer['Churn'] = np.random.choice(['Yes', 'No'])
        data.append(customer)
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_churn_request() -> Dict[str, Any]:
    """Sample API prediction request."""
    return {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'Yes',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'One year',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 65.6,
        'TotalCharges': 787.2
    }


@pytest.fixture
def temp_data_dir():
    """Temporary directory for test data files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_csv_file(sample_dataframe, temp_data_dir) -> Path:
    """Sample CSV file for testing."""
    csv_path = temp_data_dir / "test_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_sklearn_model():
    """Mock scikit-learn model for testing."""
    from sklearn.ensemble import RandomForestClassifier
    
    # Create a simple model
    model = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=42)
    
    # Dummy training data
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    
    model.fit(X, y)
    
    return model


@pytest.fixture
def mock_preprocessing_pipeline(sample_dataframe):
    """Mock preprocessing pipeline for testing."""
    from src.data.preprocessing import create_preprocessing_pipeline
    
    pipeline = create_preprocessing_pipeline()
    
    # Fit on sample data
    target = 'Churn'
    X = sample_dataframe.drop(target, axis=1)
    y = sample_dataframe[target]
    
    pipeline.fit(X, y)
    
    return pipeline


@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings."""
    return {
        'test_mode': True,
        'random_seed': 42,
        'sample_size': 10
    }


# Pytest configuration
def pytest_configure(config):
    """Pytest configuration hook."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
