"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import (
    DataCleaner,
    CategoricalEncoder,
    FeatureScaler,
    create_preprocessing_pipeline,
    preprocess_data,
)


@pytest.mark.unit
class TestDataCleaner:
    """Tests for DataCleaner transformer."""

    def test_removes_customer_id(self, sample_dataframe):
        """Test that customerID column is removed."""
        cleaner = DataCleaner()
        result = cleaner.fit_transform(sample_dataframe)

        assert "customerID" not in result.columns

    def test_converts_total_charges_to_numeric(self, sample_dataframe):
        """Test TotalCharges conversion to numeric."""
        # Add string value
        df = sample_dataframe.copy()
        df["TotalCharges"] = df["TotalCharges"].astype(str)

        cleaner = DataCleaner()
        result = cleaner.fit_transform(df)

        assert pd.api.types.is_numeric_dtype(result["TotalCharges"])

    def test_fills_missing_total_charges(self):
        """Test missing TotalCharges are filled with 0."""
        df = pd.DataFrame(
            {
                "customerID": ["C001"],
                "TotalCharges": [" "],  # Space represents missing
                "MonthlyCharges": [50.0],
            }
        )

        cleaner = DataCleaner()
        result = cleaner.fit_transform(df)

        assert result["TotalCharges"].iloc[0] == 0.0

    def test_converts_senior_citizen_to_string(self, sample_dataframe):
        """Test SeniorCitizen conversion to string."""
        cleaner = DataCleaner()
        result = cleaner.fit_transform(sample_dataframe)

        assert result["SeniorCitizen"].dtype == object


@pytest.mark.unit
class TestCategoricalEncoder:
    """Tests for CategoricalEncoder transformer."""

    def test_encodes_categorical_features(self, sample_dataframe):
        """Test categorical features are encoded."""
        encoder = CategoricalEncoder()

        X = sample_dataframe.drop("Churn", axis=1)
        y = sample_dataframe["Churn"]

        encoder.fit(X, y)
        result = encoder.transform(X)

        # Check that categorical columns are now numeric
        assert pd.api.types.is_numeric_dtype(result["gender"])
        assert pd.api.types.is_numeric_dtype(result["Contract"])

    def test_handles_unseen_categories(self, sample_dataframe):
        """Test handling of unseen categories."""
        encoder = CategoricalEncoder()

        X_train = sample_dataframe.drop("Churn", axis=1).iloc[:5]
        X_test = sample_dataframe.drop("Churn", axis=1).iloc[5:]

        encoder.fit(X_train)

        # Should not raise error with different categories
        result = encoder.transform(X_test)

        assert len(result) == len(X_test)

    def test_inverse_transform_target(self, sample_dataframe):
        """Test inverse transformation of target variable."""
        encoder = CategoricalEncoder()

        X = sample_dataframe.drop("Churn", axis=1)
        y = sample_dataframe["Churn"]

        encoder.fit(X, y)
        y_encoded = encoder.transform_target(y)
        y_decoded = encoder.inverse_transform_target(y_encoded)

        assert all(y == y_decoded)


@pytest.mark.unit
class TestFeatureScaler:
    """Tests for FeatureScaler transformer."""

    def test_scales_numerical_features(self, sample_dataframe):
        """Test numerical features are scaled."""
        scaler = FeatureScaler()

        X = sample_dataframe[["tenure", "MonthlyCharges", "TotalCharges"]]

        scaler.fit(X)
        result = scaler.transform(X)

        # Check that means are close to 0 and stds close to 1
        assert abs(result["tenure"].mean()) < 0.1
        assert abs(result["tenure"].std() - 1.0) < 0.1

    def test_handles_missing_features(self):
        """Test scaler handles missing numerical features."""
        scaler = FeatureScaler()

        # DataFrame without some numerical features
        df = pd.DataFrame({"tenure": [1, 2, 3], "other_feature": ["a", "b", "c"]})

        scaler.fit(df)
        result = scaler.transform(df)

        assert "tenure" in result.columns


@pytest.mark.unit
class TestPreprocessingPipeline:
    """Tests for complete preprocessing pipeline."""

    def test_pipeline_creation(self):
        """Test pipeline is created with all steps."""
        pipeline = create_preprocessing_pipeline()

        assert len(pipeline.steps) == 3
        assert pipeline.named_steps["cleaner"] is not None
        assert pipeline.named_steps["encoder"] is not None
        assert pipeline.named_steps["scaler"] is not None

    def test_preprocess_data_with_target(self, sample_dataframe):
        """Test preprocessing with target column."""
        X_processed, y_processed, pipeline = preprocess_data(sample_dataframe)

        assert X_processed is not None
        assert y_processed is not None
        assert "Churn" not in X_processed.columns
        assert len(X_processed) == len(sample_dataframe)

    def test_preprocess_data_without_target(self, sample_dataframe):
        """Test preprocessing without target column."""
        df_no_target = sample_dataframe.drop("Churn", axis=1)

        X_processed, y_processed, pipeline = preprocess_data(df_no_target)

        assert X_processed is not None
        assert y_processed is None
        assert len(X_processed) == len(df_no_target)

    def test_pipeline_is_reusable(self, sample_dataframe):
        """Test fitted pipeline can be reused."""
        # Fit pipeline
        _, _, pipeline = preprocess_data(sample_dataframe)

        # Use fitted pipeline on new data
        new_data = sample_dataframe.copy()
        X_new, y_new, _ = preprocess_data(new_data, fitted_pipeline=pipeline)

        assert X_new is not None
        assert len(X_new) == len(new_data)
