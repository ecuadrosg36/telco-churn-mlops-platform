"""
Unit tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import (
    TenureBinner,
    ChargesRatioCalculator,
    ServiceCountCalculator,
    LongTermCustomerIndicator,
    MonthlyChargesBinner,
    ContractValueIndicator,
    FeatureEngineer,
    engineer_features,
)


@pytest.mark.unit
class TestTenureBinner:
    """Tests for TenureBinner transformer."""

    def test_creates_tenure_bins(self):
        """Test tenure is binned correctly."""
        df = pd.DataFrame({"tenure": [6, 18, 48]})

        binner = TenureBinner()
        result = binner.fit_transform(df)

        assert "tenure_bins" in result.columns
        assert result["tenure_bins"].iloc[0] == "new"
        assert result["tenure_bins"].iloc[1] == "medium"
        assert result["tenure_bins"].iloc[2] == "long_term"

    def test_handles_edge_cases(self):
        """Test tenure binning edge cases."""
        df = pd.DataFrame({"tenure": [0, 12, 36, 100]})

        binner = TenureBinner()
        result = binner.fit_transform(df)

        assert result["tenure_bins"].iloc[0] == "new"
        assert result["tenure_bins"].iloc[1] == "new"
        assert result["tenure_bins"].iloc[2] == "medium"
        assert result["tenure_bins"].iloc[3] == "long_term"


@pytest.mark.unit
class TestChargesRatioCalculator:
    """Tests for ChargesRatioCalculator transformer."""

    def test_calculates_ratio(self):
        """Test charges ratio calculation."""
        df = pd.DataFrame(
            {"TotalCharges": [1000.0, 2000.0], "MonthlyCharges": [100.0, 50.0]}
        )

        calculator = ChargesRatioCalculator()
        result = calculator.fit_transform(df)

        assert "charges_ratio" in result.columns
        assert result["charges_ratio"].iloc[0] == 10.0
        assert result["charges_ratio"].iloc[1] == 40.0

    def test_handles_zero_monthly_charges(self):
        """Test handling of zero monthly charges."""
        df = pd.DataFrame({"TotalCharges": [1000.0], "MonthlyCharges": [0.0]})

        calculator = ChargesRatioCalculator()
        result = calculator.fit_transform(df)

        assert result["charges_ratio"].iloc[0] == 0.0


@pytest.mark.unit
class TestServiceCountCalculator:
    """Tests for ServiceCountCalculator transformer."""

    def test_counts_services(self):
        """Test service counting."""
        df = pd.DataFrame(
            {
                "PhoneService": ["Yes", "No"],
                "InternetService": ["DSL", "No"],
                "OnlineSecurity": ["Yes", "No internet service"],
                "StreamingTV": ["Yes", "No internet service"],
            }
        )

        calculator = ServiceCountCalculator()
        result = calculator.fit_transform(df)

        assert "service_count" in result.columns
        assert result["service_count"].iloc[0] == 4  # All services
        assert result["service_count"].iloc[1] == 0  # No services

    def test_ignores_no_service_values(self):
        """Test that 'No' and 'No X service' don't count."""
        df = pd.DataFrame(
            {
                "PhoneService": ["No"],
                "InternetService": ["No"],
                "OnlineSecurity": ["No internet service"],
            }
        )

        calculator = ServiceCountCalculator()
        result = calculator.fit_transform(df)

        assert result["service_count"].iloc[0] == 0


@pytest.mark.unit
class TestLongTermCustomerIndicator:
    """Tests for LongTermCustomerIndicator transformer."""

    def test_creates_indicator(self):
        """Test long-term customer indicator."""
        df = pd.DataFrame({"tenure": [12, 30]})

        indicator = LongTermCustomerIndicator(threshold=24)
        result = indicator.fit_transform(df)

        assert "is_long_term_customer" in result.columns
        assert result["is_long_term_customer"].iloc[0] == 0
        assert result["is_long_term_customer"].iloc[1] == 1


@pytest.mark.unit
class TestMonthlyChargesBinner:
    """Tests for MonthlyChargesBinner transformer."""

    def test_bins_monthly_charges(self):
        """Test monthly charges binning."""
        df = pd.DataFrame({"MonthlyCharges": [25.0, 50.0, 90.0]})

        binner = MonthlyChargesBinner()
        result = binner.fit_transform(df)

        assert "monthly_charges_tier" in result.columns
        assert result["monthly_charges_tier"].iloc[0] == "low"
        assert result["monthly_charges_tier"].iloc[1] == "medium"
        assert result["monthly_charges_tier"].iloc[2] == "high"


@pytest.mark.unit
class TestContractValueIndicator:
    """Tests for ContractValueIndicator transformer."""

    def test_identifies_long_term_contracts(self):
        """Test long-term contract identification."""
        df = pd.DataFrame({"Contract": ["Month-to-month", "One year", "Two year"]})

        indicator = ContractValueIndicator()
        result = indicator.fit_transform(df)

        assert "has_long_term_contract" in result.columns
        assert result["has_long_term_contract"].iloc[0] == 0
        assert result["has_long_term_contract"].iloc[1] == 1
        assert result["has_long_term_contract"].iloc[2] == 1


@pytest.mark.unit
class TestFeatureEngineer:
    """Tests for complete FeatureEngineer."""

    def test_adds_all_features(self, sample_dataframe):
        """Test all engineered features are added."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(sample_dataframe)

        expected_features = engineer.get_feature_names()

        for feature in expected_features:
            assert feature in result.columns

    def test_preserve_original_features(self, sample_dataframe):
        """Test original features are preserved."""
        original_columns = set(sample_dataframe.columns)

        result = engineer_features(sample_dataframe)

        # All original columns should still be there
        for col in original_columns:
            assert col in result.columns

    def test_feature_count(self, sample_dataframe):
        """Test correct number of features added."""
        original_count = len(sample_dataframe.columns)

        result = engineer_features(sample_dataframe)

        # Should have original + 6 new features
        assert len(result.columns) == original_count + 6
