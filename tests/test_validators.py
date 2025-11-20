"""
Unit tests for data validation module.
"""

import pytest
import pandas as pd
from src.data.validators import TelcoDataValidator, validate_dataframe


@pytest.mark.unit
class TestTelcoDataValidator:
    """Tests for TelcoDataValidator."""

    def test_valid_data_passes(self, sample_dataframe):
        """Test that valid data passes validation."""
        validator = TelcoDataValidator()
        result = validator.validate(sample_dataframe, strict=False)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_missing_columns_detected(self):
        """Test detection of missing required columns."""
        df = pd.DataFrame(
            {
                "tenure": [12],
                # Missing many required columns
            }
        )

        validator = TelcoDataValidator()
        result = validator.validate(df, strict=False)

        assert not result.is_valid
        assert len(result.errors) > 0

    def test_null_in_critical_feature_detected(self, sample_dataframe):
        """Test detection of nulls in critical features."""
        df = sample_dataframe.copy()
        df.loc[0, "customerID"] = None

        validator = TelcoDataValidator()
        result = validator.validate(df, strict=False)

        assert not result.is_valid
        assert any("customerID" in error for error in result.errors)

    def test_invalid_tenure_range_detected(self, sample_dataframe):
        """Test detection of invalid tenure values."""
        df = sample_dataframe.copy()
        df.loc[0, "tenure"] = -5  # Negative tenure

        validator = TelcoDataValidator()
        result = validator.validate(df, strict=False)

        assert not result.is_valid

    def test_negative_charges_detected(self, sample_dataframe):
        """Test detection of negative charges."""
        df = sample_dataframe.copy()
        df.loc[0, "MonthlyCharges"] = -50.0

        validator = TelcoDataValidator()
        result = validator.validate(df, strict=False)

        assert not result.is_valid

    def test_invalid_categorical_values_detected(self, sample_dataframe):
        """Test detection of invalid categorical values."""
        df = sample_dataframe.copy()
        df.loc[0, "gender"] = "Unknown"  # Invalid gender

        validator = TelcoDataValidator()
        result = validator.validate(df, strict=False)

        assert not result.is_valid
        assert any("gender" in error for error in result.errors)

    def test_target_distribution_in_stats(self, sample_dataframe):
        """Test target distribution is included in stats."""
        validator = TelcoDataValidator()
        result = validator.validate(sample_dataframe, strict=False)

        assert "target_distribution" in result.stats
        assert (
            "Yes" in result.stats["target_distribution"]
            or "No" in result.stats["target_distribution"]
        )

    def test_strict_mode_raises_exception(self):
        """Test strict mode raises exception on validation error."""
        df = pd.DataFrame({"tenure": [12]})  # Missing required columns

        validator = TelcoDataValidator()

        with pytest.raises(ValueError):
            validator.validate(df, strict=True)

    def test_warnings_for_high_null_percentage(self, sample_dataframe):
        """Test warnings are generated for high null percentages."""
        df = sample_dataframe.copy()
        # Add nulls to a feature (more than 5%)
        df.loc[0:6, "Partner"] = None

        validator = TelcoDataValidator()
        result = validator.validate(df, strict=False)

        assert len(result.warnings) > 0

    def test_empty_dataframe_detected(self):
        """Test detection of empty dataframe."""
        df = pd.DataFrame()

        validator = TelcoDataValidator()
        result = validator.validate(df, strict=False)

        assert not result.is_valid
        assert any("empty" in error.lower() for error in result.errors)


@pytest.mark.unit
def test_validate_dataframe_convenience_function(sample_dataframe):
    """Test the convenience validation function."""
    result = validate_dataframe(sample_dataframe, strict=False)

    assert result.is_valid
    assert result.stats["row_count"] == len(sample_dataframe)
