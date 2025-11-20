"""
Batch prediction module for Telco Churn.

Loads production model from MLflow Model Registry and generates
predictions on batch data with comprehensive logging and metadata.
"""

import pandas as pd
import numpy as np
import sys
import argparse
import joblib
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config, get_data_paths
from src.data.validators import validate_dataframe
from src.data.preprocessing import preprocess_data
from src.features.feature_engineering import engineer_features
from src.inference.model_registry import ModelRegistry, load_production_model
from src.logging_utils import setup_logger

logger = setup_logger(__name__)


def load_batch_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load batch data for prediction.

    Args:
        data_path: Path to data file (if None, loads from gold directory)

    Returns:
        DataFrame with batch data
    """
    logger.info("=" * 60)
    logger.info("STEP 1: LOAD BATCH DATA")
    logger.info("=" * 60)

    if data_path is None:
        # Load from gold directory
        data_paths = get_data_paths()
        gold_dir = data_paths["gold"]

        # Look for CSV files in gold directory
        csv_files = list(gold_dir.glob("*.csv"))

        if not csv_files:
            # Fallback to raw data if no gold data exists
            logger.warning(f"No files found in {gold_dir}, falling back to raw data")
            raw_dir = data_paths["raw"]
            config = get_config()
            data_path = raw_dir / config.get("data.kaggle_file")
        else:
            data_path = csv_files[0]

    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    return df


def prepare_features_for_inference(
    df: pd.DataFrame, pipeline_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.Series, bool]:
    """
    Prepare features for inference using saved preprocessing pipeline.

    Args:
        df: Raw dataframe
        pipeline_path: Path to saved pipeline (if None, will look in artifacts)

    Returns:
        Tuple of (processed features, target if present, has_target)
    """
    logger.info("=" * 60)
    logger.info("STEP 2: PREPARE FEATURES")
    logger.info("=" * 60)

    config = get_config()
    target_col = config.get("features.target")

    # Check if target column exists
    has_target = target_col in df.columns

    # Validate data (non-strict for inference)
    logger.info("Validating input data...")
    validate_dataframe(df, strict=False)

    # Load preprocessing pipeline
    if pipeline_path is None:
        # Try to load from artifacts directory
        artifacts_dir = Path(project_root) / "artifacts"
        pipeline_path = artifacts_dir / "preprocessing_pipeline.pkl"

        if not pipeline_path.exists():
            raise FileNotFoundError(
                f"Preprocessing pipeline not found at {pipeline_path}\n"
                f"Train a model first: python src/training/train_model.py"
            )

    logger.info(f"Loading preprocessing pipeline from: {pipeline_path}")
    pipeline = joblib.load(pipeline_path)

    # Preprocess data
    X_processed, y_processed, _ = preprocess_data(df, fitted_pipeline=pipeline)

    # Engineer features
    logger.info("Engineering features...")
    X_engineered = engineer_features(X_processed)

    logger.info(f"✓ Feature preparation complete: {X_engineered.shape}")

    return X_engineered, y_processed, has_target


def generate_predictions(
    X: pd.DataFrame,
    model_stage: str = "Production",
    model_version: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate predictions using model from registry.

    Args:
        X: Prepared features
        model_stage: Model stage to load from
        model_version: Specific model version (overrides stage)

    Returns:
        Tuple of (predictions, probabilities, model_metadata)
    """
    logger.info("=" * 60)
    logger.info("STEP 3: GENERATE PREDICTIONS")
    logger.info("=" * 60)

    # Load model from registry
    registry = ModelRegistry()

    if model_version:
        logger.info(f"Loading model version {model_version}")
        model = registry.load_model_by_version(model_version)
        metadata = registry.get_model_metadata(version=model_version)
    else:
        logger.info(f"Loading model from {model_stage} stage")
        try:
            model = registry.load_model_by_stage(model_stage)
            metadata = registry.get_model_metadata(stage=model_stage)
        except ValueError:
            logger.warning(f"No model in {model_stage} stage, loading latest")
            model = registry.load_latest_model()
            # Get metadata for latest
            versions = registry.list_model_versions()
            if versions:
                metadata = registry.get_model_metadata(
                    version=int(versions[0]["version"])
                )
            else:
                metadata = {}

    logger.info(
        f"Using model: v{metadata.get('version', 'unknown')} ({metadata.get('stage', 'unknown')})"
    )

    # Generate predictions
    logger.info(f"Generating predictions for {len(X):,} samples...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Get churn probabilities (probability of class 1)
    churn_probabilities = probabilities[:, 1]

    logger.info(f"✓ Predictions generated")
    logger.info(
        f"  Predicted churn rate: {(predictions == 1).sum() / len(predictions) * 100:.2f}%"
    )
    logger.info(f"  Avg churn probability: {churn_probabilities.mean():.4f}")

    return predictions, churn_probabilities, metadata


def save_predictions(
    df_original: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    metadata: dict,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save predictions to file with metadata.

    Args:
        df_original: Original dataframe (for customer IDs)
        predictions: Prediction array
        probabilities: Probability array
        metadata: Model metadata
        output_path: Output file path (if None, creates timestamped file)

    Returns:
        Path to saved file
    """
    logger.info("=" * 60)
    logger.info("STEP 4: SAVE PREDICTIONS")
    logger.info("=" * 60)

    # Create predictions dataframe
    predictions_df = pd.DataFrame(
        {
            "prediction_timestamp": datetime.now().isoformat(),
            "model_version": metadata.get("version", "unknown"),
            "model_stage": metadata.get("stage", "unknown"),
            "churn_prediction": predictions,
            "churn_probability": probabilities,
            "prediction_binary": (predictions == 1).astype(int),
        }
    )

    # Add customer ID if available
    if "customerID" in df_original.columns:
        predictions_df.insert(0, "customerID", df_original["customerID"].values)

    # Add risk tier based on probability
    predictions_df["risk_tier"] = pd.cut(
        predictions_df["churn_probability"],
        bins=[0, 0.3, 0.7, 1.0],
        labels=["low", "medium", "high"],
    )

    # Determine output path
    if output_path is None:
        data_paths = get_data_paths()
        predictions_dir = data_paths["predictions"]
        predictions_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = predictions_dir / f"churn_predictions_{timestamp}.csv"

    # Save to file
    logger.info(f"Saving predictions to: {output_path}")
    predictions_df.to_csv(output_path, index=False)

    logger.info(f"✓ Predictions saved")
    logger.info(f"  Total predictions: {len(predictions_df):,}")
    logger.info(
        f"  High risk customers: {(predictions_df['risk_tier'] == 'high').sum():,}"
    )
    logger.info(
        f"  Medium risk customers: {(predictions_df['risk_tier'] == 'medium').sum():,}"
    )
    logger.info(
        f"  Low risk customers: {(predictions_df['risk_tier'] == 'low').sum():,}"
    )

    return output_path


def batch_predict(
    data_path: Optional[Path] = None,
    model_stage: str = "Production",
    model_version: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Run complete batch prediction pipeline.

    Args:
        data_path: Path to input data file
        model_stage: Model stage to use
        model_version: Specific model version (overrides stage)
        output_path: Output file path

    Returns:
        Path to saved predictions file
    """
    try:
        logger.info("🚀 Starting Batch Prediction Pipeline")
        logger.info("")

        # Load data
        df = load_batch_data(data_path)

        # Prepare features
        X_prepared, y_actual, has_target = prepare_features_for_inference(df)

        # Generate predictions
        predictions, probabilities, metadata = generate_predictions(
            X_prepared, model_stage=model_stage, model_version=model_version
        )

        # If we have actual targets, evaluate performance
        if has_target and y_actual is not None:
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

            accuracy = accuracy_score(y_actual, predictions)
            f1 = f1_score(y_actual, predictions, zero_division=0)
            auc = roc_auc_score(y_actual, probabilities)

            logger.info("\nPrediction Performance on Batch Data:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  F1 Score: {f1:.4f}")
            logger.info(f"  AUC: {auc:.4f}")

        # Save predictions
        output_file = save_predictions(
            df, predictions, probabilities, metadata, output_path
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info("✓ BATCH PREDICTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {output_file}")

        return output_file

    except Exception as e:
        logger.error(f"✗ Batch prediction failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch predictions for Telco Churn"
    )
    parser.add_argument("--data-path", type=str, help="Path to input data file (CSV)")
    parser.add_argument(
        "--model-stage",
        type=str,
        default="Production",
        choices=["Staging", "Production", "Archived"],
        help="Model stage to load from registry",
    )
    parser.add_argument(
        "--model-version", type=int, help="Specific model version (overrides stage)"
    )
    parser.add_argument(
        "--output-path", type=str, help="Output file path for predictions"
    )

    args = parser.parse_args()

    batch_predict(
        data_path=Path(args.data_path) if args.data_path else None,
        model_stage=args.model_stage,
        model_version=args.model_version,
        output_path=Path(args.output_path) if args.output_path else None,
    )
