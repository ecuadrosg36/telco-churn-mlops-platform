"""
Training pipeline for Telco Churn prediction model.

Implements end-to-end training with data validation, preprocessing,
feature engineering, model training, and MLflow tracking.
"""

import pandas as pd
import numpy as np
import sys
import argparse
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config, get_data_paths, get_model_params, get_mlflow_config
from src.data.validators import validate_dataframe
from src.data.preprocessing import preprocess_data
from src.features.feature_engineering import engineer_features
from src.logging_utils import setup_logger

logger = setup_logger(__name__)


def load_and_validate_data() -> pd.DataFrame:
    """
    Load raw data and validate it.
    
    Returns:
        Validated DataFrame
    """
    logger.info("=" * 60)
    logger.info("STEP 1: LOAD AND VALIDATE DATA")
    logger.info("=" * 60)
    
    data_paths = get_data_paths()
    config = get_config()
    
    # Get data file path
    raw_dir = data_paths['raw']
    data_file = raw_dir / config.get('data.kaggle_file')
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_file}\n"
            f"Run 'python src/data/download_data.py' first to download the dataset."
        )
    
    logger.info(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Validate data
    logger.info("Validating data quality...")
    validation_result = validate_dataframe(df, strict=True)
    
    if validation_result.warnings:
        logger.warning(f"Data validation completed with {len(validation_result.warnings)} warnings")
    else:
        logger.info("✓ Data validation passed with no warnings")
    
    return df


def split_data(df: pd.DataFrame) -> tuple:
    """
    Split data into train/val/test sets.
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("=" * 60)
    logger.info("STEP 2: SPLIT DATA")
    logger.info("=" * 60)
    
    config = get_config()
    target = config.get('features.target')
    train_ratio = config.get('data.train_ratio')
    val_ratio = config.get('data.val_ratio')
    test_ratio = config.get('data.test_ratio')
    random_seed = config.get('data.random_seed')
    
    logger.info(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    # Separate features and target
    X = df.drop(target, axis=1)
    y = df[target]
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_seed,
        stratify=y_temp
    )
    
    logger.info(f"  Train set: {len(X_train):,} samples")
    logger.info(f"  Val set:   {len(X_val):,} samples")
    logger.info(f"  Test set:  {len(X_test):,} samples")
    
    # Check target distribution
    logger.info("Target distribution:")
    for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        dist = y_split.value_counts(normalize=True)
        logger.info(f"  {split_name}: {dict(dist)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_features(X_train, X_val, X_test, y_train, y_val, y_test) -> tuple:
    """
    Preprocess and engineer features.
    
    Returns:
        Tuple of (X_train_prep, X_val_prep, X_test_prep, y_train_prep, y_val_prep, y_test_prep, pipeline)
    """
    logger.info("=" * 60)
    logger.info("STEP 3: PREPROCESS AND ENGINEER FEATURES")
    logger.info("=" * 60)
    
    # Combine X and y for preprocessing
    train_df = X_train.copy()
    train_df[get_config().get('features.target')] = y_train.values
    
    # Fit preprocessing pipeline on training data
    logger.info("Fitting preprocessing pipeline on training data...")
    X_train_prep, y_train_prep, pipeline = preprocess_data(train_df)
    
    # Transform validation data
    logger.info("Transforming validation data...")
    val_df = X_val.copy()
    val_df[get_config().get('features.target')] = y_val.values
    X_val_prep, y_val_prep, _ = preprocess_data(val_df, fitted_pipeline=pipeline)
    
    # Transform test data
    logger.info("Transforming test data...")
    test_df = X_test.copy()
    test_df[get_config().get('features.target')] = y_test.values
    X_test_prep, y_test_prep, _ = preprocess_data(test_df, fitted_pipeline=pipeline)
    
    # Feature engineering (after preprocessing)
    logger.info("Engineering features...")
    X_train_eng = engineer_features(X_train_prep)
    X_val_eng = engineer_features(X_val_prep)
    X_test_eng = engineer_features(X_test_prep)
    
    logger.info(f"✓ Feature preparation complete")
    logger.info(f"  Final feature count: {X_train_eng.shape[1]}")
    
    return X_train_eng, X_val_eng, X_test_eng, y_train_prep, y_val_prep, y_test_prep, pipeline


def train_baseline_model(X_train, y_train, X_val, y_val):
    """
    Train RandomForest baseline model.
    
    Returns:
        Trained model and metrics dict
    """
    logger.info("=" * 60)
    logger.info("STEP 4A: TRAIN BASELINE MODEL (RandomForest)")
    logger.info("=" * 60)
    
    params = get_model_params('baseline')
    logger.info(f"Hyperparameters: {params}")
    
    # Create and train model
    model = RandomForestClassifier(**params)
    
    logger.info("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_train, y_train, X_val, y_val, "Baseline")
    
    return model, metrics


def train_advanced_model(X_train, y_train, X_val, y_val):
    """
    Train XGBoost advanced model.
    
    Returns:
        Trained model and metrics dict
    """
    logger.info("=" * 60)
    logger.info("STEP 4B: TRAIN ADVANCED MODEL (XGBoost)")
    logger.info("=" * 60)
    
    params = get_model_params('advanced')
    logger.info(f"Hyperparameters: {params}")
    
    # Create and train model
    model = XGBClassifier(**params)
    
    logger.info("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_train, y_train, X_val, y_val, "Advanced")
    
    return model, metrics


def evaluate_model(model, X_train, y_train, X_val, y_val, model_name: str) -> dict:
    """
    Evaluate model and return metrics.
    
    Returns:
        Dictionary of metrics
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Probabilities for AUC
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
        'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
        'train_f1': f1_score(y_train, y_train_pred, zero_division=0),
        'train_auc': roc_auc_score(y_train, y_train_proba),
        'val_accuracy': accuracy_score(y_val, y_val_pred),
        'val_precision': precision_score(y_val, y_val_pred, zero_division=0),
        'val_recall': recall_score(y_val, y_val_pred, zero_division=0),
        'val_f1': f1_score(y_val, y_val_pred, zero_division=0),
        'val_auc': roc_auc_score(y_val, y_val_proba),
    }
    
    logger.info(f"\n{model_name} Model Performance:")
    logger.info(f"  Training   - Accuracy: {metrics['train_accuracy']:.4f}, F1: {metrics['train_f1']:.4f}, AUC: {metrics['train_auc']:.4f}")
    logger.info(f"  Validation - Accuracy: {metrics['val_accuracy']:.4f}, F1: {metrics['val_f1']:.4f}, AUC: {metrics['val_auc']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    return metrics


def log_to_mlflow(model, pipeline, metrics, params, model_name: str):
    """
    Log model, metrics, and artifacts to MLflow.
    
    Returns:
        MLflow run ID
    """
    logger.info("=" * 60)
    logger.info("STEP 5: LOG TO MLFLOW")
    logger.info("=" * 60)
    
    mlflow_config = get_mlflow_config()
    mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
    mlflow.set_experiment(mlflow_config['experiment_name'])
    
    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=mlflow_config['model_registry_name']
        )
        
        # Log preprocessing pipeline
        pipeline_path = Path("artifacts/preprocessing_pipeline.pkl")
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, pipeline_path)
        mlflow.log_artifact(str(pipeline_path), artifact_path="pipeline")
        
        logger.info(f"✓ Logged to MLflow")
        logger.info(f"  Run ID: {run.info.run_id}")
        logger.info(f"  Experiment: {mlflow_config['experiment_name']}")
        
        return run.info.run_id


def main(model_type: str = 'both'):
    """
    Main training pipeline.
    
    Args:
        model_type: 'baseline', 'advanced', or 'both'
    """
    try:
        logger.info("🚀 Starting Telco Churn Training Pipeline")
        logger.info(f"   Model type: {model_type}")
        logger.info("")
        
        # Load and validate
        df = load_and_validate_data()
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
        
        # Prepare features
        X_train_prep, X_val_prep, X_test_prep, y_train_prep, y_val_prep, y_test_prep, pipeline = prepare_features(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Train models
        if model_type in ['baseline', 'both']:
            baseline_model, baseline_metrics = train_baseline_model(
                X_train_prep, y_train_prep, X_val_prep, y_val_prep
            )
            baseline_params = get_model_params('baseline')
            log_to_mlflow(baseline_model, pipeline, baseline_metrics, baseline_params, "RandomForest")
        
        if model_type in ['advanced', 'both']:
            advanced_model, advanced_metrics = train_advanced_model(
                X_train_prep, y_train_prep, X_val_prep, y_val_prep
            )
            advanced_params = get_model_params('advanced')
            run_id = log_to_mlflow(advanced_model, pipeline, advanced_metrics, advanced_params, "XGBoost")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("✓ TRAINING PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"View results: mlflow ui --port 5000")
        logger.info(f"Then navigate to: http://localhost:5000")
        
    except Exception as e:
        logger.error(f"✗ Training pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Telco Churn prediction model')
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['baseline', 'advanced', 'both'],
        default='both',
        help='Which model(s) to train'
    )
    
    args = parser.parse_args()
    main(model_type=args.model_type)
