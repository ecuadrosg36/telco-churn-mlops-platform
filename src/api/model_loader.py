"""
Model loader singleton for FastAPI application.

Manages loading and caching of production models with automatic reloading.
"""

import sys
import joblib
import mlflow
from pathlib import Path
from typing import Optional, Tuple, Any
from threading import Lock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.model_registry import ModelRegistry
from src.logging_utils import setup_logger

logger = setup_logger(__name__)


class ModelLoader:
    """
    Singleton model loader for API.

    Loads and caches model and preprocessing pipeline for fast predictions.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model = None
        self.preprocessing_pipeline = None
        self.model_metadata = None
        self.model_stage = None
        self.model_version = None
        self._initialized = True

        logger.info("ModelLoader singleton created")

    def load_model(
        self, stage: str = "Production", force_reload: bool = False
    ) -> Tuple[Any, Any, dict]:
        """
        Load model and preprocessing pipeline.

        Args:
            stage: Model stage to load from registry
            force_reload: Force reload even if model is cached

        Returns:
            Tuple of (model, pipeline, metadata)
        """
        # Return cached model if available and not forcing reload
        if not force_reload and self.model is not None:
            logger.debug("Using cached model")
            return self.model, self.preprocessing_pipeline, self.model_metadata

        logger.info(f"Loading model from {stage} stage...")

        # Load model from registry
        registry = ModelRegistry()

        try:
            self.model = registry.load_model_by_stage(stage)
            self.model_metadata = registry.get_model_metadata(stage=stage)
            self.model_stage = stage
            self.model_version = self.model_metadata.get("version")

        except ValueError:
            # Fallback to latest if no production model
            logger.warning(f"No model in {stage} stage, loading latest")
            self.model = registry.load_latest_model()

            # Get metadata for latest version
            versions = registry.list_model_versions()
            if versions:
                self.model_metadata = registry.get_model_metadata(
                    version=int(versions[0]["version"])
                )
                self.model_version = versions[0]["version"]
                self.model_stage = versions[0]["stage"]

        # Load preprocessing pipeline
        self._load_preprocessing_pipeline((self.model_metadata or {}).get("run_id"))

        logger.info(f"✓ Model loaded: v{self.model_version} ({self.model_stage})")

        return self.model, self.preprocessing_pipeline, self.model_metadata

    def _load_preprocessing_pipeline(self, run_id: Optional[str] = None) -> None:
        """Load the preprocessing pipeline fitted with this model.

        First choice is the copy logged in the model's own MLflow run, so model and pipeline always match.
        The local artifacts/ file is a fallback that may come from a different training run.
        """
        if run_id:
            try:
                path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path="pipeline/preprocessing_pipeline.pkl"
                )
                self.preprocessing_pipeline = joblib.load(path)
                logger.info(f"✓ Preprocessing pipeline loaded from MLflow run {run_id}")
                return
            except Exception as e:
                logger.warning(f"Could not load the pipeline from run {run_id}: {e}")

        pipeline_path = project_root / "artifacts" / "preprocessing_pipeline.pkl"
        if not pipeline_path.exists():
            # Without the fitted pipeline the API would fit a new one on each request; is_loaded() stays False.
            logger.error(
                f"No preprocessing pipeline in MLflow or at {pipeline_path}. Predictions are disabled."
            )
            self.preprocessing_pipeline = None
            return

        logger.warning(
            f"Using local pipeline {pipeline_path}; it may not match model v{self.model_version}."
        )
        self.preprocessing_pipeline = joblib.load(pipeline_path)

    def get_model_info(self) -> dict:
        """
        Get information about loaded model.

        Returns:
            Dictionary with model information
        """
        if not self.is_loaded():
            return {
                "loaded": False,
                "message": "Model or preprocessing pipeline not loaded",
            }

        return {
            "loaded": True,
            "version": str(self.model_version),
            "stage": self.model_stage,
            "metadata": self.model_metadata,
        }

    def is_loaded(self) -> bool:
        """True only when both the model and its fitted preprocessing pipeline are loaded."""
        return self.model is not None and self.preprocessing_pipeline is not None


# Singleton instance getter
def get_model_loader() -> ModelLoader:
    """Get ModelLoader singleton instance."""
    return ModelLoader()


if __name__ == "__main__":
    # Test model loader
    loader = get_model_loader()

    try:
        model, pipeline, metadata = loader.load_model()
        print(f"\n✓ Model loaded successfully")
        print(f"  Version: {metadata.get('version')}")
        print(f"  Stage: {metadata.get('stage')}")
        print(f"  Type: {type(model).__name__}")
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
