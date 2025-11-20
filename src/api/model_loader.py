"""
Model loader singleton for FastAPI application.

Manages loading and caching of production models with automatic reloading.
"""

import sys
import joblib
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
        self, 
        stage: str = "Production",
        force_reload: bool = False
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
            self.model_version = self.model_metadata.get('version')
            
        except ValueError:
            # Fallback to latest if no production model
            logger.warning(f"No model in {stage} stage, loading latest")
            self.model = registry.load_latest_model()
            
            # Get metadata for latest version
            versions = registry.list_model_versions()
            if versions:
                self.model_metadata = registry.get_model_metadata(
                    version=int(versions[0]['version'])
                )
                self.model_version = versions[0]['version']
                self.model_stage = versions[0]['stage']
        
        # Load preprocessing pipeline
        self._load_preprocessing_pipeline()
        
        logger.info(
            f"✓ Model loaded: v{self.model_version} ({self.model_stage})"
        )
        
        return self.model, self.preprocessing_pipeline, self.model_metadata
    
    def _load_preprocessing_pipeline(self) -> None:
        """Load preprocessing pipeline from artifacts."""
        artifacts_dir = project_root / "artifacts"
        pipeline_path = artifacts_dir / "preprocessing_pipeline.pkl"
        
        if not pipeline_path.exists():
            logger.warning(
                f"Preprocessing pipeline not found at {pipeline_path}. "
                f"Some features may not work correctly."
            )
            self.preprocessing_pipeline = None
            return
        
        logger.info(f"Loading preprocessing pipeline...")
        self.preprocessing_pipeline = joblib.load(pipeline_path)
        logger.info("✓ Preprocessing pipeline loaded")
    
    def get_model_info(self) -> dict:
        """
        Get information about loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {
                'loaded': False,
                'message': 'No model loaded'
            }
        
        return {
            'loaded': True,
            'version': str(self.model_version),
            'stage': self.model_stage,
            'metadata': self.model_metadata
        }
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


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
