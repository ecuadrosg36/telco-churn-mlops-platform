"""
MLflow Model Registry management module.

Handles loading models by stage, promoting models between stages,
and managing model versions in the registry.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import mlflow
from mlflow.tracking import MlflowClient

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config, get_mlflow_config
from src.logging_utils import setup_logger

logger = setup_logger(__name__)


class ModelRegistry:
    """Manager for MLflow Model Registry operations."""
    
    def __init__(self):
        self.mlflow_config = get_mlflow_config()
        self.model_name = self.mlflow_config['model_registry_name']
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])
        
        # Create MLflow client
        self.client = MlflowClient()
        
        logger.info(f"ModelRegistry initialized for '{self.model_name}'")
    
    def load_model_by_stage(self, stage: str = "Production") -> Any:
        """
        Load model from registry by stage.
        
        Args:
            stage: Model stage ('Staging', 'Production', 'Archived', or 'None')
            
        Returns:
            Loaded model object
            
        Raises:
            ValueError: If no model found in specified stage
        """
        model_uri = f"models:/{self.model_name}/{stage}"
        
        try:
            logger.info(f"Loading model from stage: {stage}")
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"✓ Model loaded successfully from {stage} stage")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {stage} stage: {str(e)}")
            raise ValueError(
                f"No model found in '{stage}' stage for '{self.model_name}'. "
                f"Error: {str(e)}"
            )
    
    def load_model_by_version(self, version: int) -> Any:
        """
        Load specific model version.
        
        Args:
            version: Model version number
            
        Returns:
            Loaded model object
        """
        model_uri = f"models:/{self.model_name}/{version}"
        
        try:
            logger.info(f"Loading model version: {version}")
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"✓ Model version {version} loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model version {version}: {str(e)}")
            raise ValueError(f"Model version {version} not found: {str(e)}")
    
    def load_latest_model(self) -> Any:
        """
        Load the most recent model version regardless of stage.
        
        Returns:
            Loaded model object
        """
        try:
            # Get latest versions
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            
            if not versions:
                raise ValueError(f"No versions found for model '{self.model_name}'")
            
            # Sort by version number and get latest
            latest_version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
            
            logger.info(f"Loading latest model version: {latest_version.version}")
            return self.load_model_by_version(int(latest_version.version))
            
        except Exception as e:
            logger.error(f"Failed to load latest model: {str(e)}")
            raise
    
    def get_model_metadata(self, stage: Optional[str] = None, version: Optional[int] = None) -> Dict[str, Any]:
        """
        Get metadata for a model.
        
        Args:
            stage: Model stage (if specified)
            version: Model version (if specified)
            
        Returns:
            Dictionary with model metadata
        """
        try:
            if version:
                model_version = self.client.get_model_version(self.model_name, version)
            elif stage:
                versions = self.client.get_latest_versions(self.model_name, stages=[stage])
                if not versions:
                    raise ValueError(f"No model in {stage} stage")
                model_version = versions[0]
            else:
                raise ValueError("Must specify either stage or version")
            
            # Get run info for metrics
            run = self.client.get_run(model_version.run_id)
            
            metadata = {
                'name': model_version.name,
                'version': model_version.version,
                'stage': model_version.current_stage,
                'run_id': model_version.run_id,
                'created_at': model_version.creation_timestamp,
                'last_updated': model_version.last_updated_timestamp,
                'description': model_version.description,
                'metrics': run.data.metrics,
                'params': run.data.params,
                'tags': model_version.tags,
            }
            
            logger.info(f"Retrieved metadata for {self.model_name} v{metadata['version']} ({metadata['stage']})")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get model metadata: {str(e)}")
            raise
    
    def promote_model(self, version: int, to_stage: str, archive_existing: bool = True) -> None:
        """
        Promote a model version to a new stage.
        
        Args:
            version: Model version to promote
            to_stage: Target stage ('Staging' or 'Production')
            archive_existing: Whether to archive existing models in target stage
        """
        logger.info(f"Promoting model version {version} to {to_stage}")
        
        try:
            # Archive existing models in target stage if requested
            if archive_existing and to_stage in ['Staging', 'Production']:
                existing_versions = self.client.get_latest_versions(
                    self.model_name, 
                    stages=[to_stage]
                )
                
                for existing in existing_versions:
                    logger.info(f"Archiving existing {to_stage} model v{existing.version}")
                    self.client.transition_model_version_stage(
                        name=self.model_name,
                        version=existing.version,
                        stage="Archived"
                    )
            
            # Promote new version
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage=to_stage
            )
            
            logger.info(f"✓ Model v{version} promoted to {to_stage}")
            
        except Exception as e:
            logger.error(f"Failed to promote model: {str(e)}")
            raise
    
    def list_model_versions(self, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all model versions, optionally filtered by stage.
        
        Args:
            stage: Optional stage filter
            
        Returns:
            List of model version metadata
        """
        try:
            if stage:
                versions = self.client.get_latest_versions(self.model_name, stages=[stage])
            else:
                versions = self.client.search_model_versions(f"name='{self.model_name}'")
            
            version_list = []
            for v in versions:
                version_list.append({
                    'version': v.version,
                    'stage': v.current_stage,
                    'run_id': v.run_id,
                    'created_at': v.creation_timestamp,
                })
            
            # Sort by version number
            version_list.sort(key=lambda x: int(x['version']), reverse=True)
            
            logger.info(f"Found {len(version_list)} model versions")
            return version_list
            
        except Exception as e:
            logger.error(f"Failed to list model versions: {str(e)}")
            raise
    
    def delete_model_version(self, version: int) -> None:
        """
        Delete a specific model version.
        
        Args:
            version: Model version to delete
        """
        logger.warning(f"Deleting model version {version}")
        
        try:
            self.client.delete_model_version(
                name=self.model_name,
                version=version
            )
            logger.info(f"✓ Model version {version} deleted")
            
        except Exception as e:
            logger.error(f"Failed to delete model version: {str(e)}")
            raise


def load_production_model() -> Any:
    """
    Convenience function to load the production model.
    
    Returns:
        Production model object
    """
    registry = ModelRegistry()
    
    try:
        # Try to load from Production stage
        return registry.load_model_by_stage("Production")
    except ValueError:
        # Fallback to latest model if no Production model exists
        logger.warning("No Production model found, falling back to latest version")
        return registry.load_latest_model()


if __name__ == "__main__":
    # Test model registry operations
    registry = ModelRegistry()
    
    # List all versions
    print("\nAll Model Versions:")
    versions = registry.list_model_versions()
    for v in versions:
        print(f"  v{v['version']} - {v['stage']}")
    
    # Try to load production model
    try:
        model = load_production_model()
        print(f"\n✓ Loaded production model successfully")
        print(f"  Model type: {type(model).__name__}")
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
