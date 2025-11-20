"""
Configuration loader for Telco Churn MLOps Platform.

Loads base configuration and merges environment-specific overrides
based on ENVIRONMENT variable (DEV/PROD).
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
from functools import lru_cache


class ConfigManager:
    """Singleton configuration manager for the application."""

    _instance = None
    _config: Dict[str, Any] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML files."""
        # Get project root (three levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "configs"

        # Load base configuration
        base_config_path = config_dir / "config.yaml"
        with open(base_config_path, "r") as f:
            self._config = yaml.safe_load(f)

        # Determine environment
        env = os.getenv("ENVIRONMENT", "dev").lower()

        # Load environment-specific overrides
        env_config_path = config_dir / f"config_{env}.yaml"
        if env_config_path.exists():
            with open(env_config_path, "r") as f:
                env_config = yaml.safe_load(f)
                self._merge_configs(self._config, env_config)

        # Add project root to config for path resolution
        self._config["project_root"] = str(project_root)

        print(f"✓ Configuration loaded for environment: {env.upper()}")

    def _merge_configs(self, base: Dict, override: Dict) -> None:
        """Recursively merge override config into base config."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., 'data.raw_dir')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config = ConfigManager()
            >>> config.get('model.baseline.n_estimators')
            100
            >>> config.get('mlflow.experiment_name')
            'telco-churn-prediction'
        """
        keys = key_path.split(".")
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary."""
        return self._config.copy()

    def get_absolute_path(self, relative_path: str) -> Path:
        """
        Convert relative path to absolute path based on project root.

        Args:
            relative_path: Relative path from project root

        Returns:
            Absolute Path object
        """
        project_root = Path(self._config["project_root"])
        return project_root / relative_path

    def validate_config(self) -> bool:
        """
        Validate that required configuration keys exist.

        Returns:
            True if valid, raises ValueError otherwise
        """
        required_keys = [
            "data.raw_dir",
            "data.processed_dir",
            "features.target",
            "model.baseline.type",
            "mlflow.experiment_name",
        ]

        for key_path in required_keys:
            value = self.get(key_path)
            if value is None:
                raise ValueError(f"Missing required configuration: {key_path}")

        # Validate split ratios sum to 1.0
        train = self.get("data.train_ratio", 0)
        val = self.get("data.val_ratio", 0)
        test = self.get("data.test_ratio", 0)

        if abs(train + val + test - 1.0) > 0.01:
            raise ValueError(
                f"Train/val/test ratios must sum to 1.0, got {train + val + test}"
            )

        print("✓ Configuration validation passed")
        return True


@lru_cache()
def get_config() -> ConfigManager:
    """
    Get singleton ConfigManager instance.

    This function is cached, so it will always return the same instance.

    Returns:
        ConfigManager instance
    """
    return ConfigManager()


# Convenience functions for common operations
def get_data_paths() -> Dict[str, Path]:
    """Get all data directory paths as Path objects."""
    config = get_config()
    return {
        "raw": config.get_absolute_path(config.get("data.raw_dir")),
        "processed": config.get_absolute_path(config.get("data.processed_dir")),
        "gold": config.get_absolute_path(config.get("data.gold_dir")),
        "predictions": config.get_absolute_path(config.get("data.predictions_dir")),
    }


def get_model_params(model_type: str = "baseline") -> Dict[str, Any]:
    """
    Get model hyperparameters.

    Args:
        model_type: 'baseline' or 'advanced'

    Returns:
        Dictionary of model parameters
    """
    config = get_config()
    return config.get(f"model.{model_type}", {})


def get_mlflow_config() -> Dict[str, str]:
    """Get MLflow configuration."""
    config = get_config()
    return {
        "tracking_uri": config.get("mlflow.tracking_uri"),
        "experiment_name": config.get("mlflow.experiment_name"),
        "model_registry_name": config.get("mlflow.model_registry_name"),
    }


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    config.validate_config()

    print("\nSample Configuration Values:")
    print(f"  Experiment Name: {config.get('mlflow.experiment_name')}")
    print(f"  Target Variable: {config.get('features.target')}")
    print(f"  Baseline Model: {config.get('model.baseline.type')}")
    print(f"  Train Ratio: {config.get('data.train_ratio')}")

    print("\nData Paths:")
    for name, path in get_data_paths().items():
        print(f"  {name}: {path}")
