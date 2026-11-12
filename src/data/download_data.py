"""
Data download module for Telco Churn dataset from Kaggle.

Downloads the IBM Telco Customer Churn dataset and saves it to the raw data directory.
"""

import os
import sys
import urllib.request
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config, get_data_paths
from src.logging_utils import setup_logger

# Load environment variables
load_dotenv()

# Get logger
logger = setup_logger(__name__)


DOWNLOAD_TIMEOUT_SECONDS = 60


def _fetch_bytes(url: str) -> bytes:
    """Fetch a URL over HTTPS and return the body."""
    if not url.startswith("https://"):
        raise ValueError(f"Refusing non-HTTPS download URL: {url}")
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        return response.read()


def download_public_copy(
    url: str, target: Path, fetch: Callable[[str], bytes] = _fetch_bytes
) -> Path:
    """
    Download the public copy of the dataset (no credentials needed).

    Returns:
        Path to the written CSV file

    Raises:
        RuntimeError: If the download fails or is empty
    """
    logger.info(f"Kaggle credentials not set; downloading public copy: {url}")
    try:
        content = fetch(url)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"Public dataset download failed: {e}") from e
    if not content:
        raise RuntimeError(f"Public dataset download was empty: {url}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    logger.info(f"✓ Dataset saved: {target} ({len(content) / 1024:.0f} KB)")
    return target


def download_telco_data() -> Path:
    """
    Download Telco Customer Churn dataset.

    Uses the Kaggle API when KAGGLE_USERNAME and KAGGLE_KEY are set,
    otherwise the public IBM copy of the same dataset.

    Returns:
        Path to downloaded CSV file

    Raises:
        RuntimeError: If download fails
    """
    config = get_config()
    data_paths = get_data_paths()

    # Get dataset info from config
    kaggle_dataset = config.get("data.kaggle_dataset")
    kaggle_file = config.get("data.kaggle_file")
    raw_dir = data_paths["raw"]

    if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
        return download_public_copy(
            config.get("data.public_url"), raw_dir / kaggle_file
        )

    logger.info(f"Downloading dataset: {kaggle_dataset}")
    logger.info(f"Target directory: {raw_dir}")

    # Create raw directory if it doesn't exist
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset using Kaggle API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        # Authenticate
        api = KaggleApi()
        api.authenticate()

        logger.info("Kaggle API authenticated successfully")

        # Download dataset
        api.dataset_download_files(
            kaggle_dataset, path=str(raw_dir), unzip=True, quiet=False
        )

        # Check if file exists
        downloaded_file = raw_dir / kaggle_file
        if not downloaded_file.exists():
            raise RuntimeError(
                f"Downloaded file not found: {downloaded_file}\n"
                f"Files in directory: {list(raw_dir.glob('*'))}"
            )

        file_size_mb = downloaded_file.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Dataset downloaded successfully: {downloaded_file.name}")
        logger.info(f"  File size: {file_size_mb:.2f} MB")
        logger.info(f"  Location: {downloaded_file}")

        return downloaded_file

    except ImportError:
        raise RuntimeError("Kaggle package not installed. Run: pip install kaggle")
    except Exception as e:
        logger.error(f"Failed to download dataset: {str(e)}")
        raise RuntimeError(f"Dataset download failed: {str(e)}")


def verify_data(data_path: Path) -> bool:
    """
    Verify downloaded data is valid.

    Args:
        data_path: Path to CSV file

    Returns:
        True if valid

    Raises:
        ValueError: If data validation fails
    """
    import pandas as pd

    logger.info(f"Verifying data: {data_path}")

    try:
        df = pd.read_csv(data_path)

        # Basic checks
        if df.empty:
            raise ValueError("Dataset is empty")

        logger.info(f"  Rows: {len(df):,}")
        logger.info(f"  Columns: {len(df.columns)}")
        logger.info(
            f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )

        # Check for target column
        config = get_config()
        target = config.get("features.target")

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset")

        logger.info(f"  Target column '{target}' found")
        logger.info(f"  Target distribution:\n{df[target].value_counts()}")

        logger.info("✓ Data verification passed")
        return True

    except Exception as e:
        logger.error(f"Data verification failed: {str(e)}")
        raise


def main():
    """Main function to download and verify data."""
    logger.info("=" * 60)
    logger.info("TELCO CHURN DATA DOWNLOAD")
    logger.info("=" * 60)

    try:
        # Download data
        data_path = download_telco_data()

        # Verify data
        verify_data(data_path)

        logger.info("=" * 60)
        logger.info("✓ Data download and verification complete")
        logger.info("=" * 60)

        return data_path

    except Exception as e:
        logger.error(f"✗ Data download failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
