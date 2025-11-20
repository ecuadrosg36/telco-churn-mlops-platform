"""
Drift detection module using Evidently AI.

Monitors data drift, target drift, and data quality by comparing
reference dataset with current production data.
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config, get_data_paths
from src.logging_utils import setup_logger

logger = setup_logger(__name__)


class DriftDetector:
    """Drift detection using Evidently AI."""
    
    def __init__(self):
        self.config = get_config()
        self.target = self.config.get('features.target')
        
        # Drift thresholds from config
        self.drift_share_threshold = self.config.get(
            'monitoring.drift_thresholds.dataset_drift_share', 
            0.3
        )
        self.stattest_threshold = self.config.get(
            'monitoring.drift_thresholds.stattest_threshold',
            0.05
        )
        
        logger.info("DriftDetector initialized")
    
    def load_reference_data(self, reference_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load reference dataset for drift comparison.
        
        Args:
            reference_path: Path to reference data (if None, uses config)
            
        Returns:
            Reference dataframe
        """
        if reference_path is None:
            # Try to load from config
            ref_path_str = self.config.get('monitoring.reference_data_path')
            reference_path = Path(project_root) / ref_path_str
        
        if not reference_path.exists():
            # Create reference from raw data
            logger.warning(f"Reference data not found at {reference_path}")
            logger.info("Creating reference dataset from raw data...")
            
            data_paths = get_data_paths()
            raw_dir = data_paths['raw']
            data_file = raw_dir / self.config.get('data.kaggle_file')
            
            if not data_file.exists():
                raise FileNotFoundError(
                    f"Cannot create reference data: raw data not found at {data_file}"
                )
            
            # Load and sample data
            df = pd.read_csv(data_file)
            reference_df = df.sample(n=min(1000, len(df)), random_state=42)
            
            # Save reference data
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            reference_df.to_csv(reference_path, index=False)
            logger.info(f"✓ Reference dataset created: {len(reference_df)} samples")
            
            return reference_df
        
        logger.info(f"Loading reference data from: {reference_path}")
        df = pd.read_csv(reference_path)
        logger.info(f"  Loaded {len(df)} reference samples")
        
        return df
    
    def generate_data_drift_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> Tuple[Report, Path]:
        """
        Generate data drift report comparing reference and current data.
        
        Args:
            reference_data: Reference dataset
            current_data: Current production data
            output_path: Output HTML file path
            
        Returns:
            Tuple of (report object, output path)
        """
        logger.info("=" * 60)
        logger.info("GENERATING DATA DRIFT REPORT")
        logger.info("=" * 60)
        
        # Create report
        report = Report(metrics=[
            DataDriftPreset(
                drift_share=self.drift_share_threshold,
                stattest_threshold=self.stattest_threshold
            ),
        ])
        
        # Run report
        logger.info("Analyzing data drift...")
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=None
        )
        
        # Determine output path
        if output_path is None:
            reports_dir = Path(project_root) / self.config.get('monitoring.reports_dir')
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = reports_dir / f"data_drift_report_{timestamp}.html"
        
        # Save report
        logger.info(f"Saving report to: {output_path}")
        report.save_html(str(output_path))
        
        logger.info("✓ Data drift report generated")
        
        return report, output_path
    
    def generate_data_quality_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> Tuple[Report, Path]:
        """
        Generate data quality report.
        
        Args:
            reference_data: Reference dataset
            current_data: Current production data
            output_path: Output HTML file path
            
        Returns:
            Tuple of (report object, output path)
        """
        logger.info("=" * 60)
        logger.info("GENERATING DATA QUALITY REPORT")
        logger.info("=" * 60)
        
        # Create report
        report = Report(metrics=[
            DataQualityPreset(),
        ])
        
        # Run report
        logger.info("Analyzing data quality...")
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=None
        )
        
        # Determine output path
        if output_path is None:
            reports_dir = Path(project_root) / self.config.get('monitoring.reports_dir')
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = reports_dir / f"data_quality_report_{timestamp}.html"
        
        # Save report
        logger.info(f"Saving report to: {output_path}")
        report.save_html(str(output_path))
        
        logger.info("✓ Data quality report generated")
        
        return report, output_path
    
    def generate_target_drift_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> Tuple[Optional[Report], Optional[Path]]:
        """
        Generate target drift report.
        
        Args:
            reference_data: Reference dataset (must include target)
            current_data: Current production data (must include target)
            output_path: Output HTML file path
            
        Returns:
            Tuple of (report object, output path) or (None, None) if target missing
        """
        # Check if target column exists in both datasets
        if self.target not in reference_data.columns or self.target not in current_data.columns:
            logger.warning(f"Target column '{self.target}' not found in data, skipping target drift report")
            return None, None
        
        logger.info("=" * 60)
        logger.info("GENERATING TARGET DRIFT REPORT")
        logger.info("=" * 60)
        
        # Create report
        report = Report(metrics=[
            TargetDriftPreset(),
        ])
        
        # Run report
        logger.info("Analyzing target drift...")
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=None
        )
        
        # Determine output path
        if output_path is None:
            reports_dir = Path(project_root) / self.config.get('monitoring.reports_dir')
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = reports_dir / f"target_drift_report_{timestamp}.html"
        
        # Save report
        logger.info(f"Saving report to: {output_path}")
        report.save_html(str(output_path))
        
        logger.info("✓ Target drift report generated")
        
        return report, output_path
    
    def generate_all_reports(
        self,
        current_data_path: Path,
        reference_data_path: Optional[Path] = None
    ) -> dict:
        """
        Generate all drift monitoring reports.
        
        Args:
            current_data_path: Path to current production data
            reference_data_path: Path to reference data (optional)
            
        Returns:
            Dictionary with report paths
        """
        logger.info("🔍 Starting Drift Monitoring")
        logger.info("")
        
        # Load data
        reference_data = self.load_reference_data(reference_data_path)
        
        logger.info(f"Loading current data from: {current_data_path}")
        current_data = pd.read_csv(current_data_path)
        logger.info(f"  Loaded {len(current_data)} current samples")
        logger.info("")
        
        results = {}
        
        # Generate data drift report
        _, drift_path = self.generate_data_drift_report(reference_data, current_data)
        results['data_drift'] = str(drift_path)
        logger.info("")
        
        # Generate data quality report
        _, quality_path = self.generate_data_quality_report(reference_data, current_data)
        results['data_quality'] = str(quality_path)
        logger.info("")
        
        # Generate target drift report (if target available)
        target_report, target_path = self.generate_target_drift_report(reference_data, current_data)
        if target_path:
            results['target_drift'] = str(target_path)
            logger.info("")
        
        logger.info("=" * 60)
        logger.info("✓ DRIFT MONITORING COMPLETE")
        logger.info("=" * 60)
        logger.info("Generated reports:")
        for report_type, path in results.items():
            logger.info(f"  {report_type}: {path}")
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate drift monitoring reports')
    parser.add_argument(
        '--current-data',
        type=str,
        required=True,
        help='Path to current production data CSV'
    )
    parser.add_argument(
        '--reference-data',
        type=str,
        help='Path to reference data CSV (optional)'
    )
    
    args = parser.parse_args()
    
    detector = DriftDetector()
    
    results = detector.generate_all_reports(
        current_data_path=Path(args.current_data),
        reference_data_path=Path(args.reference_data) if args.reference_data else None
    )
    
    print("\n📊 Reports generated successfully!")
    print(f"Open reports in your browser to view drift analysis.")
