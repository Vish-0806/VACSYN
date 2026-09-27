"""
Minimal S3/path utilities for SageMaker execution layer.
"""

import os
from pathlib import Path
from typing import Optional
import pandas as pd


def resolve_s3_or_local_path(path: str) -> str:
    """
    Normalize a path that may be an S3 URI or local path.
    
    Args:
        path: S3 URI (s3://...) or local file path
        
    Returns:
        Normalized path string usable by pandas/s3fs
    """
    if path.startswith("s3://"):
        return path
    return str(Path(path).resolve())


def read_tsv(path: str, **kwargs) -> pd.DataFrame:
    """
    Read a TSV file from S3 or local filesystem.
    
    Args:
        path: S3 URI or local path to TSV file
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        DataFrame with TSV contents
    """
    defaults = {
        "sep": "\t",
        "dtype": str,
        "encoding": "utf-8",
        "na_filter": False,
    }
    defaults.update(kwargs)
    return pd.read_csv(resolve_s3_or_local_path(path), **defaults)


def get_sagemaker_input_dir() -> Path:
    """
    Get the SageMaker training input directory from environment.
    
    Returns:
        Path to SM_CHANNEL_TRAIN (default: /opt/ml/input/data/train)
    """
    return Path(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))


def get_sagemaker_model_dir() -> Path:
    """
    Get the SageMaker model output directory from environment.
    
    Returns:
        Path to SM_MODEL_DIR (default: /opt/ml/model)
    """
    return Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))


def get_sagemaker_output_dir() -> Path:
    """
    Get the SageMaker output directory from environment.
    
    Returns:
        Path to SM_OUTPUT_DATA_DIR (default: /opt/ml/output/data)
    """
    return Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))


def find_required_files(input_dir: Path) -> dict:
    """
    Locate the four required training files in the input directory.
    
    Args:
        input_dir: Directory containing training files
        
    Returns:
        Dict with keys: s1, s2, s3, gt mapping to file paths
        
    Raises:
        FileNotFoundError: If any required file is missing
    """
    required = {
        "s1": "train_source1.tsv",
        "s2": "train_source2.tsv",
        "s3": "train_source3.tsv",
        "gt": "train_ground_truth.tsv",
    }
    
    found = {}
    missing = []
    
    for key, filename in required.items():
        filepath = input_dir / filename
        if filepath.exists():
            found[key] = str(filepath)
        else:
            missing.append(filename)
    
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {input_dir}: {', '.join(missing)}"
        )
    
    return found


def get_memory_usage_gb() -> float:
    """
    Get current process memory usage in GB.
    
    Returns:
        Memory usage in GB
    """
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)
    except ImportError:
        return 0.0