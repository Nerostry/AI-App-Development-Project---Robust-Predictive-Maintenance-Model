"""
pipelines
=========

Package containing the predictive maintenance data pipeline stages:

- data_ingest      : Raw dataset discovery, multi-format loading, ZIP extraction, and merging
- data_clean        : KNN-based imputation and target engineering (comp_failure -> failed)
- feature_engineer   : One-hot encoding and feature preparation for model training
- model_train        : Multimodal transformer model definition and training loop
- model_eval         : Threshold tuning, PR-AUC evaluation, and sample inference
- model_save         : Persisting trained model weights to disk

Exposing key classes/functions at the package level allows callers to do:

    from pipelines import DataIngestor, DatasetCombiner

instead of reaching into individual submodules.
"""

__version__ = "1.0.0"

__all__ = [
    "DataIngestor",
    "DatasetCombiner",
    "PredictiveMaintenancePreprocessor",
]

# Re-export commonly used classes for convenience.
# Wrapped in try/except so a missing optional dependency (e.g. torch, sklearn)
# doesn't break `import pipelines` entirely for unrelated stages.
try:
    from pipelines.data_ingest import DataIngestor, DatasetCombiner
except ImportError:
    pass

try:
    from pipelines.data_clean import PredictiveMaintenancePreprocessor
except ImportError:
    pass