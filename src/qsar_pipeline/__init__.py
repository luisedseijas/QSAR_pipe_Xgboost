"""
QSAR Pipeline Package
=====================
A comprehensive pipeline for QSAR analysis, XGBoost modeling, and prediction.
"""

__version__ = "0.1.0"

# Relative imports assuming files are in the same package directory
from .dataset_optimizer import main as run_optimization
from .xgboost_optimizer import main as run_training
from .molecular_predictor import predict_new_compounds as run_prediction

__all__ = ['run_optimization', 'run_training', 'run_prediction']
