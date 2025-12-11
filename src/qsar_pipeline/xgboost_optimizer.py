#!/Users/luisseijas/miniforge3/envs/IC-50/bin/python
# -*- coding: utf-8 -*-
"""
XGBoost Optimization Pipeline for pIC50 Prediction.

This script implements a comprehensive pipeline for training and optimizing an
XGBoost model to predict pIC50 values from molecular descriptors.
The methodology follows 'XGBoost_Facil_2.ipynb' and includes:
1.  Data Loading & Preprocessing
2.  Hyperparameter Optimization (Grid Search)
3.  Feature Selection (Importance-based)
4.  Outlier Detection & Removal (Mahalanobis Distance & Leverage)
5.  Refitting on Clean Data
6.  Evaluation (OECD Domain Applicability, Williams Plot)
7.  Model Persistence (Models, Metadata, Plots)


Author: Antigravity (Google DeepMind)
Date: 2025-12-10
"""

# ==============================================================================
import os
import sys
import json
import time
import joblib
import warnings
from datetime import datetime
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import chi2
from sklearn.model_selection import GridSearchCV, train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm.auto import tqdm

# Suppress minor warnings for cleaner output
warnings.filterwarnings('ignore')

@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'dataset_molecular_optimizado.xlsx')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
METADATA_DIR = os.path.join(RESULTS_DIR, 'model_metadata')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')


# Random state for reproducibility
RANDOM_STATE = 42

# Grid Search Hyperparameter Space
# (Reduced for demonstration/verification purposes. Expand lists for full optimization.)
# PARAM_GRID = {
#     'n_estimators': [100, 150, 200, 250, 300, 350],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'max_depth': [3, 4, 5, 6],
#     'gamma': [0.0, 0.1, 0.2, 0.5],
#     'colsample_bytree': [0.7, 0.8],
#     'min_child_weight': [1, 5],
#     'reg_alpha': [0.01, 0.1, 1],
#     'reg_lambda': [1, 5],
#     'subsample': [0.7, 0.8]
# }

PARAM_GRID = {
    'n_estimators': [150, 250, 350],        # Número de árboles (más árboles = modelo más complejo)
    'learning_rate': [0.03, 0.05],          # Tasa de aprendizaje (menor = aprendizaje más lento pero estable)
    'max_depth': [4, 5],                    # Profundidad máxima de árboles (controla complejidad)
    'min_child_weight': [1, 5],             # Peso mínimo en nodos hijo (previene overfitting)
    'subsample': [0.7, 0.8],                # Fracción de muestras usadas por árbol (previene overfitting)
    'colsample_bytree': [0.7, 0.8],         # Fracción de features usadas por árbol
    'gamma': [0.0, 0.1, 0.5],               # Reducción mínima de pérdida para hacer split (regularización)
    'reg_alpha': [0.01, 0.1],               # Regularización L1 (produce features sparse)
    'reg_lambda': [1, 5]                    # Regularización L2 (reduce pesos)
}
# PARAM_GRID = {'n_estimators': [10], 'max_depth': [3]} # Fast test


# Fast Grid for Verification (Best Params from previous run)
# PARAM_GRID = {
#     'n_estimators': [350],
#     'learning_rate': [0.05],
#     'max_depth': [5],
#     'gamma': [0.5],
#     'colsample_bytree': [0.7],
#     'min_child_weight': [1],
#     'reg_alpha': [0.1], 
#     'reg_lambda': [1],
#     'subsample': [0.7]
# }

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def setup_directories():
    """Creates necessary output directories if they don't exist."""
    for directory in [RESULTS_DIR, METADATA_DIR, PLOTS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def load_data(filepath):
    """
    Loads the optimized dataset and separates predictors (X) and target (y).
    
    Args:
        filepath (str): Path to the Excel file.
        
    Returns:
        tuple: (X_df, y_series, original_df)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    print(f"Loading data from: {filepath}")
    df = pd.read_excel(filepath)
    
    # Define columns to exclude from features
    excluded_cols = ['smiles', 'Especie', 'IC50', 'pIC50', 'Molecule Name', 'Name']
    
    # Target variable
    y = df['pIC50']
    
    # Feature matrix: drop excluded columns and any non-numeric columns
    X = df.drop(columns=[c for c in excluded_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number])
    
    print(f"Data loaded: {len(df)} samples, {X.shape[1]} features")
    return X, y, df

def evaluate_metrics(y_true, y_pred, set_name="Set"):
    """Calculates and prints regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nMetrics for {set_name}:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def calculate_mahalanobis(X_train, X_test=None):
    """
    Calculates Mahalanobis distance for train (and optionally test) sets.
    Returns distances, covariance matrix inverse, and centroid.
    """
    # Regularization to prevent singular matrix
    regularization = np.eye(X_train.shape[1]) * 1e-5
    
    # Covariance matrix and its inverse
    cov_matrix = np.cov(X_train.T)
    cov_matrix_reg = cov_matrix + regularization
    
    try:
        cov_inv = np.linalg.inv(cov_matrix_reg)
    except np.linalg.LinAlgError:
        print("  ⚠️ Covariance matrix singular, using pseudo-inverse.")
        cov_inv = np.linalg.pinv(cov_matrix_reg)
        
    centroid = np.mean(X_train, axis=0)
    
    # Calculate Distances for Train
    diff_train = X_train - centroid
    mahal_train = np.sqrt(np.sum((diff_train @ cov_inv) * diff_train, axis=1))
    
    mahal_test = None
    if X_test is not None:
        diff_test = X_test - centroid
        mahal_test = np.sqrt(np.sum((diff_test @ cov_inv) * diff_test, axis=1))
        
    return mahal_train, mahal_test, cov_inv, centroid, cov_matrix_reg

def plot_williams(leverage, residuals_std, h_crit, res_crit, set_name="Train", filename=None):
    """Generates and saves a Williams Plot (Leverage vs Std Residuals)."""
    plt.figure(figsize=(10, 6))
    plt.scatter(leverage, residuals_std, alpha=0.6, c='blue', edgecolors='k')
    
    # Critical limits
    plt.axhline(y=res_crit, color='r', linestyle='--', label=f'Residual Limit (±{res_crit})')
    plt.axhline(y=-res_crit, color='r', linestyle='--')
    plt.axvline(x=h_crit, color='orange', linestyle='--', label=f'Leverage Limit ({h_crit:.3f})')
    
    # Domain shading
    plt.axhspan(-res_crit, res_crit, 0, 1.1, alpha=0.1, color='green') # Approximation for shading
    
    plt.xlabel('Leverage (h)')
    plt.ylabel('Standardized Residuals')
    plt.title(f'Williams Plot ({set_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if filename:
        save_path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved plot: {save_path}")

def plot_mahalanobis_williams(mahal_dist, residuals_std, mahal_crit, res_crit, set_name="Train", filename=None):
    """Generates and saves a Williams-like Plot using Mahalanobis distance."""
    plt.figure(figsize=(10, 6))
    plt.scatter(mahal_dist, residuals_std, alpha=0.6, c='green', marker='s', edgecolors='k')
    
    # Critical limits
    plt.axhline(y=res_crit, color='r', linestyle='--', label=f'Residual Limit (±{res_crit})')
    plt.axhline(y=-res_crit, color='r', linestyle='--')
    plt.axvline(x=mahal_crit, color='orange', linestyle='--', label=f'Mahalanobis Limit ({mahal_crit:.2f})')
    
    plt.xlabel('Mahalanobis Distance')
    plt.ylabel('Standardized Residuals')
    plt.title(f'Williams Plot via Mahalanobis ({set_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if filename:
        save_path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved plot: {save_path}")

def plot_pred_vs_real(y_true, y_pred, title, filename=None):
    """Plots Predicted vs Real values."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, color='purple')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.xlabel('Experimental pIC50')
    plt.ylabel('Predicted pIC50')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if filename:
        save_path = os.path.join(PLOTS_DIR, filename)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved plot: {save_path}")

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    start_time = time.time()
    print("=" * 80)
    print("XGBOOST OPTIMIZATION PIPELINE FOR pIC50 PREDICTION")
    print("=" * 80)
    
    setup_directories()
    
    # 1. Load Data
    try:
        X, y, df_orig = load_data(INPUT_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Train/Test Split
    print("\n" + "-" * 40)
    print("2. Splitting Data (Train/Test)")
    print("-" * 40)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train size: {X_train.shape}")
    print(f"Test size:  {X_test.shape}")

    # 3. Hyperparameter Optimization
    print("\n" + "-" * 40)
    print("3. Hyperparameter Optimization (Grid Search)")
    print("-" * 40)
    
    # Calculate total iterations for progress bar
    n_candidates = len(ParameterGrid(PARAM_GRID))
    n_splits = 5
    total_fits = n_candidates * n_splits
    
    print(f"Hyperparameter combinations: {n_candidates}")
    print(f"Total fits to perform: {total_fits}")

    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=PARAM_GRID,
        scoring='neg_mean_squared_error',
        cv=n_splits,
        verbose=0,  # Disable verbose output to use tqdm instead
        n_jobs=-1
    )
    
    print("Starting Grid Search...")
    with tqdm_joblib(tqdm(desc="Grid Search Progress", total=total_fits)) as progress_bar:
        grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    print(f"\nBest Parameters found: {best_params}")
    
    best_model_cv = grid_search.best_estimator_

    # 4. Feature Selection
    print("\n" + "-" * 40)
    print("4. Feature Selection (Importance-based)")
    print("-" * 40)
    
    importances = best_model_cv.feature_importances_
    indices = np.argsort(importances)[::-1] # Features sorted by importance
    
    # Cumulative importance
    cum_importance = np.cumsum(importances[indices])
    n_features_95 = np.argmax(cum_importance >= 0.95) + 1
    
    # Limit features between 20 and 50 as per notebook logic
    n_features_opt = max(20, min(50, n_features_95))
    print(f"Optimal number of features (95% info or constraint): {n_features_opt}")
    
    # Select top N features
    selected_indices = indices[:n_features_opt]
    selected_features = X.columns[selected_indices].tolist()
    
    X_train_sel = X_train.iloc[:, selected_indices]
    X_test_sel = X_test.iloc[:, selected_indices]
    
    print("Top 10 Features:")
    for i in range(10):
        print(f"  {selected_features[i]}: {importances[selected_indices[i]]:.4f}")

    # 5. Outlier Removal (Mahalanobis)
    print("\n" + "-" * 40)
    print("5. Outlier Detection & Removal (Mahalanobis)")
    print("-" * 40)
    
    # Calculate Mahalanobis for *selected features* training set
    # Using numpy arrays for matrix operations
    X_train_np = X_train_sel.values
    X_test_np = X_test_sel.values
    
    mahal_train, mahal_test, cov_inv, centroid, cov_mat = calculate_mahalanobis(X_train_np, X_test_np)
    
    # Critical value (Chi-squared distribution)
    p = X_train_sel.shape[1] # Number of features
    # Using 97.5% confidence interval as common practice (or 0.999 per some cells in notebook)
    # The snippet showed 0.999 used in one cell and 0.975 in comments. Let's use 0.999 for robustness.
    mahal_crit = np.sqrt(chi2.ppf(0.975, df=p))
    print(f"Mahalanobis Critical Limit (d*): {mahal_crit:.4f}")
    
    # Identify outliers
    outliers_mask = mahal_train > mahal_crit
    n_outliers = np.sum(outliers_mask)
    print(f"Detected {n_outliers} structural outliers in Training Set.")
    
    # Remove outliers
    X_train_clean = X_train_sel[~outliers_mask]
    y_train_clean = y_train[~outliers_mask]
    print(f"Clean Training Set size: {X_train_clean.shape}")
    
    # Recalculate covariance matrix on CLEAN data for future domain applicability checks
    _, _, cov_inv_clean, centroid_clean, cov_mat_clean = calculate_mahalanobis(X_train_clean.values)

    # 6. Refit Model on Clean Data
    print("\n" + "-" * 40)
    print("6. Refitting XGBoost on Clean Data")
    print("-" * 40)
    
    # Split clean train data again for early stopping validation
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_clean, y_train_clean, test_size=0.1, random_state=RANDOM_STATE
    )
    
    # Remove parameters that conflict with explicit arguments or are not needed for final fit
    params_to_remove = ['n_estimators', 'random_state']
    # Create a copy to avoid modifying the original dictionary for metadata saving later
    final_params = best_params.copy()
    for param in params_to_remove:
        if param in final_params:
            del final_params[param]
            
    final_model = xgb.XGBRegressor(
        **final_params,
        objective='reg:squarederror',
        n_estimators=1000, # High number, controlled by early stopping
        early_stopping_rounds=20,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    final_model.fit(
        X_fit, y_fit,
        eval_set=[(X_val, y_val)],
        verbose=10
    )
    
    best_iter = final_model.best_iteration
    print(f"Model refitted. Best iteration: {best_iter}")
    
    # 7. Evaluation
    print("\n" + "-" * 40)
    print("7. Final Evaluation")
    print("-" * 40)
    
    # Predictions
    y_train_pred = final_model.predict(X_train_clean)
    y_test_pred = final_model.predict(X_test_sel)
    
    # Metrics
    metrics_train = evaluate_metrics(y_train_clean, y_train_pred, "Clean Train")
    metrics_test = evaluate_metrics(y_test, y_test_pred, "Test (All)")
    
    # Domain Applicability on Test Set
    # Identify valid test samples (within Mahalanobis domain of CLEAN model)
    diff_test = X_test_sel.values - centroid_clean
    mahal_test_clean_metric = np.sqrt(np.sum((diff_test @ cov_inv_clean) * diff_test, axis=1))
    
    # ---------------------------------------------------------
    # 1. Standard OECD Applicability Domain (Structural Only)
    # ---------------------------------------------------------
    # Only removes compounds that are structurally different (X-outliers)
    # This is the standard, rigorous way to evaluate external validity.
    test_in_domain_mask = mahal_test_clean_metric <= mahal_crit
    
    print(f"\nTest samples in domain (Structural Only - OECD): {np.sum(test_in_domain_mask)} / {len(y_test)}")
    
    if np.sum(test_in_domain_mask) > 0:
        y_test_in = y_test[test_in_domain_mask]
        y_test_pred_in = y_test_pred[test_in_domain_mask]
        metrics_test_domain = evaluate_metrics(y_test_in, y_test_pred_in, "Test (In Domain - Structural)")
    else:
        metrics_test_domain = None
        print("Warning: No test samples found within the strict applicability domain.")

    # ---------------------------------------------------------
    # 2. Notebook Replication (Structural + Residual Filter)
    # ---------------------------------------------------------
    # The notebook (XGBoost_Facil_2.ipynb) filters the test set by removing:
    #   a) Structural outliers (Mahalanobis distance > limit)
    #   b) Response outliers (Standardized Residuals > 3.0)
    # Removing samples with high prediction errors (b) artificially inflates the R2.
    # We replicate this here ONLY to show we can match the notebook's 0.66 R2.
    
    std_resid_train_clean = np.std(y_train_clean - y_train_pred)
    std_res_test = (y_test - y_test_pred) / std_resid_train_clean
    
    # Notebook Logic: both X-outlier AND Y-outlier must be absent
    # (In notebook: "outliers_severos" logic, but for "muestras validas" they usually exclude any outlier)
    # Let's check strict replication: The notebook likely excludes anything flagged as an outlier.
    # Line 1223: y_outliers_test = np.abs(residuos_std_test) > residuo_critico (3.0)
    
    mask_notebook_replication = (mahal_test_clean_metric <= mahal_crit) & (np.abs(std_res_test) <= 3.0)
    
    print(f"\nTest samples in domain (Structural + Residual Filter): {np.sum(mask_notebook_replication)} / {len(y_test)}")
    if np.sum(mask_notebook_replication) > 0:
        y_test_nb = y_test[mask_notebook_replication]
        y_test_pred_nb = y_test_pred[mask_notebook_replication]
        metrics_test_notebook = evaluate_metrics(y_test_nb, y_test_pred_nb, "Test (Structural + Residual Filter)")

    # 8. Visualization & Saving
    print("\n" + "-" * 40)
    print("8. Saving Results")
    print("-" * 40)
    
    # Generate Plots
    plot_pred_vs_real(y_train_clean, y_train_pred, 
                      f"Train (Clean) - R2={metrics_train['r2']:.3f}", 
                      "pred_vs_real_train.png")
    
    plot_pred_vs_real(y_test, y_test_pred, 
                      f"Test (All) - R2={metrics_test['r2']:.3f}", 
                      "pred_vs_real_test.png")
    
    if np.sum(mask_notebook_replication) > 0:
        plot_pred_vs_real(y_test[mask_notebook_replication], 
                          y_test_pred[mask_notebook_replication],
                          f"Test (Filtered) - R2={metrics_test_notebook['r2']:.3f}",
                          "pred_vs_real_test_filtered.png")
    
    # Standardized Residuals
    std_res_train = (y_train_clean - y_train_pred) / np.std(y_train_clean - y_train_pred)
    
    # Leverage (Hat matrix diagonal approximation for large N) - simplistic version
    # Here we stick to Mahalanobis based Williams plot as it's more robust for QSAR
    plot_mahalanobis_williams(
        mahal_train[~outliers_mask], # Distances of clean data
        std_res_train,
        mahal_crit,
        3.0, # Residual limit 3 sigma
        "Clean Train",
        "williams_plot_mahalanobis_train.png"
    )
    
    # Save Files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save Metadata JSON
    metadata = {
        'timestamp': timestamp,
        'best_params': best_params,
        'features_selected': selected_features,
        'metrics_train': metrics_train,
        'metrics_test': metrics_test,
        'metrics_test_domain_valid': metrics_test_domain,
        'mahalanobis_crit': float(mahal_crit),
        'n_features': int(p),
        'n_train_samples_clean': int(X_train_clean.shape[0]),
        'n_outliers_removed': int(n_outliers)
    }
    
    metadata_path = os.path.join(METADATA_DIR, f'metadatos_modelo_{timestamp}.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
        
    # Save Covariance Info JSON
    cov_info = {
        'centroid': centroid_clean.tolist(),
        'cov_inv': cov_inv_clean.tolist(),
        'cov_matrix_reg': cov_mat_clean.tolist()
    }
    cov_path = os.path.join(METADATA_DIR, f'covarianza_dominio_{timestamp}.json')
    with open(cov_path, 'w', encoding='utf-8') as f:
        json.dump(cov_info, f)
        
    # Save Models
    model_json_path = os.path.join(METADATA_DIR, f'modelo_xgboost_{timestamp}.json')
    final_model.get_booster().save_model(model_json_path)
    
    model_pkl_path = os.path.join(METADATA_DIR, f'modelo_completo_{timestamp}.pkl')
    joblib.dump(final_model, model_pkl_path)
    
    print("\nSaved Files:")

    print(f"  - Metadata: {metadata_path}")
    print(f"  - Covariance: {cov_path}")
    print(f"  - Model (JSON): {model_json_path}")
    print(f"  - Model (PKL): {model_pkl_path}")
    
    duration = time.time() - start_time
    print(f"\nPipeline completed in {duration:.2f} seconds.")

if __name__ == "__main__":
    main()
