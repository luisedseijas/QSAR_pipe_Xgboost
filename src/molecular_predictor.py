#!/Users/luisseijas/miniforge3/envs/IC-50/bin/python
# -*- coding: utf-8 -*-

"""
Molecular Predictor with PCA-based Applicability Domain

This script predicts pIC50 values for new compounds
using a pre-trained XGBoost model. It implements a rigorous Applicability Domain (AD)
methodology based on Principal Component Analysis (PCA) and Mahalanobis distance.

Methodology:
1.  **Data Loading**: Loads training data (to define the domain) and new compounds data.
2.  **Feature Alignment**: Ensures new data has the exact same features as the trained model.
3.  **Preprocessing**: Scales data using RobustScaler (fitted on training data).
4.  **Applicability Domain (New)**:
    -   Performs PCA on the scaled training data, retaining components that explain
        95% of the variance.
    -   Calculates the centroid and covariance matrix of the training data in this
        reduced PCA space.
    -   Computes Mahalanobis distances for all training samples to define a
        threshold (e.g., 95th percentile).
    -   Projects new compounds into this same PCA space and calculates their
        Mahalanobis distances to determine if they are "In Domain".
5.  **Prediction**: Predicts pIC50 using the XGBoost model.
6.  **Reporting**: Exports results to Excel and generates visualization plots.

Author: Antigravity (Google DeepMind)
Date: 2025-12-09
"""

import os
import glob
import json
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import chi2

from scipy.stats import chi2

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.ML.Descriptors import MoleculeDescriptors

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_STATE = 42
PCA_VARIANCE_THRESHOLD = 0.95  # Retain components explaining 95% variance
AD_CONFIDENCE_LEVEL = 0.95     # Confidence level for AD threshold (Chi-square)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_DIR = os.path.join(RESULTS_DIR, "model_metadata")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")

# Ensure output directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


def load_data(filepath):
    """
    Loads data from an Excel or CSV file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.info(f"Loading data from: {filepath}")
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        return pd.read_excel(filepath)
    elif filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    else:
        raise ValueError("Unsupported file format. Please use Excel (.xlsx) or CSV (.csv).")


def load_latest_model_and_metadata(model_dir):
    """
    Finds and loads the most recent XGBoost model (.json) and its metadata (.json).
    """
    # Find all model JSON files (excluding metadata files)
    # Note: Files are named 'modelo_xgboost_...' (Spanish)
    model_files = glob.glob(os.path.join(model_dir, "modelo_xgboost_*.json"))
    if not model_files:
        raise FileNotFoundError(f"No XGBoost model files found in {model_dir} with pattern 'modelo_xgboost_*.json'")
    
    # Sort by modification time (newest first)
    latest_model_path = max(model_files, key=os.path.getmtime)
    
    # Construct corresponding metadata path
    # Filename format: modelo_xgboost_YYYYMMDD_HHMMSS.json
    filename = os.path.basename(latest_model_path)
    # Extract everything after 'modelo_xgboost_' and before '.json'
    timestamp = filename.replace('modelo_xgboost_', '').replace('.json', '')
    
    metadata_path = os.path.join(model_dir, f"metadatos_modelo_{timestamp}.json")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found for model: {latest_model_path}")
    
    logger.info(f"Loading latest model: {os.path.basename(latest_model_path)}")
    logger.info(f"Loading metadata: {os.path.basename(metadata_path)}")
    
    # Load Model
    model = xgb.XGBRegressor()
    model.load_model(latest_model_path)
    
    # Load Metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    return model, metadata


def calculate_mahalanobis_distance(X_pca, centroid, cov_inv):
    """
    Calculates the Mahalanobis distance for each sample in the PCA space.
    
    Args:
        X_pca (np.ndarray): Data projected into PCA space (n_samples, n_components).
        centroid (np.ndarray): Mean vector of the training data in PCA space.
        cov_inv (np.ndarray): Inverse covariance matrix of the training data in PCA space.
        
    Returns:
        np.ndarray: Array of Mahalanobis distances.
    """
    # X_pca is (n_samples, n_features)
    # Centroid is (n_features,)
    # Distance = sqrt( (x - mu)^T * S^-1 * (x - mu) )
    
    diff = X_pca - centroid
    # Calculate distance: sqrt(diag(diff * cov_inv * diff.T))
    # Using cdist is often more robust but manual calculation with einsum is efficient for batch
    # Here we use a robust manual approach:
    
    left_term = np.dot(diff, cov_inv)
    # Element-wise multiplication and sum, then sqrt
    dist = np.sqrt(np.sum(left_term * diff, axis=1))
    
    return dist


def train_pca_ad(X_train_scaled, variance_threshold=0.95):
    """
    Trains the PCA-based Applicability Domain model.
    
    1. Fits PCA on scaled training data to retain 95% variance.
    2. Projects training data to PCA space.
    3. Calculates centroid and covariance matrix in PCA space.
    4. Determines the critical Mahalanobis distance threshold.
    
    Returns dictionary with AD parameters.
    """
    logger.info(f"Fitting PCA (variance threshold: {variance_threshold})...")
    
    # Fit PCA
    pca = PCA(n_components=variance_threshold, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    n_components = pca.n_components_
    explained_variance = np.sum(pca.explained_variance_ratio_)
    logger.info(f"PCA retained {n_components} components explaining {explained_variance:.2%} of variance.")
    
    # Calculate Applicability Domain Parameters in PCA Space
    centroid = np.mean(X_train_pca, axis=0)
    cov_matrix = np.cov(X_train_pca, rowvar=False)
    
    # Handle potentially singular matrix (add small jitter if needed, though PCA usually orthogonalizes well)
    # Using pseudoinverse for stability
    cov_inv = np.linalg.pinv(cov_matrix)
    
    # Calculate Distances for Training Set
    distances = calculate_mahalanobis_distance(X_train_pca, centroid, cov_inv)
    
    # Define Threshold
    # Option 1: Chi-square cut-off (theoretical)
    # threshold = chi2.ppf(AD_CONFIDENCE_LEVEL, df=n_components)
    
    # Option 2: Percentile based (empirical) - often more robust for non-normal distributions
    # We will use the Chi-square approach as it's standard for Mahalanobis AD, 
    # but we sanity check against the max distance in training.
    threshold_chi2 = chi2.ppf(AD_CONFIDENCE_LEVEL, df=n_components)
    
    # Let's log both for verification
    max_train_dist = np.max(distances)
    logger.info(f"AD Threshold (Chi2 @ {AD_CONFIDENCE_LEVEL}, df={n_components}): {threshold_chi2:.4f}")
    logger.info(f"Max Mahalanobis distance in training set: {max_train_dist:.4f}")
    
    return {
        'pca_model': pca,
        'centroid': centroid,
        'cov_inv': cov_inv,
        'threshold': threshold_chi2,
        'n_components': n_components
    }


def calculate_descriptors(df, smiles_col='smiles'):
    """
    Calculates RDKit descriptors for molecules in the DataFrame.
    """
    logger.info("Calculating molecular descriptors using RDKit...")
    
    # Initialize descriptor calculator with all available RDKit descriptors
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    descriptor_names = calc.GetDescriptorNames()
    
    descriptors_data = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        smiles = row.get(smiles_col)
        if pd.isna(smiles):
            continue
            
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Standard Descriptors
            descriptors = list(calc.CalcDescriptors(mol))
            
            # Custom Counts (to match training data features like NumN, NumBr, NumAtoms)
            # Note: Training data has both 'NumAtoms' and 'HeavyAtomCount'. 
            # calculating NumAtoms as total atoms (including H) requires AddHs.
            mol_with_h = Chem.AddHs(mol)
            
            custom_descriptors = {
                'NumN': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'N']),
                'NumBr': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'Br']),
                'NumAtoms': mol_with_h.GetNumAtoms(), 
                'NumBonds': mol.GetNumBonds(),
                'RingCount': mol.GetRingInfo().NumRings()
            }
            
            # Append custom values to descriptor list
            # We need to extend the descriptor NAMES too, but doing it inside the loop is inefficient for names.
            # We will handle names outside.
            
            # Combine
            row_values = descriptors + list(custom_descriptors.values())
            descriptors_data.append(row_values)
            valid_indices.append(idx)
        else:
            logger.warning(f"Invalid SMILES at index {idx}: {smiles}")
            
    # Update Descriptor Names
    custom_names = ['NumN', 'NumBr', 'NumAtoms', 'NumBonds', 'RingCount']
    all_names = list(descriptor_names) + custom_names
    
    # Create DataFrame with descriptors
    df_desc = pd.DataFrame(descriptors_data, columns=all_names, index=valid_indices)
    
    # Merge back with original data (keeping only valid molecules)
    df_result = df.loc[valid_indices].join(df_desc)
    
    logger.info(f"Descriptors calculated for {len(df_result)} molecules.")
    return df_result



def predict_new_compounds():
    """
    Main workflow function for New Compounds prediction.
    """
    start_time = datetime.now()
    logger.info("Starting Molecular Prediction Pipeline...")
    
    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    # Training Data (for scaling & PCA fit)
    train_file = os.path.join(DATA_DIR, "processed", "dataset_molecular_optimizado.xlsx") # Optimized dataset
    if not os.path.exists(train_file):
        # Fallback to the known raw file if optimized one isn't there, but optimized is preferred for consistency
        # Assuming dataset_optimizer.py generated this. If not, we might need to look for 'all_descriptor_results_1751.xlsx'
        # Let's assume the user has run the previous steps.
        logger.warning(f"Optimized dataset not found at {train_file}. Checking raw input.")
        train_file = os.path.join(DATA_DIR, "raw", "all_descriptor_results_1751.xlsx") # Fallback absolute path
        
    df_train = load_data(train_file)
    
    # New Compounds Data
    # Filename generic
    input_file = "new_compounds.xlsx"
    # We look for it in the Documents folder or current dir
    input_path = None
    possible_paths = [
        os.path.join(DATA_DIR, "raw", input_file),
        os.path.join(BASE_DIR, input_file),
        "/Users/luisseijas/Documents/QSAR_Andres/" + input_file
    ]
    
    for p in possible_paths:
        if os.path.exists(p):
            input_path = p
            break
            
    if not input_path:
        raise FileNotFoundError(f"Input file '{input_file}' not found in common locations.")
        
    df_new_raw = load_data(input_path)
    
    # Calculate descriptors if they don't exist
    # Determine SMILES column
    smiles_col = None
    for col in ['smiles', 'SMILES', 'Smiles', 'Canonical_Smiles']:
        if col in df_new_raw.columns:
            smiles_col = col
            break
            
    if smiles_col:
        df_new = calculate_descriptors(df_new_raw, smiles_col=smiles_col)
    else:
        # If no smiles, hopefully descriptors are already there (unlikely given user input)
        logger.warning("No SMILES column found. Assuming descriptors are present in the file.")
        df_new = df_new_raw
    
    # -------------------------------------------------------------------------
    # 2. Load Model & Metadata
    # -------------------------------------------------------------------------
    model, metadata = load_latest_model_and_metadata(MODEL_DIR)
    
    # Identify Selected Features
    try:
        selected_features = metadata['features_selected']
    except KeyError:
        # If metadata structure varies, try to infer or fallback
        logger.error("Could not find 'features_selected' in metadata.")
        return

    logger.info(f"Model uses {len(selected_features)} features.")
    
    # -------------------------------------------------------------------------
    # 3. Data Preparation
    # -------------------------------------------------------------------------
    # Ensure features exist in both datasets
    missing_features_train = [f for f in selected_features if f not in df_train.columns]
    missing_features_new = [f for f in selected_features if f not in df_new.columns]
    
    if missing_features_train:
        raise ValueError(f"Training data missing features: {missing_features_train}")
    if missing_features_new:
        # Check if we calculated descriptors but some are still missing (e.g. non-RDKit features used in training?)
        # For now, raise error to ensure strict matching as requested.
        raise ValueError(f"New data missing required features used in training: {missing_features_new}")
        
    # Extract Features
    X_train = df_train[selected_features]
    X_new = df_new[selected_features]
    
    # Handle NaNs in New Data (Crucial for robust prediction)
    # RDKit might return NaN for some descriptors on some molecules
    if X_new.isnull().values.any():
        logger.warning("Found NaN values in calculated descriptors. Imputing with column means from Training data.")
        # Impute with Training means (best practice to avoid data leakage from test set)
        train_means = X_train.mean()
        X_new = X_new.fillna(train_means)
        
        # If any NaNs remain (e.g. if train column was all NaN?), drop or fill 0
        if X_new.isnull().values.any():
             logger.warning("Still finding NaNs after mean imputation. Filling remaining with 0.")
             X_new = X_new.fillna(0)
    
    # Identifiers and SMILES
    id_col = 'Molecule Name' if 'Molecule Name' in df_new.columns else df_new.columns[0]
    ids_new = df_new[id_col].values
    
    # Extract SMILES for reporting (if present)
    # df_new should have it from descriptor calculation step or original load
    smiles_col = None
    for col in ['smiles', 'SMILES', 'Smiles', 'Canonical_Smiles']:
        if col in df_new.columns:
            smiles_col = col
            break
            
    if smiles_col:
        smiles_values = df_new[smiles_col].values
    else:
        smiles_values = [''] * len(df_new)
    
    # Scaling (Fit on Train, Transform Train & New)
    logger.info("Scaling features using RobustScaler (fitted on training data)...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_new_scaled = scaler.transform(X_new)
    
    # -------------------------------------------------------------------------
    # 4. Applicability Domain (PCA + Mahalanobis)
    # -------------------------------------------------------------------------
    logger.info("Calculating Applicability Domain parameters...")
    ad_params = train_pca_ad(X_train_scaled, variance_threshold=PCA_VARIANCE_THRESHOLD)
    
    pca = ad_params['pca_model']
    centroid = ad_params['centroid']
    cov_inv = ad_params['cov_inv']
    threshold = ad_params['threshold']
    
    # Project New Data to PCA Space
    X_new_pca = pca.transform(X_new_scaled)
    
    # Calculate Distances for New Data
    distances_new = calculate_mahalanobis_distance(X_new_pca, centroid, cov_inv)
    
    # Determine Domain Status
    in_domain = distances_new <= threshold
    
    n_inside = np.sum(in_domain)
    n_outside = len(in_domain) - n_inside
    logger.info(f"AD Results: {n_inside} compounds Inside ({n_inside/len(in_domain):.1%}), {n_outside} Outside.")
    
    # -------------------------------------------------------------------------
    # 5. Prediction
    # -------------------------------------------------------------------------
    logger.info("Predicting pIC50 values...")
    # XGBoost expects specific localized structures sometimes, but usually numpy array is fine if feature names match
    # To be safe with feature names:
    X_new_scaled_df = pd.DataFrame(X_new_scaled, columns=selected_features)
    
    y_pred_pIC50 = model.predict(X_new_scaled_df)
    
    # Convert to IC50 (nM): 10^(9 - pIC50)
    y_pred_IC50_nM = 10**(9 - y_pred_pIC50)
    
    # -------------------------------------------------------------------------
    # 6. Formatting Results
    # -------------------------------------------------------------------------
    results_df = pd.DataFrame({
        'Compound_ID': ids_new,
        'SMILES': smiles_values,
        'Predicted_pIC50': y_pred_pIC50,
        'Predicted_IC50_nM': y_pred_IC50_nM,
        'Mahalanobis_Distance': distances_new,
        'In_Applicability_Domain': in_domain
    })
    
    # Rank by pIC50 (descending)
    results_df = results_df.sort_values(by='Predicted_pIC50', ascending=False).reset_index(drop=True)
    results_df['Rank'] = results_df.index + 1
    
    # Classify Activity (Generic QSAR bins)
    def classify_activity(val):
        if val >= 8.0: return "Highly Active (<10 nM)"
        if val >= 7.0: return "Active (<100 nM)"
        if val >= 6.0: return "Moderate (<1 uM)"
        return "Inactive (>1 uM)"
        
    results_df['Activity_Class'] = results_df['Predicted_pIC50'].apply(classify_activity)
    
    # -------------------------------------------------------------------------
    # 7. Saving Output
    # -------------------------------------------------------------------------
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"predictions_{timestamp_str}.xlsx"
    output_path = os.path.join(PREDICTIONS_DIR, output_filename)
    
    logger.info(f"Saving results to {output_path}...")
    
    with pd.ExcelWriter(output_path) as writer:
        results_df.to_excel(writer, sheet_name='All_Predictions', index=False)
        results_df[results_df['In_Applicability_Domain']].to_excel(writer, sheet_name='Reliable_Predictions', index=False)
        
        # Validation/Stats Sheet
        stats_data = {
            'Metric': ['Total Compounds', 'In Domain', 'Out of Domain', 'AD Threshold', 'Max pIC50', 'Model Used'],
            'Value': [len(results_df), n_inside, n_outside, threshold, results_df['Predicted_pIC50'].max(), str(model)]
        }
        pd.DataFrame(stats_data).to_excel(writer, sheet_name='Statistics', index=False)
        
    # -------------------------------------------------------------------------
    # 8. Visualization
    # -------------------------------------------------------------------------
    logger.info("Generating plots...")
    
    # Plot 1: Histogram of Predicted pIC50
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['Predicted_pIC50'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Predicted pIC50 Values', fontsize=14)
    plt.xlabel('Predicted pIC50')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(PLOTS_DIR, f"distribucion_predicciones_{timestamp_str}.png"))
    plt.close()
    
    # Plot 2: Mahalanobis Distance vs pIC50 (AD Plot)
    plt.figure(figsize=(10, 6))
    
    # Scatter for In Domain
    plt.scatter(
        results_df.loc[results_df['In_Applicability_Domain'], 'Mahalanobis_Distance'],
        results_df.loc[results_df['In_Applicability_Domain'], 'Predicted_pIC50'],
        color='green', alpha=0.6, label='In Domain', s=20
    )
    
    # Scatter for Out of Domain
    plt.scatter(
        results_df.loc[~results_df['In_Applicability_Domain'], 'Mahalanobis_Distance'],
        results_df.loc[~results_df['In_Applicability_Domain'], 'Predicted_pIC50'],
        color='red', alpha=0.4, label='Out of Domain', s=20
    )
    
    # Threshold Line
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=1.5, label=f'AD Threshold ({threshold:.2f})')
    
    plt.title('Applicability Domain Analysis (PCA-Based)', fontsize=14)
    plt.xlabel(f'Mahalanobis Distance (PCA space, {ad_params["n_components"]} comps)')
    plt.ylabel('Predicted pIC50')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig(os.path.join(PLOTS_DIR, f"dominio_aplicabilidad_pca_{timestamp_str}.png"))
    plt.close()
    
    # Plot 3: Top 200 Candidates Grid Visualization
    logger.info("Generating top 200 candidates visual report...")
    top_200 = results_df.head(200)
    top_mols = []
    top_legends = []
    
    for _, row in top_200.iterrows():
        s = row['SMILES']
        if pd.notna(s) and s:
            m = Chem.MolFromSmiles(s)
            if m:
                top_mols.append(m)
                # Legend: Rank, ID, pIC50
                top_legends.append(f"#{row['Rank']} | {str(row['Compound_ID'])[:10]} | pIC50: {row['Predicted_pIC50']:.2f}")
    
    if top_mols:
        # Generate Grid Image (max 50 per page usually, but RDKit can do large grids)
        # We will split into chunks of 50 to resolve visibility issues or just one big image?
        # A single large image is often hard to read. Let's do top 100 in one image 10x10.
        # User asked for 200. 
        # Strategy: Save 4 images of 50 molecules each.
        
        chunk_size = 50
        for i in range(0, len(top_mols), chunk_size):
            chunk_mols = top_mols[i:i+chunk_size]
            chunk_legends = top_legends[i:i+chunk_size]
            
            img = Draw.MolsToGridImage(
                chunk_mols, 
                molsPerRow=5, 
                subImgSize=(300, 300), 
                legends=chunk_legends,
                returnPNG=False
            )
            
            # Save
            part_num = (i // chunk_size) + 1
            img_path = os.path.join(PLOTS_DIR, f"top_candidatos_part{part_num}_{timestamp_str}.png")
            img.save(img_path)
            logger.info(f"Saved visual report part {part_num}: {img_path}")
            
    else:
        logger.warning("No valid molecules found for visualization among top candidates.")
    
    duration = datetime.now() - start_time
    logger.info(f"Process completed successfully in {duration}.")
    logger.info(f"Outputs:\n1. Excel: {output_path}\n2. Plots saved in: {PLOTS_DIR}")


if __name__ == "__main__":
    predict_new_compounds()
