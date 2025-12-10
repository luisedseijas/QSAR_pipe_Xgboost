#!/Users/luisseijas/miniforge3/envs/IC-50/bin/python
# -*- coding: utf-8 -*-

"""
QSAR Dataset Optimizer Pipeline
===============================

This script implements a comprehensive pipeline for analyzing, cleaning, and optimizing
molecular datasets for QSAR (Quantitative Structure-Activity Relationship) modeling.
It is based on the logic defined in 'Optimizador_Dataset_Explicado.ipynb'.

Key Features:
-------------
1.  **Statistical Analysis**: Detailed analysis of IC50 distribution and molecular features.
2.  **Outlier Removal**: Robust statistical methods (3xIQR) to remove extreme values while preserving data integrity.
3.  **Feature Optimization**:
    *   Transformation of IC50 to pIC50 (standard in medicinal chemistry).
    *   Cleaning of low-variance and extreme-value features.
    *   Robust scaling resistant to outliers.
    *   Feature selection using F-regression.
4.  **Visualization**: Generation of comparative plots (original vs. optimized).
5.  **Data Management**:
    *   Preserves 'smiles' and 'Especie' columns.
    *   Saves optimized data to 'data/' directory.
    *   Saves figures to 'results/' directory.

Usage:
------
Ensure 'all_descriptor_results_1751.xlsx' is in the current directory or provide the path.
Run the script:
    $ python dataset_optimizer.py

Dependencies:
-------------
pandas, numpy, matplotlib, seaborn, scikit-learn, openpyxl
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.preprocessing import RobustScaler

# Configuration
# ==============================================================================
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style for professional-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Constants
INPUT_FILENAME = os.path.join('data', 'raw', 'all_descriptor_results_1751.xlsx')
DATA_DIR = os.path.join('data', 'processed')
RESULTS_DIR = 'results'
OUTPUT_FILENAME = os.path.join(DATA_DIR, 'dataset_molecular_optimizado.xlsx')


def setup_directories():
    """
    Ensure that the necessary output directories exist.
    Creates 'data/' and 'results/' if they do not exist.
    """
    for directory in [DATA_DIR, RESULTS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}/")


def load_dataset(filepath):
    """
    Load the dataset from an Excel file.

    Args:
        filepath (str): Path to the Excel file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    if not os.path.exists(filepath):
        print(f"Error: Input file '{filepath}' not found.")
        sys.exit(1)
    
    print(f"Loading dataset from {filepath}...")
    try:
        # Attempt to read 'Todas' sheet first, fallback to 'Datos' or first sheet
        try:
            df = pd.read_excel(filepath, sheet_name='Todas')
        except ValueError:
            try:
                df = pd.read_excel(filepath, sheet_name='Datos')
            except ValueError:
                df = pd.read_excel(filepath)
        
        print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)


def create_initial_distribution_analysis(df, stats, output_dir=RESULTS_DIR):
    """
    Create and save initial distribution plots for IC50.

    Args:
        df (pd.DataFrame): The dataframe containing 'IC50'.
        stats (pd.Series): Descriptive statistics of 'IC50'.
        output_dir (str): Directory to save the figures.
    """
    print("\nGenerating initial distribution analysis plots...")
    
    # 1. Histogram and Boxplot of IC50
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Histogram
    sns.histplot(df['IC50'], bins=50, kde=True, ax=ax1, color='skyblue')
    ax1.set_title('Distribución Original de IC50 (Escala Lineal)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('IC50 (nM)')
    ax1.set_ylabel('Frecuencia')
    
    # Add mean and median lines
    ax1.axvline(stats['mean'], color='red', linestyle='--', label=f"Media: {stats['mean']:.0f}")
    ax1.axvline(stats['50%'], color='green', linestyle='-', label=f"Mediana: {stats['50%']:.0f}")
    ax1.legend()
    
    # Boxplot
    sns.boxplot(x=df['IC50'], ax=ax2, color='lightgreen')
    ax2.set_xlabel('IC50 (nM)')
    ax2.set_title('Boxplot de IC50 (Detección de Outliers)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_distribucion_original_ic50.png'), dpi=300)
    plt.close()
    
    # 2. Log-scale Distribution (to show the need for transformation)
    plt.figure(figsize=(12, 6))
    
    # Filter positive values for log scale
    positive_ic50 = df[df['IC50'] > 0]['IC50']
    
    sns.histplot(np.log10(positive_ic50), bins=50, kde=True, color='purple')
    plt.title('Distribución de log10(IC50)', fontsize=14, fontweight='bold')
    plt.xlabel('log10(IC50)')
    plt.ylabel('Frecuencia')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_distribucion_log_ic50.png'), dpi=300)
    plt.close()
    
    print("Initial plots saved.")


def create_detailed_plots(df_original, df_optimized, output_dir=RESULTS_DIR):
    """
    Create and save comparative plots between original and optimized datasets.

    Args:
        df_original (pd.DataFrame): The original dataframe.
        df_optimized (pd.DataFrame): The optimized dataframe.
        output_dir (str): Directory to save the figures.
    """
    print("\nGenerating comparative analysis plots...")
    
    # 1. Comparative Histogram: IC50 vs pIC50
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original IC50 (Log scale for visibility)
    positive_ic50 = df_original[df_original['IC50'] > 0]['IC50']
    sns.histplot(positive_ic50, bins=50, kde=True, ax=ax1, color='gray', log_scale=True)
    ax1.set_title('Original: Distribución IC50 (Log Scale)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('IC50 (nM)')
    
    # Optimized pIC50
    sns.histplot(df_optimized['pIC50'], bins=50, kde=True, ax=ax2, color='teal')
    ax2.set_title('Optimizado: Distribución pIC50', fontsize=12, fontweight='bold')
    ax2.set_xlabel('pIC50')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_comparacion_distribuciones.png'), dpi=300)
    plt.close()
    
    # 2. Boxplot Comparison by Species
    if 'Especie' in df_original.columns and 'Especie' in df_optimized.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Original
        sns.boxplot(x='Especie', y='IC50', data=df_original, ax=ax1, palette='Set3')
        ax1.set_yscale('log')
        ax1.set_title('Original: IC50 por Especie (Log Scale)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('IC50 (nM)')
        
        # Optimized
        sns.boxplot(x='Especie', y='pIC50', data=df_optimized, ax=ax2, palette='viridis')
        ax2.set_title('Optimizado: pIC50 por Especie', fontsize=12, fontweight='bold')
        ax2.set_ylabel('pIC50')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '4_comparacion_especies.png'), dpi=300)
        plt.close()
    
    # 3. Correlation Heatmap of Top Features
    # Identify numeric columns in optimized df excluding metadata
    numeric_cols = df_optimized.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude IC50 if present to focus on pIC50 and features
    cols_to_plot = [c for c in numeric_cols if c != 'IC50']
    
    # If too many columns, take top 15 correlated with pIC50
    if len(cols_to_plot) > 15:
        corrs = df_optimized[cols_to_plot].corrwith(df_optimized['pIC50']).abs().sort_values(ascending=False)
        top_cols = corrs.head(15).index.tolist()
        cols_to_plot = top_cols
    
    plt.figure(figsize=(12, 10))
    corr_matrix = df_optimized[cols_to_plot].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title('Matriz de Correlación (Top Características)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_matriz_correlacion.png'), dpi=300)
    plt.close()
    
    print("Comparative plots saved.")


def comprehensive_dataset_analysis_and_optimization(df):
    """
    Execute the full optimization pipeline:
    1. Statistical Analysis
    2. Outlier Removal
    3. Feature Optimization
    4. Comparative Analysis
    5. Save Results

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        dict: Dictionary containing results and statistics.
    """
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE MOLECULAR DATASET OPTIMIZATION")
    print("="*80)

    # ========================================================================
    # PHASE 1: EXHAUSTIVE STATISTICAL ANALYSIS
    # ========================================================================
    print("\nPHASE 1: STATISTICAL ANALYSIS")
    print("="*50)
    
    print(f"General Information:")
    print(f"   Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
    
    # IC50 Analysis
    ic50_stats = df['IC50'].describe()
    print("\nIC50 Descriptive Statistics:")
    print(ic50_stats)
    
    # Calculate IQR and bounds
    Q1 = ic50_stats['25%']
    Q3 = ic50_stats['75%']
    IQR = Q3 - Q1
    
    lower_bound_3 = Q1 - 3 * IQR
    upper_bound_3 = Q3 + 3 * IQR
    
    print(f"\nOutlier Detection (IC50):")
    print(f"   IQR: {IQR:.2f}")
    print(f"   3xIQR Bounds: {lower_bound_3:.2f} - {upper_bound_3:.2f}")
    
    # Generate initial plots
    create_initial_distribution_analysis(df, ic50_stats)

    # ========================================================================
    # PHASE 2: OUTLIER REMOVAL (ROBUST STATISTICAL CRITERIA)
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2: OUTLIER REMOVAL")
    print("="*80)
    
    df_clean = df.copy()
    removal_log = []
    
    # 1. Extreme Outliers in IC50 (3xIQR)
    print(f"1. Removing Extreme IC50 Outliers (3xIQR):")
    outlier_mask = (df_clean['IC50'] < lower_bound_3) | (df_clean['IC50'] > upper_bound_3)
    n_outliers = outlier_mask.sum()
    
    print(f"   Outliers detected: {n_outliers} ({n_outliers/len(df_clean)*100:.2f}%)")
    
    if n_outliers > 0:
        # Safety rule: only remove if less than 15% of data
        if n_outliers / len(df_clean) < 0.15:
            df_clean = df_clean[~outlier_mask]
            removal_log.append(f"Extreme IC50 outliers (3xIQR): {n_outliers} records")
            print(f"   ✅ REMOVED: {n_outliers} extreme outliers")
        else:
            print(f"   ⚠️  NOT REMOVED: Too many outliers (>{15}% of data)")
            
    # 2. Impossible Values
    print(f"\n2. Removing Impossible Values:")
    impossible_ic50 = (df_clean['IC50'] <= 0) | df_clean['IC50'].isna()
    n_impossible = impossible_ic50.sum()
    
    if n_impossible > 0:
        df_clean = df_clean[~impossible_ic50]
        removal_log.append(f"IC50 <= 0 or NaN: {n_impossible} records")
        print(f"   ✅ REMOVED: {n_impossible} records with IC50 <= 0 or NaN")
    else:
        print(f"   ✅ No impossible values found in IC50")

    # ========================================================================
    # PHASE 3: ADVANCED FEATURE OPTIMIZATION
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 3: ADVANCED FEATURE OPTIMIZATION")
    print("="*80)
    
    # 1. Target Variable Transformation
    print(f"1. Target Variable Transformation:")
    # pIC50 = -log10(IC50_nM / 1e9) -> Standard in medicinal chemistry
    df_clean['pIC50'] = -np.log10(df_clean['IC50'] / 1e9)
    target_name = 'pIC50'
    print(f"   Applied transformation: pIC50 = -log10(IC50_nM / 1e9)")
    
    # 2. Feature Preparation
    print(f"\n2. Feature Preparation:")
    
    # Identify numeric features, excluding metadata
    # IMPORTANT: Preserve 'smiles' in df_clean, but exclude from numeric features for ML
    basic_cols = ['Especie', 'molecule_chembl_id', 'IC50', 'units', 'smiles', 'pIC50']
    numeric_features = []
    for col in df_clean.columns:
        if col not in basic_cols and df_clean[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_features.append(col)
            
    print(f"   Numeric features detected: {len(numeric_features)}")
    
    # Create X (features) and y (target)
    X = df_clean[numeric_features].copy()
    y = df_clean['pIC50'].copy()
    
    # Handle missing values in features
    missing_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[missing_mask]
    y = y[missing_mask]
    df_clean = df_clean[missing_mask] # Sync dataframe
    
    print(f"   Valid samples after handling missing values: {len(X)}")
    
    # 3. Feature Cleaning
    print(f"\n3. Feature Cleaning:")
    initial_features = X.shape[1]
    
    # Remove low variance features
    variance_selector = VarianceThreshold(threshold=1e-8)
    X_variance = variance_selector.fit_transform(X)
    removed_variance = initial_features - X_variance.shape[1]
    
    kept_features = X.columns[variance_selector.get_support()]
    X_clean = pd.DataFrame(X_variance, columns=kept_features, index=X.index)
    
    if removed_variance > 0:
        print(f"   ✅ Removed {removed_variance} low-variance features")
        
    # Remove extreme value features (potential calculation errors)
    extreme_features = []
    for col in X_clean.columns:
        if (X_clean[col].abs() > 1e10).any():
            extreme_features.append(col)
            
    if extreme_features:
        X_clean = X_clean.drop(columns=extreme_features)
        print(f"   ✅ Removed {len(extreme_features)} features with extreme values")
        
    print(f"   Final features after cleaning: {X_clean.shape[1]}")
    
    # 4. Robust Scaling
    print(f"\n4. Robust Scaling:")
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_clean),
        columns=X_clean.columns,
        index=X_clean.index
    )
    print(f"   ✅ RobustScaler applied")
    
    # 5. Feature Selection
    print(f"\n5. Feature Selection:")
    # Rule of thumb: max 80 features, or 1 feature per 10 samples
    n_features_available = min(80, X_scaled.shape[1], len(y) // 10)
    
    selector = SelectKBest(f_regression, k=n_features_available)
    X_selected = selector.fit_transform(X_scaled, y)
    
    selected_features = X_scaled.columns[selector.get_support()]
    feature_scores = selector.scores_[selector.get_support()]
    
    print(f"   ✅ Selected features: {n_features_available}")
    
    X_final = pd.DataFrame(X_selected, columns=selected_features, index=X_scaled.index)
    
    # Feature Ranking
    feature_ranking = pd.DataFrame({
        'feature': selected_features,
        'score': feature_scores
    }).sort_values('score', ascending=False)
    
    print(f"   Top 5 features:")
    for i, (_, row) in enumerate(feature_ranking.head(5).iterrows()):
        print(f"   {i+1}. {row['feature']}: {row['score']:.4f}")

    # 6. Create Final Optimized Dataset
    print(f"\n6. Creating Final Optimized Dataset:")
    
    # Combine metadata with optimized features
    # IMPORTANT: Include 'smiles'
    cols_to_keep = ['Especie', 'molecule_chembl_id', 'IC50', 'pIC50']
    if 'smiles' in df_clean.columns:
        cols_to_keep.append('smiles')
        
    df_optimized = df_clean.loc[X_final.index, cols_to_keep].copy()
    df_optimized = pd.concat([df_optimized, X_final], axis=1)
    
    print(f"   ✅ Optimized dataset shape: {df_optimized.shape}")

    # ========================================================================
    # PHASE 4: COMPARATIVE ANALYSIS
    # ========================================================================
    # (Visualizations generated in separate function)
    create_detailed_plots(df, df_optimized)

    # ========================================================================
    # PHASE 5: SAVE RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 5: SAVING RESULTS")
    print("="*80)
    
    with pd.ExcelWriter(OUTPUT_FILENAME, engine='openpyxl') as writer:
        df_optimized.to_excel(writer, sheet_name='Datos_Optimizados', index=False)
        
        metadata = pd.DataFrame({
            'Parameter': ['Original Samples', 'Final Samples', 'Original Features', 'Final Features', 'Target'],
            'Value': [len(df), len(df_optimized), len(numeric_features), n_features_available, target_name]
        })
        metadata.to_excel(writer, sheet_name='Metadata', index=False)
        
        feature_ranking.to_excel(writer, sheet_name='Selected_Features', index=False)
        
    print(f"   ✅ Dataset saved to: {OUTPUT_FILENAME}")
    
    return {
        'df_original': df,
        'df_optimized': df_optimized,
        'filename': OUTPUT_FILENAME
    }


def main():
    """
    Main execution function.
    """
    setup_directories()
    
    print(f"Looking for input file: {INPUT_FILENAME}")
    if not os.path.exists(INPUT_FILENAME):
        print(f"WARNING: {INPUT_FILENAME} not found.")
        # Check if user provided a path argument
        if len(sys.argv) > 1:
            input_path = sys.argv[1]
            print(f"Using provided path: {input_path}")
            df = load_dataset(input_path)
        else:
            print("Please ensure the file exists or provide the path as an argument.")
            return
    else:
        df = load_dataset(INPUT_FILENAME)
        
    comprehensive_dataset_analysis_and_optimization(df)
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
