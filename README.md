# QSAR_Andres: pIC50 Prediction Pipeline

A comprehensive Python pipeline for QSAR analysis and XGBoost model optimization. This project provides a robust, automated workflow for predicting pIC50 values from molecular descriptors, managed by an intelligent terminal interface.

## 🚀 Key Features

- **Intelligent CLI**: A central `pipeline_interface.py` that manages the workflow and checks dependencies.
- **Automated Data Pipeline**: From raw descriptors to optimized feature sets (`dataset_optimizer.py`).
- **XGBoost Optimization**: Robust hyperparameter tuning with `GridSearchCV` (`xgboost_optimizer.py`).
- **New Compound Prediction**: Predicts pIC50 for new compounds using PCA-based Applicability Domain (`molecular_predictor.py`).
- **Visualization**: Generates structure grid images and analysis plots.
- **Artifact Management**: Automatically saves models, metadata, and results in a structured format.

## 🛠️ Quick Start

This project is configured to run in the `IC-50` conda environment.

**Run the pipeline interface:**
```bash
./pipeline_interface.py
```
*(Or `python pipeline_interface.py`)*

This will launch a menu where you can check the status of each step and run them.

## 📂 Project Structure

- **`pipeline_interface.py`**: **MAIN ENTRY POINT**. Manages the entire workflow.
- **`src/`**: Contains the core logic scripts (managed by the interface).
    - `dataset_optimizer.py`: **Phase 1**. Preprocessing & feature selection.
    - `xgboost_optimizer.py`: **Phase 2**. Model training & optimization.
    - `molecular_predictor.py`: **Phase 3**. New compound prediction.
- **`data/`**:
    - **`raw/`**: Input Excel files.
    - **`processed/`**: Intermediate optimized datasets.
- **`results/`**: Output directory for models, plots, and predictions.
- **`MANUAL.md`**: Detailed user manual and troubleshooting guide.
