# QSAR Pipeline User Manual

## Overview
This pipeline allows you to build QSAR models to predict pIC50 values for chemical compounds using XGBoost. It is designed to be easy to use via a terminal interface that handles the complex steps for you.

## 🖥️ The Interface
The main way to use this tool is through the `pipeline_interface.py` script.

### Starting the Interface
Open your terminal in the project folder and run:
```bash
./pipeline_interface.py
```

### The Menu System
The interface shows you a status table with 3 steps:

| Step | Description | Status Meaning |
|------|-------------|----------------|
| **1. Optimization** | Cleans data & selects features | **MISSING INPUT**: You need the raw file in `data/raw`. <br> **READY**: Input found, ready to run. <br> **COMPLETED**: Optimizd file exists. |
| **2. Training** | Trains the AI model | **BLOCKED**: Needs Step 1 done first. <br> **OUTDATED**: Data changed, you should retrain. |
| **3. Prediction** | Predicts for new compounds | **BLOCKED**: Needs a trained model. <br> **READY**: Good to go. |

## 📝 Detailed Steps

### Step 1: Dataset Optimization (`dataset_optimizer.py`)
- **What it does**: Reads `data/raw/all_descriptor_results_1751.xlsx`, removes outliers, selects the best features, and creates `data/processed/dataset_molecular_optimizado.xlsx`.
- **When to run**: When you have new raw training data or want to reset the dataset.

### Step 2: Model Training (`xgboost_optimizer.py`)
- **What it does**: Uses the optimized data to train an XGBoost model. It searches for the best "hyperparameters" (settings) to make the model accurate.
- **Output**: Saves the model to `results/model_metadata/` and validation plots to `results/`.
- **Note**: This can take a few minutes depending on the settings.

### Step 3: Prediction (`molecular_predictor.py`)
- **What it does**: Reads `data/raw/new_compounds.xlsx` and predicts pIC50 values using the *latest* trained model.
- **Applicability Domain**: It checks if new compounds are "similar enough" to the training data. If a compound is too different (High Mahalanobis Distance), it is marked as "Out of Domain".
- **Output**: Saves an Excel file with predictions to `results/predictions/` and images of the top candidates to `results/plots/`.

## ❓ Troubleshooting

### "File Not Found" Errors
- Check `data/raw/`. You must have:
    - `all_descriptor_results_1751.xlsx` (for training phase)
    - `new_compounds.xlsx` (for prediction phase)

### Environment Issues
- The script uses the python environment it is run in.
- Best practice is to create a fresh conda environment:
  ```bash
  conda create -n qsar_env python=3.10
  conda activate qsar_env
  pip install -e .
  ```

### Dependencies
- Most dependencies (like `pandas`, `xgboost`, `rich`, `tqdm`) are automatically installed when you run `pip install -e .`.
