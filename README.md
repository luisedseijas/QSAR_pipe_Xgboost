# QSAR_Andres: Advanced QSAR & pIC50 Prediction Pipeline

**A robust, automated Python pipeline for Quantitative Structure-Activity Relationship (QSAR) analysis and XGBoost model optimization.**

This project provides a complete end-to-end workflow for discovering potential drug candidates. It takes molecular descriptors as input, optimizes the dataset, trains a high-performance XGBoost model using genetic-like hyperparameter tuning, and predicts pIC50 values for new compounds with strict Applicability Domain (AD) checks.

---

## 🚀 Key Features

- **🧠 Intelligent CLI**: A central command interface (`pipeline_interface.py`) guided by rich UI to manage the entire lifecycle.
- **🧹 Automated Data Engineering**: 
    - Removes low-variance features.
    - Filters highly correlated descriptors ($r > 0.9$) to prevent multicollinearity.
    - Standardizes data automatically.
- **⚡ XGBoost Optimization**: 
    - Uses `GridSearchCV` to explore a wide hyperparameter space (`max_depth`, `learning_rate`, `n_estimators`, etc.).
    - Implements K-Fold Cross-Validation to ensure model robustness.
- **🛡️ Applicability Domain (AD)**: 
    - **PCA-based AD**: Calculates the Mahalanobis distance in the Principal Component space.
    - **Reliability Flagging**: Automatically flags predictions as "Out of Domain" if they fall outside the chemical space of the training set.
- **📊 Advanced Visualization**: 
    - Generates correlation heatmaps, feature importance plots, and "Predicted vs Actual" regression plots.
    - Visualizes top chemical structures in a grid.

---

## 🔬 Scientific Workflow

The pipeline is divided into three distinct phases, ensuring scientific rigor at each step.

### Phase 1: Dataset Optimization (`dataset_optimizer.py`)
**Goal**: Prepare a clean, high-quality dataset for machine learning.
1.  **Ingestion**: Reads raw descriptor data from `data/raw/`.
2.  **Cleaning**:
    - **Variance Threshold**: Removes features that have the same value for >95% of compounds.
    - **Correlation Filtering**: Identifies pairs of features with correlation > 0.9 and removes one of them to reduce redundancy.
3.  **Output**: Produces `data/processed/dataset_molecular_optimizado.xlsx`.

### Phase 2: Model Training (`xgboost_optimizer.py`)
**Goal**: Train a predictive model that generalizes well to unseen data.
1.  **Data Splitting**: Splits data into Training (80%) and Test (20%) sets.
2.  **Hyperparameter Tuning**: Performs a Grid Search over parameters like:
    - `max_depth` (Tree depth)
    - `learning_rate` (Step size)
    - `subsample` (Fraction of samples used per tree)
3.  **Validation**: Evaluates the best model using $R^2$ (Coefficient of Determination) and RMSE (Root Mean Squared Error).
4.  **Model Persistence**: Saves the trained model (`.json`) and specific scalers (for standardization) to `results/model_metadata/`.


### Phase 3: Prediction & Applicability Domain (`molecular_predictor.py`)
**Goal**: Screen new compounds and assess prediction reliability.
1.  **Standardization**: Applies the *exact same* scaling parameters used during training.
2.  **Applicability Domain Check**:
    - Projects the new compound into the PCA space defined by the training set.
    - Calculates the squared Mahalanobis distance ($d^2$).
    - If $d^2 > \text{Threshold}$, the prediction is marked **Out of Domain**.
3.  **Result**: Outputs pIC50 predictions and top candidate visualizations to `results/`.

---

## 🛠️ Installation & Usage

### 1. Environment Setup
It is best practice to use a dedicated Conda environment.

```bash
# Create environment
conda create -n qsar_env python=3.10
conda activate qsar_env

# Install the package and dependencies
pip install -e .
```

### 2. Running the Pipeline
Launch the interactive dashboard:

```bash
./pipeline_interface.py
```
*(Or `python pipeline_interface.py`)*

### 3. directory Structure

- **`src/qsar_pipeline/`**: Core Python package logic.
- **`data/`**:
    - `raw/`: Place your `all_descriptor_results_1751.xlsx` and `new_compounds.xlsx` here.
    - `processed/`: Optimized datasets are saved here.
- **`results/`**: 
    - `model_metadata/`: Trained XGBoost models and JSON metadata.
    - `predictions/`: Excel files with pIC50 predictions and AD flags.
    - `plots/`: Performance comparisons and structure grids.

---

## 📚 Documentation
For a detailed step-by-step guide on how to operate the menu and troubleshoot issues, please refer to [MANUAL.md](MANUAL.md).

---

**Authors**: Luis E. Seijas & Andres Cervantes

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

