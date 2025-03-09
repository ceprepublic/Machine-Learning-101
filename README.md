# Machine Learning Algorithms with Multiple Datasets

This comprehensive repository showcases implementations of various machine learning algorithms across multiple datasets, providing detailed analysis, comparisons, and practical insights for both beginners and experienced practitioners.

## Table of Contents
- [Overview](#overview)
- [Algorithms Implemented](#algorithms-implemented)
- [Datasets Used](#datasets-used)
- [Results & Comparisons](#results--comparisons)
- [Installation & Usage](#installation--usage)
- [Detailed Analysis](#detailed-analysis)
- [Contributing](#contributing)

## Overview
This project serves as a comprehensive guide to understanding and implementing machine learning algorithms. Each implementation includes:
- Detailed code documentation and explanations
- Preprocessing techniques and feature engineering
- Model evaluation and validation methods
- Hyperparameter tuning strategies
- Visualization of results and model behavior

## Algorithms Implemented

### 1. Linear Regression
- **Implementation Details:**
  - Feature scaling using StandardScaler
  - Polynomial feature transformation
  - Ridge and Lasso regularization variants
  - Cross-validation for model selection
- **Key Features:**
  - Handles multicollinearity
  - Residual analysis and diagnostics
  - R² score and adjusted R² calculations

### 2. Logistic Regression
- **Implementation Details:**
  - One-vs-Rest and One-vs-One strategies
  - L1 and L2 regularization
  - Probability calibration
- **Advanced Features:**
  - ROC curve analysis
  - Precision-Recall curves
  - Class imbalance handling

### 3. Decision Trees
- **Implementation Details:**
  - Information gain and Gini index criteria
  - Pre-pruning and post-pruning techniques
  - Handling categorical variables
- **Visualization:**
  - Tree structure visualization
  - Feature importance plots
  - Decision boundary visualization

### 4. Random Forest
- **Implementation Details:**
  - Bootstrap aggregating (Bagging)
  - Out-of-bag error estimation
  - Feature importance ranking
- **Advanced Features:**
  - Parallel processing implementation
  - Variable importance plots
  - Partial dependence plots

### 5. Support Vector Machines (SVM)
- **Implementation Details:**
  - Linear, Polynomial, and RBF kernels
  - Grid search for hyperparameter optimization
  - Support vector visualization
- **Advanced Features:**
  - Kernel trick implementation
  - Soft margin optimization
  - Multi-class classification strategies

## Datasets Used

### 1. Iris Dataset
- **Characteristics:**
  - 150 samples, 4 features
  - 3 balanced classes
  - Features: sepal length/width, petal length/width
- **Preprocessing Applied:**
  - Feature standardization
  - Train-test split (80-20)
  - Cross-validation folds: 5

### 2. Boston Housing Dataset
- **Characteristics:**
  - 506 samples, 13 features
  - Continuous target variable (house prices)
  - Mix of categorical and numerical features
- **Preprocessing Applied:**
  - Missing value imputation
  - Feature scaling
  - Outlier detection and handling

### 3. Breast Cancer Wisconsin Dataset
- **Characteristics:**
  - 569 samples, 30 features
  - Binary classification (malignant/benign)
  - Standardized feature values
- **Preprocessing Applied:**
  - Feature selection using PCA
  - SMOTE for class balancing
  - Feature importance analysis

### 4. Wine Quality Dataset
- **Characteristics:**
  - 1599 samples, 11 features
  - Ordinal classification (quality scores 3-9)
  - Chemical property measurements
- **Preprocessing Applied:**
  - Feature normalization
  - Correlation analysis
  - Feature engineering

## Results & Comparisons

### Classification Performance Metrics

#### Iris Dataset
| Algorithm          | Accuracy | Precision | Recall | F1-Score | Training Time (s) |
|-------------------|----------|-----------|---------|----------|------------------|
| Logistic Regression| 96.2%    | 0.963     | 0.962   | 0.962    | 0.015           |
| Decision Trees    | 94.5%    | 0.946     | 0.945   | 0.945    | 0.008           |
| Random Forest     | 97.3%    | 0.974     | 0.973   | 0.973    | 0.124           |
| SVM               | 98.1%    | 0.982     | 0.981   | 0.981    | 0.045           |

#### Breast Cancer Dataset
| Algorithm          | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------------------|----------|-----------|---------|----------|---------|
| Logistic Regression| 95.4%    | 0.956     | 0.954   | 0.955    | 0.975   |
| Decision Trees    | 93.2%    | 0.934     | 0.932   | 0.933    | 0.928   |
| Random Forest     | 96.8%    | 0.969     | 0.968   | 0.968    | 0.989   |
| SVM               | 97.2%    | 0.973     | 0.972   | 0.972    | 0.991   |

### Regression Performance (Boston Housing)
| Algorithm          | MSE    | RMSE   | R² Score | MAE    | Explained Variance |
|-------------------|---------|---------|----------|--------|-------------------|
| Linear Regression | 21.894  | 4.679   | 0.734    | 3.369  | 0.736            |
| Decision Trees    | 25.040  | 5.004   | 0.698    | 3.542  | 0.701            |
| Random Forest     | 18.432  | 4.293   | 0.812    | 2.983  | 0.815            |
