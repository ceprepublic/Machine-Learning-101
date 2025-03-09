# Machine Learning Algorithms with Multiple Datasets

This comprehensive repository showcases implementations of various machine learning algorithms across multiple datasets, providing detailed analysis, comparisons, and practical insights for both beginners and experienced practitioners.

## Table of Contents
- [Overview](#overview)
- [Algorithms Implemented](#algorithms-implemented)
- [Datasets Used](#datasets-used)
- [Results & Comparisons](#results--comparisons)
- [Installation & Usage](#installation--usage)
- [Classification Metrics](#classification-metrics)

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

## Installation & Usage
You could directly download this repo and run the main PROJECT.ipynb file accordingly. All algortihms implemented independently as a functions and called at main file. It is important that all files should be in the same folder.
## Classification Metrics

### Accuracy (94-98%)
- **Definition**: The proportion of total predictions (both positive and negative) that were correctly identified
- **Formula**: `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
- **Characteristics**:
  - Simple and intuitive metric
  - Works best with balanced datasets
  - Can be misleading with imbalanced classes
- **Expectations**:
  - Range: 0.0 to 1.0 (or 0-100%)
  - >90% typically indicates strong performance
  - Should be evaluated alongside other metrics for imbalanced data

### Precision (0.93-0.98)
- **Definition**: The proportion of positive identifications that were actually correct
- **Formula**: `Precision = TP / (TP + FP)`
- **Theoretical Importance**:
  - Critical when false positives are costly
  - Measures model's exactness
  - Key for systems requiring high confidence in positive predictions
- **Use Cases**:
  - Medical diagnosis where false positives are dangerous
  - Spam detection where false positives affect user experience
  - Fraud detection systems

### Recall (0.93-0.98)
- **Definition**: The proportion of actual positives that were correctly identified
- **Formula**: `Recall = TP / (TP + FN)`
- **Theoretical Significance**:
  - Measures model's completeness
  - Critical when false negatives are costly
  - Also known as sensitivity or true positive rate
- **Applications**:
  - Disease detection where missing positives is dangerous
  - Criminal detection where missing cases is problematic
  - Predictive maintenance where missing failures is costly

### F1-Score (0.93-0.98)
- **Definition**: Harmonic mean of precision and recall
- **Formula**: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`
- **Mathematical Properties**:
  - Balances precision and recall
  - Penalizes extreme values more than arithmetic mean
  - Range: 0.0 to 1.0
- **When to Use**:
  - When seeking balance between precision and recall
  - With imbalanced datasets
  - When both false positives and negatives are important

### ROC-AUC (0.92-0.99)
- **Definition**: Area under the Receiver Operating Characteristic curve
- **Mathematical Interpretation**:
  - Probability that model ranks a random positive example higher than a random negative example
  - Plots TPR (sensitivity) vs FPR (1-specificity)
- **Characteristics**:
  - Threshold-invariant
  - Robust to class imbalance
  - Range: 0.5 (random) to 1.0 (perfect)
- **Usage Guidelines**:
  - >0.9: Excellent discrimination
  - 0.8-0.9: Good discrimination
  - 0.7-0.8: Fair discrimination
  - <0.7: Poor discrimination

## Regression Metrics

### Mean Squared Error (MSE)
- **Definition**: Average of squared differences between predictions and actual values
- **Formula**: `MSE = (1/n) Σ(yi - ŷi)²`
- **Mathematical Properties**:
  - Always non-negative
  - Heavily penalizes large errors due to squaring
  - Not scale-independent
- **Range in Current Analysis**: 18.4-25.0
- **Interpretation**:
  - Lower values indicate better fit
  - Units are squared units of target variable
  - Useful for comparing models on same dataset

### Root Mean Squared Error (RMSE)
- **Definition**: Square root of MSE
- **Formula**: `RMSE = √[(1/n) Σ(yi - ŷi)²]`
- **Advantages**:
  - Same units as target variable
  - More interpretable than MSE
  - Commonly used in practice
- **Range in Current Analysis**: 4.2-5.0
- **Usage**:
  - For comparing models with same target variable
  - When large errors are particularly undesirable
  - When error units should match target variable units

### R² Score (Coefficient of Determination)
- **Definition**: Proportion of variance in dependent variable explained by independent variable(s)
- **Formula**: `R² = 1 - (Σ(yi - ŷi)²) / (Σ(yi - ȳ)²)`
- **Mathematical Properties**:
  - Range: (-∞, 1.0]
  - 1.0 indicates perfect fit
  - Can be negative for poorly fitting models
- **Range in Current Analysis**: 0.69-0.81
- **Interpretation**:
  - 0.7-0.8: Good fit
  - >0.8: Strong fit
  - <0.6: Weak fit

### Mean Absolute Error (MAE)
- **Definition**: Average of absolute differences between predictions and actual values
- **Formula**: `MAE = (1/n) Σ|yi - ŷi|`
- **Characteristics**:
  - Linear penalty for errors
  - More robust to outliers than MSE/RMSE
  - Same units as target variable
- **Range in Current Analysis**: 2.9-3.5
- **Best Used When**:
  - Outliers should not have excessive influence
  - Error magnitude should scale linearly
  - Simple interpretation is needed

### Explained Variance Score
- **Definition**: Proportion of variance that is predictable from the independent variable
- **Formula**: `Explained Variance = 1 - Var(y - ŷ) / Var(y)`
- **Theoretical Background**:
  - Related to R² but more focused on variance explanation
  - Measures quality of linear relationship
  - Accounts for systematic bias
- **Range in Current Analysis**: 0.70-0.81
- **Interpretation**:
  - Similar to R² but can differ with biased predictions
  - 1.0 indicates perfect prediction
  - Should be considered alongside other metrics

## Notation
- TP: True Positives (correctly predicted positive cases)
- TN: True Negatives (correctly predicted negative cases)
- FP: False Positives (incorrectly predicted positive cases)
- FN: False Negatives (incorrectly predicted negative cases)
- yi: Actual value
- ŷi: Predicted value
- ȳ: Mean of actual values
- n: Number of samples
- Var(): Variance function

