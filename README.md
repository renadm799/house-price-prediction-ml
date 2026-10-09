# House Price Prediction | Machine Learning

## Project Overview

This project predicts residential house prices using machine learning techniques. It demonstrates the complete machine learning workflow, including data exploration, feature engineering, data preprocessing, model training, cross-validation, and model evaluation.

The goal was to compare different regression models, minimize prediction error, and select a reliable model while preventing data leakage.

## Dataset

**Source:** Kaggle — House Prices: Advanced Regression Techniques

- Training data: 1,460 houses
- Test data: 1,459 houses
- Target variable: `SalePrice`
- Problem type: Supervised Learning — Regression

The target variable was transformed using `log1p()` to reduce skewness and improve model training.

## Tools and Technologies

- **Python:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Preprocessing:** SimpleImputer, OneHotEncoder, ColumnTransformer
- **Model Evaluation:** K-Fold Cross-Validation, RMSE
- **Environment:** Kaggle Notebooks

## Project Workflow

### 1. Data Preparation

- Separated the target variable from the input features.
- Handled missing numerical values using median imputation.
- Handled missing categorical values using constant-value imputation.
- Encoded categorical features using one-hot encoding.
- Built preprocessing pipelines to prevent data leakage during cross-validation.

### 2. Feature Engineering

Created additional features to capture relationships between house characteristics and sale prices, including:

- `TotalSF`: Total basement and above-ground floor area
- `TotalBath`: Combined number of bathrooms
- `HouseAge`: Age of the property at sale
- `RemodAge`: Years since remodeling
- `GarageAge`: Age of the garage
- `GrLiv_Overall`: Living area multiplied by overall quality
- `TotalSF_Overall`: Total floor area multiplied by overall quality

### 3. Model Development

Trained and compared three regression algorithms:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Models were evaluated using 5-fold cross-validation with preprocessing performed separately within each training fold.

## Model Performance

| Model | Cross-Validation RMSE |
|---|---:|
| XGBoost | **0.1305** |
| Random Forest | 0.1430 |
| Linear Regression | 0.1469 |

*Lower RMSE indicates better predictive performance. Results are based on the log-transformed target variable.*

## Stacking Ensemble Experiment

An additional stacking ensemble combined:

- Random Forest
- XGBoost
- Ridge Regression
- Lasso Regression

Ridge Regression was used as the final meta-model.

| Approach | CV RMSE |
|---|---:|
| XGBoost | **0.1305** |
| Stacking Ensemble | 0.1308 |

The stacking ensemble achieved comparable performance but did not improve the average cross-validation result.

**XGBoost was selected as the final model** because it achieved slightly lower validation error with a simpler modeling approach.

## Final Output

The selected XGBoost pipeline was trained on the full training dataset and used to generate predictions for 1,459 test houses.

The predictions were transformed back to the original price scale and exported to `submission.csv`, containing the columns `Id` and `SalePrice`.

## Key Skills Demonstrated

- Data Cleaning and Preprocessing
- Feature Engineering
- Regression Modeling
- Machine Learning Pipelines
- Cross-Validation and Model Evaluation
- Data Leakage Prevention
- Ensemble Learning
- Model Selection

## Project Notebook

The complete Python implementation, experiments, explanations, and results are available in the Jupyter Notebook included in this repository.
