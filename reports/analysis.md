# Fraud Detection Analysis Report

## Preprocessing
- Used ColumnTransformer with StandardScaler for numeric features and OneHotEncoder for categorical features
- Handled missing values using SimpleImputer with median strategy
- Removed isFlaggedFraud column to prevent data leakage and identifier columns (nameOrig, nameDest)

## Class Imbalance Handling
- Algorithmic approach: Used class_weight='balanced' in Logistic Regression and Random Forest
- Resampling approach: Applied **SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic fraud samples

## Models Compared
1. Logistic Regression with L2 regularization and balanced class weighting
2. Random Forest with balanced class weighting and 100 estimators  
3. XGBoost with scale_pos_weight for imbalance handling

## Results
- Best Model: XGBoost achieved 0.9998 PR-AUC
- Fraud Detection Rate: 99.73% of fraudulent transactions identified
- Business Impact: $43,730 total error cost with $12.6M savings vs blocking all transactions

## Key Insight
Used Precision-Recall AUC instead of accuracy because fraud detection requires measuring rare class detection capability, not overall accuracy which would be misleading with 0.2% fraud rate.