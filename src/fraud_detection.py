# src/fraud_detection.py
import pandas as pd
import numpy as np
import argparse
import joblib
from sklearn.model_selection import train_test_split
from preprocessing import DataPreprocessor
from model_comparison import ModelComparator

def load_data(filepath):
    """Load transaction data from CSV file"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Remove isFlaggedFraud column to prevent data leakage
    if 'isFlaggedFraud' in df.columns:
        df = df.drop('isFlaggedFraud', axis=1)
    
    return df

def prepare_data(df):
    """Prepare data for modeling"""
    # Separate features and target
    X = df.drop('isFraud', axis=1)
    y = df['isFraud']
    
    # Remove identifier columns that shouldn't be used as features
    if 'nameOrig' in X.columns:
        X = X.drop('nameOrig', axis=1)
    if 'nameDest' in X.columns:
        X = X.drop('nameDest', axis=1)
    
    return X, y

def main():
    """Main function for fraud detection"""
    parser = argparse.ArgumentParser(description='Fraud Detection System')
    parser.add_argument('--data', type=str, default='data/transactions.csv', 
                       help='Path to transaction data')
    parser.add_argument('--sampling', type=str, default='smote', 
                       choices=['smote', 'nearmiss', 'none'],
                       help='Sampling method for class imbalance')
    parser.add_argument('--output', type=str, default='models/fraud_model.pkl',
                       help='Output path for the trained model')
    args = parser.parse_args()
    
    # Load and prepare data
    df = load_data(args.data)
    X, y = prepare_data(df)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocess data
    preprocessor = DataPreprocessor(sampling_method=args.sampling)
    X_train_processed, y_train_processed = preprocessor.fit_resample(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Compare models
    comparator = ModelComparator(X_train_processed, y_train_processed, 
                                X_test_processed, y_test)
    results = comparator.compare_models(use_randomized=True, n_iter=20)
    
    # Select best model
    best_model_name, best_model_info = comparator.select_best_model()
    print(f"\nBest model: {best_model_name}")
    print(f"Precision-Recall AUC: {best_model_info['pr_auc']:.4f}")
    print(f"Business impact: ${best_model_info['business_impact']:,.2f}")
    
    # Save the best model
    comparator.save_model(best_model_info['model'], args.output)
    print(f"Model saved to {args.output}")
    
    # Generate SHAP explanations
    feature_names = preprocessor.get_feature_names()
    shap_values = comparator.explain_model(
        best_model_info['model'], X_test_processed[:100], feature_names
    )
    
    # Save SHAP summary plot
    import  matplotlib.pyplot as plt
    import shap
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_processed[:100], feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('reports/shap_summary.png')
    plt.close()
    
    print("SHAP analysis completed and saved to reports/shap_summary.png")

if name == "main": # pyright: ignore[reportUndefinedVariable]
    main()