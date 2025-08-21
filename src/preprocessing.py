# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

def create_preprocessor():
    """
    Create a preprocessing pipeline
    """
    # Define numeric and categorical features
    numeric_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                       'oldbalanceDest', 'newbalanceDest']
    categorical_features = ['type']
    
    # Create preprocessing transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def load_and_preprocess_data(filepath, sampling_method=None):
    """
    Load and preprocess the transaction data
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Remove leakage columns and identifiers
    df = df.drop(['isFlaggedFraud', 'nameOrig', 'nameDest'], axis=1, errors='ignore')
    
    # Separate features and target
    X = df.drop('isFraud', axis=1)
    y = df['isFraud']
    
    # Create and fit preprocessor
    preprocessor = create_preprocessor()
    X_processed = preprocessor.fit_transform(X)
    
    # Apply resampling if specified
    if sampling_method == 'smote':
        sampler = SMOTE(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X_processed, y)
        return X_resampled, y_resampled, preprocessor
    elif sampling_method == 'nearmiss':
        sampler = NearMiss(version=2, n_neighbors=3)
        X_resampled, y_resampled = sampler.fit_resample(X_processed, y)
        return X_resampled, y_resampled, preprocessor
    
    return X_processed, y, preprocessor