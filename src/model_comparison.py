# src/model_comparison.py
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, auc, classification_report
from sklearn.model_selection import train_test_split
import joblib
import shap

class ModelComparator:
    def init(self, X_train, y_train, X_test, y_test, scoring='average_precision', 
                 n_jobs=-1, random_state=42):
        """
        Initialize the model comparator
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            scoring: Scoring metric for grid search
            n_jobs: Number of parallel jobs
            random_state: Random state for reproducibility
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def setup_models(self):
        """Set up the models with their parameter grids"""
        # Logistic Regression
        lr_params = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'class_weight': ['balanced', None]
        }
        
        # Random Forest
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }
        
        # XGBoost
        xgb_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'scale_pos_weight': [1, (len(self.y_train) - sum(self.y_train)) / sum(self.y_train)]
        }
        
        self.models = {
            'LogisticRegression': {
                'model': LogisticRegression(random_state=self.random_state),
                'params': lr_params
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': rf_params
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'params': xgb_params
            }
        }
    
    def compare_models(self, use_randomized=True, n_iter=20):
        """
        Compare models using grid search or randomized search
        
        Args:
            use_randomized: Whether to use randomized search (faster)
            n_iter: Number of iterations for randomized search
            
        Returns:
            Dictionary with comparison results
        """
        self.setup_models()
        self.results = {}
        
        for name, model_info in self.models.items():
            print(f"Training {name}...")
            start_time = time.time()
            
            if use_randomized:
                search = RandomizedSearchCV(
                    model_info['model'], model_info['params'], 
                    n_iter=n_iter, scoring=self.scoring, 
                    n_jobs=self.n_jobs, cv=3, random_state=self.random_state,
                    verbose=1
                )
            else:
                search = GridSearchCV(
                    model_info['model'], model_info['params'], 
                    scoring=self.scoring, n_jobs=self.n_jobs, 
                    cv=3, verbose=1
                )
            
            # Train the model
            search.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = search.best_estimator_.predict(self.X_test)
            y_pred_proba = search.best_estimator_.predict_proba(self.X_test)[:, 1]
            
            # Calculate precision-recall AUC
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            # Calculate business impact
            tn, fp, fn, tp = self._calculate_confusion_matrix(self.y_test, y_pred)
            business_impact = self._calculate_business_impact(fp, fn)
            
            # Store results
            self.results[name] = {
                'model': search.best_estimator_,
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'pr_auc': pr_auc,
                'business_impact': business_impact,
                'training_time': time.time() - start_time,
                'confusion_matrix': (tn, fp, fn, tp)
            }
            
            print(f"{name} completed in {time.time() - start_time:.2f} seconds")
            print(f"Best {self.scoring} score: {search.best_score_:.4f}")
            print(f"Precision-Recall AUC: {pr_auc:.4f}")
            print(f"Business impact: ${business_impact:,.2f}")
            print("-" * 50)
        
        return self.results
    
    def _calculate_confusion_matrix(self, y_true, y_pred):
        """Calculate confusion matrix components"""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        return tn, fp, fn, tp
    
    def _calculate_business_impact(self, fp, fn):
        """
        Calculate business impact based on the problem statement
        False positives cost 5x more than false negatives
        """
        # Assuming base cost of $1 for false negatives
        false_negative_cost = fn * 1
        false_positive_cost = fp * 5  # 5x more expensive
        return false_negative_cost + false_positive_cost
    
    def select_best_model(self):
        """Select the best model based on multiple criteria"""
        if not self.results:
            raise ValueError("No results available. Run compare_models first.")
        
        # Create a combined score considering both PR AUC and business impact
        # We want to maximize PR AUC and minimize business impact
        best_score = -float('inf')
        best_model_name = None
        
        for name, result in self.results.items():
            # Normalize business impact (lower is better)
            max_impact = max(r['business_impact'] for r in self.results.values())
            normalized_impact = 1 - (result['business_impact'] / max_impact)
            
            # Combined score (weighted average)
            combined_score = 0.7 * result['pr_auc'] + 0.3 * normalized_impact
            
            if combined_score > best_score:
                best_score = combined_score
                best_model_name = name
        
        return best_model_name, self.results[best_model_name]
    
    def explain_model(self, model, X, feature_names):
        """Generate SHAP explanations for the model"""
        explainer = shap.Explainer(model, X, feature_names=feature_names)
        shap_values = explainer(X)
        return shap_values
    
    def save_model(self, model, filename):
        """Save the model to a file"""
        joblib.dump(model, filename)