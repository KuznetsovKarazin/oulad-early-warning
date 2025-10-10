import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV


class BoostingModel:
    """Gradient Boosting model with LightGBM"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5, calibrate=True):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.calibrate = calibrate
        self.model = None
        self.feature_names = None
        self.feature_importance = None
    
    def _select_features(self, X):
        """Select features for modeling"""
        feature_cols = [
            'total_clicks', 'log_total_clicks', 'active_days_ratio',
            'avg_clicks_per_day', 'days_since_last_activity',
            'completed_first_assessment', 'first_assessment_score_norm',
            'education_level', 'age_group', 'imd_numeric',
            'num_of_prev_attempts', 'is_female', 'has_disability',
            'credit_load_norm'
        ]
        
        available_features = [f for f in feature_cols if f in X.columns]
        return X[available_features].copy()
    
    def fit(self, X, y):
        """Train gradient boosting model"""
        X_feat = self._select_features(X)
        self.feature_names = X_feat.columns.tolist()
        
        # Handle missing values
        X_feat = X_feat.fillna(X_feat.median())
        
        # Train base model
        base_model = LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42,
            verbose=-1
        )
        
        # Calibrate if requested
        if self.calibrate:
            self.model = CalibratedClassifierCV(base_model, cv=5, method='sigmoid')
        else:
            self.model = base_model
        
        self.model.fit(X_feat, y)
        
        # Extract feature importance
        if hasattr(self.model, 'calibrated_classifiers_'):
            # Average importance across CV folds
            importances = [clf.estimator.feature_importances_ for clf in self.model.calibrated_classifiers_]
            self.feature_importance = np.mean(importances, axis=0)
        else:
            self.feature_importance = self.model.feature_importances_
        
        return self
    
    def predict_proba(self, X):
        """Predict calibrated probabilities"""
        X_feat = self._select_features(X)
        X_feat = X_feat.fillna(X_feat.median())
        return self.model.predict_proba(X_feat)
    
    def predict(self, X, threshold=0.5):
        """Predict at-risk students"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def get_feature_importance(self):
        """Get feature importance as DataFrame"""
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df