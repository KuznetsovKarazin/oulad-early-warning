import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV


class LogisticModel:
    """Logistic Regression model with calibration and interpretability"""
    
    def __init__(self, C=1.0, max_iter=1000, calibrate=True):
        self.C = C
        self.max_iter = max_iter
        self.calibrate = calibrate
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.coefficients = None
        self.odds_ratios = None
    
    def _select_features(self, X):
        """Select numeric features for modeling"""
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
        """Train logistic regression model"""
        X_feat = self._select_features(X)
        self.feature_names = X_feat.columns.tolist()
        
        # Handle missing values
        X_feat = X_feat.fillna(X_feat.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_feat)
        
        # Train base model
        base_model = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=42)
        
        # Calibrate if requested
        if self.calibrate:
            self.model = CalibratedClassifierCV(base_model, cv=5, method='sigmoid')
        else:
            self.model = base_model
        
        self.model.fit(X_scaled, y)
        
        # Extract coefficients and odds ratios
        if hasattr(self.model, 'calibrated_classifiers_'):
            # Get average coefficients across CV folds
            coefs = [clf.estimator.coef_[0] for clf in self.model.calibrated_classifiers_]
            self.coefficients = np.mean(coefs, axis=0)
        else:
            self.coefficients = self.model.coef_[0]
        
        self.odds_ratios = np.exp(self.coefficients)
        
        return self
    
    def predict_proba(self, X):
        """Predict calibrated probabilities"""
        X_feat = self._select_features(X)
        X_feat = X_feat.fillna(X_feat.median())
        X_scaled = self.scaler.transform(X_feat)
        return self.model.predict_proba(X_scaled)
    
    def predict(self, X, threshold=0.5):
        """Predict at-risk students"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def get_feature_importance(self):
        """Get feature importance as DataFrame"""
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.coefficients,
            'odds_ratio': self.odds_ratios,
            'abs_coefficient': np.abs(self.coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance_df
    
    def explain_prediction(self, X, idx):
        """Explain prediction for a single student"""
        X_feat = self._select_features(X.iloc[[idx]])
        X_feat = X_feat.fillna(X_feat.median())
        X_scaled = self.scaler.transform(X_feat)
        
        contributions = X_scaled[0] * self.coefficients
        
        explanation = pd.DataFrame({
            'feature': self.feature_names,
            'value': X_feat.iloc[0].values,
            'scaled_value': X_scaled[0],
            'coefficient': self.coefficients,
            'contribution': contributions
        }).sort_values('contribution')
        
        return explanation