import pandas as pd
import numpy as np


class BaselineModel:
    """Simple threshold-based early warning rules"""
    
    def __init__(self, 
                 min_clicks=50, 
                 min_active_days_ratio=0.2,
                 require_assessment_completion=False):
        self.min_clicks = min_clicks
        self.min_active_days_ratio = min_active_days_ratio
        self.require_assessment_completion = require_assessment_completion
        self.thresholds = {}
    
    def fit(self, X, y=None):
        """Store thresholds (baseline doesn't learn)"""
        self.thresholds = {
            'min_clicks': self.min_clicks,
            'min_active_days_ratio': self.min_active_days_ratio,
            'require_assessment': self.require_assessment_completion
        }
        return self
    
    def predict_proba(self, X):
        """Return risk probabilities (0=low risk, 1=high risk)"""
        n = len(X)
        proba = np.zeros((n, 2))
        
        # Calculate risk score based on rules
        risk_flags = []
        
        # Rule 1: Low activity
        low_activity = X['total_clicks'] < self.min_clicks
        risk_flags.append(low_activity)
        
        # Rule 2: Low active days ratio
        low_engagement = X['active_days_ratio'] < self.min_active_days_ratio
        risk_flags.append(low_engagement)
        
        # Rule 3: Assessment not completed (optional)
        if self.require_assessment_completion:
            no_assessment = X['completed_first_assessment'] == 0
            risk_flags.append(no_assessment)
        
        # Combine flags: risk = proportion of rules violated
        n_rules = len(risk_flags)
        risk_score = sum(risk_flags) / n_rules
        
        # Convert to probabilities (class 0 = pass, class 1 = fail)
        proba[:, 1] = risk_score  # Probability of failure
        proba[:, 0] = 1 - risk_score  # Probability of passing
        
        return proba
    
    def predict(self, X, threshold=0.5):
        """Predict at-risk students (1=at risk, 0=not at risk)"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def get_alert_reasons(self, X):
        """Get reasons for each alert"""
        reasons = []
        
        for idx, row in X.iterrows():
            student_reasons = []
            
            if row['total_clicks'] < self.min_clicks:
                student_reasons.append(f"Low activity: {row['total_clicks']:.0f} clicks (< {self.min_clicks})")
            
            if row['active_days_ratio'] < self.min_active_days_ratio:
                student_reasons.append(f"Low engagement: {row['active_days_ratio']:.2f} active days ratio (< {self.min_active_days_ratio})")
            
            if self.require_assessment_completion and row['completed_first_assessment'] == 0:
                student_reasons.append("First assessment not completed")
            
            reasons.append("; ".join(student_reasons) if student_reasons else "No risk flags")
        
        return reasons