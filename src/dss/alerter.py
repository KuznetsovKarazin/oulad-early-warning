import pandas as pd
import numpy as np


class StudentAlerter:
    """Generate student alerts based on model predictions"""
    
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold
    
    def generate_alerts(self, X):
        """Generate alert list with probabilities"""
        
        # Get predictions
        proba = self.model.predict_proba(X)[:, 1]
        predictions = (proba >= self.threshold).astype(int)
        
        # Create alert dataframe
        alert_df = X[['code_module', 'code_presentation', 'id_student']].copy()
        alert_df['risk_probability'] = proba
        alert_df['at_risk_flag'] = predictions
        alert_df['risk_level'] = pd.cut(
            proba,
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        # Sort by risk probability (highest first)
        alert_df = alert_df.sort_values('risk_probability', ascending=False)
        
        return alert_df
    
    def get_flagged_students(self, X):
        """Get only students flagged as at-risk"""
        alert_df = self.generate_alerts(X)
        return alert_df[alert_df['at_risk_flag'] == 1].reset_index(drop=True)
    
    def get_top_n_students(self, X, n=100):
        """Get top N highest risk students regardless of threshold"""
        alert_df = self.generate_alerts(X)
        return alert_df.head(n).reset_index(drop=True)