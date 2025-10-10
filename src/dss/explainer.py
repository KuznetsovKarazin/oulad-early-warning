import pandas as pd
import numpy as np


class PredictionExplainer:
    """Explain model predictions to instructors"""
    
    def __init__(self, model, feature_importance_df=None):
        self.model = model
        self.feature_importance = feature_importance_df
    
    def explain_student(self, X_student, idx):
        """Generate human-readable explanation for single student"""
        
        student = X_student.iloc[idx]
        reasons = []
        
        # Activity-based reasons
        if student.get('total_clicks', 0) < 50:
            reasons.append(f"Very low activity: only {student['total_clicks']:.0f} clicks in first 2 weeks")
        elif student.get('total_clicks', 0) < 100:
            reasons.append(f"Low activity: {student['total_clicks']:.0f} clicks (median is 132)")
        
        if student.get('active_days_ratio', 0) < 0.2:
            reasons.append(f"Poor engagement: active only {student['active_days_ratio']*14:.0f} days out of 14")
        
        if student.get('days_since_last_activity', 14) > 7:
            reasons.append(f"Inactive recently: last activity {student['days_since_last_activity']:.0f} days ago")
        
        # Assessment-based reasons
        if student.get('completed_first_assessment', 0) == 0:
            reasons.append("Has not completed first assessment")
        elif student.get('first_assessment_score', -1) >= 0:
            score = student['first_assessment_score']
            if score < 40:
                reasons.append(f"Failed first assessment: score {score:.0f}/100")
            elif score < 60:
                reasons.append(f"Low score on first assessment: {score:.0f}/100")
        
        # Demographic risk factors
        if student.get('is_repeat', 0) == 1:
            attempts = student.get('num_of_prev_attempts', 0)
            reasons.append(f"Repeat student: {attempts} previous attempt(s)")
        
        if student.get('education_level', 2) < 2:
            reasons.append("Lower educational background may need additional support")
        
        return reasons
    
    def explain_all_flagged(self, X_flagged):
        """Generate explanations for all flagged students"""
        
        explanations = []
        for idx in range(len(X_flagged)):
            student_reasons = self.explain_student(X_flagged, idx)
            explanations.append("; ".join(student_reasons) if student_reasons else "Multiple risk factors")
        
        return explanations