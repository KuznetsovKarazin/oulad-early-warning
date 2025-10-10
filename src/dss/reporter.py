import pandas as pd
from datetime import datetime


class DSSReporter:
    """Generate reports for instructors"""
    
    @staticmethod
    def create_instructor_report(alert_df, X_full, explanations):
        """Create comprehensive instructor report"""
        
        # Get student data for flagged students only
        flagged_ids = alert_df['id_student'].values
        
        student_data = student_data.drop_duplicates(subset='id_student', keep='first')
        
        student_data = X_full[X_full['id_student'].isin(flagged_ids)].copy()
        
        # Merge with alerts - use left join on alert_df to maintain order
        report_df = alert_df.merge(
            student_data[['id_student', 'gender', 'age_band', 'highest_education', 
                         'total_clicks', 'active_days_ratio', 'completed_first_assessment',
                         'first_assessment_score']],
            on='id_student',
            how='left'
        )
        
        # Ensure we have correct number of explanations
        if len(explanations) != len(report_df):
            print(f"Warning: explanations length ({len(explanations)}) != report length ({len(report_df)})")
            # Pad with empty explanations if needed
            while len(explanations) < len(report_df):
                explanations.append("Multiple risk factors")
        
        # Add explanations
        report_df['alert_reasons'] = explanations[:len(report_df)]
        
        # Reorder columns for readability
        cols = ['id_student', 'risk_level', 'risk_probability', 'alert_reasons',
                'total_clicks', 'active_days_ratio', 'completed_first_assessment',
                'first_assessment_score', 'gender', 'age_band', 'highest_education']
        
        available_cols = [c for c in cols if c in report_df.columns]
        report_df = report_df[available_cols]
        
        return report_df
    
    @staticmethod
    def generate_email_template(student_row):
        """Generate personalized email template for student"""
        
        student_id = student_row['id_student']
        risk_level = student_row['risk_level']
        
        template = f"""
Subject: Check-in: How are you doing in [Course Name]?

Dear Student {student_id},

I hope this message finds you well. I'm reaching out because I've noticed some patterns in your early engagement with the course that I'd like to discuss.

"""
        
        # Add specific concerns based on risk level
        if risk_level == 'Critical':
            template += """I'm particularly concerned because:
- Your activity level in the first two weeks has been quite low
- This early engagement is often a strong indicator of course success

I'd like to schedule a brief 15-minute call to:
1. Understand if you're facing any challenges
2. Discuss strategies to help you succeed
3. Connect you with support resources if needed
"""
        else:
            template += """I wanted to reach out early to see if you might benefit from additional support. Many successful students find that:
- Regular engagement with course materials improves outcomes
- Completing assignments on time helps build momentum
- Early help-seeking leads to better results

Would you be interested in a brief check-in call to discuss your progress?
"""
        
        template += """
Please feel free to respond to this email or book a time slot here: [Booking Link]

I'm here to support your success!

Best regards,
[Instructor Name]
[Office Hours]
[Contact Information]
"""
        
        return template
    
    @staticmethod
    def create_summary_statistics(alert_df, X_full):
        """Create summary statistics for the report"""
        
        total_students = len(X_full)
        flagged_students = len(alert_df)
        
        # Merge to get additional stats
        merged = alert_df.merge(
            X_full[['id_student', 'total_clicks', 'completed_first_assessment']], 
            on='id_student',
            how='left'
        )
        
        summary = {
            'total_students': total_students,
            'flagged_students': flagged_students,
            'flagged_percentage': (flagged_students / total_students * 100),
            'risk_levels': alert_df['risk_level'].value_counts().to_dict(),
            'avg_risk_probability': alert_df['risk_probability'].mean(),
            'median_clicks_flagged': merged['total_clicks'].median(),
            'assessment_completion_flagged': merged['completed_first_assessment'].mean()
        }
        
        return summary