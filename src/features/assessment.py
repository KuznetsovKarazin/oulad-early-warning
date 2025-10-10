import pandas as pd
import numpy as np


class AssessmentFeatures:
    """Engineer features from assessment data"""
    
    @staticmethod
    def create_assessment_features(df):
        """Create assessment-based features for early prediction"""
        
        print("\n" + "=" * 60)
        print("Creating Assessment Features")
        print("=" * 60)
        
        df_feat = df.copy()
        
        # Has completed first assessment (within first 2 weeks)
        df_feat['completed_first_assessment'] = (
            (~df_feat['first_assessment_score'].isna()) & 
            (df_feat['first_assessment_submitted'] <= 14)
        ).astype(int)
        
        # First assessment score (normalized to 0-1)
        df_feat['first_assessment_score_norm'] = df_feat['first_assessment_score'] / 100
        df_feat['first_assessment_score_norm'] = df_feat['first_assessment_score_norm'].fillna(-1)  # -1 for missing
        
        # Performance categories for first assessment
        df_feat['first_assessment_performance'] = pd.cut(
            df_feat['first_assessment_score'],
            bins=[-1, 40, 60, 80, 101],
            labels=['fail', 'pass', 'good', 'excellent'],
            include_lowest=True
        )
        df_feat['first_assessment_performance'] = df_feat['first_assessment_performance'].cat.add_categories('not_completed')
        df_feat['first_assessment_performance'] = df_feat['first_assessment_performance'].fillna('not_completed')
        
        # Submitted on time (before or on due date)
        df_feat['first_assessment_on_time'] = (
            df_feat['first_assessment_submitted'] <= df_feat['first_assessment_date']
        ).astype(float)
        df_feat['first_assessment_on_time'] = df_feat['first_assessment_on_time'].fillna(-1)  # -1 for not submitted
        
        # Days early/late for submission
        df_feat['first_assessment_timeliness'] = (
            df_feat['first_assessment_date'] - df_feat['first_assessment_submitted']
        )
        df_feat['first_assessment_timeliness'] = df_feat['first_assessment_timeliness'].fillna(0)
        
        completed_pct = df_feat['completed_first_assessment'].mean() * 100
        avg_score = df_feat[df_feat['first_assessment_score'].notna()]['first_assessment_score'].mean()
        
        print(f"✓ Created assessment features")
        print(f"  Completed first assessment: {completed_pct:.1f}%")
        if not np.isnan(avg_score):
            print(f"  Average score (completed): {avg_score:.1f}")
        print("=" * 60)
        
        return df_feat