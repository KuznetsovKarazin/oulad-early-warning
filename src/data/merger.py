import pandas as pd


class DataMerger:
    """Merge OULAD tables into analytical datasets"""
    
    @staticmethod
    def create_student_base(data):
        """Create base student dataset with demographics and outcomes"""
        
        df = data['student_info'].copy()
        
        # Select relevant columns
        base_cols = ['code_module', 'code_presentation', 'id_student',
                     'gender', 'region', 'highest_education', 'imd_band',
                     'age_band', 'num_of_prev_attempts', 'studied_credits',
                     'disability', 'final_result', 'passed', 'failed']
        
        df_base = df[base_cols].copy()
        
        print(f"✓ Created student base: {len(df_base)} students")
        return df_base
    
    @staticmethod
    def add_first_assessment(df_base, data):
        """Add first assessment information"""
        
        # Get first assessment per module
        first_assessments = data['assessments'][data['assessments']['is_first']].copy()
        first_assessments = first_assessments[['code_module', 'code_presentation', 
                                               'id_assessment', 'date', 'assessment_type', 'weight']]
        first_assessments.columns = ['code_module', 'code_presentation', 
                                     'first_assessment_id', 'first_assessment_date', 
                                     'first_assessment_type', 'first_assessment_weight']
        
        # Merge with base
        df = df_base.merge(first_assessments, on=['code_module', 'code_presentation'], how='left')
        
        # Add student scores for first assessment
        student_scores = data['student_assessment'].copy()
        student_scores = student_scores.rename(columns={'id_assessment': 'first_assessment_id'})
        
        df = df.merge(
            student_scores[['id_student', 'first_assessment_id', 'score', 'date_submitted']],
            on=['id_student', 'first_assessment_id'],
            how='left'
        )
        
        df = df.rename(columns={'score': 'first_assessment_score',
                               'date_submitted': 'first_assessment_submitted'})
        
        print(f"✓ Added first assessment info")
        return df
    
    @staticmethod
    def add_early_activity(df, data):
        """Add early VLE activity features"""
        
        activity = data['student_vle_early'].copy()
        
        df = df.merge(activity, on=['code_module', 'code_presentation', 'id_student'], how='left')
        
        # Fill missing activity with zeros (no activity)
        activity_cols = ['total_clicks', 'active_days', 'last_active_day', 
                        'first_active_day', 'activity_recency']
        
        for col in activity_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        print(f"✓ Added early VLE activity")
        return df
    
    @staticmethod
    def create_modeling_dataset(data):
        """Create complete dataset for modeling"""
        print("\n" + "=" * 60)
        print("Creating Modeling Dataset")
        print("=" * 60)
        
        # Start with student base
        df = DataMerger.create_student_base(data)
        
        # Add features
        df = DataMerger.add_first_assessment(df, data)
        df = DataMerger.add_early_activity(df, data)
        
        # Remove students who withdrew before week 2
        initial_len = len(df)
        df = df[df['final_result'] != 'Withdrawn'].copy()  # Keep for now, filter later if needed
        
        print("=" * 60)
        print(f"✓ Modeling dataset ready: {len(df)} students")
        print(f"  Target distribution:")
        print(f"    Passed:  {df['passed'].sum():>6,} ({df['passed'].mean()*100:.1f}%)")
        print(f"    Failed:  {df['failed'].sum():>6,} ({df['failed'].mean()*100:.1f}%)")
        print("=" * 60)
        
        return df