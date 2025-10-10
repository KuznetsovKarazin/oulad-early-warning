import pandas as pd
import numpy as np


class DataCleaner:
    """Clean and preprocess OULAD data"""
    
    @staticmethod
    def clean_student_info(df):
        """Clean studentInfo table"""
        df_clean = df.copy()
        
        # Handle missing values
        df_clean['imd_band'] = df_clean['imd_band'].fillna('Unknown')
        
        # Standardize values
        df_clean['disability'] = df_clean['disability'].replace({'N': 'No', 'Y': 'Yes'})
        
        # Create target variable (Pass/Fail)
        df_clean['passed'] = df_clean['final_result'].isin(['Pass', 'Distinction']).astype(int)
        df_clean['failed'] = df_clean['final_result'].isin(['Fail', 'Withdrawn']).astype(int)
        
        print(f"✓ Cleaned student_info: {len(df_clean)} rows")
        return df_clean
    
    @staticmethod
    def clean_assessments(df):
        """Clean assessments table"""
        df_clean = df.copy()
        
        # Sort by date to identify first assessment
        df_clean = df_clean.sort_values(['code_module', 'code_presentation', 'date'])
        
        # Mark first assessment per module
        df_clean['is_first'] = df_clean.groupby(['code_module', 'code_presentation']).cumcount() == 0
        
        print(f"✓ Cleaned assessments: {len(df_clean)} rows")
        return df_clean
    
    @staticmethod
    def clean_student_assessment(df):
        """Clean studentAssessment table"""
        df_clean = df.copy()
        
        # Remove records with missing scores
        initial_len = len(df_clean)
        df_clean = df_clean.dropna(subset=['score'])
        
        # Handle submitted late
        df_clean['is_banked'] = df_clean['is_banked'].fillna(0).astype(int)
        df_clean['date_submitted'] = df_clean['date_submitted'].fillna(-1).astype(int)
        
        print(f"✓ Cleaned student_assessment: {len(df_clean)} rows (removed {initial_len - len(df_clean)} null scores)")
        return df_clean
    
    @staticmethod
    def clean_vle_activity(df):
        """Clean early VLE activity aggregation"""
        df_clean = df.copy()
        
        # Remove zero activity (shouldn't happen but just in case)
        df_clean = df_clean[df_clean['total_clicks'] > 0]
        
        # Calculate activity recency (higher = more recent)
        df_clean['activity_recency'] = df_clean['last_active_day'] - df_clean['first_active_day']
        
        print(f"✓ Cleaned VLE activity: {len(df_clean)} rows")
        return df_clean
    
    @staticmethod
    def clean_all(data_dict):
        """Clean all tables"""
        print("\n" + "=" * 60)
        print("Data Cleaning")
        print("=" * 60)
        
        cleaned = {}
        cleaned['courses'] = data_dict['courses'].copy()
        cleaned['vle'] = data_dict['vle'].copy()
        cleaned['student_registration'] = data_dict['student_registration'].copy()
        
        cleaned['student_info'] = DataCleaner.clean_student_info(data_dict['student_info'])
        cleaned['assessments'] = DataCleaner.clean_assessments(data_dict['assessments'])
        cleaned['student_assessment'] = DataCleaner.clean_student_assessment(data_dict['student_assessment'])
        cleaned['student_vle_early'] = DataCleaner.clean_vle_activity(data_dict['student_vle_early'])
        
        print("=" * 60)
        print("✓ All data cleaned")
        print("=" * 60)
        
        return cleaned