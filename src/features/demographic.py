import pandas as pd


class DemographicFeatures:
    """Engineer features from demographic data"""
    
    @staticmethod
    def create_demographic_features(df):
        """Create demographic-based features"""
        
        print("\n" + "=" * 60)
        print("Creating Demographic Features")
        print("=" * 60)
        
        df_feat = df.copy()
        
        # Binary encodings
        df_feat['is_female'] = (df_feat['gender'] == 'F').astype(int)
        df_feat['has_disability'] = (df_feat['disability'] == 'Yes').astype(int)
        df_feat['is_repeat'] = (df_feat['num_of_prev_attempts'] > 0).astype(int)
        
        # Education level numeric encoding (higher = more education)
        education_order = {
            'No Formal quals': 0,
            'Lower Than A Level': 1,
            'A Level or Equivalent': 2,
            'HE Qualification': 3,
            'Post Graduate Qualification': 4
        }
        df_feat['education_level'] = df_feat['highest_education'].map(education_order)
        
        # Age band numeric encoding
        age_order = {
            '0-35': 0,
            '35-55': 1,
            '55<=': 2
        }
        df_feat['age_group'] = df_feat['age_band'].map(age_order)
        
        # IMD band (deprivation index) - convert to numeric
        # Lower IMD = more deprived, higher = less deprived
        imd_order = {
            '0-10%': 1,
            '10-20%': 2,
            '20-30%': 3,
            '30-40%': 4,
            '40-50%': 5,
            '50-60%': 6,
            '60-70%': 7,
            '70-80%': 8,
            '80-90%': 9,
            '90-100%': 10,
            'Unknown': -1
        }
        df_feat['imd_numeric'] = df_feat['imd_band'].map(imd_order)
        
        # Credit load (normalize by typical 60 credits per year)
        df_feat['credit_load_norm'] = df_feat['studied_credits'] / 60
        
        print(f"✓ Created demographic features")
        print(f"  Gender: {df_feat['is_female'].mean()*100:.1f}% female")
        print(f"  Disability: {df_feat['has_disability'].mean()*100:.1f}%")
        print(f"  Repeat students: {df_feat['is_repeat'].mean()*100:.1f}%")
        print("=" * 60)
        
        return df_feat