import sys
sys.path.append('.')

from src.data import OULADLoader, DataCleaner, DataMerger
from src.features import ActivityFeatures, AssessmentFeatures, DemographicFeatures
import pandas as pd

def main():
    # Step 1: Load data
    loader = OULADLoader(data_dir='data/raw')
    data = loader.load_all(max_vle_day=14)
    
    # Step 2: Clean data
    data_clean = DataCleaner.clean_all(data)
    
    # Step 3: Merge into modeling dataset
    df_model = DataMerger.create_modeling_dataset(data_clean)
    
    # Step 4: Create features
    df_model = ActivityFeatures.create_activity_features(df_model)
    df_model = AssessmentFeatures.create_assessment_features(df_model)
    df_model = DemographicFeatures.create_demographic_features(df_model)
    
    # Save
    output_path = 'data/processed/modeling_dataset_early.csv'
    df_model.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df_model.shape}")
    print(f"\nColumns: {list(df_model.columns)}")
    print(f"\nFirst few rows:")
    print(df_model.head())
    
    # Basic stats
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    print(df_model.describe())

if __name__ == '__main__':
    main()