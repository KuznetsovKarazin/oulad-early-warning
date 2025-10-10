import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class OULADLoader:
    """Load OULAD dataset with memory-efficient handling of large files"""
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def load_all(self, vle_chunk_size=10_000_000, max_vle_day=14):
        """Load all tables with optimized settings"""
        print("=" * 60)
        print("OULAD Dataset Loader")
        print("=" * 60)
        
        data = {}
        
        # Small tables - load directly
        data['courses'] = self._load_csv('courses.csv')
        data['assessments'] = self._load_csv('assessments.csv')
        data['vle'] = self._load_csv('vle.csv')
        data['student_info'] = self._load_csv('studentInfo.csv')
        data['student_registration'] = self._load_csv('studentRegistration.csv')
        data['student_assessment'] = self._load_csv('studentAssessment.csv')
        
        # Large table - aggregate early to reduce size
        data['student_vle_early'] = self.load_and_aggregate_vle(
            chunk_size=vle_chunk_size,
            max_day=max_vle_day
        )
        
        print("=" * 60)
        print("✓ All data loaded successfully")
        print("=" * 60)
        return data
    
    def _load_csv(self, filename):
        """Load single CSV file"""
        path = self.data_dir / filename
        df = pd.read_csv(path)
        print(f"✓ {filename:30s} {len(df):>10,} rows")
        return df
    
    def load_and_aggregate_vle(self, chunk_size=10_000_000, max_day=14):
        """Load huge studentVle.csv and aggregate early activity"""
        path = self.data_dir / 'studentVle.csv'
        
        print(f"\n⚙ Processing studentVle.csv (chunked, max_day={max_day})...")
        
        aggregated_chunks = []
        total_rows = 0
        
        for i, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size), 1):
            total_rows += len(chunk)
            
            # Filter early days only
            early = chunk[chunk['date'] <= max_day].copy()
            
            if len(early) > 0:
                # Aggregate by student and presentation
                agg = early.groupby(['code_module', 'code_presentation', 'id_student']).agg({
                    'sum_click': 'sum',
                    'date': ['nunique', 'max', 'min'] 
                }).reset_index()
                
                agg.columns = ['code_module', 'code_presentation', 'id_student',
                              'total_clicks', 'active_days', 'last_active_day', 'first_active_day']
                
                aggregated_chunks.append(agg)
            
            print(f"  Chunk {i}: processed {total_rows:>12,} rows", end='\r')
        
        print(f"\n  Total processed: {total_rows:>12,} rows")
        
        # Combine all chunks and aggregate again (student might be in multiple chunks)
        print("  Merging chunks...")
        df_combined = pd.concat(aggregated_chunks, ignore_index=True)
        
        df_final = df_combined.groupby(['code_module', 'code_presentation', 'id_student']).agg({
            'total_clicks': 'sum',
            'active_days': 'sum',
            'last_active_day': 'max',
            'first_active_day': 'min'
        }).reset_index()
        
        print(f"✓ studentVle (early aggregated) {len(df_final):>10,} rows")
        
        return df_final