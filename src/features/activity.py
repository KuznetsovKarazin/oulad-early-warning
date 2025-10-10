import pandas as pd
import numpy as np


class ActivityFeatures:
    """Engineer features from VLE activity data"""
    
    @staticmethod
    def create_activity_features(df):
        """Create activity-based features for early prediction"""
        
        print("\n" + "=" * 60)
        print("Creating Activity Features")
        print("=" * 60)
        
        df_feat = df.copy()
        
        # Already have: total_clicks, active_days, last_active_day, first_active_day, activity_recency
        
        # Average clicks per active day
        df_feat['avg_clicks_per_day'] = np.where(
            df_feat['active_days'] > 0,
            df_feat['total_clicks'] / df_feat['active_days'],
            0
        )
        
        # Activity intensity bins
        df_feat['activity_level'] = pd.cut(
            df_feat['total_clicks'],
            bins=[-1, 0, 10, 50, 200, 1000, np.inf],
            labels=['none', 'very_low', 'low', 'medium', 'high', 'very_high']
        )
        
        # Binary: has any activity in first 2 weeks
        df_feat['has_early_activity'] = (df_feat['total_clicks'] > 0).astype(int)
        
        # Days since last activity (relative to day 14)
        df_feat['days_since_last_activity'] = 14 - df_feat['last_active_day']
        df_feat['days_since_last_activity'] = df_feat['days_since_last_activity'].clip(lower=0)
        
        # Active days ratio (out of 14 days)
        df_feat['active_days_ratio'] = df_feat['active_days'] / 14
        
        # Log-transformed clicks (helps with skewness)
        df_feat['log_total_clicks'] = np.log1p(df_feat['total_clicks'])
        
        print(f"✓ Created activity features")
        print(f"  Total clicks: min={df_feat['total_clicks'].min()}, "
              f"median={df_feat['total_clicks'].median():.0f}, "
              f"max={df_feat['total_clicks'].max()}")
        print(f"  Active days: mean={df_feat['active_days'].mean():.1f}")
        print("=" * 60)
        
        return df_feat