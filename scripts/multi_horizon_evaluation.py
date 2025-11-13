# experiments/multi_horizon_evaluation.py
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from pathlib import Path
from src.data import OULADLoader, DataCleaner, DataMerger
from src.features import ActivityFeatures, AssessmentFeatures, DemographicFeatures
from src.models import BaselineModel, LogisticModel, BoostingModel
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import joblib


HORIZONS = [14, 28, 56, 84, 112]  

results = []

for day in HORIZONS:
    print(f"\n{'='*60}")
    print(f"EVALUATING HORIZON: DAY {day}")
    print(f"{'='*60}")
    

    loader = OULADLoader(data_dir='data/raw')
    data = loader.load_all(max_vle_day=day)
    

    data_clean = DataCleaner.clean_all(data)
    df_model = DataMerger.create_modeling_dataset(data_clean)
    

    df_model = ActivityFeatures.create_activity_features(df_model)
    df_model = AssessmentFeatures.create_assessment_features(df_model)
    df_model = DemographicFeatures.create_demographic_features(df_model)
    

    presentations = df_model['code_presentation'].unique()
    train_pres = presentations[:int(len(presentations)*0.7)]
    test_pres = presentations[int(len(presentations)*0.7):]
    
    train_idx = df_model['code_presentation'].isin(train_pres)
    test_idx = df_model['code_presentation'].isin(test_pres)
    
    X_train = df_model[train_idx]
    X_test = df_model[test_idx]
    y_train = df_model[train_idx]['failed'].values
    y_test = df_model[test_idx]['failed'].values
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train fail rate: {y_train.mean()*100:.1f}%")
    print(f"Test fail rate: {y_test.mean()*100:.1f}%")
    

    print("\nTraining models...")
    baseline = BaselineModel(min_clicks=50, min_active_days_ratio=0.2)
    baseline.fit(X_train, y_train)
    
    logistic = LogisticModel(C=0.1, calibrate=True)
    logistic.fit(X_train, y_train)
    
    boosting = BoostingModel(n_estimators=100, learning_rate=0.1, max_depth=5, calibrate=True)
    boosting.fit(X_train, y_train)
    

    for name, model in [('Baseline', baseline), ('Logistic', logistic), ('Boosting', boosting)]:
        proba = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, proba)
        ap = average_precision_score(y_test, proba)
        brier = brier_score_loss(y_test, proba)
        

        thresh = np.percentile(proba, 85)
        preds = (proba >= thresh).astype(int)
        
        tp = ((preds == 1) & (y_test == 1)).sum()
        fp = ((preds == 1) & (y_test == 0)).sum()
        fn = ((preds == 0) & (y_test == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        flagged_pct = preds.sum() / len(preds) * 100
        
        results.append({
            'Day': day,
            'Model': name,
            'AUC': auc,
            'AP': ap,
            'Brier': brier,
            'Precision@15%': precision,
            'Recall@15%': recall,
            'Flagged%': flagged_pct
        })
        
        print(f"{name}: AUC={auc:.3f}, AP={ap:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")


df_results = pd.DataFrame(results)
df_results.to_csv('results/multi_horizon_results.csv', index=False)

print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(df_results.to_string(index=False))


pivot = df_results.pivot_table(
    index='Model',
    columns='Day',
    values=['AUC', 'AP', 'Precision@15%', 'Recall@15%']
)
print("\n" + "="*60)
print("PIVOT TABLE FOR PAPER")
print("="*60)
print(pivot.round(3))
pivot.to_csv('results/multi_horizon_pivot.csv')