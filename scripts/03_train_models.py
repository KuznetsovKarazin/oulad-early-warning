import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.models import BaselineModel, LogisticModel, BoostingModel

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/processed/modeling_dataset_early.csv')
    
    # Prepare features and target
    X = df.copy()
    y = df['failed'].values  # Predict FAILURE (1=at risk, 0=not at risk)
    
    # Train-test split (by presentation to avoid leakage)
    presentations = df['code_presentation'].unique()
    train_pres = presentations[:int(len(presentations)*0.7)]
    test_pres = presentations[int(len(presentations)*0.7):]
    
    train_idx = df['code_presentation'].isin(train_pres)
    test_idx = df['code_presentation'].isin(test_pres)
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    print(f"\nTrain set: {len(X_train)} students")
    print(f"Test set:  {len(X_test)} students")
    print(f"Train failure rate: {y_train.mean()*100:.1f}%")
    print(f"Test failure rate:  {y_test.mean()*100:.1f}%")
    
    # Train models
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    # 1. Baseline
    print("\n1. Baseline Model...")
    baseline = BaselineModel(min_clicks=50, min_active_days_ratio=0.2)
    baseline.fit(X_train, y_train)
    
    # 2. Logistic Regression
    print("2. Logistic Regression...")
    logistic = LogisticModel(C=0.1, calibrate=True)
    logistic.fit(X_train, y_train)
    
    # 3. Gradient Boosting
    print("3. Gradient Boosting (LightGBM)...")
    boosting = BoostingModel(n_estimators=100, learning_rate=0.1, max_depth=5, calibrate=True)
    boosting.fit(X_train, y_train)
    
    print("\n✓ All models trained!")
    
    # Quick evaluation
    print("\n" + "="*60)
    print("QUICK EVALUATION (Test Set)")
    print("="*60)
    
    for name, model in [('Baseline', baseline), ('Logistic', logistic), ('Boosting', boosting)]:
        proba = model.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc = roc_auc_score(y_test, proba)
        ap = average_precision_score(y_test, proba)
        
        print(f"\n{name}:")
        print(f"  AUC: {auc:.3f}")
        print(f"  Average Precision: {ap:.3f}")
    
    # Show feature importance
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION - TOP 10 FEATURES")
    print("="*60)
    print(logistic.get_feature_importance().head(10))
    
    print("\n" + "="*60)
    print("GRADIENT BOOSTING - TOP 10 FEATURES")
    print("="*60)
    print(boosting.get_feature_importance().head(10))
    
    # Save models
    import joblib
    joblib.dump(baseline, 'results/models/baseline_model.pkl')
    joblib.dump(logistic, 'results/models/logistic_model.pkl')
    joblib.dump(boosting, 'results/models/boosting_model.pkl')
    print("\n✓ Models saved to results/models/")

if __name__ == '__main__':
    main()