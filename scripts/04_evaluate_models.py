import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss
)
import joblib
from pathlib import Path

def plot_roc_curves(models_dict, X_test, y_test):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/07_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_precision_recall_curves(models_dict, X_test, y_test):
    """Plot Precision-Recall curves for all models"""
    plt.figure(figsize=(10, 8))
    
    baseline_precision = y_test.mean()
    
    for name, model in models_dict.items():
        proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        
        plt.plot(recall, precision, lw=2, label=f'{name} (AP = {ap:.3f})')
    
    plt.axhline(y=baseline_precision, color='k', linestyle='--', lw=2, 
                label=f'Baseline (AP = {baseline_precision:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/08_pr_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_calibration_curve(models_dict, X_test, y_test, n_bins=10):
    """Plot calibration curves"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, model in models_dict.items():
        proba = model.predict_proba(X_test)[:, 1]
        
        # Bin predictions
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        bin_indices = np.digitize(proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_sums = np.bincount(bin_indices, weights=proba, minlength=n_bins)
        bin_true = np.bincount(bin_indices, weights=y_test, minlength=n_bins)
        bin_total = np.bincount(bin_indices, minlength=n_bins)
        
        # Avoid division by zero
        nonzero = bin_total > 0
        fraction_of_positives = np.zeros(n_bins)
        mean_predicted_value = np.zeros(n_bins)
        
        fraction_of_positives[nonzero] = bin_true[nonzero] / bin_total[nonzero]
        mean_predicted_value[nonzero] = bin_sums[nonzero] / bin_total[nonzero]
        
        ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
                label=name, markersize=8)
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/09_calibration_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

def workload_analysis(model, X_test, y_test, thresholds=[0.3, 0.4, 0.5, 0.6]):
    """Analyze workload vs performance at different thresholds"""
    proba = model.predict_proba(X_test)[:, 1]
    
    results = []
    for thresh in thresholds:
        predictions = (proba >= thresh).astype(int)
        
        # Metrics
        tp = ((predictions == 1) & (y_test == 1)).sum()
        fp = ((predictions == 1) & (y_test == 0)).sum()
        tn = ((predictions == 0) & (y_test == 0)).sum()
        fn = ((predictions == 0) & (y_test == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        flagged_pct = predictions.sum() / len(predictions) * 100
        
        results.append({
            'threshold': thresh,
            'flagged_pct': flagged_pct,
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn
        })
    
    df_results = pd.DataFrame(results)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Workload vs Recall
    axes[0].plot(df_results['flagged_pct'], df_results['recall'], 'o-', 
                 markersize=10, linewidth=2, color='green')
    axes[0].set_xlabel('% Students Flagged (Workload)', fontsize=12)
    axes[0].set_ylabel('Recall (% At-Risk Caught)', fontsize=12)
    axes[0].set_title('Workload vs Recall Trade-off', fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    for _, row in df_results.iterrows():
        axes[0].annotate(f"thresh={row['threshold']:.1f}", 
                        (row['flagged_pct'], row['recall']),
                        textcoords="offset points", xytext=(0,10), ha='center')
    
    # Precision vs Recall
    axes[1].plot(df_results['recall'], df_results['precision'], 'o-', 
                 markersize=10, linewidth=2, color='blue')
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision vs Recall Trade-off', fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    for _, row in df_results.iterrows():
        axes[1].annotate(f"{row['flagged_pct']:.0f}% flagged", 
                        (row['recall'], row['precision']),
                        textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig('results/figures/10_workload_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return df_results

def main():
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/processed/modeling_dataset_early.csv')
    
    # Split same way as training
    presentations = df['code_presentation'].unique()
    train_pres = presentations[:int(len(presentations)*0.7)]
    test_pres = presentations[int(len(presentations)*0.7):]
    
    test_idx = df['code_presentation'].isin(test_pres)
    X_test = df[test_idx]
    y_test = df[test_idx]['failed'].values
    
    # Load models
    baseline = joblib.load('results/models/baseline_model.pkl')
    logistic = joblib.load('results/models/logistic_model.pkl')
    boosting = joblib.load('results/models/boosting_model.pkl')
    
    models_dict = {
        'Baseline': baseline,
        'Logistic': logistic,
        'Boosting': boosting
    }
    
    # Comprehensive metrics
    print("\n" + "="*60)
    print("COMPREHENSIVE METRICS (Test Set)")
    print("="*60)
    
    metrics_list = []
    for name, model in models_dict.items():
        proba = model.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
        
        auc_score = roc_auc_score(y_test, proba)
        ap_score = average_precision_score(y_test, proba)
        brier = brier_score_loss(y_test, proba)
        
        metrics_list.append({
            'Model': name,
            'AUC': auc_score,
            'Average Precision': ap_score,
            'Brier Score': brier
        })
    
    df_metrics = pd.DataFrame(metrics_list)
    print("\n", df_metrics.to_string(index=False))
    
    # Plot curves
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    print("\n1. ROC Curves...")
    plot_roc_curves(models_dict, X_test, y_test)
    
    print("2. Precision-Recall Curves...")
    plot_precision_recall_curves(models_dict, X_test, y_test)
    
    print("3. Calibration Curves...")
    plot_calibration_curve(models_dict, X_test, y_test)
    
    # Workload analysis for best model
    print("\n" + "="*60)
    print("WORKLOAD ANALYSIS (Boosting Model)")
    print("="*60)
    
    print("\n4. Workload Trade-offs...")
    workload_df = workload_analysis(boosting, X_test, y_test, 
                                     thresholds=[0.2, 0.3, 0.4, 0.5, 0.6])
    print("\n", workload_df.to_string(index=False))
    
    # Recommended threshold for DSS (10-20% flagged)
    print("\n" + "="*60)
    print("RECOMMENDED THRESHOLD FOR DSS")
    print("="*60)
    
    target_flagged = 0.15  # 15% of students
    proba = boosting.predict_proba(X_test)[:, 1]
    recommended_threshold = np.percentile(proba, (1 - target_flagged) * 100)
    
    predictions = (proba >= recommended_threshold).astype(int)
    tp = ((predictions == 1) & (y_test == 1)).sum()
    fp = ((predictions == 1) & (y_test == 0)).sum()
    fn = ((predictions == 0) & (y_test == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    flagged_pct = predictions.sum() / len(predictions) * 100
    
    print(f"\nTo flag ~15% of students:")
    print(f"  Recommended threshold: {recommended_threshold:.3f}")
    print(f"  Actual % flagged: {flagged_pct:.1f}%")
    print(f"  Precision: {precision:.3f} ({precision*100:.1f}% of flagged are truly at-risk)")
    print(f"  Recall: {recall:.3f} ({recall*100:.1f}% of at-risk students caught)")
    print(f"  True Positives: {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    
    print("\n" + "="*60)
    print("✓ EVALUATION COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()