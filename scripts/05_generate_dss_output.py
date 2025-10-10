import sys
sys.path.append('.')

import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from src.dss import StudentAlerter, PredictionExplainer, DSSReporter

def main():
    print("="*60)
    print("DSS OUTPUT GENERATION")
    print("="*60)
    
    # Load data and model
    print("\nLoading data and models...")
    df = pd.read_csv('data/processed/modeling_dataset_early.csv')
    boosting = joblib.load('results/models/boosting_model.pkl')
    
    # For demo, use test set (in production, use current semester data)
    presentations = df['code_presentation'].unique()
    test_pres = presentations[int(len(presentations)*0.7):]
    X_current = df[df['code_presentation'].isin(test_pres)].copy()
    
    print(f"Processing {len(X_current)} students...")
    
    # Initialize DSS components
    alerter = StudentAlerter(model=boosting, threshold=0.607)
    explainer = PredictionExplainer(model=boosting)
    reporter = DSSReporter()
    
    # Generate alerts
    print("\nGenerating alerts...")
    flagged_students = alerter.get_flagged_students(X_current)
    print(f"✓ Flagged {len(flagged_students)} students ({len(flagged_students)/len(X_current)*100:.1f}%)")
    
    # Generate explanations - make sure we match flagged_students exactly
    print("\nGenerating explanations...")
    
    # Get the actual student data for flagged students
    flagged_data = X_current[X_current['id_student'].isin(flagged_students['id_student'])].copy()
    
    # Reset index to ensure alignment
    flagged_data = flagged_data.reset_index(drop=True)
    
    # Generate explanations
    explanations = []
    for idx in range(len(flagged_students)):
        student_id = flagged_students.iloc[idx]['id_student']
        
        # Find this student in flagged_data
        student_idx = flagged_data[flagged_data['id_student'] == student_id].index
        
        if len(student_idx) > 0:
            student_reasons = explainer.explain_student(flagged_data, student_idx[0])
            explanations.append("; ".join(student_reasons) if student_reasons else "Multiple risk factors")
        else:
            explanations.append("Multiple risk factors")
    
    print(f"✓ Generated {len(explanations)} explanations")
    
    # Create instructor report
    print("\nCreating instructor report...")
    report_df = reporter.create_instructor_report(flagged_students, X_current, explanations)
    
    # Create summary
    summary = reporter.create_summary_statistics(flagged_students, X_current)
    
    # Save outputs
    output_dir = Path('results/dss_output')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save main report
    report_path = output_dir / f'instructor_alert_list_{timestamp}.csv'
    report_df.to_csv(report_path, index=False)
    print(f"\n✓ Saved instructor report: {report_path}")
    
    # Save summary
    summary_path = output_dir / f'summary_statistics_{timestamp}.txt'
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EARLY WARNING SYSTEM - SUMMARY REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total Students: {summary['total_students']}\n")
        f.write(f"Flagged Students: {summary['flagged_students']} ({summary['flagged_percentage']:.1f}%)\n\n")
        f.write(f"Risk Levels:\n")
        for level, count in summary['risk_levels'].items():
            f.write(f"  {level}: {count}\n")
        f.write(f"\nAverage Risk Probability: {summary['avg_risk_probability']:.3f}\n")
        f.write(f"Median Clicks (Flagged): {summary['median_clicks_flagged']:.0f}\n")
        f.write(f"Assessment Completion (Flagged): {summary['assessment_completion_flagged']*100:.1f}%\n")
    
    print(f"✓ Saved summary: {summary_path}")
    
    # Generate sample email templates
    print("\nGenerating sample email templates...")
    email_dir = output_dir / 'email_templates'
    email_dir.mkdir(exist_ok=True)
    
    for i in range(min(5, len(report_df))):
        student = report_df.iloc[i]
        email = reporter.generate_email_template(student)
        
        email_path = email_dir / f'student_{student["id_student"]}_template.txt'
        with open(email_path, 'w', encoding='utf-8') as f:
            f.write(email)
    
    print(f"✓ Generated {min(5, len(report_df))} sample email templates")
    
    # Display preview
    print("\n" + "="*60)
    print("PREVIEW: TOP 10 AT-RISK STUDENTS")
    print("="*60)
    
    preview_df = report_df.head(10)[['id_student', 'risk_level', 'risk_probability', 
                                      'total_clicks', 'alert_reasons']].copy()
    
    # Truncate long reasons for display
    preview_df['alert_reasons'] = preview_df['alert_reasons'].str[:80] + '...'
    
    print(preview_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total Students: {summary['total_students']}")
    print(f"Flagged Students: {summary['flagged_students']} ({summary['flagged_percentage']:.1f}%)")
    print(f"\nRisk Level Distribution:")
    for level, count in sorted(summary['risk_levels'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {level}: {count}")
    
    print("\n" + "="*60)
    print("✓ DSS OUTPUT GENERATION COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review the instructor_alert_list CSV")
    print("2. Customize email templates as needed")
    print("3. Contact flagged students within 1-2 weeks")
    print("4. Track outcomes for system improvement")

if __name__ == '__main__':
    main()