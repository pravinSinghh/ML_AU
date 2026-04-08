# Stress Level Prediction using Sleep and Health Data

This project predicts stress levels from sleep, health, and lifestyle features using a Random Forest model. The implementation is in a single notebook with preprocessing, model training, tuning, and artifact export.

## Repository Contents

- `Final_stress.ipynb` - full pipeline (EDA, preprocessing, training, tuning, visualization)
- `sleep_health_synthetic_5000.csv` - source dataset
- `stress_model.pkl` - trained model artifact
- `scaler.pkl` - fitted StandardScaler
- `encoders.pkl` - fitted LabelEncoders for categorical columns
- `columns.pkl` - feature metadata artifact

## Pipeline Summary

1. Load and inspect dataset
2. Perform EDA with distribution and correlation plots
3. Feature engineering
   - Split Blood Pressure into Systolic and Diastolic
   - Derive Pulse Pressure = Systolic - Diastolic
   - Drop Blood Pressure and Person ID
4. Encode categorical features with LabelEncoder
5. Train-test split (80:20) with stratification
6. Scale features with StandardScaler
7. Train baseline RandomForestClassifier
8. Tune hyperparameters using GridSearchCV
9. Evaluate with accuracy, F1-scores, and confusion matrix
10. Save model and preprocessing artifacts with joblib

## Dataset Insights

After preprocessing (feature engineering + null removal):

- Total rows used: 2122
- Total columns: 14
- Train/Test split: 1697 / 425

Class distribution for Stress Level:

- 3: 424
- 4: 383
- 5: 135
- 6: 39
- 7: 630
- 8: 511

Insight: the dataset is imbalanced, with Stress Level 6 having very low support.

## Model Performance Insights

Baseline model (RandomForest, n_estimators=200, max_depth=10):

- Accuracy: 0.9859

Tuned model (GridSearchCV best estimator):

- Accuracy: 0.9835
- Macro F1: 0.9315
- Weighted F1: 0.9820
- Best params: max_depth=20, min_samples_split=2, n_estimators=100

Insight: tuning did not improve accuracy over the baseline in this run, which indicates the baseline configuration is already strong for this dataset.

Confusion matrix insight:

- Most classes are predicted almost perfectly.
- The largest errors appear in minority classes (especially around stress levels 5 and 6), consistent with class imbalance.

## Top Predictive Features (from tuned model)

Top 5 feature importances:

1. Sleep Duration (0.1943)
2. Quality of Sleep (0.1391)
3. Sleep Disorder (0.1102)
4. Occupation (0.1019)
5. Heart Rate (0.0898)

Insight: sleep-related variables dominate prediction power, which aligns with expected stress-sleep relationships.

## Requirements

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Run Instructions

1. Open `Final_stress.ipynb` in VS Code or Jupyter.
2. Keep `sleep_health_synthetic_5000.csv` in the same folder.
3. Run notebook cells from top to bottom.
4. Exported artifacts are written as .pkl files in the project root.

## Future Improvements

- Handle class imbalance (for example class weights or SMOTE)
- Add per-class precision/recall tracking across experiments
- Validate on additional real-world data to test generalization

