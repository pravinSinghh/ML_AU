# Stress Level Prediction (Machine Learning Project)

This project predicts stress levels using health and lifestyle features with a Random Forest classifier.

## Project Contents

- `Final_stress.ipynb` - end-to-end notebook for EDA, preprocessing, training, tuning, and model export
- `stress_model.pkl` - trained model artifact
- `scaler.pkl` - fitted `StandardScaler`
- `encoders.pkl` - fitted label encoders for categorical features
- `columns.pkl` - saved feature metadata

## Workflow Implemented

1. Load dataset (`sleep_health_synthetic_5000.csv`)
2. Explore data (`head`, `info`, `describe`, null checks, distribution plots)
3. Feature engineering:
   - Split `Blood Pressure` into `Systolic` and `Diastolic`
   - Create `Pulse Pressure`
   - Drop unused columns (`Blood Pressure`, `Person ID`)
4. Encode categorical columns using `LabelEncoder`
5. Train/test split with stratification
6. Feature scaling using `StandardScaler`
7. Train `RandomForestClassifier`
8. Evaluate with accuracy, classification report, and confusion matrix
9. Hyperparameter tuning using `GridSearchCV`
10. Visualize feature importance
11. Export model and preprocessing artifacts using `joblib`

## Requirements

Typical Python packages used:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

Install with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## How To Run

1. Open `Final_stress.ipynb` in Jupyter or VS Code notebooks.
2. Ensure the dataset file `sleep_health_synthetic_5000.csv` is available in the project directory.
3. Run cells from top to bottom.

## Notes

- The notebook saves model artifacts as `.pkl` files in this folder.
- If you retrain on new data, regenerate and replace exported artifacts.

## Author

MCA Semester 2 ML Project
