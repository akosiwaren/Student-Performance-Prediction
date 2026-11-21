# MP2 Student Performance Predictor

Streamlit application implementing the workflow in `Untitled19.ipynb`.

## Model workflow
- Dataset: `augmented_balanced_dataset_m.csv` (3,602 records)
- Target: `rating` (High / Low)
- 8 raw LMS activity variables
- 7 engineered behavioral variables, calculated exactly as in the notebook
- Stratified 80/20 split, random seed 42
- GradientBoostingClassifier:
  - n_estimators=300
  - learning_rate=0.08
  - max_depth=6
  - subsample=0.85
  - max_features='sqrt'
  - min_samples_split=5
  - min_samples_leaf=3
  - random_state=42

## Reproduced holdout results
- Accuracy: 90.43%
- ROC-AUC: 0.958
- Training records: 2881
- Testing records: 721

## Run locally
`pip install -r requirements.txt`
`streamlit run app.py`

## Deploy on Streamlit Community Cloud
1. Upload `app.py`, `student_success_gb_model.joblib`, `model_metadata.json`, and `requirements.txt` to a GitHub repository.
2. Go to https://share.streamlit.io/ and sign in with GitHub.
3. Create an app, select the repository and branch, and set the main file path to `app.py`.
4. Deploy and copy the generated public URL.

The app accepts the 8 raw LMS activity counts and calculates the same 7 engineered features as the notebook. The predicted labels remain High and Low, matching the MP2 target. They should not be relabeled Pass/Fail without an explicit mapping in the assignment.
