# StrokeRisk Predictor

ALY6040 Final Project. A screening tool that predicts stroke risk from clinical and demographic data.

Built on a logistic regression model trained with SMOTE oversampling on the Brain Stroke Prediction Dataset (fedesoriano, 2021). Catches 82% of real stroke cases in the test set with a 5-fold cross-validated ROC-AUC of 0.84.

## Run locally

```bash
pip install -r requirements.txt
python train_model.py        # only needed once, builds stroke_model.joblib
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Files

- `app.py` — Streamlit app
- `train_model.py` — script that trains the model and saves it as `stroke_model.joblib`
- `stroke_model.joblib` — the trained pipeline (preprocessor + SMOTE + logistic regression)
- `brainStrokeDataset.csv` — source dataset
- `requirements.txt` — Python dependencies

## Disclaimer

For educational and screening purposes only. Not a medical diagnosis. Always consult a qualified clinician for personal health decisions.

## Authors

Abhishek Thadem · Kehao Gu · Mengdi Sun  
Faculty: Prof. Justin Grosz
