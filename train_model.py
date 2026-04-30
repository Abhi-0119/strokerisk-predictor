"""Train the Logit+SMOTE model and save it for the Streamlit app.
Re-run this any time the dataset changes.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

np.random.seed(42)

df = pd.read_csv("brainStrokeDataset.csv")
df = df.drop(columns=["id"])
df["bmi"] = df["bmi"].fillna(df["bmi"].median())
df = df[df["gender"] != "Other"].reset_index(drop=True)

df["glucose_high"] = (df["avg_glucose_level"] >= 125).astype(int)
df["bmi_obese"] = (df["bmi"] >= 30).astype(int)

X = df.drop(columns=["stroke"])
y = df["stroke"]

cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
num_cols = [c for c in X.columns if c not in cat_cols]

prep = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
])

pipe = ImbPipeline([
    ("prep", prep),
    ("smote", SMOTE(random_state=42)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pipe.fit(X_train, y_train)

# Quick sanity check
from sklearn.metrics import roc_auc_score, recall_score
y_proba = pipe.predict_proba(X_test)[:, 1]
y_pred = pipe.predict(X_test)
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Test Recall:  {recall_score(y_test, y_pred):.4f}")

# Population stats for peer-comparison
pop_stats = {
    "stroke_rate_overall": float(df["stroke"].mean()),
    "age_mean": float(df["age"].mean()),
    "age_std": float(df["age"].std()),
    "glucose_mean": float(df["avg_glucose_level"].mean()),
    "bmi_mean": float(df["bmi"].mean()),
    "n_total": int(len(df)),
    "n_strokes": int(df["stroke"].sum()),
}

# Save model + a tiny risk-quantile lookup so the app can do peer comparison
risk_scores = pipe.predict_proba(X)[:, 1]
quantiles = np.percentile(risk_scores, np.arange(0, 101, 1))

joblib.dump({
    "model": pipe,
    "feature_order": list(X.columns),
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "pop_stats": pop_stats,
    "risk_quantiles": quantiles.tolist(),
}, "stroke_model.joblib")

print("Saved stroke_model.joblib")
