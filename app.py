"""StrokeRisk Predictor - Streamlit app
ALY6040 Final Project | Brain Stroke Risk Prediction
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="StrokeRisk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- LOAD MODEL ----------
# Strategy: try the pre-trained joblib first (loads in ~50ms). If that fails for any
# reason (sklearn version drift, missing file, corrupt pickle), retrain in-memory.
# The retrain is optimized to finish in about 1 second.
@st.cache_resource
def load_bundle():
    import os
    if os.path.exists("stroke_model.joblib"):
        try:
            bundle = joblib.load("stroke_model.joblib")
            # Sanity-check: make sure the loaded model can actually predict.
            test = pd.DataFrame([{
                "gender": "Male", "age": 50.0, "hypertension": 0, "heart_disease": 0,
                "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
                "avg_glucose_level": 100.0, "bmi": 25.0, "smoking_status": "never smoked",
                "glucose_high": 0, "bmi_obese": 0,
            }], columns=bundle["feature_order"])
            bundle["model"].predict_proba(test)
            return bundle
        except Exception:
            pass  # Fall through to retrain
    return _retrain_from_csv()


def _retrain_from_csv():
    import pandas as pd
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
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

    cats = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    nums = [c for c in X.columns if c not in cats]

    prep = ColumnTransformer([
        ("num", StandardScaler(), nums),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cats),
    ])
    # max_iter dropped from 1000 to 200; the model converges well before 200 on this dataset
    # and skipping the train/test split lets us train on the full data, which is what we want
    # in production anyway (the test split was only for evaluation in the notebook).
    pipe = ImbPipeline([
        ("prep", prep),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=200, random_state=42, solver="lbfgs")),
    ])
    pipe.fit(X, y)

    risk_scores = pipe.predict_proba(X)[:, 1]
    quantiles = np.percentile(risk_scores, np.arange(0, 101, 1)).tolist()

    return {
        "model": pipe,
        "feature_order": list(X.columns),
        "cat_cols": cats,
        "num_cols": nums,
        "pop_stats": {
            "stroke_rate_overall": float(df["stroke"].mean()),
            "age_mean": float(df["age"].mean()),
            "age_std": float(df["age"].std()),
            "glucose_mean": float(df["avg_glucose_level"].mean()),
            "bmi_mean": float(df["bmi"].mean()),
            "n_total": int(len(df)),
            "n_strokes": int(df["stroke"].sum()),
        },
        "risk_quantiles": quantiles,
    }


bundle = load_bundle()
model = bundle["model"]
feature_order = bundle["feature_order"]
cat_cols = bundle["cat_cols"]
num_cols = bundle["num_cols"]
pop_stats = bundle["pop_stats"]
risk_quantiles = np.array(bundle["risk_quantiles"])


# ---------- STYLING ----------
# 3-color palette: NAVY (#1A237E) for primary/brand/moderate, RED (#C62828) for high risk,
# GREEN (#2E7D32) for low risk. Everything else is grayscale.
st.markdown("""
<style>
.main { background-color: #FAFAFA; }
.stButton>button {
    background-color: #1A237E;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    border: none;
    width: 100%;
}
.stButton>button:hover {
    background-color: #0D1657;
    color: white;
}
.metric-card {
    background-color: white;
    padding: 1.2rem;
    border-radius: 10px;
    border: 1px solid #E0E0E0;
    margin-bottom: 1rem;
}
.app-header {
    background-color: #1A237E;
    color: white;
    padding: 1.4rem 1.8rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
}
.app-header h1 { color: white; margin: 0; font-size: 1.7rem; }
.app-header p  { color: #C5CAE9; margin: 0.2rem 0 0; font-size: 0.9rem; }
.disclaimer {
    background-color: #F5F5F5;
    border-left: 4px solid #1A237E;
    padding: 0.8rem 1rem;
    border-radius: 6px;
    font-size: 0.85rem;
    color: #424242;
}
.recommendation {
    background-color: #F5F5F5;
    border: 1px solid #1A237E;
    padding: 1rem 1.2rem;
    border-radius: 8px;
    margin-top: 1rem;
}
.recommendation h4 { color: #1A237E; margin: 0 0 0.4rem 0; }
</style>
""", unsafe_allow_html=True)


# ---------- HEADER ----------
st.markdown("""
<div class="app-header">
  <h1>🧠 StrokeRisk Predictor</h1>
  <p>ALY6040 Final Project | Powered by clinical data on 5,109 patients</p>
</div>
""", unsafe_allow_html=True)


# ---------- SIDEBAR INPUTS ----------
st.sidebar.header("Patient Profile")
st.sidebar.caption("Enter the patient's information below.")

age = st.sidebar.slider("Age", 1, 100, 55)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
hypertension = st.sidebar.radio("Hypertension", ["No", "Yes"], horizontal=True)
heart_disease = st.sidebar.radio("Heart Disease", ["No", "Yes"], horizontal=True)
ever_married = st.sidebar.selectbox("Married", ["Yes", "No"])
work_type = st.sidebar.selectbox(
    "Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
)
residence = st.sidebar.selectbox("Residence", ["Urban", "Rural"])
glucose = st.sidebar.slider("Average Glucose Level (mg/dL)", 50, 280, 105)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 28.0, step=0.5)
smoking_status = st.sidebar.selectbox(
    "Smoking Status",
    ["never smoked", "formerly smoked", "smokes", "Unknown"]
)

predict = st.sidebar.button("🔍 Calculate My Risk")
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="disclaimer"><b>For screening only.</b> '
    'This tool is built on a public dataset for an academic project. '
    'It is not a medical diagnosis.</div>',
    unsafe_allow_html=True,
)


# ---------- BUILD FEATURE ROW ----------
def build_row():
    row = {
        "gender": gender,
        "age": float(age),
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence,
        "avg_glucose_level": float(glucose),
        "bmi": float(bmi),
        "smoking_status": smoking_status,
        "glucose_high": 1 if glucose >= 125 else 0,
        "bmi_obese": 1 if bmi >= 30 else 0,
    }
    return pd.DataFrame([row], columns=feature_order)


# ---------- FEATURE CONTRIBUTION (logit-space) ----------
def compute_contributions(X):
    """Return ranked list of (feature_label, contribution_to_log_odds)."""
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    Xt = prep.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    coefs = clf.coef_[0]
    contribs = (Xt[0] * coefs)
    feat_names = (
        num_cols
        + list(prep.named_transformers_["cat"].get_feature_names_out(cat_cols))
    )
    pretty = {
        "age": "Age",
        "avg_glucose_level": "Glucose level",
        "bmi": "BMI",
        "hypertension": "Hypertension",
        "heart_disease": "Heart disease",
        "glucose_high": "Glucose >= 125",
        "bmi_obese": "BMI >= 30 (obese)",
        "gender_Male": "Gender: Male",
        "ever_married_Yes": "Married",
        "work_type_Private": "Work: Private",
        "work_type_Self-employed": "Work: Self-employed",
        "work_type_children": "Work: children",
        "work_type_Never_worked": "Work: Never worked",
        "Residence_type_Urban": "Residence: Urban",
        "smoking_status_formerly smoked": "Formerly smoked",
        "smoking_status_never smoked": "Never smoked",
        "smoking_status_smokes": "Currently smokes",
    }
    df = pd.DataFrame({
        "feature": [pretty.get(f, f) for f in feat_names],
        "contribution": contribs,
    })
    df["abs"] = df["contribution"].abs()
    return df.sort_values("abs", ascending=False)


# ---------- BAND HELPERS ----------
# Headline color is strictly binary: GREEN if low risk (good), RED otherwise (bad).
# The Low / Moderate / High label still shows the three bands for context.
GOOD_GREEN = "#2E7D32"
BAD_RED = "#C62828"
BAND_THRESHOLD = 0.20  # below this is green (good), at or above is red (bad)


def risk_band(prob):
    if prob < BAND_THRESHOLD:
        label = "Low"
    elif prob < 0.50:
        label = "Moderate"
    else:
        label = "High"
    color = GOOD_GREEN if prob < BAND_THRESHOLD else BAD_RED
    return label, color


def percentile_rank(prob):
    return int(np.searchsorted(risk_quantiles, prob, side="left"))


# Plain-English reason for each feature, used by the personalized explainer.
# Returned reason depends on whether the user has the trait (push direction).
def feature_reason(feature_label, contribution_sign, user_inputs):
    age = user_inputs["age"]
    glucose = user_inputs["glucose"]
    bmi = user_inputs["bmi"]
    if feature_label == "Age":
        if contribution_sign > 0:
            return (f"At age {int(age)}, the data shows stroke risk roughly doubles every "
                    "decade after 55. This is the single strongest signal in the dataset.")
        return (f"At age {int(age)}, you're below the age range where stroke risk climbs sharply.")
    if feature_label == "Glucose level":
        if contribution_sign > 0:
            return (f"Your glucose level of {int(glucose)} mg/dL is in or near the diabetic range, "
                    "which the data links to higher stroke rates.")
        return (f"Your glucose level of {int(glucose)} mg/dL is in the safe range.")
    if feature_label == "Glucose >= 125":
        return ("You're flagged as above the diabetic glucose threshold, a clinically established "
                "stroke risk factor.")
    if feature_label == "BMI":
        if contribution_sign > 0:
            return (f"A BMI of {bmi:.1f} sits on the high side. Higher BMI correlates with stroke "
                    "in the training data.")
        return f"A BMI of {bmi:.1f} is in a healthy range."
    if feature_label == "BMI >= 30 (obese)":
        return "You're flagged as obese (BMI ≥ 30), which raises stroke risk in the data."
    if feature_label == "Hypertension":
        if contribution_sign > 0:
            return ("Patients with hypertension had 13.3% stroke rate vs. 4.0% for those without. "
                    "It's one of the top clinical risk factors.")
        return "No hypertension means you avoid one of the strongest clinical risk factors."
    if feature_label == "Heart disease":
        if contribution_sign > 0:
            return ("Patients with heart disease had 17.0% stroke rate vs. 4.2% for those without. "
                    "Heart disease shares a lot of underlying biology with stroke.")
        return "No heart disease means you avoid one of the strongest clinical risk factors."
    if feature_label == "Married":
        if contribution_sign > 0:
            return ("In this dataset, married patients had higher stroke rates, partly because "
                    "they tend to be older.")
        return "Being unmarried is associated with lower stroke risk in the data, mostly an age effect."
    if feature_label == "Currently smokes":
        return "Smoking is a known stroke risk factor."
    if feature_label == "Formerly smoked":
        if contribution_sign > 0:
            return ("Former smokers had the highest stroke rate in the dataset (7.9%), likely "
                    "because many quit after a health scare.")
        return "Former smokers had a higher stroke rate, but this didn't apply strongly to your profile."
    if feature_label == "Never smoked":
        if contribution_sign < 0:
            return "Never-smokers had a lower stroke rate. The model treats this as protective."
        return "Never-smoker status didn't move the needle much for your profile."
    if feature_label == "Gender: Male":
        if contribution_sign > 0:
            return "Male gender slightly raises predicted risk in this dataset."
        return "Female gender is associated with slightly lower stroke risk in the data."
    if feature_label == "Residence: Urban":
        if contribution_sign > 0:
            return "Urban residence shows a small association with higher stroke rate, possibly tied to lifestyle factors."
        return "Rural residence shows a small protective effect in the data."
    if feature_label.startswith("Work:"):
        return f"{feature_label} is associated with this risk direction in the dataset."
    return "This trait shifts predicted risk based on patterns in the training data."


# ---------- MAIN PANEL ----------
if not predict:
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("How it works")
        st.write("""
        Fill in the patient profile on the left, then click **Calculate My Risk**.
        The app uses a logistic regression model trained with SMOTE oversampling on
        5,109 patient records. It returns a stroke-risk percentage, the factors driving
        that score, where the patient sits compared to the population, and a recommended
        next step.
        """)
        st.markdown("**Model performance (test set)**")
        st.write("""
        - 5-fold cross-validated ROC-AUC: 0.84
        - Recall on stroke cases: 82% (catches 41 of 50 real stroke patients)
        - Trained on Kaggle's Brain Stroke Prediction Dataset (fedesoriano, 2021)
        """)
    with col_right:
        st.subheader("Population snapshot")
        st.metric("Patients in training data", f"{pop_stats['n_total']:,}")
        st.metric("Patients with stroke", f"{pop_stats['n_strokes']:,}")
        st.metric("Overall stroke rate", f"{pop_stats['stroke_rate_overall']*100:.2f}%")
        st.metric("Average age", f"{pop_stats['age_mean']:.1f} yrs")

else:
    X_input = build_row()
    proba = model.predict_proba(X_input)[0, 1]
    band, band_color = risk_band(proba)
    pct = percentile_rank(proba)

    # ============ TOP RESULT CARD ============
    st.subheader("Your Risk Profile")
    g1, g2, g3 = st.columns([2, 2, 2])
    with g1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 6px solid {band_color};">
          <div style="font-size: 0.85rem; color: #607D8B;">Predicted Stroke Risk</div>
          <div style="font-size: 2.3rem; font-weight: bold; color: {band_color}; margin: 0.2rem 0;">
            {proba*100:.1f}%
          </div>
          <div style="font-size: 1rem; color: {band_color}; font-weight: bold;">{band} Risk Band</div>
        </div>
        """, unsafe_allow_html=True)
    # The "Top X%" metric: smaller number = closer to the highest-risk slice = bad.
    # Use red if user is in the top half (top X% < 50), green if bottom half.
    top_pct = 100 - pct
    pop_color = BAD_RED if top_pct < 50 else GOOD_GREEN
    with g2:
        st.markdown(f"""
        <div class="metric-card">
          <div style="font-size: 0.85rem; color: #607D8B;">Compared to Population</div>
          <div style="font-size: 2.3rem; font-weight: bold; color: {pop_color}; margin: 0.2rem 0;">
            Top {top_pct}%
          </div>
          <div style="font-size: 0.85rem; color: #455A64;">of stroke risk in the training data</div>
        </div>
        """, unsafe_allow_html=True)
    with g3:
        st.markdown(f"""
        <div class="metric-card">
          <div style="font-size: 0.85rem; color: #607D8B;">Population Average Risk</div>
          <div style="font-size: 2.3rem; font-weight: bold; color: #455A64; margin: 0.2rem 0;">
            {pop_stats['stroke_rate_overall']*100:.2f}%
          </div>
          <div style="font-size: 0.85rem; color: #455A64;">across all 5,109 patients</div>
        </div>
        """, unsafe_allow_html=True)

    # ============ PERSONALIZED EXPLAINER ============
    with st.expander(f"ℹ️  Why the model says your risk is {band.lower()} ({proba*100:.1f}%)"):
        # Pull this user's actual contributions
        contribs_df = compute_contributions(X_input)
        risk_drivers = contribs_df[contribs_df["contribution"] > 0].head(3)
        protectives = contribs_df[contribs_df["contribution"] < 0].head(3)

        user_inputs = {"age": age, "glucose": glucose, "bmi": bmi}

        st.markdown(f"#### What pushed your score **up**")
        if len(risk_drivers) == 0:
            st.markdown("Nothing in your profile flagged as a stroke risk factor.")
        else:
            for _, row in risk_drivers.iterrows():
                reason = feature_reason(row["feature"], row["contribution"], user_inputs)
                st.markdown(
                    f"- **{row['feature']}** &nbsp; "
                    f"<span style='color:{BAD_RED};font-size:0.85rem;'>"
                    f"contribution: {row['contribution']:+.2f}</span><br>"
                    f"<span style='color:#455A64;font-size:0.92rem;'>{reason}</span>",
                    unsafe_allow_html=True
                )

        st.markdown(f"#### What pushed your score **down**")
        if len(protectives) == 0:
            st.markdown("Nothing in your profile reduced your stroke risk.")
        else:
            for _, row in protectives.iterrows():
                reason = feature_reason(row["feature"], row["contribution"], user_inputs)
                st.markdown(
                    f"- **{row['feature']}** &nbsp; "
                    f"<span style='color:{GOOD_GREEN};font-size:0.85rem;'>"
                    f"contribution: {row['contribution']:+.2f}</span><br>"
                    f"<span style='color:#455A64;font-size:0.92rem;'>{reason}</span>",
                    unsafe_allow_html=True
                )

        # Reach the conclusion
        net = float(contribs_df["contribution"].sum())
        if proba >= BAND_THRESHOLD:
            verdict = (
                f"The risk-raising factors outweighed the protective ones (net log-odds **{net:+.2f}**). "
                f"After running through the model's sigmoid function, that net score maps to a "
                f"**{proba*100:.1f}%** prediction. Above our 20% screening threshold, so the app flags "
                f"this profile as worth a clinician's attention."
            )
        else:
            verdict = (
                f"The protective factors outweighed the risk-raising ones (net log-odds **{net:+.2f}**). "
                f"After the sigmoid step, that maps to a **{proba*100:.1f}%** prediction. Below our 20% "
                f"screening threshold, so the app flags this profile as low risk."
            )
        st.markdown(f"#### Bottom line")
        st.markdown(verdict)

        st.markdown("---")
        st.markdown(
            "<span style='color:#607D8B;font-size:0.85rem;'>"
            "<b>How the math works:</b> the model is a logistic regression trained with SMOTE oversampling "
            "on 5,109 patient records. For each input, it learned a coefficient during training. Your "
            "inputs get multiplied by those coefficients, the contributions are summed, and the total is "
            "passed through a sigmoid to produce the percentage above. The bars in the Top Risk Drivers "
            "panel below show those individual contributions in order. SMOTE shifts the raw probability "
            "scale upward, so the percentage reads as 'how far above average risk' rather than a literal "
            "X-out-of-100 probability."
            "</span>",
            unsafe_allow_html=True,
        )

    # ============ RISK BAR ============
    st.markdown("**Risk gauge**")
    bar_col1, bar_col2, bar_col3 = st.columns([1, 6, 1])
    with bar_col2:
        st.progress(min(float(proba), 1.0))
    legend = st.columns(3)
    with legend[0]:
        st.markdown('<div style="text-align:left;color:#2E7D32;">Low &lt; 20% (good)</div>',
                    unsafe_allow_html=True)
    with legend[1]:
        st.markdown('<div style="text-align:center;color:#C62828;">Moderate 20-50%</div>',
                    unsafe_allow_html=True)
    with legend[2]:
        st.markdown('<div style="text-align:right;color:#C62828;">High &gt; 50% (bad)</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # ============ TOP DRIVERS ============
    drivers_col, peer_col = st.columns([3, 2])

    with drivers_col:
        st.subheader("Top Risk Drivers for You")
        st.caption("How much each factor pushed the score up (red) or down (green).")
        c = compute_contributions(X_input).head(8).copy()
        c["direction"] = c["contribution"].apply(lambda x: "↑ Increases risk" if x > 0 else "↓ Decreases risk")
        c["color"] = c["contribution"].apply(lambda x: "#C62828" if x > 0 else "#2E7D32")
        max_abs = c["abs"].max()
        for _, r in c.iterrows():
            pct_bar = abs(r["contribution"]) / max_abs * 100
            st.markdown(f"""
            <div style="margin-bottom: 0.55rem;">
              <div style="display:flex; justify-content:space-between; font-size:0.92rem;">
                <span><b>{r['feature']}</b></span>
                <span style="color:{r['color']}; font-size:0.85rem;">{r['direction']}</span>
              </div>
              <div style="background:#ECEFF1; border-radius:4px; height:10px; margin-top:3px;">
                <div style="width:{pct_bar:.0f}%; height:10px; background:{r['color']}; border-radius:4px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with peer_col:
        st.subheader("How You Compare")
        st.caption("Where this risk score falls in the patient population.")
        # If user beats more than half the population, that's worse risk = red.
        peer_color = BAD_RED if pct >= 50 else GOOD_GREEN
        st.markdown(f"""
        <div class="metric-card" style="text-align:center;">
          <div style="font-size:0.9rem; color:#607D8B; margin-bottom:0.3rem;">
            You scored higher than
          </div>
          <div style="font-size:3rem; font-weight:bold; color:{peer_color};">{pct}%</div>
          <div style="font-size:0.9rem; color:#455A64;">
            of the {pop_stats['n_total']:,} patients in the training data.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ---- Recommendation banner (color matches the band) ----
        rec_color = BAD_RED if proba >= BAND_THRESHOLD else GOOD_GREEN
        rec_bg = "#FBEAEA" if proba >= BAND_THRESHOLD else "#E8F5E9"
        if proba >= 0.50:
            rec_title = "Schedule a clinical review soon"
            rec_body = (
                "Two or more of your top drivers are clinically modifiable. "
                "A clinician can confirm the risk with a blood pressure check, "
                "fasting glucose panel, and lifestyle assessment."
            )
        elif proba >= 0.20:
            rec_title = "Worth discussing at your next visit"
            rec_body = (
                "Bring your blood pressure and glucose history to your next "
                "annual physical. The model flagged moderate risk."
            )
        else:
            rec_title = "Continue routine monitoring"
            rec_body = (
                "Your profile sits in the lower-risk band. Maintain regular "
                "check-ups and keep an eye on weight and blood pressure."
            )
        st.html(
            f'<div style="background:{rec_bg};border:1px solid {rec_color};'
            f'padding:1rem 1.2rem;border-radius:8px;margin-top:0.8rem;">'
            f'<h4 style="color:{rec_color};margin:0 0 0.4rem 0;">{rec_title}</h4>'
            f'<p style="margin:0; font-size:0.92rem; color:#263238;">{rec_body}</p>'
            f'</div>'
        )

    # ============ FULL-WIDTH TIPS SECTION ============
    st.markdown("")
    st.subheader("Tips to reduce your stroke risk")
    st.caption("Personalized to your modifiable risk factors.")

    tips = []
    if hypertension == "Yes":
        tips.append(("Manage your blood pressure",
                     "Monitor daily; target under 130/80."))
    if glucose >= 125:
        tips.append(("Get a fasting glucose / A1C panel",
                     f"{int(glucose)} mg/dL is diabetic range. Diabetes is a top stroke driver."))
    elif glucose >= 100:
        tips.append(("Watch your blood sugar",
                     "Cut sugary drinks and refined carbs."))
    if bmi >= 30:
        tips.append(("Work toward a healthier weight",
                     f"BMI {bmi:.1f} is obese; even 5-10% weight loss measurably drops risk."))
    elif bmi >= 25:
        tips.append(("Aim to bring BMI under 25",
                     "Modest weight loss plus 150 min/week of exercise."))
    if smoking_status == "smokes":
        tips.append(("Quit smoking",
                     "Quitting halves stroke risk within 5 years."))
    elif smoking_status == "formerly smoked":
        tips.append(("Stay quit",
                     "Risk keeps dropping the longer you stay smoke-free."))
    if heart_disease == "Yes":
        tips.append(("Stay on top of your heart care",
                     "Keep cardiology follow-ups and medications on schedule."))
    if proba < 0.20 and not tips:
        tips.append(("Keep up your healthy habits",
                     "Regular check-ups, balanced diet, 150+ min/week exercise."))
    if not tips:
        tips.append(("Annual screening",
                     "Most of your top drivers (like age) are not directly modifiable."))

    # Build the inner grid HTML in a single line per tip so Streamlit's markdown parser
    # doesn't treat indented HTML as a code block.
    tip_cards = ""
    for title, body in tips:
        tip_cards += (
            f'<div><div style="font-weight:500;font-size:14px;color:{rec_color};margin-bottom:4px;">'
            f'&bull; {title}</div>'
            f'<div style="font-size:13px;color:#455A64;line-height:1.5;">{body}</div></div>'
        )
    st.html(
        f'<div style="background:{rec_bg};border-left:4px solid {rec_color};'
        f'border-radius:6px;padding:18px 22px;">'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:18px 24px;">'
        f'{tip_cards}'
        f'</div></div>'
    )

    st.markdown("---")
    with st.expander("See the raw input the model received"):
        st.dataframe(X_input.T.rename(columns={0: "value"}))

    st.caption(
        "🩺 Built for ALY6040 Final Project — not a medical diagnosis. "
        "Always consult a qualified clinician for personal health decisions."
    )
