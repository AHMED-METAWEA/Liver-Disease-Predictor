import streamlit as st
import numpy as np
import pickle
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Liver Disease Predictor",
    page_icon="🫀",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 100%);
}

/* Card container */
.predict-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}

/* Section label */
.section-label {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #7c84a3;
    margin-bottom: 1rem;
}

/* Result boxes */
.result-positive {
    background: linear-gradient(135deg, #ff4d4d22, #ff000011);
    border: 1px solid #ff4d4d66;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    color: #ff6b6b;
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
}

.result-negative {
    background: linear-gradient(135deg, #00e68822, #00ff9911);
    border: 1px solid #00e68866;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    color: #2ecc88;
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
}

.confidence-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 8px;
    margin-top: 0.5rem;
}

.stSlider > div > div { color: #a0a8c3; }
stNumberInput input { background: rgba(255,255,255,0.05); }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for path in [
        os.path.join(base_dir, "..", "models", "model.pkl"),
        os.path.join(base_dir, "models", "model.pkl"),
    ]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                obj = pickle.load(f)
            # If it's a dict, extract the model
            if isinstance(obj, dict):
                return obj.get("model") or obj.get("classifier") or list(obj.values())[0]
            return obj
    return None

model = load_model()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🫀 Liver Disease Predictor")
st.markdown(
    "<p style='color:#7c84a3; margin-top:-0.5rem; margin-bottom:2rem;'>"
    "Enter patient lab values below to predict liver disease risk.</p>",
    unsafe_allow_html=True,
)

if model is None:
    st.warning(
        "⚠️ `models/model.pkl` not found. "
        "Run `train_model.py` first, then relaunch this app.",
        icon="⚠️",
    )


# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="predict-card">', unsafe_allow_html=True)
st.markdown('<p class="section-label">Patient Demographics</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45, step=1)
with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])

st.markdown('</div>', unsafe_allow_html=True)

# ── Liver function tests ───────────────────────────────────────────────────────
st.markdown('<div class="predict-card">', unsafe_allow_html=True)
st.markdown('<p class="section-label">Liver Function Tests</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    total_bilirubin       = st.number_input("Total Bilirubin (mg/dL)",        min_value=0.0,  max_value=75.0,  value=0.9,  step=0.1)
    direct_bilirubin      = st.number_input("Direct Bilirubin (mg/dL)",       min_value=0.0,  max_value=20.0,  value=0.2,  step=0.1)
    alkaline_phosphotase  = st.number_input("Alkaline Phosphotase (IU/L)",    min_value=0,    max_value=2000,  value=187,  step=1)
    alamine_aminotransferase = st.number_input("Sgpt Alamine Aminotransferase (IU/L)", min_value=0, max_value=2000, value=16,  step=1)

with col2:
    aspartate_aminotransferase = st.number_input("Sgot Aspartate Aminotransferase (IU/L)", min_value=0, max_value=5000, value=18, step=1)
    total_proteins        = st.number_input("Total Proteins (g/dL)",          min_value=0.0,  max_value=10.0,  value=6.8,  step=0.1)
    albumin               = st.number_input("Albumin (g/dL)",                 min_value=0.0,  max_value=6.0,   value=3.3,  step=0.1)
    albumin_globulin_ratio = st.number_input("Albumin/Globulin Ratio",        min_value=0.0,  max_value=3.0,   value=0.9,  step=0.01)

st.markdown('</div>', unsafe_allow_html=True)


# ── Predict ───────────────────────────────────────────────────────────────────
gender_encoded = 1 if gender == "Male" else 0

features = np.array([[
    age,
    gender_encoded,
    total_bilirubin,
    direct_bilirubin,
    alkaline_phosphotase,
    alamine_aminotransferase,
    aspartate_aminotransferase,
    total_proteins,
    albumin,
    albumin_globulin_ratio,
]])

predict_btn = st.button("🔍 Run Prediction", use_container_width=True, type="primary")

if predict_btn:
    if model is None:
        st.error("Model not loaded. Please train the model first.")
    else:
        with st.spinner("Running model inference…"):
            prediction = model.predict(features)[0]
            has_proba  = hasattr(model, "predict_proba")
            confidence = model.predict_proba(features)[0].max() * 100 if has_proba else None

        st.markdown("---")
        st.markdown("### Result")

        if prediction == 1:
            st.markdown(
                '<div class="result-positive">'
                '⚠️ Liver Disease Detected<br>'
                '<small style="font-family:\'DM Sans\',sans-serif; font-size:0.85rem; opacity:0.75;">'
                'This patient may have liver disease. Please consult a specialist.</small>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-negative">'
                '✅ No Liver Disease Detected<br>'
                '<small style="font-family:\'DM Sans\',sans-serif; font-size:0.85rem; opacity:0.75;">'
                'Results appear normal. Continue routine monitoring.</small>'
                '</div>',
                unsafe_allow_html=True,
            )

        if confidence is not None:
            st.markdown(f"**Model Confidence:** `{confidence:.1f}%`")
            st.progress(int(confidence))

        # Feature summary
        with st.expander("📋 Input Summary"):
            import pandas as pd
            summary = pd.DataFrame({
                "Feature": [
                    "Age", "Gender", "Total Bilirubin", "Direct Bilirubin",
                    "Alkaline Phosphotase", "Alamine Aminotransferase",
                    "Aspartate Aminotransferase", "Total Proteins",
                    "Albumin", "Albumin/Globulin Ratio"
                ],
                "Value": [
                    age, gender, total_bilirubin, direct_bilirubin,
                    alkaline_phosphotase, alamine_aminotransferase,
                    aspartate_aminotransferase, total_proteins,
                    albumin, albumin_globulin_ratio,
                ]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#444a66; font-size:0.75rem;'>"
    "For clinical reference only — not a substitute for professional medical advice."
    "</p>",
    unsafe_allow_html=True,
)