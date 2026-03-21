import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model
model = joblib.load("heart_disease_model.joblib")

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# 🎥 BACKGROUND + FULL UI FIX
st.markdown("""
<style>

/* Background */
.stApp {
    background:
    linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
    url("https://media.giphy.com/media/I9blWoLkS46CQ/giphy.gif");
    background-size: cover;
    background-position: center;
}

/* Headings */
h1, h2, h3 {
    color: white !important;
}

/* Labels */
label {
    color: #f0f0f0 !important;
}

/* Fix cursor (VERY IMPORTANT) */
input, textarea {
    caret-color: white !important;
}

/* Input box */
input {
    background-color: #1e1e1e !important;
    color: white !important;
}

/* Selectbox container */
div[data-baseweb="select"] > div {
    background-color: #1e1e1e !important;
    color: white !important;
    border-radius: 5px;
}

/* Dropdown list */
div[role="listbox"] {
    background-color: #2b2b2b !important;
    color: white !important;
}

/* Dropdown options */
div[role="option"] {
    color: white !important;
}

/* Force arrow to appear */
div[data-baseweb="select"] svg {
    fill: white !important;
}

/* Result cards */
.stAlert {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# Title
st.title("❤️ Heart Disease Prediction System")

# ---------------- INPUTS ---------------- #

age = st.number_input("Age")

sex_option = st.selectbox("Sex", ["Female", "Male"])
sex = 0 if sex_option == "Female" else 1

cp_option = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
)
cp_map = {"Typical Angina":0,"Atypical Angina":1,"Non-anginal Pain":2,"Asymptomatic":3}
cp = cp_map[cp_option]

trestbps = st.number_input("Resting Blood Pressure")

chol = st.number_input("Cholesterol")

fbs_option = st.selectbox("Fasting Blood Sugar >120", ["No","Yes"])
fbs = 1 if fbs_option == "Yes" else 0

restecg_option = st.selectbox(
    "Rest ECG",
    ["Normal","ST-T Abnormality","LV Hypertrophy"]
)
restecg_map = {"Normal":0,"ST-T Abnormality":1,"LV Hypertrophy":2}
restecg = restecg_map[restecg_option]

thalach = st.number_input("Max Heart Rate")

exang_option = st.selectbox("Exercise Angina", ["No","Yes"])
exang = 1 if exang_option == "Yes" else 0

oldpeak = st.number_input("Oldpeak")

slope_option = st.selectbox(
    "ST Slope",
    ["Upsloping","Flat","Downsloping"]
)
slope_map = {"Upsloping":0,"Flat":1,"Downsloping":2}
slope = slope_map[slope_option]

# ---------------- PREDICTION ---------------- #

if st.button("🔍 Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol,
                            fbs, restecg, thalach,
                            exang, oldpeak, slope]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    result = "Disease" if prediction[0] == 1 else "No Disease"

    # Result
    st.subheader("📊 Result")

    if prediction[0] == 1:
        st.error(f"⚠️ Heart Disease Detected ({probability*100:.2f}% risk)")
    else:
        st.success(f"✅ No Heart Disease ({(1-probability)*100:.2f}% safe)")

    # ---------------- 🧠 AI EXPLANATION ---------------- #

    st.subheader("🧠 AI Explanation (Why this result?)")

    reasons = []

    if age > 50:
        reasons.append("Age is high")

    if chol > 240:
        reasons.append("High cholesterol")

    if trestbps > 140:
        reasons.append("High blood pressure")

    if thalach < 100:
        reasons.append("Low heart rate")

    if oldpeak > 2:
        reasons.append("High ST depression")

    if exang == 1:
        reasons.append("Exercise-induced angina present")

    if cp == 3:
        reasons.append("Asymptomatic chest pain (risky)")

    if reasons:
        st.markdown("""
        <div style="
        background: rgba(255,165,0,0.2);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid orange;
        color: white;">
        ⚠️ <b>Possible Reasons:</b>
        </div>
        """, unsafe_allow_html=True)

        for r in reasons:
            st.markdown(f"<p style='color:white;'>• {r}</p>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
        background: rgba(0,200,0,0.2);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid lime;
        color: white;">
        ✅ No major risk factors detected
        </div>
        """, unsafe_allow_html=True)

    # ---------------- CSV DOWNLOAD ---------------- #

    df = pd.DataFrame([{
        "Age": age,
        "Sex": sex_option,
        "Chest Pain": cp_option,
        "BP": trestbps,
        "Cholesterol": chol,
        "FBS": fbs_option,
        "ECG": restecg_option,
        "Max HR": thalach,
        "Angina": exang_option,
        "Oldpeak": oldpeak,
        "Slope": slope_option,
        "Prediction": result,
        "Risk %": round(probability*100, 2)
    }])

    st.download_button(
        "⬇️ Download Report CSV",
        data=df.to_csv(index=False),
        file_name="heart_prediction.csv",
        mime="text/csv"
    )
