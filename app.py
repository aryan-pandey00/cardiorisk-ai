import streamlit as st
import numpy as np
import joblib

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# ------------------ LOAD MODEL ------------------
model = joblib.load("heart_rf_model.pkl")

# ------------------ HEADER ------------------
st.title("❤️ Heart Disease Risk Predictor")
st.caption("Machine Learning–based Heart Disease Risk Estimation")

st.divider()

# -------- PATIENT INFORMATION --------
st.markdown("#### Personal Information")
col1, col2 = st.columns(2)

with col1:
    patient_name = st.text_input("Patient Name", placeholder="Enter name")
    age = st.number_input("Age (years)", 20, 100, 45)

with col2:
    blood_group = st.selectbox(
        "Blood Group",
        ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    )
    sex = st.selectbox("Sex", ["Male", "Female"])
    sex = 1 if sex == "Male" else 0

st.divider()

# -------- SYMPTOMS --------
st.markdown("#### Symptoms")
col1, col2 = st.columns(2)

with col1:
    cp_label = st.selectbox(
        "Chest Pain Type (cp)",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    )
    cp_map = {
        "Typical Angina": 1,
        "Atypical Angina": 2,
        "Non-anginal Pain": 3,
        "Asymptomatic": 4
    }
    cp = cp_map[cp_label]

with col2:
    exang = st.selectbox("Exercise-Induced Angina (exang)", ["No", "Yes"])
    exang = 1 if exang == "Yes" else 0

st.divider()

# -------- VITALS & LABS --------
st.markdown("#### Vitals & Laboratory")
col1, col2 = st.columns(2)

with col1:
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)(trestbps)", 80, 250, 130)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 700, 240)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl (fbs)", ["No", "Yes"])
    fbs = 1 if fbs == "Yes" else 0
    thalach = st.number_input("Maximum Heart Rate (thalach)", 60, 250, 150)

st.divider()

# -------- CARDIAC TESTS --------
st.markdown("#### Cardiac Tests")
col1, col2 = st.columns(2)

with col1:
    restecg_label = st.selectbox(
        "Resting ECG (restecg)",
        ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"]
    )
    restecg_map = {
        "Normal": 0,
        "ST-T Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    restecg = restecg_map[restecg_label]

    slope_label = st.selectbox(
        "ST Segment Slope",
        ["Upsloping", "Flat", "Downsloping"]
    )
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    slope = slope_map[slope_label]

    ca = st.slider("Number of Major Vessels (ca)", 0, 4, 0)

with col2:
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, step=0.1)

    thal_label = st.selectbox(
        "Thalassemia (thal)",
        ["Normal", "Fixed Defect", "Reversible Defect"]
    )
    thal_map = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}
    thal = thal_map[thal_label]

import plotly.graph_objects as go

# ================== PREDICT ==================
if st.button("Predict Heart Disease Risk", use_container_width=True):

    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]])

    probability = model.predict_proba(input_data)[0][1]
    threshold = 0.4
    prediction = 1 if probability >= threshold else 0

    st.subheader("Prediction Result")

    # -------- RISK CATEGORY --------

    if probability < 0.20:
        st.success(f"Risk Level: Low ({probability:.1%})")
    elif probability < 0.50:
        st.warning(f"Risk Level: Moderate ({probability:.1%})")
    elif probability < 0.75:
        st.error(f"Risk Level: High ({probability:.1%})")
    else:
        st.error(f"Risk Level: Very High ({probability:.1%})")

    # -------- GAUGE --------

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#111827"},
            'steps': [
                {'range': [0, 20], 'color': "#2ecc71"},
                {'range': [20, 50], 'color': "#f1c40f"},
                {'range': [50, 75], 'color': "#e67e22"},
                {'range': [75, 100], 'color': "#e74c3c"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # -------- PATIENT SUMMARY --------
    name_row = f"<div style='margin-bottom:6px;'><b>Name:</b> {patient_name}</div>" if patient_name else ""
    bg_row = f"<div style='margin-bottom:6px;'><b>Blood Group:</b> {blood_group}</div>" if blood_group else ""
    
    st.markdown("**Patient Summary**")

    summary_html = f"""
    <div style="
        background-color: rgba(100, 116, 139, 0.12);
        padding:20px;
        border-radius:10px;
        border:1px solid rgba(100, 116, 139, 0.25);
        margin-top:8px;
        margin-bottom:16px;
    ">
    {name_row}
    {bg_row}
    <div style="margin-bottom:6px;"><b>Age:</b> {age} yrs</div>
    <div style="margin-bottom:6px;"><b>Chest Pain:</b> {cp_label}</div>
    <div style="margin-bottom:6px;"><b>Blood Pressure:</b> {trestbps} mmHg</div>
    <div style="margin-bottom:6px;"><b>Cholesterol:</b> {chol} mg/dl</div>
    <div><b>Major Vessels:</b> {ca}</div>
    </div>
    """
        
    st.markdown(summary_html, unsafe_allow_html=True)

        # -------- CONTRIBUTING FACTORS --------
    factors = []

    if trestbps >= 140:
        factors.append("Elevated resting blood pressure")
    if chol >= 240:
        factors.append("High serum cholesterol")
    if exang == 1:
        factors.append("Exercise-induced angina")
    if oldpeak >= 2:
        factors.append("ST-segment depression")
    if ca >= 2:
        factors.append("Multiple major vessels involvement")
    if cp == 4:
        factors.append("Asymptomatic chest pain pattern")

    st.markdown("**Key Contributing Clinical Factors**")


    if probability < 0.20:
        # Low risk 
        factors_html = (
        "<div>No major clinical risk indicators are strongly elevated based on the entered values.</div>"
        "<div style='margin-top:6px;'>"
        "The current parameter combination does not show a dominant high-risk pattern."
        "</div>"
    )

    elif probability < 0.50:
        # Moderate risk 
        if len(factors) >= 2:
            factors_html = "<ul style='margin:0; padding-left:18px;'>"
            for f in factors:
                factors_html += f"<li>{f}</li>"
            factors_html += "</ul>"
        elif len(factors) == 1:
            factors_html = (
                "<div>One clinical parameter is elevated. "
                "However, the overall combination suggests moderate estimated risk.</div>"
            )
        else:
            factors_html = (
                "<div>No dominant individual risk factors detected. "
                "The prediction is influenced by combined clinical pattern.</div>"
            )

    else:
        # High / Very High risk 
        if factors:
            factors_html = "<ul style='margin:0; padding-left:18px;'>"
            for f in factors:
                factors_html += f"<li>{f}</li>"
            factors_html += "</ul>"
        else:
            factors_html = (
                "<div>The elevated risk is influenced by combined clinical interactions "
                "rather than a single dominant parameter.</div>"
            )

    factors_card = f"""
    <div style="
        background-color: rgba(100, 116, 139, 0.12);
        padding:18px 20px;
        border-radius:10px;
        border:1px solid rgba(100, 116, 139, 0.25);
        margin-top:8px;
        margin-bottom:16px;
    ">
    {factors_html}
    </div>
    """

    st.markdown(factors_card, unsafe_allow_html=True)

    # -------- INTERPRETATION --------
    
    if probability < 0.20:
        st.info("The entered clinical parameters resemble patterns commonly associated with lower likelihood of heart disease.")
    elif probability < 0.50:
        st.info("Some clinical risk indicators are present. Further medical evaluation may be beneficial.")
    elif probability < 0.75:
        st.info("The clinical profile shows multiple features associated with increased likelihood of heart disease.")
    else:
        st.info("This result indicates a strong pattern similarity to cases with diagnosed heart disease. Immediate clinical evaluation is recommended.")

    # -------- DISCLAIMER --------
    
    st.caption(
        "This application provides machine-learning–based heart disease risk "
        "estimation for educational purposes only and is not a medical diagnosis."
    )

