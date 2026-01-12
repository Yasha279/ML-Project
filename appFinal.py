# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np

# # --- 1. Load the Saved Model and Scaler ---
# @st.cache_resource # This keeps the model in memory so it doesn't reload every time
# def load_assets():
#     model = joblib.load('cardio_model.joblib')
#     scaler = joblib.load('scaler.joblib')
#     return model, scaler

# model, scaler = load_assets()

# # --- 2. UI Setup ---
# st.set_page_config(page_title="Cardio Health Predictor", layout="centered")

# st.title("❤️ Cardiovascular Disease Predictor")
# st.write("Enter the patient's health details below to check for cardiovascular risk.")

# # Create a form for user inputs
# with st.form("prediction_form"):
#     st.header("Patient Information")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         age = st.number_input("Age (Years)", min_value=1, max_value=120, value=50)
#         gender = st.selectbox("Gender", options=["Female", "Male"])
#         height = st.number_input("Height (cm)", min_value=50, max_value=250, value=165)
#         weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70)
#         active = st.radio("Physical Activity", options=["Inactive", "Active"])

#     with col2:
#         ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=80, max_value=250, value=120)
#         ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=80)
#         cholesterol = st.selectbox("Cholesterol Level", options=["Normal", "Above Normal", "Well Above Normal"])
#         gluc = st.selectbox("Glucose Level", options=["Normal", "Above Normal", "Well Above Normal"])
#         habits = st.multiselect("Habits", options=["Smoker", "Alcohol Consumption"])

#     submit_button = st.form_submit_button(label="Predict Risk")

# # --- 3. Prediction Logic ---
# if submit_button:
#     # Prepare Binary/Categorical mapping based on training (One-Hot Encoding logic)
#     # Gender: Female=1, Male=2 (In training). gender_2 is the column after drop_first.
#     gender_2 = 1 if gender == "Male" else 0
    
#     # Habits
#     smoke = 1 if "Smoker" in habits else 0
#     alco = 1 if "Alcohol Consumption" in habits else 0
#     active_val = 1 if active == "Active" else 0
    
#     # Cholesterol & Glucose Mapping
#     # Logic: Normal=1, Above Normal=2, Well Above Normal=3
#     # cholesterol_2 and cholesterol_3 are the Dummy columns
#     c2, c3 = (1, 0) if cholesterol == "Above Normal" else (0, 1) if cholesterol == "Well Above Normal" else (0, 0)
#     g2, g3 = (1, 0) if gluc == "Above Normal" else (0, 1) if gluc == "Well Above Normal" else (0, 0)
    
#     # Create input DataFrame (order must match training exactly!)
#     # Numerical: age, height, weight, ap_hi, ap_lo
#     # Binary/OHE: smoke, alco, active, gender_2, cholesterol_2, cholesterol_3, gluc_2, gluc_3
#     input_data = pd.DataFrame([[
#         age, height, weight, ap_hi, ap_lo, 
#         smoke, alco, active_val, gender_2, 
#         c2, c3, g2, g3
#     ]], columns=['age', 'height', 'weight', 'ap_hi', 'ap_lo', 
#                  'smoke', 'alco', 'active', 'gender_2', 
#                  'cholesterol_2', 'cholesterol_3', 'gluc_2', 'gluc_3'])
    
#     # Scale numerical columns
#     num_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
#     input_data[num_cols] = scaler.transform(input_data[num_cols])
    
#     # Get Prediction
#     prediction = model.predict(input_data)[0]
#     probability = model.predict_proba(input_data)[0][1]
    
#     # Display Result
#     st.divider()
#     if prediction == 1:
#         st.error(f"### High Risk Detected ⚠️")
#         st.write(f"Confidence: **{probability*100:.2f}%**")
#         st.write("Please consult a healthcare professional for further evaluation.")
#     else:
#         st.success(f"### Low Risk Detected ✅")
#         st.write(f"Confidence: **{(1-probability)*100:.2f}%**")
#         st.write("Maintain a healthy lifestyle and regular checkups!")


# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np
# # @st.cache_resource
# st.set_page_config(page_title="CardioCare | Professional Risk Assessment", page_icon="❤️", layout="wide")
# # --- 1. Load the Saved Model and Scaler ---

# def load_assets():
#     model = joblib.load('cardio_model.joblib')
#     scaler = joblib.load('scaler.joblib')
#     return model, scaler

# model, scaler = load_assets()

# # --- 2. Professional UI Setup ---


# # Custom CSS for a "Website" feel
# st.markdown("""
#     <style>
#     .main {
#         background-color: #f8f9fa;
#     }
#     .stButton>button {
#         width: 100%;
#         border-radius: 5px;
#         height: 3em;
#         background-color: #ff4b4b;
#         color: white;
#         font-weight: bold;
#         border: none;
#     }
#     .stButton>button:hover {
#         background-color: #ff3333;
#         border: none;
#     }
#     .prediction-card {
#         padding: 20px;
#         border-radius: 10px;
#         background-color: white;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#         margin-bottom: 20px;
#     }
#     h1 {
#         color: #1e3a8a;
#         font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Navigation Bar Style Header
# st.markdown("""
#     <div style="background-color:#1e3a8a;padding:20px;border-radius:10px;margin-bottom:25px;">
#         <h1 style="color:white;text-align:center;margin:0;">CardioCare AI Portal</h1>
#         <p style="color:white;text-align:center;margin:0;opacity:0.8;">Clinical-Grade Cardiovascular Risk Assessment Tool</p>
#     </div>
#     """, unsafe_allow_html=True)

# # Main Form logic wrapped in columns for centered look
# col_left, col_mid, col_right = st.columns([1, 6, 1])

# with col_mid:
#     with st.form("prediction_form"):
#         st.subheader("📋 Patient Diagnostic Inputs")
        
#         # Grid layout for inputs
#         sub_col1, sub_col2 = st.columns(2)
        
#         with sub_col1:
#             st.markdown("**Personal Details**")
#             age = st.number_input("Age (Years)", min_value=1, max_value=120, value=50)
#             gender = st.selectbox("Gender", options=["Female", "Male"])
#             height = st.number_input("Height (cm)", min_value=50, max_value=250, value=165)
#             weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70)
#             active = st.radio("Daily Physical Activity", options=["Inactive", "Active"], horizontal=True)

#         with sub_col2:
#             st.markdown("**Clinical Vitals**")
#             ap_hi = st.number_input("Systolic BP (top number)", min_value=80, max_value=250, value=120)
#             ap_lo = st.number_input("Diastolic BP (bottom number)", min_value=40, max_value=150, value=80)
#             cholesterol = st.selectbox("Cholesterol Status", options=["Normal", "Above Normal", "Well Above Normal"])
#             gluc = st.selectbox("Glucose/Sugar Status", options=["Normal", "Above Normal", "Well Above Normal"])
#             habits = st.multiselect("Lifestyle Factors", options=["Smoker", "Alcohol Consumption"])

#         st.markdown("<br>", unsafe_allow_html=True)
#         submit_button = st.form_submit_button(label="GENERATE MEDICAL REPORT")

# # --- 3. Prediction Logic (Unchanged) ---
# if submit_button:
#     gender_2 = 1 if gender == "Male" else 0
#     smoke = 1 if "Smoker" in habits else 0
#     alco = 1 if "Alcohol Consumption" in habits else 0
#     active_val = 1 if active == "Active" else 0
    
#     c2, c3 = (1, 0) if cholesterol == "Above Normal" else (0, 1) if cholesterol == "Well Above Normal" else (0, 0)
#     g2, g3 = (1, 0) if gluc == "Above Normal" else (0, 1) if gluc == "Well Above Normal" else (0, 0)
    
#     input_data = pd.DataFrame([[
#         age, height, weight, ap_hi, ap_lo, 
#         smoke, alco, active_val, gender_2, 
#         c2, c3, g2, g3
#     ]], columns=['age', 'height', 'weight', 'ap_hi', 'ap_lo', 
#                  'smoke', 'alco', 'active', 'gender_2', 
#                  'cholesterol_2', 'cholesterol_3', 'gluc_2', 'gluc_3'])
    
#     # Scale numerical columns
#     num_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
#     input_data[num_cols] = scaler.transform(input_data[num_cols])
    
#     # Get Prediction
#     prediction = model.predict(input_data)[0]
#     probability = model.predict_proba(input_data)[0][1]
    
#     # --- Professional Result Display ---
#     with col_mid:
#         st.markdown("<br><h3>Diagnostic Results</h3>", unsafe_allow_html=True)
        
#         if prediction == 1:
#             st.warning("Analysis indicates a presence of cardiovascular risk factors.")
#             st.error(f"**RISK STATUS: HIGH RISK**")
#             st.progress(probability)
#             st.write(f"Statistical Probability: **{probability*100:.1f}%**")
#             st.info("💡 **Recommendation:** Schedule a consultation with a cardiologist for a comprehensive lipid profile and stress test.")
#         else:
#             st.success(f"**RISK STATUS: LOW RISK**")
#             st.progress(probability)
#             st.write(f"Statistical Probability of Risk: **{probability*100:.1f}%**")
#             st.balloons()
#             st.info("💡 **Recommendation:** Continue maintaining current exercise and dietary habits. Annual screenings are advised.")

# # Footer
# st.markdown("""
#     <hr>
#     <div style="text-align:center; color:grey; font-size: 0.8em;">
#         © 2026 CardioCare AI. For educational purposes only. Not a substitute for professional medical advice.
#     </div>
#     """, unsafe_allow_html=True)




# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np

# # ---------------- PAGE CONFIG (FIRST COMMAND) ----------------
# st.set_page_config(
#     page_title="HeartLens AI | Cardiac Risk Intelligence",
#     page_icon="🫀",
#     layout="wide"
# )

# # ---------------- LOAD MODEL ----------------
# def load_assets():
#     model = joblib.load("cardio_model.joblib")
#     scaler = joblib.load("scaler.joblib")
#     return model, scaler

# model, scaler = load_assets()

# # ---------------- CUSTOM THEME ----------------
# st.markdown("""
# <style>
# body {
#     background: linear-gradient(135deg, #e0f2fe, #f8fafc);
# }
# .glass {
#     background: rgba(255, 255, 255, 0.65);
#     backdrop-filter: blur(10px);
#     border-radius: 16px;
#     padding: 25px;
#     box-shadow: 0 8px 32px rgba(0,0,0,0.1);
# }
# .metric-card {
#     background: white;
#     border-radius: 14px;
#     padding: 20px;
#     text-align: center;
#     box-shadow: 0 6px 18px rgba(0,0,0,0.08);
# }
# .title {
#     font-size: 2.6rem;
#     font-weight: 800;
#     color: #0f172a;
# }
# .subtitle {
#     color: #475569;
# }
# .footer {
#     text-align:center;
#     font-size:0.8rem;
#     color:#64748b;
# }
# </style>
# """, unsafe_allow_html=True)

# # ---------------- HERO SECTION ----------------
# st.markdown("""
# <div class="glass">
#     <div class="title">HeartLens AI</div>
#     <div class="subtitle">
#         Intelligent Cardiovascular Risk Screening using Machine Learning
#     </div>
# </div>
# """, unsafe_allow_html=True)

# st.markdown("<br>", unsafe_allow_html=True)

# # ---------------- SIDEBAR INPUTS ----------------
# st.sidebar.header("🧾 Patient Inputs")

# age = st.sidebar.slider("Age (Years)", 1, 120, 50)
# gender = st.sidebar.radio("Gender", ["Female", "Male"])
# height = st.sidebar.slider("Height (cm)", 140, 210, 165)
# weight = st.sidebar.slider("Weight (kg)", 40, 160, 70)

# st.sidebar.subheader("🩺 Vitals")
# ap_hi = st.sidebar.slider("Systolic BP", 80, 220, 120)
# ap_lo = st.sidebar.slider("Diastolic BP", 50, 140, 80)

# st.sidebar.subheader("🧪 Lab Values")
# cholesterol = st.sidebar.selectbox(
#     "Cholesterol",
#     ["Normal", "Above Normal", "Well Above Normal"]
# )
# gluc = st.sidebar.selectbox(
#     "Glucose",
#     ["Normal", "Above Normal", "Well Above Normal"]
# )

# st.sidebar.subheader("🏃 Lifestyle")
# smoke = st.sidebar.toggle("Smoker")
# alco = st.sidebar.toggle("Alcohol Intake")
# active = st.sidebar.toggle("Physically Active")

# predict_btn = st.sidebar.button("🔍 Analyze Risk", use_container_width=True)

# # ---------------- MAIN DASHBOARD ----------------
# col1, col2 = st.columns([2, 3])

# with col1:
#     st.markdown("<div class='glass'>", unsafe_allow_html=True)
#     st.subheader("👤 Patient Summary")
#     st.write(f"**Age:** {age}")
#     st.write(f"**Gender:** {gender}")
#     st.write(f"**Height:** {height} cm")
#     st.write(f"**Weight:** {weight} kg")
#     st.write(f"**Blood Pressure:** {ap_hi}/{ap_lo}")
#     st.markdown("</div>", unsafe_allow_html=True)

# # ---------------- PREDICTION ----------------
# if predict_btn:

#     gender_2 = 1 if gender == "Male" else 0
#     smoke = int(smoke)
#     alco = int(alco)
#     active_val = int(active)

#     c2, c3 = (1, 0) if cholesterol == "Above Normal" else (0, 1) if cholesterol == "Well Above Normal" else (0, 0)
#     g2, g3 = (1, 0) if gluc == "Above Normal" else (0, 1) if gluc == "Well Above Normal" else (0, 0)

#     input_df = pd.DataFrame([[
#         age, height, weight, ap_hi, ap_lo,
#         smoke, alco, active_val, gender_2,
#         c2, c3, g2, g3
#     ]], columns=[
#         'age','height','weight','ap_hi','ap_lo',
#         'smoke','alco','active','gender_2',
#         'cholesterol_2','cholesterol_3','gluc_2','gluc_3'
#     ])

#     num_cols = ['age','height','weight','ap_hi','ap_lo']
#     input_df[num_cols] = scaler.transform(input_df[num_cols])

#     pred = model.predict(input_df)[0]
#     prob = model.predict_proba(input_df)[0][1]

#     with col2:
#         st.markdown("<div class='glass'>", unsafe_allow_html=True)
#         st.subheader("📊 Risk Assessment")

#         if pred == 1:
#             st.error("HIGH CARDIOVASCULAR RISK")
#             st.progress(prob)
#             st.metric("Risk Probability", f"{prob*100:.1f}%")
#             st.info("Recommendation: Immediate cardiology consultation advised.")
#         else:
#             st.success("LOW CARDIOVASCULAR RISK")
#             st.progress(prob)
#             st.metric("Risk Probability", f"{prob*100:.1f}%")
#             st.info("Recommendation: Maintain healthy lifestyle & annual checkups.")

#         st.markdown("</div>", unsafe_allow_html=True)

# # ---------------- FOOTER ----------------
# st.markdown("<br><hr>", unsafe_allow_html=True)
# st.markdown("""
# <div class="footer">
# © 2026 HeartLens AI · Educational & Research Use Only · Not a Medical Diagnosis
# </div>
# """, unsafe_allow_html=True)



import streamlit as st
import joblib
import pandas as pd
import numpy as np



# ---------------- PAGE CONFIG (MUST BE FIRST) ----------------
st.set_page_config(
    page_title="HeartLens AI Pro",
    page_icon="🫀",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
def load_assets():
    model = joblib.load("cardio_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_assets()

# ---------------- GLOBAL STYLES ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #e0f7fa, #f8fafc);
}
.glass {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.12);
}
.hero {
    background: linear-gradient(90deg, #0f766e, #1e40af);
    padding: 35px;
    border-radius: 20px;
    color: white;
}
.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.85rem;
    margin: 4px 4px 4px 0;
}
.green { background:#dcfce7; color:#166534; }
.red { background:#fee2e2; color:#991b1b; }
.blue { background:#e0f2fe; color:#075985; }
.metric {
    font-size:1.6rem;
    font-weight:700;
}
.footer {
    text-align:center;
    font-size:0.8rem;
    color:#64748b;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <h1>HeartLens AI Pro</h1>
    <p>Real-time Cardiovascular Risk Intelligence System</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("🧾 Patient Inputs")

age = st.sidebar.slider("Age (Years)", 1, 120, 45)
gender = st.sidebar.radio("Gender", ["Female", "Male"])
height = st.sidebar.slider("Height (cm)", 140, 210, 165)
weight = st.sidebar.slider("Weight (kg)", 40, 160, 70)

st.sidebar.subheader("🩺 Blood Pressure")
ap_hi = st.sidebar.slider("Systolic", 80, 220, 120)
ap_lo = st.sidebar.slider("Diastolic", 50, 140, 80)

st.sidebar.subheader("🧪 Metabolic Health")
cholesterol = st.sidebar.selectbox(
    "Cholesterol Level",
    ["Normal", "Above Normal", "Well Above Normal"]
)
gluc = st.sidebar.selectbox(
    "Glucose Level",
    ["Normal", "Above Normal", "Well Above Normal"]
)

st.sidebar.subheader("🏃 Lifestyle Factors")
smoke = st.sidebar.toggle("Smoker")
alco = st.sidebar.toggle("Alcohol Consumption")
active = st.sidebar.toggle("Physically Active")

predict_btn = st.sidebar.button("🔍 Analyze Risk", use_container_width=True)

# ---------------- MAIN LAYOUT ----------------
col_profile, col_result = st.columns([2, 3])

# ---------------- LIVE PATIENT PROFILE ----------------
with col_profile:
    st.subheader("👤 Live Patient Profile")

    st.markdown(f"**Age:** {age} years")
    st.markdown(f"**Gender:** {gender}")
    st.markdown(f"**Height / Weight:** {height} cm / {weight} kg")
    st.markdown(f"**Blood Pressure:** {ap_hi}/{ap_lo} mmHg")

    st.markdown("### 🧬 Health Indicators")

    # Smoking
    if smoke:
        st.write("🚬 Smoking: Yes")
    else:
        st.write("🚭 Smoking: No")

    # Alcohol
    if alco:
        st.write("🍷 Alcohol Consumption: Yes")
    else:
        st.write("🚫 Alcohol Consumption: No")

    # Physical Activity
    if active:
        st.write("🏃 Physical Activity: Active")
    else:
        st.write("🛋️ Physical Activity: Inactive")

    # Cholesterol
    if cholesterol == "Normal":
        st.write("🧬 Cholesterol Level: Normal")
    elif cholesterol == "Above Normal":
        st.write("🧬 Cholesterol Level: Above Normal")
    else:
        st.write("🧬 Cholesterol Level: Well Above Normal")

    # Glucose
    if gluc == "Normal":
        st.write("🍬 Glucose Level: Normal")
    elif gluc == "Above Normal":
        st.write("🍬 Glucose Level: Above Normal")
    else:
        st.write("🍬 Glucose Level: Well Above Normal")


# ---------------- PREDICTION ----------------
if predict_btn:
    gender_2 = 1 if gender == "Male" else 0
    smoke_v = int(smoke)
    alco_v = int(alco)
    active_v = int(active)

    c2, c3 = (1, 0) if cholesterol == "Above Normal" else (0, 1) if cholesterol == "Well Above Normal" else (0, 0)
    g2, g3 = (1, 0) if gluc == "Above Normal" else (0, 1) if gluc == "Well Above Normal" else (0, 0)

    input_df = pd.DataFrame([[
        age, height, weight, ap_hi, ap_lo,
        smoke_v, alco_v, active_v, gender_2,
        c2, c3, g2, g3
    ]], columns=[
        'age','height','weight','ap_hi','ap_lo',
        'smoke','alco','active','gender_2',
        'cholesterol_2','cholesterol_3','gluc_2','gluc_3'
    ])

    num_cols = ['age','height','weight','ap_hi','ap_lo']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    with col_result:
        st.subheader("📊 AI Diagnostic Report")

        if pred == 1:
            st.error("🚨 HIGH CARDIOVASCULAR RISK DETECTED")
            st.metric("Risk Probability", f"{prob*100:.1f}%")
            st.progress(prob)
            st.warning("Immediate medical consultation recommended.")
        else:
            st.success("✅ LOW CARDIOVASCULAR RISK")
            st.metric("Risk Probability", f"{prob*100:.1f}%")
            st.progress(prob)
            st.info("Maintain healthy habits and routine screening.")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
© 2026 HeartLens AI Pro · For Academic & Demonstration Use Only
</div>
""", unsafe_allow_html=True)







