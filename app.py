import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------
# Load the trained model and feature columns
# -------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load('student_exam_rf_model.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

model, features = load_model()

# -------------------------------------------
# Streamlit App UI
# -------------------------------------------
st.set_page_config(page_title="Student Exam Score Predictor", page_icon="🎓")

st.title("🎓 Student Exam Score Predictor")
st.markdown("Enter your study details below and find out your **predicted exam score!**")

# Input fields
hours_studied = st.number_input("📘 Hours Studied", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
previous_score = st.number_input("📈 Previous Score", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
sleep_hours = st.number_input("💤 Sleep Hours per Day", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
extracurricular = st.selectbox("🎭 Extracurricular Activities", ["Yes", "No"])

# -------------------------------------------
# Prepare input for prediction
# -------------------------------------------
# Create a dataframe with a single row
input_data = pd.DataFrame([{
    'Hours_Studied': hours_studied,
    'Previous_Score': previous_score,
    'Sleep_Hours': sleep_hours,
    'Extracurricular_Activities_Yes': 1 if extracurricular == "Yes" else 0
}])

# Make sure columns match training features
for col in features:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[features]

# -------------------------------------------
# Predict button
# -------------------------------------------
if st.button("🔮 Predict Exam Score"):
    prediction = model.predict(input_data)[0]
    st.success(f"🎯 **Predicted Exam Score:** {prediction:.2f}")