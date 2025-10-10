import streamlit as st
import pandas as pd
import joblib

# --- Load your trained model ---
model = joblib.load("trained_models/best_pipeline_random_forest.joblib")

st.title("📊 Final Exam Score Predictor")
st.write("Enter your details below to predict your final exam score.")

# --- Example inputs (replace these with your real feature names) ---
study_time = st.number_input("Study Time (hours per week)", min_value=0, max_value=50, value=10)
attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=85)
assignments_completed = st.number_input("Assignments Completed", min_value=0, max_value=20, value=10)
previous_grade = st.number_input("Previous Exam Grade", min_value=0, max_value=100, value=75)

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    "study_time": [study_time],
    "attendance": [attendance],
    "assignments_completed": [assignments_completed],
    "previous_grade": [previous_grade]
})

# --- Predict ---
if st.button("Predict Final Exam Score"):
    prediction = model.predict(input_data)[0]
    st.success(f"🎯 Predicted Final Exam Score: **{prediction:.2f}**")