import streamlit as st
import pandas as pd
import joblib

# Load model & encoder
try:
    model = joblib.load("model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    st.error(f"Error loading model or label encoder: {e}")
    st.stop()

st.title("Career Prediction App")
st.markdown("Fill in the details below to predict your **Career Path**:")

age = st.number_input("Age", min_value=0, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
high_school_gpa = st.slider("High School GPA", 0.0, 4.0, 3.0, 0.1)
univ_rank = st.number_input("University Ranking", 1, 500, 50)
univ_gpa = st.slider("University GPA", 0.0, 4.0, 3.0, 0.1)
field_of_study = st.selectbox("Field of Study", ['Arts', 'Law', 'Medicine', 'Computer Science', 'Business', 'Mathematics', 'Engineering'])
internships = st.number_input("Internships Completed", 0, 10, 1)
projects = st.number_input("Projects Completed", 0, 50, 2)
certifications = st.number_input("Certifications", 0, 20, 1)
soft_skills = st.slider("Soft Skills Score", 0.0, 10.0, 5.0, 0.1)
starting_salary = st.number_input("Starting Salary", 0.0, value=30000.0)

if st.button("Predict"):
    try:
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender.capitalize(),
            'High_School_GPA': high_school_gpa,
            'University_Ranking': univ_rank,
            'University_GPA': univ_gpa,
            'Field_of_Study': field_of_study,
            'Internships_Completed': internships,
            'Projects_Completed': projects,
            'Certifications': certifications,
            'Soft_Skills_Score': soft_skills,
            'Starting_Salary': starting_salary
        }])

        prediction = model.predict(input_data)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        confidence = round(float(model.predict_proba(input_data).max()) * 100, 1)

        st.success(f"**Prediction:** {predicted_label}")
        st.info(f"**Confidence:** {confidence}%")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
