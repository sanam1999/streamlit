import streamlit as st
import pandas as pd
import pickle



try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    st.stop()


try:
    with open('label_encoder.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
except Exception as e:
    st.error(f"Error loading label_encoders.pkl: {e}")
    st.stop()


st.title("Career Prediction App")
st.markdown("Fill in the details below to predict your **Career Path**:")

# Input fields
age = st.number_input("Age", min_value=0, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
high_school_gpa = st.slider("High School GPA", 0.0, 4.0, 3.0, 0.1)
univ_rank = st.number_input("University Ranking (1 = best)", min_value=1, max_value=500, value=50)
univ_gpa = st.slider("University GPA", 0.0, 4.0, 3.0, 0.1)

# Dropdown for Field of Study
field_of_study = st.selectbox(
    "Field of Study",
    ['Arts', 'Law', 'Medicine', 'Computer Science', 'Business', 'Mathematics', 'Engineering']
)

internships = st.number_input("Internships Completed", min_value=0, max_value=10, value=1)
projects = st.number_input("Projects Completed", min_value=0, max_value=50, value=2)
certifications = st.number_input("Certifications", min_value=0, max_value=20, value=1)
soft_skills = st.slider("Soft Skills Score", 0.0, 10.0, 5.0, 0.1)
starting_salary = st.number_input("Starting Salary", min_value=0.0, value=30000.0)

if st.button("Predict"):
    try:
        input_data = pd.DataFrame([{
            'Age': float(age),
            'Gender': gender.capitalize(),
            'High_School_GPA': float(high_school_gpa),
            'University_Ranking': float(univ_rank),
            'University_GPA': float(univ_gpa),
            'Field_of_Study': field_of_study,
            'Internships_Completed': int(internships),
            'Projects_Completed': int(projects),
            'Certifications': int(certifications),
            'Soft_Skills_Score': float(soft_skills),
            'Starting_Salary': float(starting_salary)
        }])

        prediction = model.predict(input_data)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        confidence = round(float(model.predict_proba(input_data).max()) * 100, 1)

        st.success(f"**Prediction:** {predicted_label}")
        st.info(f"**Confidence:** {confidence}%")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
