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
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
except Exception as e:
    st.error(f"Error loading label_encoders.pkl: {e}")
    st.stop()


try:
    with open('feature_order.pkl', 'rb') as f:
        feature_order = pickle.load(f)
except Exception as e:
    st.error(f"Error loading feature_order.pkl: {e}")
    st.stop()


try:
    data = pd.read_csv('./data/dataset.csv.csv')
except FileNotFoundError:
    st.error("CSV file not found. Please ensure 'dataset.csv.csv' is in the 'data/' folder.")
    st.stop()


st.title("Job Level Prediction App")
st.markdown("Enter your profile details to predict your **Job Level**:")


univ_rank = st.selectbox("University Ranking (1=best)", sorted(data['University_Ranking'].unique()))
univ_gpa = st.slider("University GPA", 0.0, 4.0, 3.0, 0.1)
field = st.selectbox("Field of Study", label_encoders['Field_of_Study'].classes_)
internships = st.slider("Internships Completed", 0, 10, 1)
projects = st.slider("Projects Completed", 0, 20, 2)
certs = st.slider("Certifications", 0, 10, 1)


field_encoded = label_encoders['Field_of_Study'].transform([field])[0]


input_data = {
    'University_Ranking': [univ_rank],
    'University_GPA': [univ_gpa],
    'Field_of_Study': [field_encoded],
    'Projects_Completed': [projects],
    'Internships_Completed': [internships],
    'Certifications': [certs],
}

input_df = pd.DataFrame(input_data)


input_df = input_df[feature_order]


if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        raw = round(prediction)

        if raw >= 1:
            label = "Entry"
        elif raw >= 2:
            label = "Mid"
        elif raw <= 2:
            label = "Senior"
        else:
            label = f"Unknown ({raw})"

        st.write(f"Raw model output: `{prediction:.2f}`")
        st.success(f"Predicted Job Level: **{label}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
