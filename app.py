import streamlit as st
import numpy as np
import joblib

# Load models and scalers
human_model = joblib.load("breast_cancer_model.pkl")
human_scaler = joblib.load("scaler.pkl")
cat_model = joblib.load("cat_cancer_model.pkl")
cat_scaler = joblib.load("cat_scaler.pkl")

# Feature lists
human_features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

cat_features = [
    'Alanine', 'Arginine', 'Asparagine', 'Aspartate', 'Citrate',
    'Creatine', 'Glucose', 'Glutamate', 'Glutamine', 'Glycerol',
    'Glycine', 'Lactate', 'Leucine', 'Lysine', 'Methionine',
    'Phenylalanine', 'Pyruvate', 'Serine', 'Tyrosine', 'Valine'
]

# UI
st.title("Cancer Detection App")
option = st.selectbox("Select Model:", ("Human (Breast Cancer)", "Cat (Cancer Detection)"))

# Collect Inputs
def input_section(features):
    inputs = []
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            inputs.append(st.number_input(feature, value=0.0))
    return np.array(inputs).reshape(1, -1)

if option == "Human (Breast Cancer)":
    st.subheader("Enter 30 Breast Cancer Features")
    input_data = input_section(human_features)
    input_data_std = human_scaler.transform(input_data)
    prediction = human_model.predict(input_data_std)
    prediction_label = np.argmax(prediction)

    if st.button("Predict"):
        st.subheader("Prediction Result")
        if prediction_label == 0:
            st.error("The tumor is Malignant (cancerous)")
        else:
            st.success("The tumor is Benign (non-cancerous)")

elif option == "Cat (Cancer Detection)":
    st.subheader("Enter 20 Metabolite Features for Cat")
    input_data = input_section(cat_features)
    input_data_std = cat_scaler.transform(input_data)
    prediction = cat_model.predict(input_data_std)
    prediction_label = np.argmax(prediction)

    if st.button("Predict"):
        st.subheader("Prediction Result")
        if prediction_label == 0:
            st.error("Cancer Detected in Cat")
        else:
            st.success("No Cancer Detected in Cat")

# Sidebar info
st.sidebar.header("About")
st.sidebar.info("This app lets you detect cancer in humans (breast cancer) and cats (based on metabolite profiles).")
