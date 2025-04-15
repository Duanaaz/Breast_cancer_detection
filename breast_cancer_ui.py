import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Set Streamlit page config
st.set_page_config(page_title="Breast Cancer Detector", page_icon="🩺", layout="wide")

# Try to load model and scaler
try:
    model = load_model('breast_cancer_model.h5')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"⚠️ Error loading model or scaler: {e}")
    st.stop()

# Feature names
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

st.title("🩺 Breast Cancer Detection")
st.markdown("Use the scrollable panel to input medical data and predict tumor type.")

with st.sidebar:
    st.header("📥 Input Features")
    st.markdown("Scroll through and adjust feature values.")
    input_data = []

    # Group inputs by 10s in expandable sections
    for i in range(0, 30, 10):
        with st.expander(f"Features {i+1} - {i+10}"):
            for feature in feature_names[i:i+10]:
                val = st.slider(
                    label=feature,
                    min_value=0.0,
                    max_value=100.0,
                    step=0.01,
                    value=1.0,
                    key=feature
                )
                input_data.append(val)

# Predict
if st.button("🔍 Predict Tumor Type"):
    try:
        input_array = np.array([input_data])
        input_std = scaler.transform(input_array)
        prediction = model.predict(input_std)

        predicted_class = np.argmax(prediction, axis=1)[0]
        malignant_conf = float(prediction[0][0]) * 100
        benign_conf = float(prediction[0][1]) * 100

        st.subheader("🔬 Prediction Result")
        if predicted_class == 0:
            st.error("🔴 The tumor is **Malignant**")
        else:
            st.success("🟢 The tumor is **Benign**")

        st.subheader("📊 Confidence Scores")
        col1, col2 = st.columns(2)
        col1.metric("Malignant", f"{malignant_conf:.2f}%")
        col2.metric("Benign", f"{benign_conf:.2f}%")

        st.progress(int(benign_conf if predicted_class == 1 else malignant_conf))

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
