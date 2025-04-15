import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("cat_cancer_model.pkl")

# Load feature names from your dataset
df = pd.read_csv("fmc_dataset_expanded.csv")
features = df.drop(columns=["target"]).columns  # assuming 'target' is the label column

st.title("🐱 Cat Cancer Detection")

st.markdown("""
Enter the following lab test values (metabolite features) to predict if the cat has cancer.
""")

# Create number inputs for each feature
input_data = []
cols = st.columns(3)
for i, col in enumerate(features):
    with cols[i % 3]:
        val = st.number_input(f"{col}", value=0.0)
        input_data.append(val)

# Prediction
if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    result = model.predict(input_array)
    
    st.subheader("Prediction Result:")
    if result[0] == 1:
        st.error("⚠️ The cat is likely to have cancer.")
    else:
        st.success("✅ The cat is not showing signs of cancer.")
