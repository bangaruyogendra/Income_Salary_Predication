import streamlit as st
import pickle

with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)


st.title("Income Prediction App")
st.write("Predict whether income >50K or <=50K")

input_data = {}
for col in label_encoders:
    options = label_encoders[col].classes_
    input_data[col] = st.selectbox(f"Select {col}", options)

# Predict button
if st.button("Predict"):

    encoded_input = []
    for col in label_encoders:
        le = label_encoders[col]
        encoded_val = le.transform([input_data[col]])[0]
        encoded_input.append(encoded_val)

    pred = knn_model.predict([encoded_input])[0]
    result = ">50K" if pred == 1 else "<=50K"
    
    st.success(f"Predicted Income: {result}")
