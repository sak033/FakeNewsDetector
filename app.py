import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit page setup
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.write("Enter any news article text below to check if it's Fake or Real.")

# Text input
user_input = st.text_area("Enter News Text", height=200)

# Predict button
if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some news text!")
    else:
        # Vectorize and predict
        vector_input = vectorizer.transform([user_input])
        prediction = model.predict(vector_input)[0]

        if prediction == 1:
            st.success("✅ The news is **Real**.")
        else:
            st.error("❌ The news is **Fake**.")
