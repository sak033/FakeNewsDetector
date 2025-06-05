import streamlit as st
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# --- Custom CSS styling ---
st.markdown("""
    <style>
        .main { background-color: #f5f7fa; }
        .stButton > button {
            color: white;
            background-color: #0066cc;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title and Description ---
st.markdown("<h1 style='text-align: center; color: navy;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a news article below to check if it's real or fake using ML.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Text Input ---
user_input = st.text_area("🧾 Enter News Text", height=200)

# --- Predict Button ---
if st.button("🔍 Check News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news text!")
    else:
        vector_input = vectorizer.transform([user_input])
        prediction = model.predict(vector_input)[0]
        proba = model.decision_function([user_input])
        confidence = round((abs(proba[0]) / max(abs(proba))) * 100, 2)

        if prediction == 1:
            st.success("✅ The news is Real.")
        else:
            st.error("❌ The news is Fake.")

        st.info(f"🧠 Confidence Score: {confidence}%")

# --- Word Cloud Section ---
st.sidebar.header("🌀 WordCloud Viewer")
word_type = st.sidebar.selectbox("Choose news type:", ["Fake", "Real"])

if st.sidebar.button("Generate WordCloud"):
    import pandas as pd
    fake = pd.read_csv("Fake.csv")
    real = pd.read_csv("True.csv")
    if word_type == "Fake":
        text = " ".join(fake['text'].astype(str).values)
    else:
        text = " ".join(real['text'].astype(str).values)

    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    st.subheader(f"🖼 WordCloud for {word_type} News")
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center;'>Built with ❤️ by Sakshi Kuthe</p>", unsafe_allow_html=True)
