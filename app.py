import streamlit as st
import joblib
from newspaper import Article
import nltk

# Download punkt tokenizer for newspaper3k if not already
nltk.download('punkt')

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# --- Streamlit Page Config ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# --- Sidebar ---
theme = st.sidebar.radio("🌗 Theme Mode", ["Light", "Dark"], index=0)

if theme == "Dark":
    st.markdown("""
    <style>
        body, .stApp { background-color: #0E1117; color: #FAFAFA; }
        .stButton>button { background-color: #1f77b4; color: white; }
        .stTextArea textarea { background-color: #2b2b2b; color: white; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        body, .stApp { background-color: #f5f7fa; color: black; }
        .stButton>button { background-color: #0066cc; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.markdown("### 👩‍💻 Developed by")
st.sidebar.image("https://i.imgur.com/N4YvZ5J.jpg", width=150, caption="Sakshi Kuthe")
st.sidebar.markdown("**Founder | Developer**")
st.sidebar.markdown("---")

# --- Title ---
st.markdown("<h1 style='text-align: center; color: navy;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Detect whether news is Fake or Real from Text or URL</p>", unsafe_allow_html=True)
st.markdown("---")

# --- News Text Area ---
st.subheader("🧾 Enter News Text")
user_input = st.text_area("", height=200)

if st.button("🔍 Detect from Text"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news text!")
    else:
        vector_input = vectorizer.transform([user_input])
        prediction = model.predict(vector_input)[0]
        proba = model.decision_function(vector_input)
        confidence = round((abs(proba[0]) / max(abs(proba))) * 100, 2)

        st.markdown("### 📊 Prediction Result:")
        if prediction == 1:
            st.success("✅ The news is Real.")
        else:
            st.error("❌ The news is Fake.")
        st.info(f"🧠 Confidence Score: {confidence}%")

st.markdown("---")

# --- News from URL ---
st.subheader("🌐 Or Paste a News Article URL")
news_url = st.text_input("Paste URL of a news article")

if st.button("📥 Detect from URL"):
    if news_url.strip() == "":
        st.warning("⚠️ Please paste a URL.")
    else:
        try:
            article = Article(news_url)
            article.download()
            article.parse()
            text = article.text

            if not text.strip():
                st.warning("⚠️ Could not extract article text.")
            else:
                vector_input = vectorizer.transform([text])
                prediction = model.predict(vector_input)[0]
                proba = model.decision_function(vector_input)
                confidence = round((abs(proba[0]) / max(abs(proba))) * 100, 2)

                st.markdown("### 📄 Extracted Article Preview:")
                st.info(text[:600] + "..." if len(text) > 600 else text)

                st.markdown("### 📊 Prediction Result:")
                if prediction == 1:
                    st.success("✅ The news is Real.")
                else:
                    st.error("❌ The news is Fake.")
                st.info(f"🧠 Confidence Score: {confidence}%")

        except Exception as e:
            st.error(f"🚫 Error reading URL: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center;'>Built with ❤️ by Sakshi Kuthe</p>", unsafe_allow_html=True)
