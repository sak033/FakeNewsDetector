import streamlit as st
import joblib
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests

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
        proba = model.decision_function(vector_input)
        confidence = round((abs(proba[0]) / max(abs(proba))) * 100, 2)

        if prediction == 1:
            st.success("✅ The news is Real.")
        else:
            st.error("❌ The news is Fake.")

        st.info(f"🧠 Confidence Score: {confidence}%")

# --- Sidebar WordCloud ---
st.sidebar.header("🌀 WordCloud Viewer")
word_type = st.sidebar.selectbox("Choose news type:", ["Fake", "Real"])

if st.sidebar.button("Generate WordCloud"):
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

# --- Live News Headlines ---
st.sidebar.header("🗞️ Live News Headlines")
if st.sidebar.button("Fetch Top Headlines"):
    api_key = "85ca277914ad42da9cbad5e32cd8189b"  # Replace with your actual API key
    country = "in"

    url = f"https://newsapi.org/v2/top-headlines?country={country}&apiKey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if data["status"] == "ok":
            st.subheader("📰 Top Headlines:")
            for i, article in enumerate(data["articles"][:5]):
                st.markdown(f"### {i+1}. {article['title']}")
                st.write(article["description"] or "No description available")
                st.write(f"[Read more]({article['url']})")
                st.markdown("---")
        else:
            st.error("❌ Failed to fetch news. Please check your API key or try again later.")
    except Exception as e:
        st.error(f"⚠️ Error fetching news: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center;'>Built with ❤️ by Sakshi Kuthe</p>", unsafe_allow_html=True)
