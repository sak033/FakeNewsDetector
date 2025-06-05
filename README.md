# 📰 Fake News Detection Web App

A machine learning-based web application built using Python and Streamlit to detect whether a given news article headline is **Real** or **Fake**.

![Streamlit App Screenshot](https://fakenewsdetector-jxysfyjibgagkgmqvtvdka.streamlit.app/)

## 🚀 Live Demo

👉 [Click here to try the live app](https://fakenewsdetector-jxysfyjibgagkgmqvtvdka.streamlit.app/)

## 📌 Features

- Input a news headline and get a prediction: **Real** or **Fake**
- Built with Scikit-learn's `PassiveAggressiveClassifier`
- Clean and simple web interface using **Streamlit**
- Trained on combined datasets of real and fake news

## 📁 Dataset

- [`True.csv`](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- [`Fake.csv`](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Both datasets were labeled and merged to train a binary classifier.

## 🧠 Model

- **Vectorizer**: `TfidfVectorizer` (with English stopwords)
- **Classifier**: `PassiveAggressiveClassifier`
- **Accuracy**: ~99%

The trained model and vectorizer are saved as:
- `model.pkl`
- `vectorizer.pkl`

