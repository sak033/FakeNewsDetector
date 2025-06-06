import pandas as pd
import joblib
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Function to load and label datasets
def load_dataset(path, label):
    df = pd.read_csv(path)
    if 'text' not in df.columns:
        if 'content' in df.columns:
            df = df.rename(columns={'content': 'text'})
        elif 'article' in df.columns:
            df = df.rename(columns={'article': 'text'})
    df['label'] = label
    return df[['text', 'label']]

# Load all datasets
datasets = [
    load_dataset('Fake.csv', 0),
    load_dataset('True.csv', 1),
    load_dataset('BuzzFeed_fake_news_content.csv', 0),
    load_dataset('BuzzFeed_real_news_content.csv', 1),
    load_dataset('PolitiFact_fake_news_content.csv', 0),
    load_dataset('PolitiFact_real_news_content.csv', 1)
]

# Combine all into one DataFrame
data = pd.concat(datasets, ignore_index=True)

# Drop missing or empty text entries
data.dropna(subset=['text'], inplace=True)
data = data[data['text'].str.strip() != '']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=0.7)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(acc * 100, 2)}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
