import pandas as pd
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import joblib


# Step 1: Load the data
fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')

# Step 2: Label the data
fake['label'] = 0
real['label'] = 1

# Step 3: Combine both datasets
data = pd.concat([fake, real])
data = data[['text', 'label']]  # Only keep relevant columns

# Step 4: Preprocess and vectorize
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
from nltk.corpus import stopwords

# Get stopwords list
stop_words = stopwords.words('english')

# Convert text to numerical data
vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=0.7)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42)

# Fit and transform training text, transform test text
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Step 5: Train the model
model = PassiveAggressiveClassifier()
model.fit(X_train, y_train)

# Step 6: Predict on test data
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(acc * 100, 2)}%")

# Optional: Show confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
