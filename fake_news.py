import pandas as pd
import joblib
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Function to load datasets with flexible format
def load_dataset(path, label, text_column_guess=['text', 'content', 'article', 'statement']):
    try:
        if path.endswith('.xlsx'):
            df = pd.read_excel(path)
        elif path.endswith('.tsv'):
            df = pd.read_csv(path, sep='\t', quoting=3)
        else:
            df = pd.read_csv(path)
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return pd.DataFrame(columns=['text', 'label'])

    # Try to identify the correct text column
    for col in text_column_guess:
        if col in df.columns:
            df = df.rename(columns={col: 'text'})
            break
    else:
        print(f"⚠️ No valid text column found in {path}")
        return pd.DataFrame(columns=['text', 'label'])

    df['label'] = label
    return df[['text', 'label']]

# Special loader for LIAR dataset
def load_liar(path, label_map={'false': 0, 'pants-fire': 0, 'barely-true': 0, 'half-true': 1, 'mostly-true': 1, 'true': 1}):
    try:
        liar = pd.read_csv(path, sep='\t', header=None, quoting=3)
        liar.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party',
                        'barely_true_ct', 'false_ct', 'half_true_ct', 'mostly_true_ct', 'pants_on_fire_ct',
                        'context']
        liar['label'] = liar['label'].map(label_map)
        liar = liar[['statement', 'label']].rename(columns={'statement': 'text'})
        return liar.dropna()
    except Exception as e:
        print(f"❌ Error loading LIAR from {path}: {e}")
        return pd.DataFrame(columns=['text', 'label'])

# Load datasets
datasets = [
    load_dataset('Fake.csv', 0),
    load_dataset('True.csv', 1),
    load_dataset('BuzzFeed_fake_news_content.xlsx', 0),
    load_dataset('BuzzFeed_real_news_content.xlsx', 1),
    load_dataset('PolitiFact_fake_news_content.xlsx', 0),
    load_dataset('PolitiFact_real_news_content.xlsx', 1),
    load_dataset('WELFake_Dataset.csv', 0),  # assume WELFake only contains fake; if mixed, update accordingly
    load_liar('liar_train.tsv'),
    load_liar('liar_test.tsv'),
    load_liar('liar_valid.tsv')
]

# Combine and clean
data = pd.concat(datasets, ignore_index=True)
data.dropna(subset=['text'], inplace=True)
data = data[data['text'].str.strip() != '']
data = shuffle(data).reset_index(drop=True)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# Vectorize
vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=0.7)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {round(acc * 100, 2)}%")
print("📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("\n💾 Model and vectorizer saved successfully!")
