# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

# Loads the dataset
dataset_path = '/content/amazon_books_Data.csv'
df = pd.read_csv(dataset_path)

# Handling Typos and Inconsistencies
df['Sentiment_books'] = df['Sentiment_books'].replace('negaitve', 'negative')

# Encoding Sentiment Labels
df['label'] = (df['Sentiment_books'] == 'positive').astype(int)

# Text Preprocessing
def preprocess_text(text):
    """
    Preprocesses the input text by converting to lowercase and removing non-alphabetic characters.

    Parameters:
    text (str): The input text.

    Returns:
    str: The preprocessed text.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['review_body'] = df['review_body'].apply(preprocess_text)

# Feature extraction using TF-IDF with stop words
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['review_body'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# Creates and trains a Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', min_samples_split=5)
classifier.fit(X_train, y_train)

# Makes predictions on the test set
predictions = classifier.predict(X_test)

# Evaluates the model
print("Classification Report:\n", classification_report(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
