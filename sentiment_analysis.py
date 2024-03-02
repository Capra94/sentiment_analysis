# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from imblearn.over_sampling import RandomOverSampler  
import re

# Load the dataset
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

# Oversample the minority class
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Create and train a Random Forest classifier with hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), param_grid, cv=5)
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best parameters
best_params = grid_search.best_params_

# Train the classifier with the best parameters on the resampled data
classifier = RandomForestClassifier(n_estimators=best_params['n_estimators'], 
                                    min_samples_split=best_params['min_samples_split'],
                                    random_state=42, class_weight='balanced')
classifier.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
predictions = classifier.predict(X_test)

# Evaluate the model
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predictions, average='weighted', zero_division=1)
print("Best Parameters:", best_params)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

# TODO: Address the issue of predicting minority class (label 0)
# The model still struggles to correctly predict instances of the minority class.
