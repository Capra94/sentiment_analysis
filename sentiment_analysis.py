# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from imblearn.over_sampling import RandomOverSampler  
from sklearn.svm import SVC
from xgboost import XGBClassifier
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
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['review_body'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# Oversample the minority class
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Fine-tune Support Vector Machine (SVM) parameters with class weights
svm_classifier = SVC(kernel='linear', C=1, gamma='scale', class_weight='balanced', random_state=42)
svm_classifier.fit(X_train_resampled, y_train_resampled)
predictions_svm = svm_classifier.predict(X_test)

# Evaluate the SVM model
precision_svm, recall_svm, f1_score_svm, _ = precision_recall_fscore_support(y_test, predictions_svm, average='weighted', zero_division=1)
print("Support Vector Machine (SVM):")
print("Precision:", precision_svm)
print("Recall:", recall_svm)
print("F1 Score:", f1_score_svm)
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions_svm))

# Fine-tune Gradient Boosting (XGBoost) parameters
xgb_classifier = XGBClassifier(scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(), learning_rate=0.1, max_depth=3, n_estimators=100, random_state=42)
xgb_classifier.fit(X_train_resampled, y_train_resampled)
predictions_xgb = xgb_classifier.predict(X_test)

# Evaluate the XGBoost model
precision_xgb, recall_xgb, f1_score_xgb, _ = precision_recall_fscore_support(y_test, predictions_xgb, average='weighted', zero_division=1)
print("\nGradient Boosting (XGBoost):")
print("Precision:", precision_xgb)
print("Recall:", recall_xgb)
print("F1 Score:", f1_score_xgb)
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions_xgb))
