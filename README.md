# Machine Learning Experiment - Sentiment Analysis on Amazon Book Reviews - Test-Branch

## Overview
This project is a simple machine learning experiment aimed at performing sentiment analysis on Amazon book reviews. The goal is to predict whether a review is positive or negative based on its content.

## Dataset

The dataset used in this project is sourced from Amazon book reviews and is provided under the [Creative Commons CC0 1.0 Universal (CC0 1.0) Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).

### Attribution
The dataset is used without any attribution requirements under the CC0 1.0 Universal license. However, we acknowledge and appreciate the efforts of the creators in making this dataset publicly available.
Link to the dataset: https://www.kaggle.com/datasets/anushabellam/amazon-reviews-dataset/data

### Data Preprocessing
- Typos and inconsistencies in sentiment labels are handled by replacing 'negaitve' with 'negative'.
- Sentiment labels are encoded as binary values: 1 for positive and 0 for negative.
- Text data is preprocessed by converting to lowercase and removing non-alphabetic characters.

## Feature Extraction
- TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert the text data into numerical features.
- English stop words are removed during the TF-IDF vectorization process.

## Model Training
A Random Forest classifier is trained on the TF-IDF features to predict sentiment labels. The model is configured with 100 trees, balanced class weights, and a minimum of 5 samples required to split an internal node.

## Evaluation
The model is evaluated using the following metrics:
- Classification Report: Provides precision, recall, and F1-score for each class.
- Confusion Matrix: Displays true positive, true negative, false positive, and false negative counts.

## Limitations
- The current model may have limitations in precision and F1-score due to the simplicity of the approach and the size of the dataset.
- Limited feature engineering and model tuning have been applied.

## Future Work
- Experiment with different machine learning models (e.g., deep learning) and hyperparameter tuning for improved performance.
- Explore additional text preprocessing techniques, such as lemmatization or more advanced tokenization.
- Consider using more sophisticated embeddings or pre-trained language models.
- Gather and use a larger and more diverse dataset to enhance model generalization.
- Address class imbalances and explore techniques to handle skewed data distribution.

## Usage
1. Install the required dependencies: pandas, scikit-learn.
   ```bash
   pip install pandas scikit-learn
