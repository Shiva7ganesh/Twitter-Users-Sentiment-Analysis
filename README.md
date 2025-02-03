# Sentiment Analysis of Tweets Using Machine Learning

## Project Overview
This project aims to classify tweets into two categories: **positive** or **negative**, utilizing machine learning techniques. Given the vast amount of user-generated content on platforms like Twitter, sentiment analysis enables businesses, policymakers, and individuals to make informed decisions based on public opinion trends.

## Objective
The goal of this project is to demonstrate how data-driven models can efficiently classify tweets and extract meaningful insights. Applications include:
- Brand sentiment evaluation
- Tracking public reactions to major events
- Identifying social media trends
- Understanding customer feedback

## Scope
- **Programming Language:** Python
- **Machine Learning Model:** Logistic Regression
- **Dataset:** Sentiment140 (1.6 million tweets, sourced from Kaggle)
- **Development Environment:** Google Colaboratory (Google Colab)

## Features
1. **Data Collection**
   - Using the Sentiment140 dataset from Kaggle
   - Downloading data using the Kaggle API
2. **Data Preprocessing**
   - Cleaning text (removing URLs, special characters, stopwords)
   - Tokenization and stemming (using NLTK’s PorterStemmer)
3. **Feature Engineering**
   - Text vectorization using **TF-IDF (Term Frequency-Inverse Document Frequency)**
4. **Model Training and Evaluation**
   - Training a **logistic regression** model
   - Measuring accuracy, precision, recall, and F1-score
5. **Visualization**
   - Sentiment distribution graphs
   - Trend analysis charts

## System Requirements
### Hardware
- Standard laptop/PC (Google Colab will handle computations)

### Software
- **Python 3.x**
- **Google Colaboratory** (no installation needed)
- **Kaggle API Key** (for dataset access)

### Required Python Libraries
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `nltk`: Natural Language Processing (stopwords, stemming)
- `scikit-learn`: Model building, evaluation, data splitting

## Methodology
1. **Data Collection**
   - Dataset: **Sentiment140 (1.6 million tweets)**
   - Download via Kaggle API
2. **Data Preprocessing**
   - Convert text to lowercase
   - Remove special characters, URLs, and stopwords
   - Tokenize and stem words using **PorterStemmer**
   - Convert sentiment labels: **(4 → 1 for positive, 0 for negative)**
3. **Splitting Data**
   - 80% training, 20% testing
   - Stratified sampling ensures balanced distribution
4. **Feature Engineering**
   - Apply **TF-IDF vectorization** to convert text into numerical form
5. **Model Training and Evaluation**
   - Train **logistic regression** model
   - Evaluate using accuracy, precision, recall, and F1-score
6. **Model Deployment & Prediction**
   - Save the trained model using **pickle**
   - Load saved model for future predictions

## Implementation Steps
### Setting Up Environment in Google Colab
1. **Install Kaggle API**
   ```sh
   !pip install kaggle
   ```
2. **Upload Kaggle API Token**
   ```python
   from google.colab import files
   files.upload()  # Upload kaggle.json
   ```
3. **Download Dataset**
   ```sh
   !kaggle datasets download -d kazanova/sentiment140
   ```
4. **Extract Data**
   ```python
   import zipfile
   with zipfile.ZipFile("sentiment140.zip", 'r') as zip_ref:
       zip_ref.extractall()
   ```

### Preprocessing Tweets
```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')

port_stem = PorterStemmer()
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    text = [port_stem.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)
```

### Training the Model
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, Y_train)

train_accuracy = accuracy_score(Y_train, model.predict(X_train_tfidf))
test_accuracy = accuracy_score(Y_test, model.predict(X_test_tfidf))
```

### Saving & Loading the Model
```python
import pickle
pickle.dump(model, open('sentiment_model.sav', 'wb'))
loaded_model = pickle.load(open('sentiment_model.sav', 'rb'))
```

## Results
- **Training Accuracy:** ~81%
- **Test Accuracy:** ~77.8%
- **Observations:**
  - Model performs well but struggles with neutral/sarcastic tweets
  - Some overfitting observed (training accuracy > test accuracy)

## Future Enhancements
- Implement **multi-class classification** (e.g., neutral sentiment)
- Use advanced models like **BERT** for deeper text understanding
- Optimize model with **hyperparameter tuning**

## References
- [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Library](https://www.nltk.org/)
