# pip install pandas beautifulsoup4 nltk scikit-learn


import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Download and set up NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to fetch and clean text from SEC filings
def fetch_filing(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
    return ' '.join(text.split())

# Load dataset (For illustration, a small sample dataset is created)
data = {
    'url': [
        'https://www.sec.gov/Archives/edgar/data/320193/000032019320000096/a202010-k.htm',  # Apple Inc. 10-K
        'https://www.sec.gov/Archives/edgar/data/789019/000156459020039059/msft-10k_20200630.htm'  # Microsoft Corp 10-K
    ],
    'label': [0, 1]  # Labels for classification (0: Apple, 1: Microsoft)
}

df = pd.DataFrame(data)

# Fetch and preprocess the filings
df['text'] = df['url'].apply(fetch_filing)
df['text'] = df['text'].str.replace(r'\n', ' ').str.replace(r'\s+', ' ')

# Remove stopwords and vectorize the text
vectorizer = CountVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
