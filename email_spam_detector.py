
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("C:/Users/Ram/Downloads/archive (3)/SPAM.csv", encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) 
    return text

df['message'] = df['message'].apply(clean_text)

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

def predict_spam(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]
    return "Spam" if result == 1 else "Not Spam"

a=predict_spam(input("Enter Email:-"))
print(a)
b=predict_spam(input("Enter Email:-"))
print(b)

#now lets test the model to predict spam or not spam