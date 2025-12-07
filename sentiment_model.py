import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk

# --- Data Simulation (In a real project, this would be a loaded CSV) ---
data = {
    'Text': [
        "This is an amazing product!", 
        "The service was terrible and slow.", 
        "It's just okay, nothing special.", 
        "I love the quality and design.", 
        "The worst experience of my life."
    ],
    'Sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative']
}
df = pd.DataFrame(data)

# --- Feature Extraction and Model Training ---

# 1. Clean Text (Placeholder for NLTK operations)
df['Text'] = df['Text'].str.lower() 

# 2. Convert text to numerical features (Bag-of-Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Text'])
y = df['Sentiment']

# 3. Model Training (Using a simple classification model)
model = MultinomialNB()
model.fit(X, y)

# --- Prediction Example ---
test_text = ["This product is fantastic."]
test_vector = vectorizer.transform(test_text)
prediction = model.predict(test_vector)

print(f"Test Sentence: '{test_text[0]}'")
print(f"Predicted Sentiment: {prediction[0]}")
