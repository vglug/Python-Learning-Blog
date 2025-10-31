import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
nltk.download('punkt')
nltk.download('punkt_tab')  
nltk.download('stopwords')
data = {
    "text": [
        "I love this product! It's absolutely fantastic.",
        "This is the worst service I've ever had.",
        "The movie was okay, not great but not terrible.",
        "I am so happy with my new phone.",
        "I hate waiting in long queues."
    ],
    "label": ["positive", "negative", "neutral", "positive", "negative"]
}
df = pd.DataFrame(data)
def preprocess(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    clean_tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(clean_tokens)
df['clean_text'] = df['text'].apply(preprocess)
print("\n🔹 Transformer Sentiment Analysis (DistilBERT):")
sentiment_model = pipeline("sentiment-analysis")
for i, sentence in enumerate(df['text']):
    result = sentiment_model(sentence)[0]
    print(f"Text: {sentence}")
    print(f"→ Label: {result['label']}, Score: {result['score']:.3f}\n")
print("\n🔹 Traditional Machine Learning (Logistic Regression):")
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.3, random_state=42)
tfidf = TfidfVectorizer(max_features=500)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))
