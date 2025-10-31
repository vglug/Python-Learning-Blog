# -------------------------------------------------------
# Sentiment Analysis using Pretrained Models (DistilBERT)
# and Traditional ML (Logistic Regression)
# -------------------------------------------------------

# Import required libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd

# -------------------------
# Download necessary NLTK data
# -------------------------
nltk.download('punkt')        # For tokenization
nltk.download('punkt_tab')    # Fix for newer NLTK versions
nltk.download('stopwords')    # For removing stop words

# -------------------------
# Create a small sample dataset
# -------------------------
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

# Convert dataset to DataFrame
df = pd.DataFrame(data)

# -------------------------
# Preprocessing Function
# -------------------------
def preprocess(text):
    tokens = word_tokenize(text.lower())             # Convert to lowercase and tokenize
    stop_words = set(stopwords.words('english'))     # Get English stopwords
    clean_tokens = [w for w in tokens if w.isalpha() and w not in stop_words]  # Remove punctuation & stopwords
    return " ".join(clean_tokens)                    # Join tokens back into a clean string

# Apply preprocessing to each text
df['clean_text'] = df['text'].apply(preprocess)

# -------------------------
# Sentiment Analysis using Pretrained Transformer (DistilBERT)
# -------------------------
print("\n🔹 Transformer Sentiment Analysis (DistilBERT):")

# Load pretrained model pipeline
sentiment_model = pipeline("sentiment-analysis")

# Analyze each sentence
for i, sentence in enumerate(df['text']):
    result = sentiment_model(sentence)[0]
    print(f"Text: {sentence}")
    print(f"→ Label: {result['label']}, Score: {result['score']:.3f}\n")

# -------------------------
# Traditional Machine Learning Method (TF-IDF + Logistic Regression)
# -------------------------
print("\n🔹 Traditional Machine Learning (Logistic Regression):")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.3, random_state=42
)

# Convert text into numerical features using TF-IDF
tfidf = TfidfVectorizer(max_features=500)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict sentiment on test data
y_pred = model.predict(X_test_tfidf)

# Display evaluation report
print(classification_report(y_test, y_pred))
