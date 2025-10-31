# File: SentimentAnalysis.py
# Project: Sentiment Analysis using Pretrained Models


from transformers import pipeline

def sentiment_analysis(text):
    # Load pretrained model and tokenizer (DistilBERT)
    sentiment_model = pipeline("sentiment-analysis")

    # Get prediction
    result = sentiment_model(text)[0]

    # Print and return result
    print(f"\n🧠 Input Text: {text}")
    print(f"📊 Sentiment: {result['label']} (Confidence: {result['score']:.2f})")

    return result

if __name__ == "__main__":
    print("=== Sentiment Analysis using Pretrained Models ===")
    user_input = input("Enter a sentence or paragraph: ")
    sentiment_analysis(user_input)
