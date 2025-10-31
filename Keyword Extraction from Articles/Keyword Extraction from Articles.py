# Keyword Extraction from Articles

import nltk
nltk.download('stopwords')

from rake_nltk import Rake

text = """Natural Language Processing is a branch of Artificial Intelligence
that helps computers understand human language."""

r = Rake()

r.extract_keywords_from_text(text)

keywords = r.get_ranked_phrases()

print("Extracted Keywords:")
for word in keywords:
    print(word)
