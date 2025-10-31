# Project #288: Translate text using googletrans
from googletrans import Translator

# Create a translator object
translator = Translator()

# Input text
text = input("Enter text to translate: ")

# Choose target language (e.g., 'es' for Spanish, 'fr' for French, 'ta' for Tamil)
target_lang = input("Enter target language code (e.g., 'es', 'fr', 'ta'): ")

# Perform translation
translation = translator.translate(text, dest=target_lang)

# Show result
print(f"\nOriginal: {text}")
print(f"Translated ({target_lang}): {translation.text}")
print("\n✅ Translation completed.")