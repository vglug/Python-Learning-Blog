from googletrans import Translator
translator = Translator()
text = input("Enter text to translate: ")
target_lang = input("Enter target language code (e.g., 'es', 'fr', 'ta'): ")
translation = translator.translate(text, dest=target_lang)
print(f"\nOriginal: {text}")
print(f"Translated ({target_lang}): {translation.text}")
print("\nTranslation completed.")
