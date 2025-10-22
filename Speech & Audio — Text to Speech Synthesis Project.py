import pyttsx3
import sys

try:
    import pyperclip
except ImportError:
    pyperclip = None

def list_voices(engine):
    voices = engine.getProperty('voices')
    for idx, voice in enumerate(voices):
        print(f"{idx}: {voice.name} ({voice.gender}) [{voice.id}]")
    return voices

def get_text():
    source = input(
        "Enter '1' for manual input,
"
        "'2' to read from a text file,
"
        "'3' for clipboard (if available): "
    )
    if source == '1':
        return input("Type your text: ")
    elif source == '2':
        path = input("Enter file path: ")
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    elif source == '3' and pyperclip is not None:
        return pyperclip.paste()
    else:
        print("Invalid choice or clipboard not available.")
        sys.exit(1)

def main():
    engine = pyttsx3.init()
    print("Listing available voices...")
    voices = list_voices(engine)
    selection = input("Select voice (by number): ")
    try:
        selection = int(selection)
        engine.setProperty('voice', voices[selection].id)
    except (ValueError, IndexError):
        print("Using default voice.")

    rate = input("Enter speech rate (default 200): ")
    try:
        rate = int(rate)
        engine.setProperty('rate', rate)
    except ValueError:
        pass

    text = get_text()
    if not text.strip():
        print("No text provided.")
        return

    save_option = input("Save output to audio file? (y/n): ").lower()
    if save_option == 'y':
        filename = input("Enter filename (e.g. output.wav): ")
        engine.save_to_file(text, filename)
        print("Speech saved to", filename)
    else:
        engine.say(text)
        print("Speaking...")
        engine.runAndWait()
        print("Done.")

if __name__ == '__main__':
    main()
