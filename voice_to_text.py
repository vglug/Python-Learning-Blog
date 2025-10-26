import speech_recognition as sr
import pyttsx3
from datetime import datetime

def speak(text):
    """Function to make the system speak text."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def record_voice():
    """Record audio from microphone and convert to text."""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Adjusting for background noise... Please wait...")
        recognizer.adjust_for_ambient_noise(source)
        print("🎙 Speak now...")
        audio = recognizer.listen(source)

    try:
        print("Transcribing your speech...")
        text = recognizer.recognize_google(audio)
        print("✅ You said:", text)
        return text

    except sr.UnknownValueError:
        print("Sorry, I couldn’t understand your speech.")
        speak("Sorry, I couldn’t understand your speech.")
        return None

    except sr.RequestError:
        print("Network error. Check your internet connection.")
        speak("Network error. Please check your internet connection.")
        return None


def save_to_file(text):
    """Save the recognized text to a file with timestamp."""
    if text:
        filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w") as file:
            file.write(text)
        print(f"📁 Transcription saved as: {filename}")
        speak("Your speech has been saved successfully.")
    else:
        print("⚠ No text to save.")

def main():
    print("==== VOICE TO TEXT TRANSCRIPTION ====")
    speak("Welcome to Voice to Text Transcription Project")
    
    text = record_voice()
    save_to_file(text)

    print("🎉 Task completed!")

if __name__ == "__main__":
    main()