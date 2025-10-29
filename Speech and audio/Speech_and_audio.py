import sounddevice as sd
import numpy as np

duration = 0.5  
threshold = 0.3  

print("Listening for Clap....")

while True:
    audio = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
    sd.wait()

    volume = np.linalg.norm(audio)
    
    if volume > threshold:
        print("Clap Detected! ")
