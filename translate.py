import pyaudio
import whisper
import requests
import requests

#pyaudio
#Audioaufnahme
#pip install pyaudio

#whisper
#Spracherkennung (OpenAI)
#pip install whisper

#requests
#HTTP-Anfragen an Ollama
#pip install requests

#ollama
#(Optional) Offizielle Ollama-Python-Bibliothek
#pip install ollama


# 1. Audio aufnehmen
def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Aufnahme läuft...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * 5)):  # 5 Sekunden
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    return b''.join(frames)

def translate_with_ollama(text):
    response = requests.post(
        'http://eva:11434/api/generate',
        json={
            'model': 'quen3:4b',
            'prompt': f'Übersetze folgenden Text ins Englische: {text}'
        }
    )
    return response.json()['response']

# 2. Audio zu Text
model = whisper.load_model("base")
audio = record_audio()
result = model.transcribe("audio.wav", fp16=False)
text = result["text"]

# 3. Text übersetzen
translation = translate_with_ollama(text)
print(translation)