import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

path = r"app\datasets\human-screaming\3c.wav"
sr = 16000
n_mels = 64
n_fft = 1024
hop = 256

y, _ = librosa.load(path, sr=sr, mono=True)
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop)
mel_db = librosa.power_to_db(mel, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db, sr=sr, hop_length=hop, x_axis="time", y_axis="mel")
plt.colorbar(format="%+2.0f dB")
plt.title("Mel-spectrogram: 3c.wav")
plt.tight_layout()
plt.show()