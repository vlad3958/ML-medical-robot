import os
import json
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import librosa
import speech_recognition as sr

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from keras.models import load_model

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "artifacts/models/model_transformer.keras"
LABELS_PATH = ROOT / "artifacts/models/labels_transformer.json"
CFG_PATH = ROOT / "artifacts/models/cfg_transformer.json"

ASR_LANGS = ("uk-UA",)
STEP_MS = 128


def transcribe_audio(y, sr_hz):
    recognizer = sr.Recognizer()
    pcm16 = np.clip(y, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    audio = sr.AudioData(pcm16.tobytes(), sample_rate=sr_hz, sample_width=2)

    for lang in ASR_LANGS:
        try:
            text = recognizer.recognize_google(audio, language=lang)
            if text:
                return text
        except sr.UnknownValueError:
            continue
    return None


def preprocess_windows(y, cfg):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=cfg["sr"],
        n_mels=cfg["n_mels"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop"],
    )

    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel + 80.0) / 80.0
    mel = np.nan_to_num(mel).T
    mel = np.clip(mel, 0.0, 1.0)

    frames = cfg["frames"]
    step = max(1, int(round((STEP_MS / 1000.0) * cfg["sr"] / cfg["hop"])))

    windows = []

    if len(mel) < frames:
        windows.append(np.pad(mel, ((0, frames - len(mel)), (0, 0))))
    else:
        for i in range(0, len(mel) - frames + 1, step):
            windows.append(mel[i:i + frames])

    return np.array(windows)[..., None]


def load_audio(path, cfg):
    try:
        y, _ = librosa.load(path, sr=cfg["sr"], mono=True)
        return y
    except Exception as exc:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                "Failed to read audio. Install ffmpeg or re-encode the file to WAV."
            ) from exc

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "decoded.wav"
            cmd = [
                ffmpeg,
                "-y",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                str(cfg["sr"]),
                str(wav_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg decode failed: {result.stderr.strip()}"
                ) from exc
            y, _ = librosa.load(wav_path, sr=cfg["sr"], mono=True)
            return y


def main(path):
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))

    model = load_model(MODEL_PATH)

    y = load_audio(path, cfg)
    y = y / (np.max(np.abs(y)) + 1e-6)

    x = preprocess_windows(y, cfg)

    probs = model.predict(x, verbose=0).mean(axis=0)

    pred_idx = int(np.argmax(probs))
    pred_label = labels[pred_idx]
    confidence = probs[pred_idx]

    print(f"\nPrediction: {pred_label} ({confidence:.2%})\n")

    if pred_label == "speech":
        try:
            text = transcribe_audio(y, cfg["sr"])
            if text:
                print(f"Recognized text: {text}\n")
            else:
                print("Recognized text: <not recognized>\n")
        except sr.RequestError:
            print("Recognized text: <stt service unavailable>\n")

    for i, label in enumerate(labels):
        print(f"{label}: {probs[i]:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])