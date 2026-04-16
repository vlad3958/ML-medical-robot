import os
import json
import queue
import time
import threading
import numpy as np
import sounddevice as sd
import librosa
import speech_recognition as sr
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from keras.models import load_model

# === PATHS ===
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts/models/model.keras"
LABELS_PATH = ROOT / "artifacts/models/labels.json"
CFG_PATH = ROOT / "artifacts/models/cfg.json"

# === LOAD ===
model = load_model(MODEL_PATH)
labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))

SR = cfg["sr"]
FRAMES = cfg["frames"]

# === AUDIO BUFFER ===
DURATION = 2.0  # секунди
BUFFER_SIZE = int(SR * DURATION)

audio_queue = queue.Queue()
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

asr_queue = queue.Queue(maxsize=1)
asr_result_queue = queue.Queue(maxsize=1)
ASR_LANGS = ("uk-UA",)
ASR_START_SPEECH_PROB = 0.68
ASR_END_SILENCE_SEC = 0.9
ASR_MIN_UTTERANCE_SEC = 1.2
ASR_MAX_UTTERANCE_SEC = 20.0
ASR_PRE_SPEECH_SEC = 1.0


# === PREPROCESS ===
def preprocess(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=cfg["n_mels"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop"],
    )

    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel + 80.0) / 80.0
    mel = np.nan_to_num(mel).T
    mel = np.clip(mel, 0.0, 1.0)

    if len(mel) < FRAMES:
        mel = np.pad(mel, ((0, FRAMES - len(mel)), (0, 0)))
    else:
        mel = mel[-FRAMES:]

    return mel[None, ..., None]


# === CALLBACK ===
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())


# === SMOOTHING ===
history = []


def smooth(probs):
    history.append(probs)
    if len(history) > 5:
        history.pop(0)
    return np.mean(history, axis=0)


def transcribe_audio(y):
    recognizer = sr.Recognizer()
    pcm16 = np.clip(y, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    audio = sr.AudioData(pcm16.tobytes(), sample_rate=SR, sample_width=2)

    for lang in ASR_LANGS:
        try:
            text = recognizer.recognize_google(audio, language=lang)
            if text:
                return text
        except sr.UnknownValueError:
            continue
    return None


def asr_worker():
    while True:
        chunk = asr_queue.get()
        try:
            text = transcribe_audio(chunk)
        except sr.RequestError:
            text = "<stt service unavailable>"
        if text:
            while not asr_result_queue.empty():
                try:
                    asr_result_queue.get_nowait()
                except queue.Empty:
                    break
            asr_result_queue.put(text)


# === MAIN LOOP ===
def main():
    global audio_buffer
    last_text = ""
    speech_idx = labels.index("speech") if "speech" in labels else 1
    in_utterance = False
    utterance_chunks = []
    utterance_samples = 0
    last_speech_ts = 0.0
    pre_speech_chunks = []
    pre_speech_samples = 0
    pre_speech_limit = int(SR * ASR_PRE_SPEECH_SEC)

    print("🎤 Listening... (Ctrl+C to stop)\n")

    threading.Thread(target=asr_worker, daemon=True).start()

    with sd.InputStream(
        samplerate=SR,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=1024,
    ):
        while True:
            data = audio_queue.get().flatten()

            # оновлюємо буфер
            audio_buffer = np.roll(audio_buffer, -len(data))
            audio_buffer[-len(data):] = data

            # нормалізація
            y = audio_buffer / (np.max(np.abs(audio_buffer)) + 1e-6)

            x = preprocess(y)
            probs = model.predict(x, verbose=0)[0]
            probs = smooth(probs)

            pred_idx = int(np.argmax(probs))
            pred_label = labels[pred_idx]
            speech_prob = float(probs[speech_idx])

            # Зберігаємо короткий буфер до початку мовлення, щоб не втрачати перші слова.
            pre_speech_chunks.append(data.copy())
            pre_speech_samples += len(data)
            while pre_speech_samples > pre_speech_limit and pre_speech_chunks:
                removed = pre_speech_chunks.pop(0)
                pre_speech_samples -= len(removed)

            # === логіка категорій ===
            if pred_label == "human_screaming":
                status = "🚨 DISTRESS"
            elif pred_label == "speech":
                status = "🗣️ SPEECH"
            else:
                status = "🏠 ENVIRONMENT"

            now = time.time()
            is_speech_now = speech_prob >= ASR_START_SPEECH_PROB
            just_started = False
            if is_speech_now:
                last_speech_ts = now
                if not in_utterance:
                    in_utterance = True
                    utterance_chunks = [c.copy() for c in pre_speech_chunks]
                    utterance_samples = sum(len(c) for c in utterance_chunks)
                    just_started = True

            if in_utterance:
                if not just_started:
                    utterance_chunks.append(data.copy())
                    utterance_samples += len(data)

                if utterance_samples >= int(SR * ASR_MAX_UTTERANCE_SEC):
                    if asr_queue.empty():
                        asr_queue.put(np.concatenate(utterance_chunks, axis=0))
                    in_utterance = False
                    utterance_chunks = []
                    utterance_samples = 0
                elif now - last_speech_ts >= ASR_END_SILENCE_SEC:
                    dur = utterance_samples / SR
                    if dur >= ASR_MIN_UTTERANCE_SEC and asr_queue.empty():
                        asr_queue.put(np.concatenate(utterance_chunks, axis=0))
                    in_utterance = False
                    utterance_chunks = []
                    utterance_samples = 0

            try:
                new_text = asr_result_queue.get_nowait()
                if new_text and new_text != last_text:
                    last_text = new_text
                    print(f"\nRecognized text: {last_text}")
            except queue.Empty:
                pass

            print(
                f"\r{status} | "
                f"env={probs[0]:.2f} "
                f"speech={speech_prob:.2f} "
                f"distress={probs[2]:.2f}",
                end="",
            )


if __name__ == "__main__":
    main()