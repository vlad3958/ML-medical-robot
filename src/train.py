import os
import json
from pathlib import Path
from collections import defaultdict

import librosa
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from keras import callbacks, layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

tf.get_logger().setLevel("ERROR")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/models"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT / "model.keras"
LABELS_PATH = OUT / "labels.json"
CFG_PATH = OUT / "cfg.json"
METRICS_PATH = OUT / "metrics.json"

LABELS = ["environment", "speech", "human_screaming"]

DATASETS = {
    "environment": ROOT / "datasets/environment",
    "speech": ROOT / "datasets/ukrainian-speech",
    "human_screaming": ROOT / "datasets/human-screaming",
}

SR = 16000
N_MELS = 64
N_FFT = 1024
HOP = 256
FRAMES = 192
MAX_LOAD_SEC = 6.0

EPOCHS = 15
BATCH = 32
EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
SPEECH_MAX_FILES = 800
SEED = 42
MIN_AUDIO_SEC = 0.3

BALANCE_HUMAN_SCREAMING_BY_SOURCE = True
HUMAN_SCREAMING_SOURCE_TARGET = None


def augment(y):
    if y.size == 0:
        return y
    if np.random.rand() < 0.4:
        y = y + np.random.normal(0, 0.01, size=y.shape)
    if np.random.rand() < 0.3:
        y = y * np.random.uniform(0.8, 1.2)
    if np.random.rand() < 0.35:
        y = librosa.effects.pitch_shift(y, sr=SR, n_steps=np.random.uniform(-2.0, 2.0))

    if np.random.rand() < 0.35:
        rate = np.random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y, rate=rate)

    if np.random.rand() < 0.35:
        shift = int(np.random.uniform(-0.1, 0.1) * len(y))
        y = np.roll(y, shift)

    if np.random.rand() < 0.25 and len(y) > SR // 4:
        cut = np.random.randint(SR // 20, SR // 4)
        start = np.random.randint(0, max(1, len(y) - cut))
        y[start:start + cut] = 0.0

    if np.random.rand() < 0.4:
        noise = np.random.normal(0, 1, size=y.shape)
        signal_rms = np.sqrt(np.mean(y**2) + 1e-8)
        noise_rms = np.sqrt(np.mean(noise**2) + 1e-8)
        target_snr_db = np.random.uniform(8.0, 20.0)
        scale = signal_rms / (noise_rms * (10 ** (target_snr_db / 20.0)) + 1e-8)
        y = y + noise * scale

    return np.clip(y, -1.0, 1.0)


def to_sample(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel + 80.0) / 80.0
    mel = np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0).T
    mel = np.clip(mel, 0.0, 1.0)

    if mel.shape[0] < FRAMES:
        mel = np.pad(mel, ((0, FRAMES - mel.shape[0]), (0, 0)))
    else:
        start = np.random.randint(0, mel.shape[0] - FRAMES + 1)
        mel = mel[start:start + FRAMES]

    return mel[..., None].astype(np.float32)


def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True, duration=MAX_LOAD_SEC)
    if y.size < int(SR * MIN_AUDIO_SEC):
        raise ValueError("audio too short")
    peak = np.max(np.abs(y))
    if peak < 1e-6:
        raise ValueError("near-silent audio")
    y = y / (peak + 1e-6)
    if not np.isfinite(y).all():
        raise ValueError("non-finite samples")
    return y


def source_bucket(path, class_name):
    if class_name != "human_screaming":
        return "default"
    n = path.name.lower()
    if n.endswith(".mp3") and path.stem.isdigit():
        return "hs_numeric_mp3"
    if n.endswith((".wav", ".ogg")) and path.stem.endswith("c") and path.stem[:-1].isdigit():
        return "hs_numeric_c"
    if n.endswith(".wav") and "_" in path.stem:
        return "hs_underscore"
    return "hs_other"


def collect_audio_files(class_name):
    files = [p for p in DATASETS[class_name].rglob("*") if p.suffix.lower() in EXT]
    if class_name != "human_screaming" or not BALANCE_HUMAN_SCREAMING_BY_SOURCE:
        return files

    by_source = defaultdict(list)
    for p in files:
        by_source[source_bucket(p, class_name)].append(p)

    present = {k: v for k, v in by_source.items() if v}
    if len(present) <= 1:
        return files

    target = HUMAN_SCREAMING_SOURCE_TARGET
    if target is None:
        target = min(len(v) for v in present.values())

    selected = []
    for src in sorted(present.keys()):
        group = sorted(present[src])
        np.random.shuffle(group)
        selected.extend(group[:target])

    return selected


def load_dataset():
    X, y = [], []
    counts = {name: 0 for name in LABELS}
    source_counts = {name: 0 for name in LABELS}
    bad_files = {name: 0 for name in LABELS}

    for class_id, name in enumerate(LABELS):
        for path in collect_audio_files(name):
            if name == "speech" and source_counts[name] >= SPEECH_MAX_FILES:
                break
            try:
                audio = load_audio(path)
                source_counts[name] += 1

                X.append(to_sample(audio))
                y.append(class_id)
                counts[name] += 1

                X.append(to_sample(augment(audio)))
                y.append(class_id)
                counts[name] += 1

            except Exception:
                bad_files[name] += 1

    print("Source files used:", source_counts)
    print("Bad files skipped:", bad_files)
    return np.array(X), np.array(y), counts


def report_per_class_metrics(model, X_val, y_val):
    probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    recalls = recall_score(y_val, y_pred, labels=np.arange(len(LABELS)), average=None, zero_division=0)
    f1s = f1_score(y_val, y_pred, labels=np.arange(len(LABELS)), average=None, zero_division=0)

    per_class = {}
    print("Validation per-class metrics:")
    for i, label in enumerate(LABELS):
        print(f"  {label}: recall={recalls[i]:.4f}, f1={f1s[i]:.4f}")
        per_class[label] = {"recall": float(recalls[i]), "f1": float(f1s[i])}

    return per_class


def save_metrics(history, val_loss, val_accuracy, per_class):
    hist = history.history
    metrics = {
        "train": {
            "loss": [float(x) for x in hist.get("loss", [])],
            "accuracy": [float(x) for x in hist.get("accuracy", [])],
        },
        "validation": {
            "loss": [float(x) for x in hist.get("val_loss", [])],
            "accuracy": [float(x) for x in hist.get("val_accuracy", [])],
            "final_loss": float(val_loss),
            "final_accuracy": float(val_accuracy),
        },
        "per_class": per_class,
    }
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def build_model():
    inp = layers.Input((FRAMES, N_MELS, 1))

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(96, activation="relu")(x)
    x = layers.Dropout(0.35)(x)

    out = layers.Dense(len(LABELS), activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    X, y, counts = load_dataset()
    print("Loaded:", counts)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
        shuffle=True,
    )

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    print("Class weights:", class_weights)

    model = build_model()

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        class_weight=class_weights,
        callbacks=cbs,
    )

    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    per_class = report_per_class_metrics(model, X_val, y_val)
    save_metrics(history, val_loss, val_accuracy, per_class)

    model.save(MODEL_PATH)
    LABELS_PATH.write_text(json.dumps(LABELS, ensure_ascii=False), encoding="utf-8")
    CFG_PATH.write_text(
        json.dumps(
            {
                "sr": SR,
                "n_mels": N_MELS,
                "n_fft": N_FFT,
                "hop": HOP,
                "frames": FRAMES,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Saved")
    print(f"Metrics saved: {METRICS_PATH}")


if __name__ == "__main__":
    main()