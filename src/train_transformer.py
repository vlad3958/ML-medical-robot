import os
import json
from pathlib import Path

import librosa
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from keras import callbacks, layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

tf.get_logger().setLevel("ERROR")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/models"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT / "model_transformer.keras"
LABELS_PATH = OUT / "labels_transformer.json"
CFG_PATH = OUT / "cfg_transformer.json"
METRICS_PATH = OUT / "metrics_transformer.json"

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

EPOCHS = 20
BATCH = 32
EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
SEED = 42
MIN_AUDIO_SEC = 0.3

GLOBAL_MAX_FILES_PER_CLASS: int | None = None

LOW_PITCH_AUG_CLASS = "human_screaming"
LOW_PITCH_AUG_FRACTION = 0.45
LOW_PITCH_STEPS_MIN = -7.0
LOW_PITCH_STEPS_MAX = -3.0


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment(y):
    if y.size == 0:
        return y
    if np.random.rand() < 0.4:
        y = y + np.random.normal(0, 0.01, size=y.shape)
    if np.random.rand() < 0.3:
        y = y * np.random.uniform(0.8, 1.2)
    if np.random.rand() < 0.35:
        y = librosa.effects.pitch_shift(y, sr=SR, n_steps=np.random.uniform(-2.0, 1.0))
    if np.random.rand() < 0.35:
        rate = np.random.uniform(0.7, 1.1)
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


def augment_low_pitch(y):
    if y.size == 0:
        return y
    y = librosa.effects.pitch_shift(
        y,
        sr=SR,
        n_steps=np.random.uniform(LOW_PITCH_STEPS_MIN, LOW_PITCH_STEPS_MAX),
    )
    return np.clip(y, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

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
    return mel.astype(np.float32)


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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def collect_files_for_class(class_name: str, global_limit: int) -> list[Path]:
    files = [p for p in DATASETS[class_name].rglob("*") if p.suffix.lower() in EXT]
    np.random.shuffle(files)
    return files[:global_limit]


def compute_global_limit() -> int:
    if GLOBAL_MAX_FILES_PER_CLASS is not None:
        return GLOBAL_MAX_FILES_PER_CLASS
    counts = {}
    for name in LABELS:
        files = [p for p in DATASETS[name].rglob("*") if p.suffix.lower() in EXT]
        counts[name] = len(files)
    print("Total files per class:", counts)
    return min(counts.values())


def load_dataset():
    global_limit = compute_global_limit()
    print(f"Global file limit per class: {global_limit}")

    X, y = [], []
    counts = {name: 0 for name in LABELS}
    source_counts = {name: 0 for name in LABELS}
    bad_files = {name: 0 for name in LABELS}

    for class_id, name in enumerate(LABELS):
        files = collect_files_for_class(name, global_limit)
        for path in files:
            try:
                audio = load_audio(path)
                source_counts[name] += 1

                X.append(to_sample(audio))
                y.append(class_id)
                counts[name] += 1

                X.append(to_sample(augment(audio)))
                y.append(class_id)
                counts[name] += 1

                if name == LOW_PITCH_AUG_CLASS and np.random.rand() < LOW_PITCH_AUG_FRACTION:
                    X.append(to_sample(augment_low_pitch(audio)))
                    y.append(class_id)
                    counts[name] += 1

            except Exception:
                bad_files[name] += 1

    print("Source files used:", source_counts)
    print("Bad files skipped:", bad_files)
    print("Samples per class:", counts)
    return np.array(X), np.array(y), counts


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def report_per_class_metrics(model, X_val, y_val):
    probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    cm = confusion_matrix(y_val, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(OUT / "confusion_matrix_transformer.png")
    plt.show()

    recalls = recall_score(y_val, y_pred, labels=np.arange(len(LABELS)), average=None, zero_division=0)
    f1s = f1_score(y_val, y_pred, labels=np.arange(len(LABELS)), average=None, zero_division=0)
    per_class = {}
    print("Validation per-class metrics:")
    for i, label in enumerate(LABELS):
        print(f"  {label}: recall={recalls[i]:.4f}, f1={f1s[i]:.4f}")
        per_class[label] = {"recall": float(recalls[i]), "f1": float(f1s[i])}
    return per_class


def save_metrics(history, val_loss, val_accuracy, per_class):
    name_map = {"human_screaming": "distress"}
    per_class = {name_map.get(k, k): v for k, v in per_class.items()}
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


# ---------------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------------

def positional_encoding(length: int, depth: int) -> tf.Tensor:
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (depths // 2)) / np.float32(depth))
    angle_rads = positions * angle_rates
    pos_encoding = np.zeros_like(angle_rads)
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.constant(pos_encoding, dtype=tf.float32)


def transformer_block(x, num_heads: int, key_dim: int, ff_dim: int, dropout: float):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(x.shape[-1])(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x


def build_model():
    inp = layers.Input((FRAMES, N_MELS))

    x = layers.Dense(128, activation="relu")(inp)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    pos = positional_encoding(FRAMES, int(x.shape[-1]))
    x = x + pos

    x = transformer_block(x, num_heads=4, key_dim=32, ff_dim=256, dropout=0.2)
    x = transformer_block(x, num_heads=4, key_dim=32, ff_dim=256, dropout=0.2)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(96, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
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
        X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True,
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
        X_train, y_train,
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
        json.dumps({"sr": SR, "n_mels": N_MELS, "n_fft": N_FFT, "hop": HOP, "frames": FRAMES},
                   ensure_ascii=False),
        encoding="utf-8",
    )

    print("Saved")
    print(f"Metrics saved: {METRICS_PATH}")


if __name__ == "__main__":
    main()
