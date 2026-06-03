import os
import json
import time
import math
from pathlib import Path
from collections import defaultdict

import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from keras import callbacks, layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, recall_score, precision_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

tf.get_logger().setLevel("ERROR")

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "environment":    ROOT / "datasets/environment",
    "speech":         ROOT / "datasets/ukrainian-speech",
    "human_screaming": ROOT / "datasets/human-screaming",
}

OUT_MODELS = ROOT / "artifacts/models"
OUT_EXP    = ROOT / "artifacts/experiments"
OUT_PLOTS  = OUT_EXP / "plots"

for d in (OUT_MODELS, OUT_EXP, OUT_PLOTS):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Фиксированные гиперпараметры (не зависят от длины окна)
# ---------------------------------------------------------------------------
LABELS   = ["environment", "speech", "human_screaming"]
SR       = 16000
N_MELS   = 64
N_FFT    = 1024
HOP      = 256
EPOCHS   = 20
BATCH    = 32
SEED     = 42
EXT      = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
MIN_AUDIO_SEC      = 0.3
GLOBAL_MAX_FILES_PER_CLASS = None   # None = мин. среди классов

LOW_PITCH_AUG_CLASS    = "human_screaming"
LOW_PITCH_AUG_FRACTION = 0.45
LOW_PITCH_STEPS_MIN    = -7.0
LOW_PITCH_STEPS_MAX    = -3.0

# ---------------------------------------------------------------------------
# Длины окон для эксперимента
# ---------------------------------------------------------------------------
WINDOW_DURATIONS = [1.0, 2.0, 4.0]   # секунды


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def frames_for_duration(duration_sec: float) -> int:
    """Число mel-frames, соответствующее длине окна."""
    return math.ceil(duration_sec * SR / HOP)


def augment(y: np.ndarray) -> np.ndarray:
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
        signal_rms = np.sqrt(np.mean(y ** 2) + 1e-8)
        noise_rms  = np.sqrt(np.mean(noise ** 2) + 1e-8)
        snr_db = np.random.uniform(8.0, 20.0)
        scale  = signal_rms / (noise_rms * (10 ** (snr_db / 20.0)) + 1e-8)
        y = y + noise * scale
    return np.clip(y, -1.0, 1.0)


def augment_low_pitch(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y
    y = librosa.effects.pitch_shift(
        y, sr=SR,
        n_steps=np.random.uniform(LOW_PITCH_STEPS_MIN, LOW_PITCH_STEPS_MAX),
    )
    return np.clip(y, -1.0, 1.0)


def load_audio(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True, duration=max(WINDOW_DURATIONS) + 1.0)
    if y.size < int(SR * MIN_AUDIO_SEC):
        raise ValueError("audio too short")
    peak = np.max(np.abs(y))
    if peak < 1e-6:
        raise ValueError("near-silent audio")
    y = y / (peak + 1e-6)
    if not np.isfinite(y).all():
        raise ValueError("non-finite samples")
    return y


def to_sample(y: np.ndarray, frames: int) -> np.ndarray:
    """Преобразует сырой аудиосигнал в mel-спектрограмму заданного числа frames."""
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel + 80.0) / 80.0
    mel = np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0).T
    mel = np.clip(mel, 0.0, 1.0)
    if mel.shape[0] < frames:
        mel = np.pad(mel, ((0, frames - mel.shape[0]), (0, 0)))
    else:
        start = np.random.randint(0, mel.shape[0] - frames + 1)
        mel = mel[start:start + frames]
    return mel[..., None].astype(np.float32)


# ---------------------------------------------------------------------------
# Датасет
# ---------------------------------------------------------------------------

def compute_global_limit() -> int:
    if GLOBAL_MAX_FILES_PER_CLASS is not None:
        return GLOBAL_MAX_FILES_PER_CLASS
    counts = {}
    for name in LABELS:
        files = [p for p in DATASETS[name].rglob("*") if p.suffix.lower() in EXT]
        counts[name] = len(files)
    print(f"  Всего файлов по классам: {counts}")
    return min(counts.values())


def load_dataset(frames: int):
    """
    Загружает датасет и преобразует в mel-фичи под заданное число frames.
    Возвращает X (N, frames, N_MELS, 1) и y (N,).
    """
    global_limit = compute_global_limit()
    print(f"  Лимит файлов на класс: {global_limit}")

    X, y = [], []

    for class_id, name in enumerate(LABELS):
        files = [p for p in DATASETS[name].rglob("*") if p.suffix.lower() in EXT]
        np.random.shuffle(files)
        files = files[:global_limit]

        ok, bad = 0, 0
        for path in files:
            try:
                audio = load_audio(path)
                ok += 1

                X.append(to_sample(audio, frames))
                y.append(class_id)

                X.append(to_sample(augment(audio), frames))
                y.append(class_id)

                if name == LOW_PITCH_AUG_CLASS and np.random.rand() < LOW_PITCH_AUG_FRACTION:
                    X.append(to_sample(augment_low_pitch(audio), frames))
                    y.append(class_id)

            except Exception:
                bad += 1

        print(f"    {name}: {ok} ok, {bad} bad")

    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------

def build_model(frames: int) -> tf.keras.Model:
    inp = layers.Input((frames, N_MELS, 1))
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


# ---------------------------------------------------------------------------
# Оценка
# ---------------------------------------------------------------------------

def evaluate_model(model: tf.keras.Model, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """Возвращает словарь с per-class метриками и inference latency."""
    # Метрики качества
    probs  = model.predict(X_val, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    recalls    = recall_score   (y_val, y_pred, labels=np.arange(len(LABELS)), average=None, zero_division=0)
    f1s        = f1_score       (y_val, y_pred, labels=np.arange(len(LABELS)), average=None, zero_division=0)
    precisions = precision_score(y_val, y_pred, labels=np.arange(len(LABELS)), average=None, zero_division=0)

    per_class = {}
    for i, label in enumerate(LABELS):
        per_class[label] = {
            "recall":    round(float(recalls[i]),    4),
            "f1":        round(float(f1s[i]),        4),
            "precision": round(float(precisions[i]), 4),
        }

    # Inference latency: среднее по 50 одиночным predict
    latencies = []
    for _ in range(50):
        sample = X_val[np.random.randint(len(X_val))][None]
        t0 = time.perf_counter()
        model.predict(sample, verbose=0)
        latencies.append((time.perf_counter() - t0) * 1000.0)  # мс

    return {
        "per_class":          per_class,
        "macro_f1":           round(float(np.mean(f1s)), 4),
        "macro_recall":       round(float(np.mean(recalls)), 4),
        "inference_mean_ms":  round(float(np.mean(latencies)), 2),
        "inference_std_ms":   round(float(np.std(latencies)),  2),
        "y_pred":             y_pred.tolist(),
        "y_true":             y_val.tolist(),
    }


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------

def plot_recall_f1(results: dict):
    durations = sorted(results.keys())
    label_names = LABELS
    colors = {"environment": "#378ADD", "speech": "#1D9E75", "human_screaming": "#D85A30"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ["recall", "f1"]):
        for label in label_names:
            vals = [results[d]["per_class"][label][metric] for d in durations]
            ax.plot([f"{d}s" for d in durations], vals,
                    marker="o", label=label.replace("_", " "),
                    color=colors[label], linewidth=2, markersize=7)
        ax.set_title(f"{'Recall' if metric == 'recall' else 'F1-score'} по классам", fontsize=13)
        ax.set_xlabel("Длина окна")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "recall_f1_by_window.png", dpi=150)
    plt.close()
    print(f"  Сохранён: {OUT_PLOTS / 'recall_f1_by_window.png'}")


def plot_latency(results: dict):
    durations = sorted(results.keys())
    means = [results[d]["inference_mean_ms"] for d in durations]
    stds  = [results[d]["inference_std_ms"]  for d in durations]

    # Полная latency = DURATION (буфер) + inference
    full_lat = [d * 1000 + results[d]["inference_mean_ms"] for d in durations]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(durations))
    ax.bar(x - 0.18, [d * 1000 for d in durations], 0.35,
           label="длина буфера (мс)", color="#B5D4F4")
    ax.bar(x + 0.18, means, 0.35, yerr=stds, capsize=4,
           label="inference (мс)", color="#378ADD")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}s" for d in durations])
    ax.set_ylabel("мс")
    ax.set_title("Задержка: буфер vs inference")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for i, fl in enumerate(full_lat):
        ax.text(i, fl + 20, f"total\n{fl:.0f} мс", ha="center", fontsize=9, color="#0C447C")

    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "latency_by_window.png", dpi=150)
    plt.close()
    print(f"  Сохранён: {OUT_PLOTS / 'latency_by_window.png'}")


def plot_confusion(y_true, y_pred, duration: float):
    cm = confusion_matrix(y_true, y_pred)
    short_labels = ["env", "speech", "distress"]
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=short_labels, yticklabels=short_labels,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix — окно {duration}с")
    plt.tight_layout()
    name = f"confusion_{duration}s.png".replace(".", "_").replace("_png", ".png")
    plt.savefig(OUT_PLOTS / name, dpi=150)
    plt.close()
    print(f"  Сохранён: {OUT_PLOTS / name}")


def plot_macro_summary(results: dict):
    """Сводный bar-chart macro-F1 и macro-recall для трёх окон."""
    durations = sorted(results.keys())
    macro_f1     = [results[d]["macro_f1"]     for d in durations]
    macro_recall = [results[d]["macro_recall"] for d in durations]

    x = np.arange(len(durations))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.18, macro_f1,     0.35, label="macro F1",     color="#1D9E75")
    ax.bar(x + 0.18, macro_recall, 0.35, label="macro Recall", color="#D85A30")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}s" for d in durations])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Macro F1 и Recall по длинам окна")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for i, (f1, rc) in enumerate(zip(macro_f1, macro_recall)):
        ax.text(i - 0.18, f1  + 0.01, f"{f1:.3f}",  ha="center", fontsize=9)
        ax.text(i + 0.18, rc  + 0.01, f"{rc:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "macro_summary.png", dpi=150)
    plt.close()
    print(f"  Сохранён: {OUT_PLOTS / 'macro_summary.png'}")


# ---------------------------------------------------------------------------
# Главный цикл
# ---------------------------------------------------------------------------

def main():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    print("=" * 60)
    print("Загрузка датасета (один раз, для максимального окна)")
    print("=" * 60)

    # Загружаем датасет один раз под максимальное окно,
    # для меньших окон сохраняем те же файлы (одинаковый val-сплит).
    # Но mel-фичи пересчитываем под каждое окно — это честный эксперимент.

    # Сначала получим индексы train/val через загрузку под max окно,
    # чтобы val-сплит был одинаковым для всех экспериментов.
    max_dur = max(WINDOW_DURATIONS)
    max_frames = frames_for_duration(max_dur)

    print(f"\nПредварительная загрузка под {max_dur}с (frames={max_frames}) для split-индексов...")
    X_base, y_base = load_dataset(max_frames)
    print(f"  Всего образцов: {len(y_base)}")

    # Фиксируем split-индексы
    indices = np.arange(len(y_base))
    idx_train, idx_val = train_test_split(
        indices, test_size=0.2, random_state=SEED, stratify=y_base, shuffle=True
    )
    print(f"  Train: {len(idx_train)}, Val: {len(idx_val)}")

    all_results = {}

    for duration in WINDOW_DURATIONS:
        frames = frames_for_duration(duration)
        tag    = f"{duration}s"
        print(f"\n{'=' * 60}")
        print(f"ОКНО {tag}  (FRAMES={frames})")
        print("=" * 60)

        # Пересчитываем mel-фичи под текущее окно
        print("  Загрузка датасета под текущее окно...")
        X, y = load_dataset(frames)

        # Используем те же индексы split
        # Если размер датасета совпал — берём напрямую
        if len(X) == len(X_base):
            X_train, X_val = X[idx_train], X[idx_val]
            y_train, y_val = y[idx_train], y[idx_val]
        else:
            # Если аугментация дала другое число (редко) — делаем новый split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=SEED, stratify=y, shuffle=True
            )

        # Веса классов
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
        print(f"  Class weights: {class_weights}")

        # Обучение
        model = build_model(frames)
        print(f"  Параметров модели: {model.count_params():,}")

        cbs = [
            callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ]

        t_train_start = time.time()
        hist = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH,
            class_weight=class_weights,
            callbacks=cbs,
            verbose=1,
        )
        train_time_sec = time.time() - t_train_start
        print(f"  Время обучения: {train_time_sec:.1f}с")

        # Сохранение модели
        model_path = OUT_MODELS / f"model_{tag}.keras"
        model.save(model_path)
        print(f"  Модель сохранена: {model_path}")

        # Оценка
        print("  Оценка на val-сете...")
        metrics = evaluate_model(model, X_val, y_val)

        print(f"  Macro F1: {metrics['macro_f1']:.4f}  |  Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"  Inference latency: {metrics['inference_mean_ms']:.1f} ± {metrics['inference_std_ms']:.1f} мс")
        for label, m in metrics["per_class"].items():
            print(f"    {label:20s}: recall={m['recall']:.4f}  f1={m['f1']:.4f}  precision={m['precision']:.4f}")

        # Confusion matrix
        plot_confusion(metrics["y_true"], metrics["y_pred"], duration)

        # Сохраняем историю обучения (без больших массивов)
        metrics["train_history"] = {
            "loss":     [round(v, 4) for v in hist.history.get("loss", [])],
            "val_loss": [round(v, 4) for v in hist.history.get("val_loss", [])],
            "accuracy":     [round(v, 4) for v in hist.history.get("accuracy", [])],
            "val_accuracy": [round(v, 4) for v in hist.history.get("val_accuracy", [])],
        }
        metrics["train_time_sec"] = round(train_time_sec, 1)
        metrics["frames"]         = frames
        metrics["duration_sec"]   = duration

        # Убираем большие списки из финального JSON
        metrics.pop("y_pred", None)
        metrics.pop("y_true", None)

        all_results[duration] = metrics

    # ---------------------------------------------------------------------------
    # Итоговый JSON
    # ---------------------------------------------------------------------------
    result_path = OUT_EXP / "window_results.json"
    result_path.write_text(
        json.dumps(
            {"windows": {str(k): v for k, v in all_results.items()}},
            ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    print(f"\nРезультаты сохранены: {result_path}")

    # ---------------------------------------------------------------------------
    # Сводные графики
    # ---------------------------------------------------------------------------
    print("\nПостроение сводных графиков...")
    plot_recall_f1(all_results)
    plot_latency(all_results)
    plot_macro_summary(all_results)

    # ---------------------------------------------------------------------------
    # Сводная таблица в консоль
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ ТАБЛИЦА")
    print("=" * 60)
    header = f"{'Окно':>6}  {'frames':>6}  {'macroF1':>8}  {'macroRec':>9}  {'lat_ms':>8}  {'scream_R':>9}  {'speech_R':>9}  {'env_R':>7}"
    print(header)
    print("-" * len(header))
    for dur in sorted(all_results.keys()):
        m = all_results[dur]
        print(
            f"{dur:>5}s"
            f"  {m['frames']:>6}"
            f"  {m['macro_f1']:>8.4f}"
            f"  {m['macro_recall']:>9.4f}"
            f"  {m['inference_mean_ms']:>8.1f}"
            f"  {m['per_class']['human_screaming']['recall']:>9.4f}"
            f"  {m['per_class']['speech']['recall']:>9.4f}"
            f"  {m['per_class']['environment']['recall']:>7.4f}"
        )

    print("\nГотово. Все файлы в artifacts/experiments/")


if __name__ == "__main__":
    main()