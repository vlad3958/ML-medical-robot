import json
import os
import queue
import threading
import time
import atexit
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import sounddevice as sd
import speech_recognition as sr

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from keras.models import load_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts/models/model.keras"
LABELS_PATH = ROOT / "artifacts/models/labels.json"
CFG_PATH = ROOT / "artifacts/models/cfg.json"
LATENCY_EXPORT_DIR = ROOT / "artifacts" / "latency"

model = load_model(MODEL_PATH)
labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))

SR = cfg["sr"]
FRAMES = cfg["frames"]

DURATION = 2.0
BUFFER_SIZE = int(SR * DURATION)

ASR_LANGS = ("uk-UA",)
ASR_START_SPEECH_PROB = 0.68
ASR_END_SILENCE_SEC = 0.9
ASR_MIN_UTTERANCE_SEC = 1.2
ASR_MAX_UTTERANCE_SEC = 20.0
ASR_PRE_SPEECH_SEC = 1.0
LABEL_UI_NAME = {
    "human_screaming": "distress alert",
    "human-screaming": "distress alert",
}

audio_queue = queue.Queue()
asr_queue = queue.Queue(maxsize=1)
asr_result_queue = queue.Queue(maxsize=1)
history = []
MAX_LATENCY_POINTS = 800

state = {
    "running": False,
    "status": "Idle",
    "top_label": "-",
    "top_conf": 0.0,
    "top_input_ts": None,
    "speech_text": "Waiting for speech...",
    "speech_pending_input_ts": None,
    "probs": {label: 0.0 for label in labels},
    "latency_ms": {
        "environment": None,
        "human-screaming": None,
        "speech": None,
    },
    "latency_history": {
        "environment": [],
        "human-screaming": [],
        "speech": [],
    },
}
state_lock = threading.Lock()
stop_event = threading.Event()
ui_cache_lock = threading.Lock()
ui_cache = {
    "status": "",
    "top_class": "",
    "probs": None,
    "speech_text": "",
    "latency_text": "",
    "avg_text": "",
    "saved_report": "",
    "runtime": "Stopped",
}


def normalize_output_key(pred_label):
    if pred_label in ("human_screaming", "human-screaming"):
        return "human-screaming"
    if pred_label == "speech":
        return "speech"
    return "environment"


def format_latency_ms(value):
    if value is None:
        return "N/A"
    return f"{value:.1f} ms"


def append_latency_unlocked(stream_key, value_ms):
    state["latency_ms"][stream_key] = value_ms
    stream_history = state["latency_history"][stream_key]
    stream_history.append(value_ms)
    if len(stream_history) > MAX_LATENCY_POINTS:
        del stream_history[:-MAX_LATENCY_POINTS]


def build_avg_text(latency_history):
    lines = []
    for stream_key in ("environment", "human-screaming", "speech"):
        series = latency_history.get(stream_key, [])
        if not series:
            lines.append(f"{stream_key}: avg N/A (n=0)")
            continue
        avg = float(np.mean(series))
        lines.append(f"{stream_key}: avg {avg:.1f} ms (n={len(series)})")
    return "\n".join(lines)


def build_latency_report_payload(latency_history):
    summary = {}
    for stream_key in ("environment", "human-screaming", "speech"):
        series = latency_history.get(stream_key, [])
        if series:
            summary[stream_key] = {
                "count": len(series),
                "avg_ms": round(float(np.mean(series)), 3),
                "min_ms": round(float(np.min(series)), 3),
                "max_ms": round(float(np.max(series)), 3),
            }
        else:
            summary[stream_key] = {
                "count": 0,
                "avg_ms": None,
                "min_ms": None,
                "max_ms": None,
            }

    return {
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sample_rate": SR,
        "metrics": {
            "latency_history_ms": latency_history,
            "summary": summary,
        },
    }


def save_latency_report():
    with state_lock:
        latency_history = {
            key: [round(float(v), 3) for v in values]
            for key, values in state["latency_history"].items()
        }

    has_data = any(latency_history[key] for key in latency_history)
    if not has_data:
        return None

    payload = build_latency_report_payload(latency_history)
    LATENCY_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = LATENCY_EXPORT_DIR / f"latency_report_{ts}.json"
    latest_path = LATENCY_EXPORT_DIR / "latency_report_latest.json"

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    report_path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")
    return str(report_path)


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


def audio_callback(indata, _frames, _time_info, status):
    if status:
        return
    audio_queue.put((indata.copy(), time.perf_counter()))


def asr_worker():
    while not stop_event.is_set():
        try:
            chunk, utterance_start_ts = asr_queue.get(timeout=0.2)
        except queue.Empty:
            continue

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
            asr_result_queue.put((text, utterance_start_ts))


def update_state(probs, pred_label, speech_prob, input_ts):
    top_conf = float(np.max(probs))

    if pred_label in LABEL_UI_NAME:
        status = "Distress alert"
    elif pred_label == "speech":
        status = "Speech detected"
    else:
        status = "Environment detected"

    with state_lock:
        state["status"] = status
        state["top_label"] = pred_label
        state["top_conf"] = top_conf
        state["top_input_ts"] = input_ts
        state["probs"] = {
            label: float(probs[idx])
            for idx, label in enumerate(labels)
        }


def inference_worker():
    audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    speech_idx = labels.index("speech") if "speech" in labels else 0

    in_utterance = False
    utterance_chunks = []
    utterance_samples = 0
    last_speech_ts = 0.0

    pre_speech_chunks = []
    pre_speech_samples = 0
    pre_speech_limit = int(SR * ASR_PRE_SPEECH_SEC)

    with sd.InputStream(
        samplerate=SR,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=1024,
    ):
        while not stop_event.is_set():
            try:
                chunk, input_ts = audio_queue.get(timeout=0.2)
                data = chunk.flatten()
            except queue.Empty:
                continue

            audio_buffer = np.roll(audio_buffer, -len(data))
            audio_buffer[-len(data):] = data

            y = audio_buffer / (np.max(np.abs(audio_buffer)) + 1e-6)
            probs = model.predict(preprocess(y), verbose=0)[0]
            probs = smooth(probs)

            pred_idx = int(np.argmax(probs))
            pred_label = labels[pred_idx]
            speech_prob = float(probs[speech_idx])
            update_state(probs, pred_label, speech_prob, input_ts)

            pre_speech_chunks.append((data.copy(), input_ts))
            pre_speech_samples += len(data)
            while pre_speech_samples > pre_speech_limit and pre_speech_chunks:
                removed_chunk, _removed_ts = pre_speech_chunks.pop(0)
                pre_speech_samples -= len(removed_chunk)

            now = time.time()
            is_speech_now = speech_prob >= ASR_START_SPEECH_PROB
            just_started = False

            if is_speech_now:
                last_speech_ts = now
                if not in_utterance:
                    in_utterance = True
                    utterance_chunks = [(chunk.copy(), ts) for chunk, ts in pre_speech_chunks]
                    utterance_samples = sum(len(chunk) for chunk, _ts in utterance_chunks)
                    just_started = True

            if in_utterance:
                if not just_started:
                    utterance_chunks.append((data.copy(), input_ts))
                    utterance_samples += len(data)

                if utterance_samples >= int(SR * ASR_MAX_UTTERANCE_SEC):
                    if asr_queue.empty():
                        utterance_start_ts = utterance_chunks[0][1] if utterance_chunks else input_ts
                        utterance_audio = np.concatenate([audio_chunk for audio_chunk, _ts in utterance_chunks], axis=0)
                        asr_queue.put((utterance_audio, utterance_start_ts))
                    in_utterance = False
                    utterance_chunks = []
                    utterance_samples = 0
                elif now - last_speech_ts >= ASR_END_SILENCE_SEC:
                    duration_sec = utterance_samples / SR
                    if duration_sec >= ASR_MIN_UTTERANCE_SEC and asr_queue.empty():
                        utterance_start_ts = utterance_chunks[0][1] if utterance_chunks else input_ts
                        utterance_audio = np.concatenate([audio_chunk for audio_chunk, _ts in utterance_chunks], axis=0)
                        asr_queue.put((utterance_audio, utterance_start_ts))
                    in_utterance = False
                    utterance_chunks = []
                    utterance_samples = 0

            try:
                new_text, utterance_start_ts = asr_result_queue.get_nowait()
                if new_text:
                    with state_lock:
                        state["speech_text"] = new_text
                        state["speech_pending_input_ts"] = utterance_start_ts
            except queue.Empty:
                pass


def start_runtime():
    if state["running"]:
        return "Running"

    stop_event.clear()
    history.clear()

    with state_lock:
        state["running"] = True
        state["status"] = "Starting microphone stream..."
        state["speech_text"] = "Waiting for speech..."
        state["speech_pending_input_ts"] = None
        state["top_label"] = "-"
        state["top_conf"] = 0.0
        state["top_input_ts"] = None
        state["probs"] = {label: 0.0 for label in labels}
        state["latency_ms"] = {
            "environment": None,
            "human-screaming": None,
            "speech": None,
        }
        state["latency_history"] = {
            "environment": [],
            "human-screaming": [],
            "speech": [],
        }

    threading.Thread(target=asr_worker, daemon=True).start()
    threading.Thread(target=inference_worker, daemon=True).start()
    return "Running"


def stop_runtime():
    if not state["running"]:
        return "Stopped"

    stop_event.set()
    report_path = save_latency_report()
    with state_lock:
        state["running"] = False
        state["status"] = "Stopped"
    if report_path:
        return f"Stopped | saved: {report_path}"
    return "Stopped"


def save_latency_report_on_exit():
    try:
        save_latency_report()
    except Exception:
        pass


def poll_ui():
    now = time.perf_counter()

    with state_lock:
        running = state["running"]
        status = state["status"]
        top_label = state["top_label"]
        top_conf = state["top_conf"]
        top_input_ts = state["top_input_ts"]
        speech_text = state["speech_text"]
        speech_pending_input_ts = state["speech_pending_input_ts"]
        probs = dict(state["probs"])
        latencies = dict(state["latency_ms"])
        latency_history = {
            key: list(values)
            for key, values in state["latency_history"].items()
        }

        if top_input_ts is not None:
            output_key = normalize_output_key(top_label)
            if output_key in ("environment", "human-screaming"):
                latency_ms = max(0.0, (now - top_input_ts) * 1000.0)
                append_latency_unlocked(output_key, latency_ms)
                latencies[output_key] = latency_ms
                latency_history[output_key] = list(state["latency_history"][output_key])

    top_label_ui = LABEL_UI_NAME.get(top_label, top_label.replace("_", " "))
    top_class = f"{top_label_ui} ({top_conf * 100:.1f}%)"

    runtime = "Running" if running else "Stopped"

    status_out = gr.skip()
    top_class_out = gr.skip()
    probs_out = gr.skip()
    transcript_out = gr.skip()
    latency_out = gr.skip()
    avg_out = gr.skip()
    runtime_out = gr.skip()

    if speech_text != ui_cache["speech_text"] and speech_pending_input_ts is not None:
        speech_latency_ms = max(0.0, (now - speech_pending_input_ts) * 1000.0)
        with state_lock:
            append_latency_unlocked("speech", speech_latency_ms)
            state["speech_pending_input_ts"] = None
            latencies["speech"] = speech_latency_ms
            latency_history["speech"] = list(state["latency_history"]["speech"])

    latency_text = (
        f"environment: {format_latency_ms(latencies['environment'])}\n"
        f"human-screaming: {format_latency_ms(latencies['human-screaming'])}\n"
        f"speech: {format_latency_ms(latencies['speech'])}"
    )
    avg_text = build_avg_text(latency_history)

    with ui_cache_lock:
        if status != ui_cache["status"]:
            ui_cache["status"] = status
            status_out = status

        if top_class != ui_cache["top_class"]:
            ui_cache["top_class"] = top_class
            top_class_out = top_class

        display_probs = {
            LABEL_UI_NAME.get(label, label.replace("_", " ")): value
            for label, value in probs.items()
        }

        if ui_cache["probs"] is None:
            ui_cache["probs"] = probs
            probs_out = display_probs
        else:
            any_change = any(
                abs(probs.get(label, 0.0) - ui_cache["probs"].get(label, 0.0)) >= 0.01
                for label in labels
            )
            if any_change:
                ui_cache["probs"] = probs
                probs_out = display_probs

        if speech_text != ui_cache["speech_text"]:
            ui_cache["speech_text"] = speech_text
            transcript_out = speech_text

        if latency_text != ui_cache["latency_text"]:
            ui_cache["latency_text"] = latency_text
            latency_out = latency_text

        if avg_text != ui_cache["avg_text"]:
            ui_cache["avg_text"] = avg_text
            avg_out = avg_text

        if runtime != ui_cache["runtime"]:
            ui_cache["runtime"] = runtime
            runtime_out = runtime

    return (
        status_out,
        top_class_out,
        probs_out,
        transcript_out,
        latency_out,
        avg_out,
        runtime_out,
    )


def build_app():
    css = """
    body, .gradio-container {
        font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
        background: radial-gradient(circle at 10% 20%, #d6f7ff 0%, #f7fbff 35%, #f4f9ef 100%);
    }
    .shell {
        max-width: 1000px;
        margin: 0 auto;
        border-radius: 24px;
        padding: 20px;
        background: rgba(255, 255, 255, 0.82);
        box-shadow: 0 18px 50px rgba(29, 59, 87, 0.18);
        backdrop-filter: blur(10px);
    }
    .footer-note {
        font-size: 0.9rem;
        color: #32516a;
    }
    .dashboard-title {
        color: #000000 !important;
        font-weight: 700;
    }
    .dashboard-title * {
        color: #000000 !important;
    }
    """

    with gr.Blocks(title="Realtime Audio Inference", css=css) as demo:
        with gr.Column(elem_classes=["shell"]):
            gr.Markdown("## Realtime Audio Inference Dashboard", elem_classes=["dashboard-title"])

            with gr.Row():
                start_btn = gr.Button("Start", variant="primary")
                stop_btn = gr.Button("Stop", variant="secondary")
                runtime_state = gr.Textbox(label="Engine", value="Stopped", interactive=False)

            with gr.Row():
                status_box = gr.Textbox(label="Status", value="Idle", interactive=False)
                top_box = gr.Textbox(label="Top class", value="-", interactive=False)

            probs_label = gr.Label(
                label="Class probabilities",
                value={LABEL_UI_NAME.get(label, label.replace("_", " ")): 0.0 for label in labels},
                num_top_classes=len(labels),
            )

            transcript = gr.Textbox(
                label="Recognized Speech",
                value="Waiting for speech...",
                lines=3,
                interactive=False,
            )

            latency_box = gr.Textbox(
                label="Signal -> screen latency (ms)",
                value="environment: N/A\nhuman-screaming: N/A\nspeech: N/A",
                lines=4,
                interactive=False,
            )

            avg_box = gr.Textbox(
                label="Average latency (ms)",
                value="environment: avg N/A (n=0)\nhuman-screaming: avg N/A (n=0)\nspeech: avg N/A (n=0)",
                lines=4,
                interactive=False,
            )

            gr.Markdown("<div class='footer-note'>Tip: use headphones to reduce microphone echo.</div>")

        start_btn.click(fn=start_runtime, outputs=[runtime_state])
        stop_btn.click(fn=stop_runtime, outputs=[runtime_state])

        timer = gr.Timer(value=0.30)
        timer.tick(
            fn=poll_ui,
            outputs=[
                status_box,
                top_box,
                probs_label,
                transcript,
                latency_box,
                avg_box,
                runtime_state,
            ],
        )

    return demo


if __name__ == "__main__":
    atexit.register(save_latency_report_on_exit)
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, show_api=False)