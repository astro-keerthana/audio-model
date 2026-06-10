# ============================================================
# WUALT Audio Analysis Engine — v3.1
# Streamlit Application
# ============================================================

import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch
import librosa
import soundfile as sf
import whisper
import logging
import tempfile
import os
import json
import csv
import io
import requests
import warnings
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import find_peaks
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoFeatureExtractor,
    ASTForAudioClassification,
    AutoModelForAudioClassification,
    Wav2Vec2FeatureExtractor,
    pipeline,
)
from datetime import datetime

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(funcName)s: %(message)s")
log = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="WUALT Audio Analysis",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp { background: #0a0e1a; color: #e2e8f0; }

  /* Header */
  .wualt-header {
    background: linear-gradient(135deg, #0f1728 0%, #1a2540 50%, #0f1728 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 36px 40px 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
  }
  .wualt-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4, #8b5cf6);
  }
  .wualt-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
    margin: 0 0 4px 0;
  }
  .wualt-subtitle {
    font-size: 0.95rem;
    color: #64748b;
    font-weight: 400;
    margin: 0;
  }
  .wualt-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #60a5fa;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-right: 8px;
  }

  /* Status pill */
  .status-normal  { background:#052e16; color:#4ade80; border:1px solid #166534; padding:6px 18px; border-radius:999px; font-weight:600; font-size:1.1rem; display:inline-block; }
  .status-stress  { background:#422006; color:#fb923c; border:1px solid #9a3412; padding:6px 18px; border-radius:999px; font-weight:600; font-size:1.1rem; display:inline-block; }
  .status-distress{ background:#450a0a; color:#f87171; border:1px solid #991b1b; padding:6px 18px; border-radius:999px; font-weight:600; font-size:1.1rem; display:inline-block; }

  /* Section cards */
  .card {
    background: #0f1728;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
  }
  .card-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 16px;
  }

  /* Metric tiles */
  .metric-grid { display: flex; flex-wrap: wrap; gap: 12px; }
  .metric-tile {
    background: #161d2e;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 14px 16px;
    min-width: 140px;
    flex: 1;
  }
  .metric-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing:0.07em; margin-bottom: 4px; }
  .metric-value { font-size: 1.35rem; font-weight: 600; color: #f1f5f9; font-family: 'JetBrains Mono', monospace; }
  .metric-unit  { font-size: 0.72rem; color: #64748b; margin-left: 3px; }

  /* Score bar */
  .score-row { display:flex; align-items:center; gap:12px; margin-bottom:10px; }
  .score-label { width:130px; font-size:0.8rem; color:#94a3b8; flex-shrink:0; }
  .score-bar-wrap { flex:1; background:#1e3a5f; border-radius:4px; height:8px; overflow:hidden; }
  .score-bar { height:8px; border-radius:4px; transition:width 0.4s ease; }
  .score-val { width:48px; text-align:right; font-size:0.8rem; font-family:'JetBrains Mono',monospace; color:#e2e8f0; flex-shrink:0; }

  /* Transcript box */
  .transcript-box {
    background: #161d2e;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 1rem;
    color: #e2e8f0;
    font-style: italic;
    line-height: 1.6;
  }
  .lang-tag {
    display:inline-block;
    background:#1e3a5f;
    color:#60a5fa;
    font-size:0.7rem;
    padding:2px 8px;
    border-radius:4px;
    font-weight:600;
    text-transform:uppercase;
    margin-right:8px;
    font-style:normal;
    font-family:'JetBrains Mono',monospace;
  }

  /* Detection rows */
  .detect-row {
    display:flex; align-items:center; gap:10px;
    padding: 8px 0;
    border-bottom: 1px solid #1a2540;
  }
  .detect-rank { color:#3b82f6; font-size:0.72rem; font-family:'JetBrains Mono',monospace; width:24px; }
  .detect-label { flex:1; font-size:0.85rem; color:#cbd5e1; }
  .detect-score { font-size:0.82rem; font-family:'JetBrains Mono',monospace; color:#94a3b8; width:52px; text-align:right; }

  /* Warning boxes */
  .warn-box {
    background:#1c1007;
    border:1px solid #92400e;
    border-left:3px solid #f59e0b;
    border-radius:8px;
    padding:12px 16px;
    font-size:0.85rem;
    color:#fbbf24;
    margin-top:8px;
  }
  .blocker-box {
    background:#1a0505;
    border:1px solid #991b1b;
    border-left:3px solid #ef4444;
    border-radius:8px;
    padding:12px 16px;
    font-size:0.85rem;
    color:#fca5a5;
    margin-top:8px;
  }

  /* Upload area */
  .stFileUploader > div > div {
    background: #0f1728 !important;
    border: 2px dashed #1e3a5f !important;
    border-radius: 12px !important;
  }

  /* Hide streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 24px; }
</style>
""", unsafe_allow_html=True)

# ── Configuration (identical to notebook Cell 4) ──────────────
MIC = {
    "sensitivity_dbfs": -36.0,
    "aop_db": 128.0,
    "noise_floor_db": 22.0,
    "snr_db": 72.0,
    "sample_rate": 16000,
}

FUSION_WEIGHTS = {
    "overall": {"distress": 0.25, "noise": 0.20, "danger": 0.30, "physio": 0.10, "linguistic": 0.15},
    "fusion":  {"distress": 0.40, "noise": 0.15, "danger": 0.20, "physio": 0.10, "linguistic": 0.15},
    "vector":  {"distress": 0.35, "noise": 0.15, "danger": 0.20, "physio": 0.10, "linguistic": 0.20},
}

PHYSIO_OVERRIDE_THRESHOLD = 0.55

AMBIENT_PHYSIO = {
    "Breathing", "Snoring", "Heartbeat",
    "Stomach rumble", "Heart murmur", "Sniff", "Hiccup",
}

DANGER_ENV = [
    "Screaming", "Glass breaking", "Gunshot, gunfire", "Explosion", "Crash",
    "Alarm", "Siren", "Fire alarm", "Smoke detector", "Slam", "Thud",
    "Splash, splatter", "Growling", "Thunder", "Baby cry, infant cry", "Shout",
    "Yell", "Bellow", "Children shouting", "Police car (siren)", "Ambulance (siren)",
    "Fire engine, fire truck (siren)", "Car alarm", "Air horn, truck horn",
    "Reversing beeps", "Chainsaw", "Drill", "Burst, pop",
]

DANGER_PHYSIO = [
    "Cough", "Sneeze", "Wheeze", "Breathing", "Heartbeat",
    "Crying, sobbing", "Groan", "Gasp", "Whimper",
    "Wail, moan", "Pant", "Throat clearing",
    "Baby cry, infant cry", "Snoring", "Hiccup", "Sniff",
    "Stomach rumble", "Heart murmur",
]

ALL_DANGER = list(dict.fromkeys(DANGER_ENV + DANGER_PHYSIO))

CLINICAL = {
    "jitter_normal_max": 0.0104,
    "shimmer_normal_max": 0.0381,
    "hnr_normal_min": 20.0,
}

SUPERB_EMOTION_MAP = {"ang": "angry", "hap": "happy", "neu": "neutral", "sad": "sad"}

DISTRESS_WEIGHTS = {"angry": 0.85, "sad": 0.75, "neutral": 0.00, "happy": 0.00}

PITCH_CFG = {
    "min_f0": 75.0, "max_f0": 600.0,
    "window_size": 0.05, "hop_size": 0.01, "voiced_thr": 0.25,
}

XNLI_LABELS = [
    "danger", "distress", "threat", "emergency", "violence",
    "normal conversation", "calm", "everyday activity", "neutral statement", "safe situation",
]

DANGER_PHRASES = [
    "help me", "I need help", "please help",
    "I am in danger", "someone is hurting me",
    "I am scared", "I am afraid", "I am being attacked",
    "call the police", "call an ambulance",
    "I can't breathe", "I am dying", "I am going to die",
    "leave me alone", "stop hurting me", "don't touch me",
    "get away from me", "I am being followed",
    "I don't feel safe", "something is wrong",
    "I am not okay", "this is not okay",
    "I am trapped", "I can't get out",
    "nobody is listening", "nobody cares",
    "I want to disappear", "I can't take this anymore",
    "aidez moi", "au secours", "hilfe", "ich brauche hilfe",
    "ayuda", "socorro", "aiuto", "mi aiuti",
    "помогите", "на помощь", "مساعدة", "النجدة",
    "助けて", "たすけて", "도와주세요", "살려주세요",
    "救命", "帮助我", "मदद करो", "बचाओ",
    "உதவி", "என்னை காப்பாற்றுங்கள்",
    "யாராவது உதவுங்கள்", "என்னால் தாங்க முடியவில்லை",
    "நான் பயமாக இருக்கிறேன்", "போலீஸை அழையுங்கள்",
    "என் உயிர் போகிறது",
]

# ── Model loading (cached) ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all_models():
    models = {}
    with st.spinner("Loading Whisper medium…"):
        models["whisper"] = whisper.load_model("medium")
    with st.spinner("Loading XLM-RoBERTa XNLI…"):
        models["xnli"] = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli",
            tokenizer="joeddav/xlm-roberta-large-xnli",
            device=0 if torch.cuda.is_available() else -1,
        )
    with st.spinner("Loading Multilingual MPNet…"):
        mpnet = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        models["mpnet"] = mpnet
        models["danger_embeddings"] = mpnet.encode(
            DANGER_PHRASES, convert_to_tensor=True, show_progress_bar=False
        )
    with st.spinner("Loading SER model…"):
        models["ser_extractor"] = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
        models["ser_model"] = AutoModelForAudioClassification.from_pretrained("superb/wav2vec2-base-superb-er")
        models["ser_model"].eval()
    with st.spinner("Loading AST…"):
        models["ast_extractor"] = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        models["ast_model"] = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        models["ast_model"].eval()
    with st.spinner("Loading YAMNet…"):
        models["yamnet"] = hub.load("https://tfhub.dev/google/yamnet/1")
        r = requests.get(
            "https://raw.githubusercontent.com/tensorflow/models/master/"
            "research/audioset/yamnet/yamnet_class_map.csv"
        )
        models["yamnet_classes"] = [row[2] for row in csv.reader(io.StringIO(r.text))][1:]
    return models

# ── Audio functions (identical logic to notebook) ─────────────
def load_audio(path, sr=16000):
    audio, sr = librosa.load(path, sr=sr, mono=True)
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    return audio, sr

def load_audio_raw(path, sr=16000):
    return librosa.load(path, sr=sr, mono=True)

def get_db_level(path):
    try:
        audio, _ = load_audio(path)
        audio = audio - np.mean(audio)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-10: return None
        dbfs = 20 * np.log10(rms)
        db_spl = dbfs - MIC["sensitivity_dbfs"] + 94.0
        return round(float(np.clip(db_spl, MIC["noise_floor_db"], MIC["aop_db"])), 1)
    except Exception as e:
        log.warning(f"get_db_level: {e}"); return None

def get_twa_dose(path):
    try:
        db = get_db_level(path)
        if db is None: return 0.0
        if db >= 90:
            permitted = 8 / (2 ** ((db - 90) / 5))
            return round(min(999.9, (1 / permitted) * 100), 1)
        return round((db / 90.0) * 100 * 0.5, 1)
    except: return 0.0

def get_noise_score(path):
    try:
        db = get_db_level(path)
        if db is None: return 0.0
        return round(min(1.0, max(0.0, (db - MIC["noise_floor_db"]) / (MIC["aop_db"] - MIC["noise_floor_db"]))), 4)
    except: return 0.0

def get_noise_status(path):
    db = get_db_level(path)
    if db is None: return "No Signal"
    if db < 70: return "Safe"
    elif db < 85: return "Moderate"
    else: return "Dangerous"

def _run_fusion(path, models):
    try:
        audio, _ = load_audio(path)
        audio_tf = tf.constant(audio.astype(np.float32))
        scores, _, _ = models["yamnet"](audio_tf)
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()

        yamnet_hits, yamnet_sum = {}, 0.0
        for cls in ALL_DANGER:
            for i, yc in enumerate(models["yamnet_classes"]):
                if cls.lower() in yc.lower():
                    v = float(mean_scores[i])
                    yamnet_hits[cls] = v; yamnet_sum += v; break

        inputs = models["ast_extractor"](audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = models["ast_model"](**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        id2label = models["ast_model"].config.id2label

        ast_hits, ast_sum = {}, 0.0
        for cls in ALL_DANGER:
            for idx, label in id2label.items():
                if cls.lower() in label.lower():
                    v = float(probs[idx])
                    ast_hits[cls] = v; ast_sum += v; break

        total = yamnet_sum + ast_sum + 1e-9
        wy, wa = yamnet_sum / total, ast_sum / total
        fused = {
            cls: round(yamnet_hits.get(cls, 0) * wy + ast_hits.get(cls, 0) * wa, 4)
            for cls in ALL_DANGER
        }
        return fused, round(wy, 3), round(wa, 3)
    except Exception as e:
        log.warning(f"_run_fusion: {e}")
        return {cls: 0.0 for cls in ALL_DANGER}, 0.5, 0.5

def _physio_override_triggered(fused, physio_score):
    if physio_score <= PHYSIO_OVERRIDE_THRESHOLD: return False
    physio_scores = {c: fused.get(c, 0.0) for c in DANGER_PHYSIO}
    top_physio = max(physio_scores, key=physio_scores.get)
    return top_physio not in AMBIENT_PHYSIO

def _get_voiced_f0(audio, sr):
    frame_len = int(PITCH_CFG["window_size"] * sr)
    hop_len = int(PITCH_CFG["hop_size"] * sr)
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio, fmin=PITCH_CFG["min_f0"], fmax=PITCH_CFG["max_f0"],
            sr=sr, frame_length=frame_len, hop_length=hop_len, fill_na=np.nan
        )
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0) & (voiced_prob > PITCH_CFG["voiced_thr"])]
        if len(voiced_f0) < 3:
            f0_yin = librosa.yin(audio, fmin=PITCH_CFG["min_f0"], fmax=PITCH_CFG["max_f0"],
                                 sr=sr, frame_length=frame_len, hop_length=hop_len)
            voiced_f0 = f0_yin[(f0_yin >= PITCH_CFG["min_f0"]) & (f0_yin <= PITCH_CFG["max_f0"])]
        return voiced_f0
    except: return np.array([])

def _build_f0_per_sample(audio, sr):
    frame_len = int(PITCH_CFG["window_size"] * sr)
    hop_len = int(PITCH_CFG["hop_size"] * sr)
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio, fmin=PITCH_CFG["min_f0"], fmax=PITCH_CFG["max_f0"],
            sr=sr, frame_length=frame_len, hop_length=hop_len, fill_na=np.nan
        )
    except: return np.full(len(audio), np.nan)
    n_frames = len(f0)
    n_samples = len(audio)
    frame_centres = np.arange(n_frames) * hop_len + hop_len // 2
    voiced_mask = voiced_flag & ~np.isnan(f0) & (voiced_prob > PITCH_CFG["voiced_thr"])
    voiced_idx = np.where(voiced_mask)[0]
    if len(voiced_idx) < 2: return np.full(n_samples, np.nan)
    f0_per_sample = np.full(n_samples, np.nan)
    for i in range(len(voiced_idx) - 1):
        idx_a, idx_b = voiced_idx[i], voiced_idx[i + 1]
        if (idx_b - idx_a) > 3: continue
        s_a, s_b = int(frame_centres[idx_a]), int(frame_centres[idx_b])
        f_a, f_b = f0[idx_a], f0[idx_b]
        if s_b <= s_a: continue
        interp = np.arange(s_a, min(s_b, n_samples))
        t = (interp - s_a) / (s_b - s_a)
        f0_per_sample[interp] = f_a + t * (f_b - f_a)
    return f0_per_sample

def _get_glottal_cycles(audio, sr):
    try:
        f0_per_sample = _build_f0_per_sample(audio, sr)
        n_samples = len(audio)
        periods, amplitudes = [], []
        cursor = 0
        while cursor < n_samples:
            f0_val = f0_per_sample[cursor]
            if np.isnan(f0_val):
                remaining = f0_per_sample[cursor:]
                next_v = np.argwhere(~np.isnan(remaining))
                if len(next_v) == 0: break
                cursor += int(next_v[0][0]); continue
            period_samples = int(round(sr / f0_val))
            if period_samples < 2: cursor += 1; continue
            end = cursor + period_samples
            if end > n_samples: break
            cycle = audio[cursor:end]
            peak_amp = float(np.max(np.abs(cycle)))
            if peak_amp < 0.001: cursor += period_samples; continue
            periods.append(1.0 / f0_val)
            amplitudes.append(peak_amp)
            cursor += period_samples
        return np.array(periods), np.array(amplitudes)
    except: return np.array([]), np.array([])

def get_vocal_metrics(path):
    audio, sr = load_audio_raw(path)
    voiced = _get_voiced_f0(audio, sr)
    periods, amplitudes = _get_glottal_cycles(audio, sr)

    pitch_hz = round(float(np.mean(voiced)), 1) if len(voiced) > 0 else 0.0
    pitch_sd = round(float(np.std(voiced)), 1) if len(voiced) > 1 else 0.0
    pitch_min = round(float(np.min(voiced)), 1) if len(voiced) > 0 else 0.0
    pitch_max = round(float(np.max(voiced)), 1) if len(voiced) > 0 else 0.0

    jitter = 0.0
    if len(periods) >= 3:
        p = periods[(periods >= 0.0001) & (periods <= 0.02)]
        if len(p) >= 3:
            ratios = p[1:] / (p[:-1] + 1e-10)
            valid = (ratios < 1.3) & (ratios > 1 / 1.3)
            p_clean = p[:-1][valid]
            if len(p_clean) >= 2:
                jitter = round(float((np.mean(np.abs(np.diff(p_clean))) / (np.mean(p_clean) + 1e-10)) * 100), 3)

    shimmer = 0.0
    if len(amplitudes) >= 3:
        valid_mask = (periods >= 0.0001) & (periods <= 0.02)
        a = amplitudes[valid_mask]
        if len(a) >= 3:
            ar = a[1:] / (a[:-1] + 1e-10)
            valid = (ar < 1.6) & (ar > 1 / 1.6)
            a_clean = a[:-1][valid]
            if len(a_clean) >= 2:
                shimmer = round(float((np.mean(np.abs(np.diff(a_clean))) / (np.mean(a_clean) + 1e-10)) * 100), 3)

    hnr = 0.0
    try:
        frame_len = int(0.04 * sr)
        hop_len = int(0.01 * sr)
        min_lag = int(sr / PITCH_CFG["max_f0"])
        max_lag = int(sr / PITCH_CFG["min_f0"])
        window = np.hanning(frame_len)
        hnr_vals = []
        for start in range(0, len(audio) - frame_len, hop_len):
            frame = audio[start:start + frame_len].copy()
            if np.max(np.abs(frame)) < 0.001: continue
            frame *= window
            frame -= np.mean(frame)
            acf = np.correlate(frame, frame, mode='full')
            acf = acf[len(acf) // 2:]
            if acf[0] < 1e-10: continue
            acf_norm = acf / acf[0]
            if max_lag >= len(acf_norm): continue
            seg = acf_norm[min_lag:max_lag]
            if len(seg) == 0: continue
            r = np.clip(float(np.max(seg)), 1e-9, 1.0 - 1e-9)
            hnr_vals.append(10.0 * np.log10(r / (1.0 - r)))
        hnr = round(float(np.mean(hnr_vals)), 2) if hnr_vals else 0.0
    except: pass

    intensity = 0.0
    try:
        frame_len = int(0.01 * sr)
        rms_vals = []
        for start in range(0, len(audio) - frame_len, frame_len):
            frame = audio[start:start + frame_len]
            rms = np.sqrt(np.mean(frame ** 2))
            if rms > 1e-10: rms_vals.append(rms)
        if rms_vals:
            mean_rms = np.sqrt(np.mean(np.array(rms_vals) ** 2))
            intensity = round(float(20.0 * np.log10(mean_rms + 1e-10) + 94.0), 2)
    except: pass

    speaking_rate = 0.0
    try:
        audio_n, sr_n = load_audio(path)
        hop = 512
        energy = np.array([np.sum(np.abs(audio_n[i:i + hop]) ** 2)
                           for i in range(0, len(audio_n) - hop, hop)])
        peaks, _ = find_peaks(energy, height=np.mean(energy) * 0.5, distance=4)
        duration = len(audio_n) / sr_n
        speaking_rate = round(len(peaks) / duration if duration > 0 else 0.0, 2)
    except: pass

    flags = []
    if jitter > CLINICAL["jitter_normal_max"] * 100: flags.append("High Jitter")
    if shimmer > CLINICAL["shimmer_normal_max"] * 100: flags.append("High Shimmer")
    if hnr < CLINICAL["hnr_normal_min"]: flags.append("Low HNR")
    vocal_health = "Normal" if not flags else " | ".join(flags)

    return {
        "pitch_hz": pitch_hz, "pitch_sd": pitch_sd,
        "pitch_min": pitch_min, "pitch_max": pitch_max,
        "jitter": jitter, "shimmer": shimmer, "hnr": hnr,
        "intensity": intensity, "speaking_rate": speaking_rate,
        "vocal_health": vocal_health,
    }

def get_emotion(path, models):
    try:
        audio, _ = load_audio(path)
        inputs = models["ser_extractor"](audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = models["ser_model"](**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        id2label = models["ser_model"].config.id2label
        top_idx = torch.argmax(probs).item()
        def normalize(l): return SUPERB_EMOTION_MAP.get(l.lower().strip(), l.lower().strip())
        all_emotions = {
            normalize(id2label.get(i, str(i))): round(float(probs[i]), 4)
            for i in range(len(probs))
        }
        all_emotions = dict(sorted(all_emotions.items(), key=lambda x: x[1], reverse=True))
        return {
            "top_emotion": normalize(id2label.get(top_idx, str(top_idx))),
            "confidence": round(float(probs[top_idx]), 4),
            "all_emotions": all_emotions,
        }
    except Exception as e:
        log.warning(f"get_emotion: {e}")
        return {"top_emotion": "unknown", "confidence": 0.0, "all_emotions": {}}

def get_linguistic(path, models):
    try:
        result = models["whisper"].transcribe(
            path, task="transcribe",
            fp16=torch.cuda.is_available(), verbose=False
        )
        text = result.get("text", "").strip()
        language = result.get("language", "unknown")
    except Exception as e:
        log.warning(f"whisper: {e}")
        text, language = "", "unknown"

    intent_score = 0.0
    if len(text.strip()) >= 3:
        try:
            res = models["xnli"](text, candidate_labels=XNLI_LABELS, multi_label=False)
            label_weights = {
                "danger": 1.00, "distress": 0.90, "threat": 0.85,
                "emergency": 0.80, "violence": 0.75,
                "normal conversation": 0.00, "calm": 0.00,
            }
            intent_score = round(float(max(
                (label_weights.get(l.lower(), 0.0) * s
                 for l, s in zip(res["labels"], res["scores"])
                 if label_weights.get(l.lower(), 0.0) > 0),
                default=0.0
            )), 4)
        except Exception as e:
            log.warning(f"xnli: {e}")

    semantic_score = 0.0
    if len(text.strip()) >= 3:
        try:
            q_emb = models["mpnet"].encode(text, convert_to_tensor=True, show_progress_bar=False)
            cos = util.cos_sim(q_emb, models["danger_embeddings"])[0]
            semantic_score = round(min(1.0, max(0.0, float(torch.max(cos)))), 4)
        except Exception as e:
            log.warning(f"mpnet: {e}")

    linguistic_score = round(min(1.0, intent_score * 0.60 + semantic_score * 0.40), 4)

    return {
        "text": text, "language": language,
        "intent_score": intent_score,
        "semantic_score": semantic_score,
        "linguistic_score": linguistic_score,
    }

def run_full_analysis(path, models):
    """Full pipeline — returns all results dict."""
    # Acoustic
    db = get_db_level(path)
    twa = get_twa_dose(path)
    noise_score = get_noise_score(path)
    noise_status = get_noise_status(path)

    # Fusion
    fused, wy, wa = _run_fusion(path, models)
    danger_score = round(max(fused.values()), 4)
    env_score = round(max((fused.get(c, 0) for c in DANGER_ENV), default=0.0), 4)
    physio_score = round(max((fused.get(c, 0) for c in DANGER_PHYSIO), default=0.0), 4)
    top_detections = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:5]

    # Vocal
    vocal = get_vocal_metrics(path)

    # Emotion
    emotion = get_emotion(path, models)

    # Linguistic
    ling = get_linguistic(path, models)

    # Distress score
    emotion_score = min(1.0, sum(
        DISTRESS_WEIGHTS.get(l.lower(), 0.0) * p
        for l, p in emotion["all_emotions"].items()
    ))
    j = vocal["jitter"] / (CLINICAL["jitter_normal_max"] * 100)
    s = vocal["shimmer"] / (CLINICAL["shimmer_normal_max"] * 100)
    h = max(0, CLINICAL["hnr_normal_min"] - vocal["hnr"]) / CLINICAL["hnr_normal_min"]
    vocal_score = min(1.0, (j + s + h) / 3)

    rate = vocal["speaking_rate"]
    if rate > 6.0: rate_score = min(1.0, (rate - 6.0) / 4.0)
    elif 0 < rate < 1.0: rate_score = min(1.0, 1.0 - rate)
    else: rate_score = 0.0

    distress_score = round(min(1.0, (
        emotion_score * 0.35 +
        vocal_score * 0.20 +
        physio_score * 0.15 +
        rate_score * 0.10 +
        ling["linguistic_score"] * 0.20
    )), 4)

    if distress_score < 0.25: distress_status = "Calm"
    elif distress_score < 0.50: distress_status = "Mild Distress"
    elif distress_score < 0.75: distress_status = "Moderate Distress"
    else: distress_status = "Severe Distress"

    # Overall risk
    w = FUSION_WEIGHTS["overall"]
    overall_score = round(min(1.0, (
        distress_score * w["distress"] +
        noise_score * w["noise"] +
        danger_score * w["danger"] +
        physio_score * w["physio"] +
        ling["linguistic_score"] * w["linguistic"]
    )), 4)

    physio_override = _physio_override_triggered(fused, physio_score)

    if danger_score > 0.25 or ling["linguistic_score"] > 0.70 or physio_override:
        overall_status = "🚨 Distress"
    elif overall_score >= 0.55:
        overall_status = "🚨 Distress"
    elif overall_score >= 0.25 or emotion["top_emotion"] in ("angry", "sad") or ling["linguistic_score"] > 0.40:
        overall_status = "⚠️ Stress"
    else:
        overall_status = "✅ Normal"

    return {
        "db": db, "twa": twa, "noise_score": noise_score, "noise_status": noise_status,
        "danger_score": danger_score, "env_score": env_score, "physio_score": physio_score,
        "top_detections": top_detections, "yamnet_weight": wy, "ast_weight": wa,
        "vocal": vocal,
        "emotion": emotion,
        "ling": ling,
        "distress_score": distress_score, "distress_status": distress_status,
        "overall_score": overall_score, "overall_status": overall_status,
    }

# ── Helper: score bar HTML ────────────────────────────────────
def score_bar(label, value, color="#3b82f6"):
    pct = int(value * 100)
    return f"""
    <div class="score-row">
      <div class="score-label">{label}</div>
      <div class="score-bar-wrap"><div class="score-bar" style="width:{pct}%;background:{color}"></div></div>
      <div class="score-val">{value:.3f}</div>
    </div>"""

def score_color(v):
    if v < 0.25: return "#4ade80"
    elif v < 0.55: return "#fb923c"
    else: return "#f87171"

# ── Waveform chart ─────────────────────────────────────────────
def waveform_chart(path):
    try:
        audio, sr = load_audio(path)
        duration = len(audio) / sr
        t = np.linspace(0, duration, num=min(len(audio), 4000))
        a = audio[np.linspace(0, len(audio)-1, num=len(t), dtype=int)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t, y=a, mode='lines',
            line=dict(color='#3b82f6', width=1),
            fill='tozeroy', fillcolor='rgba(59,130,246,0.08)',
            name='waveform'
        ))
        fig.update_layout(
            height=120, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            showlegend=False,
        )
        return fig
    except: return None

def emotion_chart(emotion_data):
    labels = list(emotion_data.keys())
    values = list(emotion_data.values())
    colors = {"angry": "#f87171", "sad": "#60a5fa", "neutral": "#94a3b8", "happy": "#4ade80"}
    bar_colors = [colors.get(l, "#6366f1") for l in labels]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in values],
        textposition='outside',
        textfont=dict(color='#94a3b8', size=11, family='JetBrains Mono'),
    ))
    fig.update_layout(
        height=160, margin=dict(l=0, r=40, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 1.1]),
        yaxis=dict(showgrid=False, tickfont=dict(color='#94a3b8', size=12)),
        showlegend=False,
    )
    return fig

# ═══════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="wualt-header">
  <span class="wualt-badge">v3.1</span>
  <span class="wualt-badge">6 Models</span>
  <span class="wualt-badge">EN · TA · HI · +97</span>
  <h1 class="wualt-title" style="margin-top:14px">WUALT Audio Analysis Engine</h1>
  <p class="wualt-subtitle">Distress · Danger · Vocal Health · Emotion · Linguistic — zero-input autonomous pipeline</p>
</div>
""", unsafe_allow_html=True)

# Load models
with st.spinner("Initialising models — first load takes 2–3 minutes…"):
    models = load_all_models()

st.success("All 6 models ready.", icon="✅")

# Upload
st.markdown('<div class="card"><div class="card-title">Audio Input</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
    help="WAV, MP3, M4A, OGG, FLAC or WebM. Mono or stereo — converted internally to 16 kHz mono.",
    label_visibility="collapsed",
)

if uploaded:
    st.audio(uploaded)

st.markdown('</div>', unsafe_allow_html=True)

if not uploaded:
    st.markdown("""
    <div style="text-align:center;padding:60px 0;color:#334155">
      <div style="font-size:3rem;margin-bottom:16px">🎙️</div>
      <div style="font-size:1rem;font-weight:500">Upload an audio file to begin analysis</div>
      <div style="font-size:0.8rem;margin-top:8px;color:#1e3a5f">Supports WAV · MP3 · M4A · OGG · FLAC · WebM</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Run analysis
with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

with st.spinner("Running full 6-model analysis pipeline…"):
    try:
        r = run_full_analysis(tmp_path, models)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        os.unlink(tmp_path)
        st.stop()

os.unlink(tmp_path)

# ── Waveform ──────────────────────────────────────────────────
with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp2:
    uploaded.seek(0)
    tmp2.write(uploaded.read())
    tmp2_path = tmp2.name

fig_wave = waveform_chart(tmp2_path)
os.unlink(tmp2_path)
if fig_wave:
    st.plotly_chart(fig_wave, use_container_width=True, config={"displayModeBar": False})

# ── Overall status banner ──────────────────────────────────────
status = r["overall_status"]
if "Distress" in status: css_cls = "status-distress"
elif "Stress" in status: css_cls = "status-stress"
else: css_cls = "status-normal"

st.markdown(f"""
<div class="card" style="text-align:center;padding:28px">
  <div style="font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;color:#64748b;margin-bottom:12px">Overall Assessment</div>
  <div class="{css_cls}">{status}</div>
  <div style="margin-top:16px;display:flex;justify-content:center;gap:32px">
    <div><div style="font-size:0.7rem;color:#64748b;margin-bottom:2px">OVERALL SCORE</div>
      <div style="font-size:1.6rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:#f1f5f9">{r['overall_score']:.3f}</div></div>
    <div><div style="font-size:0.7rem;color:#64748b;margin-bottom:2px">DISTRESS</div>
      <div style="font-size:1.6rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:#f1f5f9">{r['distress_status']}</div></div>
    <div><div style="font-size:0.7rem;color:#64748b;margin-bottom:2px">EMOTION</div>
      <div style="font-size:1.6rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:#f1f5f9">{r['emotion']['top_emotion'].upper()}</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── dB SPL blocker warning ─────────────────────────────────────
st.markdown("""
<div class="blocker-box">
  ⚠️ <strong>Pre-Deployment Note:</strong> dB SPL values require MIC sensitivity recalibration against your target hardware before deployment.
  Set <code>sensitivity_dbfs = measured_dBFS − 94.0</code> using a 94 dB SPL 1 kHz reference tone.
</div>
""", unsafe_allow_html=True)

# ── Score overview ─────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">Score Overview</div>', unsafe_allow_html=True)
bars_html = (
    score_bar("Distress",    r["distress_score"],              score_color(r["distress_score"])) +
    score_bar("Danger",      r["danger_score"],                score_color(r["danger_score"])) +
    score_bar("Linguistic",  r["ling"]["linguistic_score"],    score_color(r["ling"]["linguistic_score"])) +
    score_bar("Noise",       r["noise_score"],                 score_color(r["noise_score"])) +
    score_bar("Physio",      r["physio_score"],                score_color(r["physio_score"]))
)
st.markdown(bars_html, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Row 1: Acoustic + Vocal ────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card"><div class="card-title">📊 Acoustic Measurements</div>', unsafe_allow_html=True)
    ns_color = {"Safe": "#4ade80", "Moderate": "#fb923c", "Dangerous": "#f87171", "No Signal": "#64748b"}
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-tile">
        <div class="metric-label">dB SPL</div>
        <div class="metric-value">{r['db'] if r['db'] is not None else '—'}<span class="metric-unit">dB</span></div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">TWA Dose</div>
        <div class="metric-value">{r['twa']}<span class="metric-unit">%</span></div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Noise Status</div>
        <div class="metric-value" style="color:{ns_color.get(r['noise_status'],'#e2e8f0')};font-size:1.1rem">{r['noise_status']}</div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Noise Score</div>
        <div class="metric-value">{r['noise_score']:.4f}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    v = r["vocal"]
    vh_color = "#4ade80" if v["vocal_health"] == "Normal" else "#fb923c"
    st.markdown(f"""
    <div class="card"><div class="card-title">🎤 Vocal Health</div>
    <div class="metric-grid">
      <div class="metric-tile">
        <div class="metric-label">Pitch Mean</div>
        <div class="metric-value">{v['pitch_hz']}<span class="metric-unit">Hz</span></div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Pitch SD</div>
        <div class="metric-value">{v['pitch_sd']}<span class="metric-unit">Hz</span></div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Jitter</div>
        <div class="metric-value">{v['jitter']}<span class="metric-unit">%</span></div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Shimmer</div>
        <div class="metric-value">{v['shimmer']}<span class="metric-unit">%</span></div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">HNR</div>
        <div class="metric-value">{v['hnr']}<span class="metric-unit">dB</span></div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Speaking Rate</div>
        <div class="metric-value">{v['speaking_rate']}<span class="metric-unit">syl/s</span></div>
      </div>
    </div>
    <div style="margin-top:12px;padding:8px 12px;background:#161d2e;border-radius:6px;font-size:0.82rem;color:{vh_color}">
      Vocal Health: <strong>{v['vocal_health']}</strong>
    </div>
    </div>
    """, unsafe_allow_html=True)

# ── Row 2: Danger + Emotion ───────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    env_color = "#f87171" if r["danger_score"] > 0.25 else "#4ade80"
    st.markdown(f"""
    <div class="card"><div class="card-title">🚨 Danger Detection — YAMNet + AST</div>
    <div class="metric-grid">
      <div class="metric-tile">
        <div class="metric-label">Danger Score</div>
        <div class="metric-value" style="color:{score_color(r['danger_score'])}">{r['danger_score']:.4f}</div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Env Score</div>
        <div class="metric-value">{r['env_score']:.4f}</div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Physio Score</div>
        <div class="metric-value">{r['physio_score']:.4f}</div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Env Status</div>
        <div class="metric-value" style="color:{env_color};font-size:1rem">{'UNSAFE' if r['danger_score']>0.25 else 'SAFE'}</div>
      </div>
    </div>
    <div style="margin-top:14px">
      <div style="font-size:0.7rem;color:#64748b;letter-spacing:0.07em;text-transform:uppercase;margin-bottom:8px">Top Detections</div>
    """, unsafe_allow_html=True)
    for i, (cls, sc) in enumerate(r["top_detections"]):
        st.markdown(f"""
        <div class="detect-row">
          <div class="detect-rank">#{i+1}</div>
          <div class="detect-label">{cls}</div>
          <div class="detect-score">{sc:.4f}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-top:10px;font-size:0.72rem;color:#64748b">
      YAMNet weight: <span style="font-family:'JetBrains Mono',monospace;color:#94a3b8">{r['yamnet_weight']}</span> &nbsp;|&nbsp;
      AST weight: <span style="font-family:'JetBrains Mono',monospace;color:#94a3b8">{r['ast_weight']}</span>
    </div>
    </div></div>""", unsafe_allow_html=True)

with col4:
    emo = r["emotion"]
    emo_color = {"angry": "#f87171", "sad": "#60a5fa", "neutral": "#94a3b8", "happy": "#4ade80"}
    ec = emo_color.get(emo["top_emotion"], "#e2e8f0")
    st.markdown(f"""
    <div class="card"><div class="card-title">😤 Speech Emotion — wav2vec2 SER</div>
    <div style="margin-bottom:16px">
      <span style="font-size:1.8rem;font-weight:700;color:{ec};font-family:'JetBrains Mono',monospace">{emo['top_emotion'].upper()}</span>
      <span style="font-size:1rem;color:#64748b;margin-left:12px">{emo['confidence']*100:.1f}% confidence</span>
    </div>
    """, unsafe_allow_html=True)
    fig_emo = emotion_chart(emo["all_emotions"])
    st.plotly_chart(fig_emo, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# ── Linguistic ────────────────────────────────────────────────
ling = r["ling"]
st.markdown(f"""
<div class="card"><div class="card-title">🗣️ Linguistic Analysis — Whisper + XLM-RoBERTa + MPNet</div>
<div class="transcript-box">
  <span class="lang-tag">{ling['language'].upper()}</span>
  {ling['text'] if ling['text'] else '<span style="color:#475569;font-style:italic">No speech detected</span>'}
</div>
<div class="metric-grid" style="margin-top:14px">
  <div class="metric-tile">
    <div class="metric-label">Intent Score</div>
    <div class="metric-value" style="color:{score_color(ling['intent_score'])}">{ling['intent_score']:.4f}</div>
  </div>
  <div class="metric-tile">
    <div class="metric-label">Semantic Score</div>
    <div class="metric-value" style="color:{score_color(ling['semantic_score'])}">{ling['semantic_score']:.4f}</div>
  </div>
  <div class="metric-tile">
    <div class="metric-label">Linguistic Score</div>
    <div class="metric-value" style="color:{score_color(ling['linguistic_score'])}">{ling['linguistic_score']:.4f}</div>
  </div>
</div>
</div>
""", unsafe_allow_html=True)

# ── Distress breakdown ────────────────────────────────────────
ds_color = score_color(r["distress_score"])
st.markdown(f"""
<div class="card"><div class="card-title">😰 Distress Analysis — 6-model Fusion</div>
<div style="display:flex;align-items:center;gap:24px;margin-bottom:20px">
  <div>
    <div style="font-size:0.72rem;color:#64748b;margin-bottom:4px">DISTRESS SCORE</div>
    <div style="font-size:2.2rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:{ds_color}">{r['distress_score']:.4f}</div>
  </div>
  <div>
    <div style="font-size:0.72rem;color:#64748b;margin-bottom:4px">STATUS</div>
    <div style="font-size:1.2rem;font-weight:600;color:{ds_color}">{r['distress_status']}</div>
  </div>
</div>
</div>
""", unsafe_allow_html=True)

# ── Feature vector (expandable) ───────────────────────────────
with st.expander("📦 Raw Feature Vector (JSON)"):
    vector = {
        "distress_score":     r["distress_score"],
        "noise_score":        r["noise_score"],
        "danger_score":       r["danger_score"],
        "physio_score":       r["physio_score"],
        "emotion_score":      round(min(1.0, sum(
                                DISTRESS_WEIGHTS.get(l, 0.0) * p
                                for l, p in r["emotion"]["all_emotions"].items())), 4),
        "vocal_health_score": round(min(1.0, (
                                r["vocal"]["jitter"] / (CLINICAL["jitter_normal_max"] * 100) +
                                r["vocal"]["shimmer"] / (CLINICAL["shimmer_normal_max"] * 100) +
                                max(0, CLINICAL["hnr_normal_min"] - r["vocal"]["hnr"]) / CLINICAL["hnr_normal_min"]
                              ) / 3), 4),
        "linguistic_score":   r["ling"]["linguistic_score"],
        "audio_state":        ("Distress" if "Distress" in r["overall_status"] else
                               "Stress"   if "Stress"   in r["overall_status"] else "Normal"),
        "timestamp":          datetime.now().isoformat(),
    }
    st.json(vector)

# ── Full report (expandable) ──────────────────────────────────
with st.expander("📋 Full Analysis Report (all 32 fields)"):
    full = {
        "01_db_spl":          r["db"],
        "02_twa_dose_pct":    r["twa"],
        "03_noise_status":    r["noise_status"],
        "04_noise_score":     r["noise_score"],
        "05_danger_score":    r["danger_score"],
        "06_env_score":       r["env_score"],
        "07_physio_score":    r["physio_score"],
        "08_env_status":      "UNSAFE" if r["danger_score"] > 0.25 else "SAFE",
        "09_top_detections":  r["top_detections"],
        "10_yamnet_weight":   r["yamnet_weight"],
        "11_ast_weight":      r["ast_weight"],
        "12_pitch_hz":        r["vocal"]["pitch_hz"],
        "13_pitch_sd":        r["vocal"]["pitch_sd"],
        "14_pitch_min":       r["vocal"]["pitch_min"],
        "15_pitch_max":       r["vocal"]["pitch_max"],
        "16_jitter_pct":      r["vocal"]["jitter"],
        "17_shimmer_pct":     r["vocal"]["shimmer"],
        "18_hnr_db":          r["vocal"]["hnr"],
        "19_speaking_rate":   r["vocal"]["speaking_rate"],
        "20_intensity_mean":  r["vocal"]["intensity"],
        "21_vocal_health":    r["vocal"]["vocal_health"],
        "22_transcript":      r["ling"]["text"],
        "23_language":        r["ling"]["language"].upper(),
        "24_intent_score":    r["ling"]["intent_score"],
        "25_semantic_score":  r["ling"]["semantic_score"],
        "26_linguistic_score":r["ling"]["linguistic_score"],
        "27_top_emotion":     r["emotion"]["top_emotion"],
        "28_emotion_confidence": r["emotion"]["confidence"],
        "29_distress_score":  r["distress_score"],
        "30_distress_status": r["distress_status"],
        "31_overall_score":   r["overall_score"],
        "32_overall_status":  r["overall_status"],
    }
    st.json(full)

# ── Footer ─────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:32px 0 16px;border-top:1px solid #1e3a5f;margin-top:24px">
  <span style="font-size:0.72rem;color:#334155">WUALT Audio Analysis Engine · v3.1 · 6 Models · {datetime.now().strftime('%Y-%m-%d')}</span>
</div>
""", unsafe_allow_html=True)
