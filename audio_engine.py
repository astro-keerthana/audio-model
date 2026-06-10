# audio_engine.py
# ============================================================
# All model loading + analysis logic ported from v3.0.ipynb
# ============================================================

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch
import librosa
import soundfile as sf
import whisper
import logging
from scipy.signal import find_peaks
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoFeatureExtractor,
    ASTForAudioClassification,
    AutoModelForAudioClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Wav2Vec2FeatureExtractor,
    pipeline,
)
import requests, csv, io, os, warnings, json
from datetime import datetime

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s] %(funcName)s: %(message)s"
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────
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
    "Screaming", "Glass breaking", "Gunshot, gunfire",
    "Explosion", "Crash", "Alarm", "Siren", "Fire alarm",
    "Smoke detector", "Slam", "Thud", "Splash, splatter",
    "Growling", "Thunder", "Baby cry, infant cry", "Shout",
    "Yell", "Bellow", "Children shouting",
    "Police car (siren)", "Ambulance (siren)",
    "Fire engine, fire truck (siren)", "Car alarm",
    "Air horn, truck horn", "Reversing beeps",
    "Chainsaw", "Drill", "Burst, pop",
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

SUPERB_EMOTION_MAP = {
    "ang": "angry", "hap": "happy", "neu": "neutral", "sad": "sad"
}

DISTRESS_WEIGHTS = {
    "angry": 0.85, "sad": 0.75, "neutral": 0.00, "happy": 0.00,
}
DISTRESS_EMOTIONS = set(k for k, v in DISTRESS_WEIGHTS.items() if v > 0)

PITCH_CFG = {
    "min_f0": 75.0, "max_f0": 600.0,
    "window_size": 0.05, "hop_size": 0.01, "voiced_thr": 0.25,
}

XNLI_LABELS = [
    "danger", "distress", "threat", "emergency", "violence",
    "normal conversation", "calm", "everyday activity",
    "neutral statement", "safe situation",
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

# ── Global model cache ─────────────────────────────────────────
_MODELS = {}
_CACHE  = {}


def load_models():
    """Load all 6 models once and cache them globally."""
    global _MODELS

    if _MODELS:
        return _MODELS  # already loaded

    log.warning("Loading Whisper medium...")
    _MODELS["whisper"] = whisper.load_model("medium")

    log.warning("Loading XLM-RoBERTa XNLI...")
    _MODELS["xnli"] = pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli",
        tokenizer="joeddav/xlm-roberta-large-xnli",
        device=0 if torch.cuda.is_available() else -1,
    )

    log.warning("Loading Multilingual MPNet...")
    _MODELS["mpnet"] = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    _MODELS["danger_embeddings"] = _MODELS["mpnet"].encode(
        DANGER_PHRASES, convert_to_tensor=True, show_progress_bar=False
    )

    log.warning("Loading SER model...")
    _MODELS["ser_extractor"] = Wav2Vec2FeatureExtractor.from_pretrained(
        "superb/wav2vec2-base-superb-er"
    )
    _MODELS["ser_model"] = AutoModelForAudioClassification.from_pretrained(
        "superb/wav2vec2-base-superb-er"
    )
    _MODELS["ser_model"].eval()

    log.warning("Loading AST...")
    _MODELS["ast_extractor"] = AutoFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    _MODELS["ast_model"] = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    _MODELS["ast_model"].eval()

    log.warning("Loading YAMNet...")
    _MODELS["yamnet"] = hub.load("https://tfhub.dev/google/yamnet/1")

    log.warning("Loading YAMNet class map...")
    r = requests.get(
        "https://raw.githubusercontent.com/tensorflow/models/master/"
        "research/audioset/yamnet/yamnet_class_map.csv"
    )
    _MODELS["yamnet_classes"] = [row[2] for row in csv.reader(io.StringIO(r.text))][1:]

    return _MODELS


# ── Audio loaders ──────────────────────────────────────────────
def load_audio(audio_path, sr=MIC["sample_rate"]):
    audio, sr = librosa.load(audio_path, sr=sr, mono=True)
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    return audio, sr

def load_audio_raw(audio_path, sr=MIC["sample_rate"]):
    audio, sr = librosa.load(audio_path, sr=sr, mono=True)
    return audio, sr

def _clear_cache(audio_path=None):
    global _CACHE
    if audio_path:
        _CACHE.pop(audio_path, None)
    else:
        _CACHE = {}


# ── Noise / dB ─────────────────────────────────────────────────
def get_db_level(audio_path):
    try:
        audio, sr = load_audio(audio_path)
        audio = audio - np.mean(audio)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-10:
            return None
        dbfs = 20 * np.log10(rms)
        db_spl = dbfs - MIC["sensitivity_dbfs"] + 94.0
        return round(float(np.clip(db_spl, MIC["noise_floor_db"], MIC["aop_db"])), 1)
    except Exception as e:
        log.warning(f"get_db_level failed: {e}")
        return None

def get_twa_dose(audio_path):
    try:
        db_spl = get_db_level(audio_path)
        if db_spl is None:
            return 0.0
        if db_spl >= 90:
            permitted_hours = 8 / (2 ** ((db_spl - 90) / 5))
            return round(min(999.9, (1 / permitted_hours) * 100), 1)
        return round((db_spl / 90.0) * 100 * 0.5, 1)
    except Exception as e:
        log.warning(f"get_twa_dose failed: {e}")
        return 0.0

def get_noise_status(audio_path):
    try:
        db = get_db_level(audio_path)
        if db is None: return "No Signal"
        if db < 70:    return "Safe"
        elif db < 85:  return "Moderate"
        else:          return "Dangerous"
    except Exception as e:
        log.warning(f"get_noise_status failed: {e}")
        return "Unknown"

def get_noise_score(audio_path):
    try:
        db = get_db_level(audio_path)
        if db is None: return 0.0
        dynamic_range = MIC["aop_db"] - MIC["noise_floor_db"]
        score = (db - MIC["noise_floor_db"]) / dynamic_range
        return round(min(1.0, max(0.0, score)), 4)
    except Exception as e:
        log.warning(f"get_noise_score failed: {e}")
        return 0.0


# ── Danger Detection ───────────────────────────────────────────
def _run_fusion(audio_path):
    global _CACHE
    if audio_path in _CACHE and "fusion" in _CACHE[audio_path]:
        return _CACHE[audio_path]["fusion"]

    m = load_models()
    try:
        audio, sr = load_audio(audio_path)
        audio_tf = tf.constant(audio.astype(np.float32))
        scores, _, _ = m["yamnet"](audio_tf)
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()

        yamnet_hits, yamnet_sum = {}, 0.0
        for cls in ALL_DANGER:
            for i, yc in enumerate(m["yamnet_classes"]):
                if cls.lower() in yc.lower():
                    v = float(mean_scores[i])
                    yamnet_hits[cls] = v
                    yamnet_sum += v
                    break

        inputs = m["ast_extractor"](audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = m["ast_model"](**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        id2label = m["ast_model"].config.id2label

        ast_hits, ast_sum = {}, 0.0
        for cls in ALL_DANGER:
            for idx, label in id2label.items():
                if cls.lower() in label.lower():
                    v = float(probs[idx])
                    ast_hits[cls] = v
                    ast_sum += v
                    break

        total = yamnet_sum + ast_sum + 1e-9
        wy, wa = yamnet_sum / total, ast_sum / total

        fused = {
            cls: round(yamnet_hits.get(cls, 0) * wy + ast_hits.get(cls, 0) * wa, 4)
            for cls in ALL_DANGER
        }
        result = (fused, round(wy, 3), round(wa, 3))
    except Exception as e:
        log.warning(f"_run_fusion failed: {e}")
        result = ({cls: 0.0 for cls in ALL_DANGER}, 0.5, 0.5)

    _CACHE.setdefault(audio_path, {})["fusion"] = result
    return result

def get_danger_score(audio_path):
    try:
        fused, _, _ = _run_fusion(audio_path)
        return round(max(fused.values()), 4)
    except:
        return 0.0

def get_env_score(audio_path):
    try:
        fused, _, _ = _run_fusion(audio_path)
        return round(max((fused.get(c, 0) for c in DANGER_ENV), default=0.0), 4)
    except:
        return 0.0

def get_physio_score(audio_path):
    try:
        fused, _, _ = _run_fusion(audio_path)
        return round(max((fused.get(c, 0) for c in DANGER_PHYSIO), default=0.0), 4)
    except:
        return 0.0

def get_env_status(audio_path, threshold=0.25):
    return "UNSAFE" if get_danger_score(audio_path) > threshold else "SAFE"

def get_top_detections(audio_path, top_n=5):
    try:
        fused, _, _ = _run_fusion(audio_path)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_n]
    except:
        return []


# ── Vocal Health ───────────────────────────────────────────────
def _get_voiced_f0(audio, sr):
    frame_len = int(PITCH_CFG["window_size"] * sr)
    hop_len   = int(PITCH_CFG["hop_size"] * sr)
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio, fmin=PITCH_CFG["min_f0"], fmax=PITCH_CFG["max_f0"],
            sr=sr, frame_length=frame_len, hop_length=hop_len, fill_na=np.nan
        )
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0) & (voiced_prob > PITCH_CFG["voiced_thr"])]
        if len(voiced_f0) < 3:
            f0_yin = librosa.yin(
                audio, fmin=PITCH_CFG["min_f0"], fmax=PITCH_CFG["max_f0"],
                sr=sr, frame_length=frame_len, hop_length=hop_len,
            )
            voiced_f0 = f0_yin[(f0_yin >= PITCH_CFG["min_f0"]) & (f0_yin <= PITCH_CFG["max_f0"])]
        return voiced_f0
    except Exception as e:
        log.warning(f"_get_voiced_f0 failed: {e}")
        return np.array([])

def _build_f0_per_sample(audio, sr):
    frame_len = int(PITCH_CFG["window_size"] * sr)
    hop_len   = int(PITCH_CFG["hop_size"] * sr)
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio, fmin=PITCH_CFG["min_f0"], fmax=PITCH_CFG["max_f0"],
            sr=sr, frame_length=frame_len, hop_length=hop_len, fill_na=np.nan
        )
    except Exception as e:
        log.warning(f"_build_f0_per_sample pyin failed: {e}")
        return np.full(len(audio), np.nan)

    n_frames      = len(f0)
    n_samples     = len(audio)
    frame_centres = np.arange(n_frames) * hop_len + hop_len // 2
    voiced_mask   = voiced_flag & ~np.isnan(f0) & (voiced_prob > PITCH_CFG["voiced_thr"])
    voiced_idx    = np.where(voiced_mask)[0]

    if len(voiced_idx) < 2:
        return np.full(n_samples, np.nan)

    f0_per_sample = np.full(n_samples, np.nan)
    for i in range(len(voiced_idx) - 1):
        idx_a = voiced_idx[i]
        idx_b = voiced_idx[i + 1]
        if (idx_b - idx_a) > 3:
            continue
        s_a = int(frame_centres[idx_a])
        s_b = int(frame_centres[idx_b])
        f_a, f_b = f0[idx_a], f0[idx_b]
        if s_b <= s_a:
            continue
        interp_samples = np.arange(s_a, min(s_b, n_samples))
        t = (interp_samples - s_a) / (s_b - s_a)
        f0_per_sample[interp_samples] = f_a + t * (f_b - f_a)
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
                next_voiced = np.argwhere(~np.isnan(remaining))
                if len(next_voiced) == 0:
                    break
                cursor += int(next_voiced[0][0])
                continue
            period_samples = int(round(sr / f0_val))
            if period_samples < 2:
                cursor += 1
                continue
            end = cursor + period_samples
            if end > n_samples:
                break
            cycle = audio[cursor:end]
            peak_amp = float(np.max(np.abs(cycle)))
            if peak_amp < 0.001:
                cursor += period_samples
                continue
            periods.append(1.0 / f0_val)
            amplitudes.append(peak_amp)
            cursor += period_samples
        return np.array(periods), np.array(amplitudes)
    except Exception as e:
        log.warning(f"_get_glottal_cycles failed: {e}")
        return np.array([]), np.array([])

def get_pitch_hz(audio_path):
    try:
        audio, sr = load_audio_raw(audio_path)
        voiced = _get_voiced_f0(audio, sr)
        return round(float(np.mean(voiced)), 1) if len(voiced) > 0 else 0.0
    except:
        return 0.0

def get_pitch_sd(audio_path):
    try:
        audio, sr = load_audio_raw(audio_path)
        voiced = _get_voiced_f0(audio, sr)
        return round(float(np.std(voiced)), 1) if len(voiced) > 1 else 0.0
    except:
        return 0.0

def get_pitch_min(audio_path):
    try:
        audio, sr = load_audio_raw(audio_path)
        voiced = _get_voiced_f0(audio, sr)
        return round(float(np.min(voiced)), 1) if len(voiced) > 0 else 0.0
    except:
        return 0.0

def get_pitch_max(audio_path):
    try:
        audio, sr = load_audio_raw(audio_path)
        voiced = _get_voiced_f0(audio, sr)
        return round(float(np.max(voiced)), 1) if len(voiced) > 0 else 0.0
    except:
        return 0.0

def get_jitter(audio_path):
    try:
        audio, sr = load_audio_raw(audio_path)
        periods, _ = _get_glottal_cycles(audio, sr)
        if len(periods) < 3: return 0.0
        periods = periods[(periods >= 0.0001) & (periods <= 0.02)]
        if len(periods) < 3: return 0.0
        ratios = periods[1:] / (periods[:-1] + 1e-10)
        valid = (ratios < 1.3) & (ratios > (1.0 / 1.3))
        p_clean = periods[:-1][valid]
        if len(p_clean) < 2: return 0.0
        diffs = np.abs(np.diff(p_clean))
        mean_period = np.mean(p_clean)
        return round(float((np.mean(diffs) / (mean_period + 1e-10)) * 100.0), 3)
    except:
        return 0.0

def get_shimmer(audio_path):
    try:
        audio, sr = load_audio_raw(audio_path)
        periods, amplitudes = _get_glottal_cycles(audio, sr)
        if len(amplitudes) < 3: return 0.0
        valid_mask = (periods >= 0.0001) & (periods <= 0.02)
        amplitudes = amplitudes[valid_mask]
        if len(amplitudes) < 3: return 0.0
        amp_ratios = amplitudes[1:] / (amplitudes[:-1] + 1e-10)
        valid = (amp_ratios < 1.6) & (amp_ratios > (1.0 / 1.6))
        a_clean = amplitudes[:-1][valid]
        if len(a_clean) < 2: return 0.0
        diffs = np.abs(np.diff(a_clean))
        mean_amp = np.mean(a_clean)
        return round(float((np.mean(diffs) / (mean_amp + 1e-10)) * 100.0), 3)
    except:
        return 0.0

def get_hnr(audio_path):
    try:
        audio, sr = load_audio_raw(audio_path)
        frame_len = int(0.04 * sr)
        hop_len   = int(0.01 * sr)
        min_lag   = int(sr / PITCH_CFG["max_f0"])
        max_lag   = int(sr / PITCH_CFG["min_f0"])
        window    = np.hanning(frame_len)
        hnr_vals  = []
        for start in range(0, len(audio) - frame_len, hop_len):
            frame = audio[start: start + frame_len].copy()
            if np.max(np.abs(frame)) < 0.001: continue
            frame *= window
            frame -= np.mean(frame)
            acf = np.correlate(frame, frame, mode='full')
            acf = acf[len(acf) // 2:]
            if acf[0] < 1e-10: continue
            acf_norm = acf / acf[0]
            if max_lag >= len(acf_norm): continue
            segment = acf_norm[min_lag: max_lag]
            if len(segment) == 0: continue
            r = np.clip(float(np.max(segment)), 1e-9, 1.0 - 1e-9)
            hnr_vals.append(10.0 * np.log10(r / (1.0 - r)))
        return round(float(np.mean(hnr_vals)), 2) if hnr_vals else 0.0
    except:
        return 0.0

def get_intensity_mean(audio_path):
    try:
        audio, sr = load_audio_raw(audio_path)
        frame_len = int(0.01 * sr)
        rms_vals  = []
        for start in range(0, len(audio) - frame_len, frame_len):
            frame = audio[start: start + frame_len]
            rms = np.sqrt(np.mean(frame ** 2))
            if rms > 1e-10: rms_vals.append(rms)
        if not rms_vals: return 0.0
        mean_rms = np.sqrt(np.mean(np.array(rms_vals) ** 2))
        return round(float(20.0 * np.log10(mean_rms + 1e-10) + 94.0), 2)
    except:
        return 0.0

def get_speaking_rate(audio_path):
    try:
        audio, sr = load_audio(audio_path)
        hop = 512
        energy = np.array([
            np.sum(np.abs(audio[i: i + hop]) ** 2)
            for i in range(0, len(audio) - hop, hop)
        ])
        peaks, _ = find_peaks(energy, height=np.mean(energy) * 0.5, distance=4)
        duration = len(audio) / sr
        return round(len(peaks) / duration if duration > 0 else 0.0, 2)
    except:
        return 0.0

def get_vocal_health(audio_path):
    j, s, h = get_jitter(audio_path), get_shimmer(audio_path), get_hnr(audio_path)
    flags = []
    if j > CLINICAL["jitter_normal_max"] * 100:  flags.append("High Jitter")
    if s > CLINICAL["shimmer_normal_max"] * 100: flags.append("High Shimmer")
    if h < CLINICAL["hnr_normal_min"]:            flags.append("Low HNR")
    return "Normal" if not flags else " | ".join(flags)


# ── Linguistic ─────────────────────────────────────────────────
def get_transcript(audio_path):
    global _CACHE
    if audio_path in _CACHE and "transcript" in _CACHE[audio_path]:
        return _CACHE[audio_path]["transcript"]
    m = load_models()
    try:
        result = m["whisper"].transcribe(
            audio_path, task="transcribe",
            fp16=torch.cuda.is_available(), verbose=False,
        )
        data = {"text": result.get("text", "").strip(), "language": result.get("language", "unknown")}
    except Exception as e:
        log.warning(f"get_transcript failed: {e}")
        data = {"text": "", "language": "unknown"}
    _CACHE.setdefault(audio_path, {})["transcript"] = data
    return data

def get_intent_score(text):
    if not text or len(text.strip()) < 3:
        return 0.0
    m = load_models()
    try:
        result = m["xnli"](text, candidate_labels=XNLI_LABELS, multi_label=False)
        label_weights = {
            "danger": 1.00, "distress": 0.90, "threat": 0.85,
            "emergency": 0.80, "violence": 0.75,
            "normal conversation": 0.00, "calm": 0.00,
        }
        top_danger_score = max(
            (label_weights.get(l.lower(), 0.0) * s
             for l, s in zip(result["labels"], result["scores"])
             if label_weights.get(l.lower(), 0.0) > 0),
            default=0.0
        )
        return round(float(top_danger_score), 4)
    except Exception as e:
        log.warning(f"get_intent_score failed: {e}")
        return 0.0

def get_semantic_score(text):
    if not text or len(text.strip()) < 3:
        return 0.0
    m = load_models()
    try:
        query_emb = m["mpnet"].encode(text, convert_to_tensor=True, show_progress_bar=False)
        cosine_scores = util.cos_sim(query_emb, m["danger_embeddings"])[0]
        return round(min(1.0, max(0.0, float(torch.max(cosine_scores)))), 4)
    except Exception as e:
        log.warning(f"get_semantic_score failed: {e}")
        return 0.0

def get_linguistic_score(audio_path):
    global _CACHE
    if audio_path in _CACHE and "linguistic" in _CACHE[audio_path]:
        return _CACHE[audio_path]["linguistic"]
    transcript     = get_transcript(audio_path)
    text           = transcript["text"]
    language       = transcript["language"]
    intent_score   = get_intent_score(text)
    semantic_score = get_semantic_score(text)
    linguistic_score = round(min(1.0, (intent_score * 0.60 + semantic_score * 0.40)), 4)
    data = {
        "transcript": text, "language": language,
        "intent_score": intent_score, "semantic_score": semantic_score,
        "linguistic_score": linguistic_score,
    }
    _CACHE.setdefault(audio_path, {})["linguistic"] = data
    return data


# ── Emotion ────────────────────────────────────────────────────
def get_emotion(audio_path):
    global _CACHE
    if audio_path in _CACHE and "emotion" in _CACHE[audio_path]:
        return _CACHE[audio_path]["emotion"]
    m = load_models()
    try:
        audio, sr = load_audio(audio_path)
        inputs = m["ser_extractor"](audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = m["ser_model"](**inputs).logits
        probs    = torch.softmax(logits, dim=-1)[0]
        id2label = m["ser_model"].config.id2label
        top_idx  = torch.argmax(probs).item()

        def normalize(label):
            return SUPERB_EMOTION_MAP.get(label.lower().strip(), label.lower().strip())

        all_emotions = {
            normalize(id2label.get(i, str(i))): round(float(probs[i]), 4)
            for i in range(len(probs))
        }
        all_emotions = dict(sorted(all_emotions.items(), key=lambda x: x[1], reverse=True))
        data = {
            "top_emotion": normalize(id2label.get(top_idx, str(top_idx))),
            "confidence": round(float(probs[top_idx]), 4),
            "all_emotions": all_emotions,
        }
    except Exception as e:
        log.warning(f"get_emotion failed: {e}")
        data = {"top_emotion": "unknown", "confidence": 0.0, "all_emotions": {}}
    _CACHE.setdefault(audio_path, {})["emotion"] = data
    return data


# ── Distress ───────────────────────────────────────────────────
def get_distress_score(audio_path):
    try:
        emotion_data  = get_emotion(audio_path)
        emotion_score = min(1.0, sum(
            DISTRESS_WEIGHTS.get(l.lower(), 0.0) * p
            for l, p in emotion_data["all_emotions"].items()
        ))
        j = get_jitter(audio_path)  / (CLINICAL["jitter_normal_max"]  * 100)
        s = get_shimmer(audio_path) / (CLINICAL["shimmer_normal_max"] * 100)
        h = max(0, CLINICAL["hnr_normal_min"] - get_hnr(audio_path)) / CLINICAL["hnr_normal_min"]
        vocal_score  = min(1.0, (j + s + h) / 3)
        physio_score = get_physio_score(audio_path)
        rate = get_speaking_rate(audio_path)
        if rate > 6.0:
            rate_score = min(1.0, (rate - 6.0) / 4.0)
        elif 0 < rate < 1.0:
            rate_score = min(1.0, (1.0 - rate))
        else:
            rate_score = 0.0
        ling_data        = get_linguistic_score(audio_path)
        linguistic_score = ling_data["linguistic_score"]
        return round(min(1.0, (
            emotion_score    * 0.35 +
            vocal_score      * 0.20 +
            physio_score     * 0.15 +
            rate_score       * 0.10 +
            linguistic_score * 0.20
        )), 4)
    except Exception as e:
        log.warning(f"get_distress_score failed: {e}")
        return 0.0

def get_distress_status(audio_path):
    score = get_distress_score(audio_path)
    if score < 0.25:   return "Calm"
    elif score < 0.50: return "Mild Distress"
    elif score < 0.75: return "Moderate Distress"
    else:              return "Severe Distress"


# ── Overall Risk ───────────────────────────────────────────────
def _physio_override_triggered(audio_path):
    try:
        fused, _, _  = _run_fusion(audio_path)
        physio_score = get_physio_score(audio_path)
        if physio_score <= PHYSIO_OVERRIDE_THRESHOLD:
            return False
        physio_scores = {c: fused.get(c, 0.0) for c in DANGER_PHYSIO}
        top_physio = max(physio_scores, key=physio_scores.get)
        if top_physio in AMBIENT_PHYSIO:
            return False
        return True
    except:
        return False

def get_overall_risk_score(audio_path):
    try:
        w = FUSION_WEIGHTS["overall"]
        ling_data = get_linguistic_score(audio_path)
        return round(min(1.0, (
            get_distress_score(audio_path) * w["distress"] +
            get_noise_score(audio_path)    * w["noise"]    +
            get_danger_score(audio_path)   * w["danger"]   +
            get_physio_score(audio_path)   * w["physio"]   +
            ling_data["linguistic_score"]  * w["linguistic"]
        )), 4)
    except:
        return 0.0

def get_overall_risk_status(audio_path):
    try:
        score        = get_overall_risk_score(audio_path)
        danger_score = get_danger_score(audio_path)
        ling_data    = get_linguistic_score(audio_path)
        ling_score   = ling_data["linguistic_score"]
        emotion_data = get_emotion(audio_path)
        top_emotion  = emotion_data["top_emotion"]
        if danger_score > 0.25:                      return "🚨 Distress"
        if ling_score > 0.70:                        return "🚨 Distress"
        if _physio_override_triggered(audio_path):   return "🚨 Distress"
        if score >= 0.55:                            return "🚨 Distress"
        elif score >= 0.25 or top_emotion in ("angry", "sad") or ling_score > 0.40:
            return "⚠️ Stress"
        else:
            return "✅ Normal"
    except:
        return "✅ Normal"


# ── Master analysis ────────────────────────────────────────────
def analyze_audio(audio_path, top_n=5):
    _clear_cache(audio_path)
    results = {}

    # Acoustic
    db     = get_db_level(audio_path)
    twa    = get_twa_dose(audio_path)
    noise  = get_noise_status(audio_path)
    nscore = get_noise_score(audio_path)
    results.update({"db_spl": db, "twa_dose_pct": twa, "noise_status": noise, "noise_score": nscore})

    # Danger
    fused, wy, wa = _run_fusion(audio_path)
    dscore  = round(max(fused.values()), 4)
    escore  = round(max((fused.get(c, 0) for c in DANGER_ENV),    default=0.0), 4)
    pscore  = round(max((fused.get(c, 0) for c in DANGER_PHYSIO), default=0.0), 4)
    estatus = "UNSAFE" if dscore > 0.25 else "SAFE"
    top_det = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_n]
    results.update({
        "danger_score": dscore, "env_score": escore, "physio_score": pscore,
        "env_status": estatus, "yamnet_weight": wy, "ast_weight": wa, "top_detections": top_det
    })

    # Vocal
    results.update({
        "pitch_hz": get_pitch_hz(audio_path),
        "pitch_sd": get_pitch_sd(audio_path),
        "pitch_min": get_pitch_min(audio_path),
        "pitch_max": get_pitch_max(audio_path),
        "jitter_pct": get_jitter(audio_path),
        "shimmer_pct": get_shimmer(audio_path),
        "hnr_db": get_hnr(audio_path),
        "speaking_rate": get_speaking_rate(audio_path),
        "intensity_mean": get_intensity_mean(audio_path),
        "vocal_health": get_vocal_health(audio_path),
    })

    # Linguistic
    ling = get_linguistic_score(audio_path)
    results.update({
        "transcript": ling["transcript"], "language": ling["language"],
        "intent_score": ling["intent_score"], "semantic_score": ling["semantic_score"],
        "linguistic_score": ling["linguistic_score"],
    })

    # Emotion
    emotion_data = get_emotion(audio_path)
    results.update({
        "emotion": emotion_data["top_emotion"],
        "emotion_confidence": emotion_data["confidence"],
        "emotion_breakdown": emotion_data["all_emotions"],
    })

    # Distress
    results.update({
        "distress_score": get_distress_score(audio_path),
        "distress_status": get_distress_status(audio_path),
    })

    # Overall
    results.update({
        "overall_risk_score": get_overall_risk_score(audio_path),
        "overall_risk_status": get_overall_risk_status(audio_path),
    })

    return results
