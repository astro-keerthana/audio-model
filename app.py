# app.py
# ============================================================
# Streamlit Audio Analysis App — v3.1
# Mirrors all 32 metrics from v3.0.ipynb
# ============================================================

import streamlit as st
import tempfile
import os
import time
from datetime import datetime

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Audio Analysis — v3.1",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 6px 0;
        border-left: 4px solid #7c3aed;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #a0a0b0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .status-distress {
        background: #3b0a0a;
        border-left: 4px solid #ef4444;
        border-radius: 10px;
        padding: 16px 20px;
    }
    .status-stress {
        background: #2d1f00;
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 16px 20px;
    }
    .status-normal {
        background: #0a2d1a;
        border-left: 4px solid #22c55e;
        border-radius: 10px;
        padding: 16px 20px;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #c4b5fd;
        border-bottom: 1px solid #3f3f5f;
        padding-bottom: 6px;
        margin: 18px 0 10px 0;
    }
    .bar-container {
        background: #2d2d3d;
        border-radius: 4px;
        height: 8px;
        margin-top: 4px;
    }
    .bar-fill {
        height: 8px;
        border-radius: 4px;
    }
    .transcript-box {
        background: #1a1a2e;
        border: 1px solid #3f3f5f;
        border-radius: 8px;
        padding: 14px 18px;
        font-style: italic;
        color: #c4b5fd;
        font-size: 1.05rem;
    }
    .detection-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        border-bottom: 1px solid #2d2d3d;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎙️ Audio Analysis")
    st.markdown("**v3.1** · 6-Model Fusion")
    st.markdown("---")
    st.markdown("""
    **Models loaded:**
    - 🔊 Whisper Medium
    - 🧠 XLM-RoBERTa XNLI
    - 📐 Multilingual MPNet
    - 🎭 wav2vec2 SER
    - 🔍 AST AudioSet
    - 🌐 YAMNet
    """)
    st.markdown("---")
    st.markdown("""
    **Fusion Schemas:**
    - `overall` → alert layer
    - `fusion`  → state classifier
    - `vector`  → ML feature store
    """)
    st.markdown("---")
    st.caption(f"Session started: {datetime.now().strftime('%H:%M:%S')}")


# ── Helper renderers ──────────────────────────────────────────
def status_color(status: str) -> str:
    s = status.lower()
    if "distress" in s: return "#ef4444"
    if "stress"   in s: return "#f59e0b"
    if "normal"   in s: return "#22c55e"
    if "dangerous" in s: return "#ef4444"
    if "moderate"  in s: return "#f59e0b"
    if "safe"      in s: return "#22c55e"
    return "#a0a0b0"

def score_bar(score: float, color: str = "#7c3aed"):
    pct = int(score * 100)
    st.markdown(f"""
    <div class="bar-container">
        <div class="bar-fill" style="width:{pct}%; background:{color};"></div>
    </div>
    <div style="font-size:0.7rem; color:#a0a0b0; text-align:right;">{score:.4f}</div>
    """, unsafe_allow_html=True)

def metric_card(label: str, value, unit: str = ""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}{' ' + unit if unit else ''}</div>
    </div>
    """, unsafe_allow_html=True)

def render_status_banner(status: str):
    s = status.lower()
    css_class = "status-distress" if "distress" in s else ("status-stress" if "stress" in s else "status-normal")
    icon = "🚨" if "distress" in s else ("⚠️" if "stress" in s else "✅")
    st.markdown(f"""
    <div class="{css_class}">
        <span style="font-size:2rem;">{icon}</span>
        <span style="font-size:1.5rem; font-weight:800; margin-left:12px;">{status}</span>
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────
st.title("🎙️ Audio Analysis Dashboard")
st.markdown("Upload a `.wav` / `.mp3` / `.flac` file to run the full 6-model analysis pipeline.")

uploaded_file = st.file_uploader(
    "Drop your audio file here",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    help="Supported: WAV, MP3, FLAC, OGG, M4A"
)

if uploaded_file:
    # ── Audio preview ─────────────────────────────────────────
    st.markdown("### 🔈 Audio Preview")
    st.audio(uploaded_file, format=uploaded_file.type)

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("File Name", uploaded_file.name)
    col_info2.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
    col_info3.metric("Format", uploaded_file.type.split("/")[-1].upper())

    st.markdown("---")

    # ── Run analysis ──────────────────────────────────────────
    if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):

        # Save to temp file
        suffix = os.path.splitext(uploaded_file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            # Load models (cached after first run)
            with st.spinner("⏳ Loading models (first run only — ~60s)..."):
                from audio_engine import load_models, analyze_audio, CLINICAL
                load_models()

            # Run analysis with progress
            progress_bar = st.progress(0, text="🔍 Starting analysis...")
            steps = [
                (10,  "📊 Computing acoustic measurements..."),
                (25,  "🚨 Running YAMNet + AST danger detection..."),
                (45,  "🎤 Extracting vocal health features..."),
                (60,  "🗣️ Transcribing with Whisper..."),
                (75,  "🧠 Running XLM-RoBERTa + MPNet linguistic analysis..."),
                (88,  "🎭 Classifying speech emotion..."),
                (95,  "🔀 Computing distress + overall risk fusion..."),
                (100, "✅ Analysis complete!"),
            ]

            start_time = time.time()

            # Kick off actual analysis
            with st.spinner("🧠 Running 6-model analysis pipeline..."):
                for pct, msg in steps[:4]:
                    progress_bar.progress(pct, text=msg)
                    time.sleep(0.1)

                results = analyze_audio(tmp_path)

                for pct, msg in steps[4:]:
                    progress_bar.progress(pct, text=msg)
                    time.sleep(0.05)

            elapsed = round(time.time() - start_time, 1)
            st.success(f"✅ Analysis completed in **{elapsed}s**")

            # ══════════════════════════════════════════════════
            # SECTION 0 — STATUS BANNER
            # ══════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("## 📋 Overall Assessment")
            render_status_banner(results["overall_risk_status"])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall Risk Score",  f"{results['overall_risk_score']:.4f}")
            c2.metric("Distress Status",     results["distress_status"])
            c3.metric("Top Emotion",         results["emotion"].upper())
            c4.metric("Vocal Health",        results["vocal_health"])

            # ══════════════════════════════════════════════════
            # SECTION 1 — ACOUSTIC MEASUREMENTS
            # ══════════════════════════════════════════════════
            st.markdown("---")
            st.markdown('<div class="section-header">📊 Acoustic Measurements</div>', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                db_color = status_color(results["noise_status"])
                st.markdown(f'<div class="metric-card"><div class="metric-label">dB Level [01]</div><div class="metric-value" style="color:{db_color}">{results["db_spl"]} dB SPL</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">TWA Dose [02]</div><div class="metric-value">{results["twa_dose_pct"]} %</div></div>', unsafe_allow_html=True)
            with col3:
                ns_color = status_color(results["noise_status"])
                st.markdown(f'<div class="metric-card"><div class="metric-label">Noise Status [03]</div><div class="metric-value" style="color:{ns_color}">{results["noise_status"]}</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Noise Score [04]</div><div class="metric-value">{results["noise_score"]}</div></div>', unsafe_allow_html=True)
                score_bar(results["noise_score"], "#f59e0b")

            # ══════════════════════════════════════════════════
            # SECTION 2 — DANGER DETECTION
            # ══════════════════════════════════════════════════
            st.markdown("---")
            st.markdown('<div class="section-header">🚨 Danger Detection — YAMNet + AST Fusion</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                d_color = "#ef4444" if results["danger_score"] > 0.25 else "#22c55e"
                st.markdown(f'<div class="metric-card"><div class="metric-label">Danger Score [05]</div><div class="metric-value" style="color:{d_color}">{results["danger_score"]}</div></div>', unsafe_allow_html=True)
                score_bar(results["danger_score"], d_color)
            with col2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Env Score [06]</div><div class="metric-value">{results["env_score"]}</div></div>', unsafe_allow_html=True)
                score_bar(results["env_score"], "#f59e0b")
            with col3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Physio Score [07]</div><div class="metric-value">{results["physio_score"]}</div></div>', unsafe_allow_html=True)
                score_bar(results["physio_score"], "#a78bfa")

            col4, col5, col6 = st.columns(3)
            with col4:
                es_color = "#ef4444" if results["env_status"] == "UNSAFE" else "#22c55e"
                st.markdown(f'<div class="metric-card"><div class="metric-label">Env Status [08]</div><div class="metric-value" style="color:{es_color}">{results["env_status"]}</div></div>', unsafe_allow_html=True)
            with col5:
                st.markdown(f'<div class="metric-card"><div class="metric-label">YAMNet Weight [10]</div><div class="metric-value">{results["yamnet_weight"]}</div></div>', unsafe_allow_html=True)
            with col6:
                st.markdown(f'<div class="metric-card"><div class="metric-label">AST Weight [11]</div><div class="metric-value">{results["ast_weight"]}</div></div>', unsafe_allow_html=True)

            st.markdown("**[09] Top Detections**")
            det_cols = st.columns(len(results["top_detections"]))
            for i, (cls, sc) in enumerate(results["top_detections"]):
                with det_cols[i]:
                    bar_color = "#ef4444" if sc > 0.25 else ("#f59e0b" if sc > 0.10 else "#6b7280")
                    st.markdown(f"""
                    <div style="text-align:center; padding:10px; background:#1e1e2e; border-radius:8px;">
                        <div style="font-size:0.7rem; color:#a0a0b0;">{cls}</div>
                        <div style="font-size:1.1rem; font-weight:700; color:{bar_color};">{sc:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ══════════════════════════════════════════════════
            # SECTION 3 — VOCAL HEALTH
            # ══════════════════════════════════════════════════
            st.markdown("---")
            st.markdown('<div class="section-header">🎤 Vocal Health — Praat-style (librosa)</div>', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                metric_card("Pitch Mean [12]", f"{results['pitch_hz']} Hz")
                metric_card("Pitch SD [13]",   f"{results['pitch_sd']} Hz")
            with col2:
                metric_card("Pitch Min [14]",  f"{results['pitch_min']} Hz")
                metric_card("Pitch Max [15]",  f"{results['pitch_max']} Hz")
            with col3:
                j_color = "#ef4444" if results["jitter_pct"] > CLINICAL["jitter_normal_max"] * 100 else "#22c55e"
                s_color = "#ef4444" if results["shimmer_pct"] > CLINICAL["shimmer_normal_max"] * 100 else "#22c55e"
                st.markdown(f'<div class="metric-card"><div class="metric-label">Jitter [16] (normal &lt; {CLINICAL["jitter_normal_max"]*100:.2f}%)</div><div class="metric-value" style="color:{j_color}">{results["jitter_pct"]} %</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><div class="metric-label">Shimmer [17] (normal &lt; {CLINICAL["shimmer_normal_max"]*100:.2f}%)</div><div class="metric-value" style="color:{s_color}">{results["shimmer_pct"]} %</div></div>', unsafe_allow_html=True)
            with col4:
                h_color = "#ef4444" if results["hnr_db"] < CLINICAL["hnr_normal_min"] else "#22c55e"
                st.markdown(f'<div class="metric-card"><div class="metric-label">HNR [18] (normal &gt; {CLINICAL["hnr_normal_min"]} dB)</div><div class="metric-value" style="color:{h_color}">{results["hnr_db"]} dB</div></div>', unsafe_allow_html=True)
                metric_card("Speaking Rate [19]", f"{results['speaking_rate']} syl/s")

            col5, col6 = st.columns(2)
            with col5:
                metric_card("Intensity Mean [20]", f"{results['intensity_mean']} dB")
            with col6:
                vh_color = "#22c55e" if results["vocal_health"] == "Normal" else "#ef4444"
                st.markdown(f'<div class="metric-card"><div class="metric-label">Vocal Health [21]</div><div class="metric-value" style="color:{vh_color}">{results["vocal_health"]}</div></div>', unsafe_allow_html=True)

            # ══════════════════════════════════════════════════
            # SECTION 4 — LINGUISTIC ANALYSIS
            # ══════════════════════════════════════════════════
            st.markdown("---")
            st.markdown('<div class="section-header">🗣️ Linguistic Analysis — Whisper + XLM-RoBERTa + MPNet</div>', unsafe_allow_html=True)

            st.markdown("**[22] Transcript**")
            st.markdown(f'<div class="transcript-box">"{results["transcript"]}"</div>', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                metric_card("Language [23]", results["language"].upper())
           
            with col2:
                i_color = "#ef4444" if results["intent_score"] > 0.50 else ("#f59e0b" if results["intent_score"] > 0.20 else "#22c55e")
                st.markdown(f'<div class="metric-card"><div class="metric-label">Intent Score [24] (XLM-RoBERTa)</div><div class="metric-value" style="color:{i_color}">{results["intent_score"]}</div></div>', unsafe_allow_html=True)
                score_bar(results["intent_score"], i_color)
            with col3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Semantic Score [25] (MPNet cosine)</div><div class="metric-value">{results["semantic_score"]}</div></div>', unsafe_allow_html=True)
                score_bar(results["semantic_score"], "#a78bfa")
            with col4:
                l_color = "#ef4444" if results["linguistic_score"] > 0.50 else ("#f59e0b" if results["linguistic_score"] > 0.30 else "#22c55e")
                st.markdown(f'<div class="metric-card"><div class="metric-label">Linguistic Score [26]</div><div class="metric-value" style="color:{l_color}">{results["linguistic_score"]}</div></div>', unsafe_allow_html=True)
                score_bar(results["linguistic_score"], l_color)

            # ══════════════════════════════════════════════════
            # SECTION 5 — SPEECH EMOTION
            # ══════════════════════════════════════════════════
            st.markdown("---")
            st.markdown('<div class="section-header">😤 Speech Emotion — wav2vec2 SER</div>', unsafe_allow_html=True)

            col1, col2 = st.columns([1, 2])
            with col1:
                emotion_colors = {
                    "angry":   "#ef4444",
                    "sad":     "#60a5fa",
                    "happy":   "#22c55e",
                    "neutral": "#a0a0b0",
                }
                top_emo   = results["emotion"]
                top_conf  = results["emotion_confidence"]
                emo_color = emotion_colors.get(top_emo, "#a78bfa")
                st.markdown(f"""
                <div style="text-align:center; padding:30px 20px; background:#1e1e2e;
                            border-radius:12px; border: 2px solid {emo_color};">
                    <div style="font-size:3rem;">
                        {"😡" if top_emo=="angry" else "😢" if top_emo=="sad" else "😊" if top_emo=="happy" else "😐"}
                    </div>
                    <div style="font-size:1.6rem; font-weight:800; color:{emo_color}; margin-top:8px;">
                        {top_emo.upper()}
                    </div>
                    <div style="font-size:1rem; color:#a0a0b0; margin-top:4px;">
                        [27] Confidence: {top_conf*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("**[28] Emotion Breakdown**")
                for emo, prob in results["emotion_breakdown"].items():
                    bar_color = emotion_colors.get(emo, "#a78bfa")
                    pct = int(prob * 100)
                    st.markdown(f"""
                    <div style="margin: 6px 0;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                            <span style="color:#e2e8f0; font-size:0.9rem; text-transform:capitalize;">{emo}</span>
                            <span style="color:{bar_color}; font-weight:700; font-size:0.9rem;">{prob:.4f}</span>
                        </div>
                        <div style="background:#2d2d3d; border-radius:4px; height:10px;">
                            <div style="width:{pct}%; background:{bar_color}; height:10px; border-radius:4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ══════════════════════════════════════════════════
            # SECTION 6 — DISTRESS ANALYSIS
            # ══════════════════════════════════════════════════
            st.markdown("---")
            st.markdown('<div class="section-header">😰 Distress Analysis — 6-Model Fusion</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                ds = results["distress_score"]
                ds_color = "#ef4444" if ds >= 0.75 else ("#f59e0b" if ds >= 0.50 else ("#fbbf24" if ds >= 0.25 else "#22c55e"))
                st.markdown(f'<div class="metric-card"><div class="metric-label">Distress Score [29]</div><div class="metric-value" style="color:{ds_color}">{ds}</div></div>', unsafe_allow_html=True)
                score_bar(ds, ds_color)
            with col2:
                dst_color = status_color(results["distress_status"])
                st.markdown(f'<div class="metric-card"><div class="metric-label">Distress Status [30]</div><div class="metric-value" style="color:{dst_color}">{results["distress_status"]}</div></div>', unsafe_allow_html=True)

            # ══════════════════════════════════════════════════
            # SECTION 7 — OVERALL RISK
            # ══════════════════════════════════════════════════
            st.markdown("---")
            st.markdown('<div class="section-header">🔰 Overall Risk Assessment</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                ors = results["overall_risk_score"]
                ors_color = "#ef4444" if ors >= 0.55 else ("#f59e0b" if ors >= 0.25 else "#22c55e")
                st.markdown(f'<div class="metric-card"><div class="metric-label">Overall Risk Score [31]</div><div class="metric-value" style="color:{ors_color}">{ors}</div></div>', unsafe_allow_html=True)
                score_bar(ors, ors_color)
            with col2:
                render_status_banner(results["overall_risk_status"])

            # ══════════════════════════════════════════════════
            # SECTION 8 — SUMMARY TABLE
            # ══════════════════════════════════════════════════
            st.markdown("---")
            st.markdown('<div class="section-header">📋 Summary</div>', unsafe_allow_html=True)

            summary_data = {
                "Field":  [
                    "State", "Distress", "Emotion",
                    "Vocal Health", "Transcript", "Language",
                    "Overall Score", "Distress Score",
                    "Danger Score", "Linguistic Score",
                ],
                "Value": [
                    results["overall_risk_status"],
                    results["distress_status"],
                    f"{results['emotion'].upper()} ({results['emotion_confidence']*100:.1f}%)",
                    results["vocal_health"],
                    f'"{results["transcript"]}"',
                    results["language"].upper(),
                    str(results["overall_risk_score"]),
                    str(results["distress_score"]),
                    str(results["danger_score"]),
                    str(results["linguistic_score"]),
                ],
            }
            st.table(summary_data)

            # ══════════════════════════════════════════════════
            # SECTION 9 — ASSERTION CHECK (mirrors Cell 18)
            # ══════════════════════════════════════════════════
            st.markdown("---")
            st.markdown('<div class="section-header">🧪 Assertion Check (mirrors Cell 18)</div>', unsafe_allow_html=True)

            INTENT_MAX_NEUTRAL = 0.20
            LING_MAX_NEUTRAL   = 0.30
            EXPECTED_STATUSES  = ["✅ Normal", "⚠️ Stress"]

            intent_pass = results["intent_score"]   < INTENT_MAX_NEUTRAL
            ling_pass   = results["linguistic_score"] < LING_MAX_NEUTRAL
            status_pass = results["overall_risk_status"] in EXPECTED_STATUSES
            all_pass    = intent_pass and ling_pass and status_pass

            a1, a2, a3 = st.columns(3)
            with a1:
                st.markdown(f"""
                <div style="padding:14px; background:{'#0a2d1a' if intent_pass else '#3b0a0a'};
                            border-radius:10px; border-left:4px solid {'#22c55e' if intent_pass else '#ef4444'};">
                    <div style="font-size:0.75rem; color:#a0a0b0;">Intent Score (must be &lt; {INTENT_MAX_NEUTRAL})</div>
                    <div style="font-size:1.2rem; font-weight:700; color:{'#22c55e' if intent_pass else '#ef4444'};">
                        {results['intent_score']:.4f} {'✅ PASS' if intent_pass else '❌ FAIL'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with a2:
                st.markdown(f"""
                <div style="padding:14px; background:{'#0a2d1a' if ling_pass else '#3b0a0a'};
                            border-radius:10px; border-left:4px solid {'#22c55e' if ling_pass else '#ef4444'};">
                    <div style="font-size:0.75rem; color:#a0a0b0;">Linguistic Score (must be &lt; {LING_MAX_NEUTRAL})</div>
                    <div style="font-size:1.2rem; font-weight:700; color:{'#22c55e' if ling_pass else '#ef4444'};">
                        {results['linguistic_score']:.4f} {'✅ PASS' if ling_pass else '❌ FAIL'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with a3:
                st.markdown(f"""
                <div style="padding:14px; background:{'#0a2d1a' if status_pass else '#3b0a0a'};
                            border-radius:10px; border-left:4px solid {'#22c55e' if status_pass else '#ef4444'};">
                    <div style="font-size:0.75rem; color:#a0a0b0;">Overall Status (must not be Distress)</div>
                    <div style="font-size:1.2rem; font-weight:700; color:{'#22c55e' if status_pass else '#ef4444'};">
                        {results['overall_risk_status']} {'✅ PASS' if status_pass else '❌ FAIL'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if all_pass:
                st.success("🎉 ALL ASSERTIONS PASSED")
            else:
                st.error("🔴 ASSERTIONS FAILED — review intent fix")

            # ══════════════════════════════════════════════════
            # SECTION 10 — JSON EXPORT
            # ══════════════════════════════════════════════════
            st.markdown("---")
            st.markdown('<div class="section-header">💾 Export Results</div>', unsafe_allow_html=True)

            # Prepare JSON-safe results (top_detections is list of tuples)
            export = {k: v for k, v in results.items() if k != "top_detections"}
            export["top_detections"] = [
                {"class": cls, "score": sc} for cls, sc in results["top_detections"]
            ]
            export["analyzed_file"] = uploaded_file.name
            export["analyzed_at"]   = datetime.now().isoformat()

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                import json
                st.download_button(
                    label="⬇️ Download JSON Report",
                    data=json.dumps(export, indent=2),
                    file_name=f"audio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with col_dl2:
                import csv as csv_module
                import io as io_module
                flat_rows = [(k, str(v)) for k, v in export.items() if k != "top_detections"]
                flat_rows += [(f"top_detection_{i+1}", f"{d['class']} ({d['score']})") for i, d in enumerate(export["top_detections"])]
                csv_buf = io_module.StringIO()
                writer  = csv_module.writer(csv_buf)
                writer.writerow(["field", "value"])
                writer.writerows(flat_rows)
                st.download_button(
                    label="⬇️ Download CSV Report",
                    data=csv_buf.getvalue(),
                    file_name=f"audio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        finally:
            # Always clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

else:
    # ── Empty state ───────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#6b7280;">
        <div style="font-size:4rem;">🎙️</div>
        <div style="font-size:1.3rem; margin-top:12px;">Upload an audio file to begin analysis</div>
        <div style="font-size:0.9rem; margin-top:8px;">Supports WAV · MP3 · FLAC · OGG · M4A</div>
    </div>
    """, unsafe_allow_html=True)

