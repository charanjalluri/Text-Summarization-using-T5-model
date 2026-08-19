from typing import Any, Dict

import streamlit as st

from app.core.config import settings


def render_header() -> None:
    """Renders the top branding header and description for the application."""
    st.markdown('<div class="brand-header">🚀 Text Summarization using T5 model</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">An elegant, multi-backend AI summarization platform powered by T5 and Gemini</div>', unsafe_allow_html=True)

def render_sidebar() -> Dict[str, Any]:
    """Renders the settings sidebar controls and returns configuration values."""
    st.sidebar.markdown('<h2 style="margin-bottom: 0.5rem;">⚙️ Control Hub</h2>', unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # 1. Main summarizer backend choice
    backend = st.sidebar.selectbox(
        "Summarization Provider",
        ["t5", "gemini"],
        format_func=lambda x: "🤖 Local T5 (Offline)" if x == "t5" else "⚡ Google Gemini (Cloud)"
    )

    # 2. Dynamic model details based on backend
    model_name = None
    gemini_key = None
    device = "auto"
    style = "concise"

    if backend == "t5":
        model_name = st.sidebar.selectbox(
            "T5 Model Size",
            ["t5-small", "t5-base", "google/flan-t5-base"],
            help="Select the pre-trained HuggingFace model weight."
        )
        device = st.sidebar.selectbox(
            "Inference Device",
            ["auto", "cpu", "cuda"],
            format_func=lambda x: "Auto Detect (GPU/CPU)" if x == "auto" else x.upper()
        )
    else:
        model_name = st.sidebar.selectbox(
            "Gemini Model",
            ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
            help="Select the cloud Gemini LLM engine."
        )
        # Handle API keys securely
        env_key = settings.GEMINI_API_KEY or ""
        gemini_key = st.sidebar.text_input(
            "Gemini API Key",
            value=env_key,
            type="password",
            help="Retrieve your key from Google AI Studio."
        )
        style = st.sidebar.selectbox(
            "Summary Style",
            ["concise", "bullet-heavy", "academic", "narrative"],
            format_func=lambda x: x.replace("-", " ").capitalize()
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🛠️ Fine-Tuning Parameters")

    # Advanced generation tuning limits
    min_length = st.sidebar.slider("Min Summary Tokens", 10, 100, settings.MIN_SUMMARY_TOKENS, step=5)
    max_length = st.sidebar.slider("Max Summary Tokens", 30, 300, settings.MAX_SUMMARY_TOKENS, step=5)

    if min_length > max_length:
        st.sidebar.warning("⚠️ Warning: Min Tokens > Max Tokens. The pipeline will automatically adjust to prevent errors.")

    num_beams = st.sidebar.slider("Beam Search Count", 1, 8, settings.NUM_BEAMS, help="Higher values improve quality but increase inference duration.")
    length_penalty = st.sidebar.slider("Length Penalty", 0.5, 3.0, settings.LENGTH_PENALTY, step=0.1, help="Adjust penalty for generating longer summaries.")

    # Target language translator setup
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🌐 Language Settings")
    target_lang = st.sidebar.text_input(
        "Target ISO Lang Code",
        value=settings.DEFAULT_TARGET_LANGUAGE,
        help="Use 2-letter codes: 'en' for English, 'es' for Spanish, 'fr' for French, 'hi' for Hindi, etc."
    )

    return {
        "backend": backend,
        "model_name": model_name,
        "device": device,
        "style": style,
        "gemini_api_key": gemini_key,
        "min_length": min_length,
        "max_length": max_length,
        "num_beams": num_beams,
        "length_penalty": length_penalty,
        "target_lang": target_lang.strip().lower()
    }

def render_metrics_dashboard(metrics: Dict[str, Any]) -> None:
    """Displays visual metrics detailing the summarization performance."""
    st.markdown("### 📊 Inference Analytics")
    col1, col2, col3, col4 = st.columns(4)

    # 1. Processing Time
    elapsed = metrics.get("elapsed_time", 0.0)
    col1.markdown(f"""
    <div class="metric-panel">
        <div class="metric-title">Duration</div>
        <div class="metric-value">{elapsed:.2f}s</div>
        <div class="metric-sub">Inference processing time</div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Compression Ratio
    ratio = metrics.get("compression_ratio", 0.0)
    percentage = (1 - ratio) * 100
    col2.markdown(f"""
    <div class="metric-panel" style="border-left-color: #33CCFF;">
        <div class="metric-title">Compression</div>
        <div class="metric-value">{percentage:.1f}%</div>
        <div class="metric-sub">Reduction in words count</div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Model & Hardware device info
    dev = metrics.get("device", "cpu").upper()
    col3.markdown(f"""
    <div class="metric-panel" style="border-left-color: #FF9933;">
        <div class="metric-title">Hardware</div>
        <div class="metric-value">{dev}</div>
        <div class="metric-sub">{metrics.get("model", "T5")}</div>
    </div>
    """, unsafe_allow_html=True)

    # 4. Words & Token processed counts
    words = metrics.get("output_words", 0)
    col4.markdown(f"""
    <div class="metric-panel" style="border-left-color: #38A169;">
        <div class="metric-title font-bold">Output Stats</div>
        <div class="metric-value">{words} W</div>
        <div class="metric-sub">{metrics.get("output_tokens", 0)} est. tokens</div>
    </div>
    """, unsafe_allow_html=True)

def render_summary_box(summary_content: str, metadata: Dict[str, Any]) -> None:
    """Renders the summary container box with action triggers (copy, download)."""
    st.markdown("---")
    st.markdown(f"""
    <div class="summary-box">
        <div class="summary-title">📄 Generated Summary ({metadata.get("target_lang", "en").upper()})</div>
        <div>{summary_content}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
