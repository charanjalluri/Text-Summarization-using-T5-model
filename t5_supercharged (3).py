# -*- coding: utf-8 -*-
"""T5_Supercharged (Local Version — Ngrok Removed)"""

import os
import io
import tempfile
from typing import List, Dict, Optional

import streamlit as st
from PIL import Image
import numpy as np
import fitz  # PyMuPDF
import pdfplumber
from PyPDF2 import PdfReader
import speech_recognition as sr
from pydub import AudioSegment
import langdetect
from deep_translator import GoogleTranslator
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
from gtts import gTTS
from reportlab.pdfgen import canvas as rl_canvas
from docx import Document
import pytesseract
from google import genai
from google.genai import types

# -----------------------------
# Streamlit Configuration
# -----------------------------
st.set_page_config(
    page_title="🚀 T5/Gemini Summarization Tool",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Gemini Configuration
# -----------------------------
GEMINI_DEFAULT_MODEL = "gemini-1.5-flash"
GEMINI_VISION_MODEL = "gemini-1.5-pro-vision"
GEMINI_EMBED_MODEL = "text-embedding-004"

def ensure_gemini(api_key: Optional[str]):
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        st.warning("⚠️ Please enter your Gemini API key in the sidebar.")
        return None
    try:
        return genai.Client(api_key=key)
    except Exception as e:
        st.error(f"Gemini initialization failed: {e}")
        return None

def get_gemini_client():
    key = os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=key)

def gemini_summarize(text: str, style="concise", target_lang="en", model=GEMINI_DEFAULT_MODEL) -> str:
    if not text.strip():
        return ""
    client = get_gemini_client()
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[types.Part(text=f"Summarize this in {target_lang}, style={style}:\n\n{text}")]
        )
        return resp.text or ""
    except Exception as e:
        st.error(f"Gemini summarization failed: {e}")
        return ""

# -----------------------------
# Cached Models
# -----------------------------
@st.cache_resource
def load_t5(model_name: str):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

# -----------------------------
# Helper Functions
# -----------------------------
def detect_language(text: str) -> str:
    try:
        return langdetect.detect(text)
    except Exception:
        return "unknown"

def translate_text(text: str, target_lang: str = "en") -> str:
    if not text:
        return text
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception:
        return text

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 120) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - overlap)
    return chunks

def extract_text_from_pdf(file) -> Dict[str, any]:
    text_all = []
    tables = []
    file_bytes = file.read()
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text_all.append(page.get_text("text") or "")
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for p in pdf.pages:
            try:
                tbs = p.extract_tables() or []
                for tb in tbs:
                    if tb:
                        tables.append(tb)
            except Exception:
                pass
    return {"text": "\n".join(text_all), "tables": tables}

def transcribe_audio(uploaded_audio) -> str:
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(io.BytesIO(uploaded_audio.read()))
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(buf.getvalue())
        tmp.flush()
        with sr.AudioFile(tmp.name) as source:
            audio_data = recognizer.record(source)
            try:
                return recognizer.recognize_google(audio_data)
            except Exception:
                return ""

def abstractive_t5(text: str, model_name: str, max_length=150, min_length=30) -> str:
    tok, mdl, dev = load_t5(model_name)
    input_text = "summarize: " + text
    inputs = tok.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(dev)
    out_ids = mdl.generate(inputs, max_length=max_length, min_length=min_length, num_beams=4)
    return tok.decode(out_ids[0], skip_special_tokens=True)

# -----------------------------
# Streamlit App UI
# -----------------------------
def sidebar_settings():
    st.sidebar.header("⚙️ Settings")
    input_type = st.sidebar.selectbox("Input Type:", ["Text", "Image", "PDF", "Audio"])
    use_gemini = st.sidebar.checkbox("Use Gemini (Recommended)", value=True)
    gem_key = st.sidebar.text_input("Gemini API Key", type="password")
    client = ensure_gemini(gem_key) if use_gemini else None
    t5_name = st.sidebar.selectbox("T5 Model", ["t5-small", "t5-base", "google/flan-t5-base"])
    style = st.sidebar.selectbox("Summarization Style", ["concise", "bullet-heavy", "academic", "narrative"])
    target_lang = st.sidebar.text_input("Target Language Code", "en")
    return dict(input_type=input_type, use_gemini=use_gemini and client, t5_name=t5_name, style=style, target_lang=target_lang)

def main():
    st.title("🚀 T5 Summarizer — Local Version (Ngrok Removed)")
    cfg = sidebar_settings()

    if cfg["input_type"] == "Text":
        text = st.text_area("Enter text to summarize:", height=250)
        if st.button("Summarize"):
            if not text.strip():
                st.warning("Please enter some text.")
                return
            result = gemini_summarize(text, cfg["style"], cfg["target_lang"]) if cfg["use_gemini"] else abstractive_t5(text, cfg["t5_name"])
            st.subheader("📄 Summary")
            st.write(result)

    elif cfg["input_type"] == "PDF":
        pdf = st.file_uploader("Upload a PDF", type=["pdf"])
        if pdf and st.button("Summarize PDF"):
            data = extract_text_from_pdf(pdf)
            text = data["text"]
            result = gemini_summarize(text, cfg["style"], cfg["target_lang"]) if cfg["use_gemini"] else abstractive_t5(text, cfg["t5_name"])
            st.subheader("📄 PDF Summary")
            st.write(result)

    elif cfg["input_type"] == "Audio":
        audio = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])
        if audio and st.button("Transcribe & Summarize"):
            text = transcribe_audio(audio)
            st.subheader("🎙️ Transcribed Text")
            st.write(text)
            result = gemini_summarize(text, cfg["style"], cfg["target_lang"]) if cfg["use_gemini"] else abstractive_t5(text, cfg["t5_name"])
            st.subheader("📄 Summary")
            st.write(result)

    elif cfg["input_type"] == "Image":
        image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        if image and st.button("Extract & Summarize"):
            img = Image.open(image)
            st.image(img, caption="Uploaded Image", use_container_width=True)
            extracted = pytesseract.image_to_string(img)
            st.write("📝 Extracted Text:")
            st.write(extracted)
            result = gemini_summarize(extracted, cfg["style"], cfg["target_lang"]) if cfg["use_gemini"] else abstractive_t5(extracted, cfg["t5_name"])
            st.subheader("📄 Summary")
            st.write(result)

# -----------------------------
# Local Streamlit Launcher
# -----------------------------
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    import sys
    sys.argv = ["streamlit", "run", __file__, "--server.port", "8501", "--server.address", "0.0.0.0"]
    sys.exit(stcli.main())

