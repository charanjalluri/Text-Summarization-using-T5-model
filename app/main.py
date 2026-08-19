import os
import sys

# Ensure the parent directory of 'app' is in sys.path so imports function correctly
# when running 'python app/main.py' directly.
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import streamlit as st

from app.core.logging_config import get_logger, setup_logging
from app.services.document_processor import extract_text_from_file
from app.services.summarization import SummarizationService
from app.ui.components import render_header, render_metrics_dashboard, render_sidebar, render_summary_box
from app.ui.styles import inject_custom_styles
from app.utils.helpers import generate_pdf_bytes, time_execution
from app.utils.metrics import calculate_compression_ratio, calculate_counts, estimate_tokens
from app.utils.validators import validate_inference_params, validate_uploaded_file

# Initialize global logging configuration
setup_logging()
logger = get_logger(__name__)

def run_app():
    """Builds and runs the Streamlit UI application."""
    # Custom Streamlit setups
    st.set_page_config(
        page_title="AntiGravity Summarizer - Portfolio AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject visual styling and header branding
    inject_custom_styles()
    render_header()

    # Load Sidebar configuration inputs
    cfg = render_sidebar()

    # Form Tabs
    tab_text, tab_file = st.tabs(["📝 Paste Text Content", "📁 Upload Document File"])

    input_text = ""
    source_name = "Text Input"

    with tab_text:
        text_area_content = st.text_area(
            "Enter text to analyze and summarize:",
            height=280,
            placeholder="Type or paste paragraphs here (articles, notes, papers, reports)..."
        )
        if text_area_content.strip():
            input_text = text_area_content
            source_name = "Manual Input"

    with tab_file:
        uploaded_file = st.file_uploader(
            "Choose a document (supported format: PDF, DOCX, TXT, MD, WAV, MP3, M4A, PNG, JPG, JPEG):",
            type=["txt", "text", "md", "pdf", "docx", "wav", "mp3", "m4a", "flac", "png", "jpg", "jpeg"]
        )
        if uploaded_file:
            source_name = uploaded_file.name
            try:
                # Fetch uploaded file bytes
                file_bytes = uploaded_file.read()

                # Perform extension and size validations
                validate_uploaded_file(uploaded_file.name, len(file_bytes))

                # Process extraction
                with st.spinner("Extracting textual content from file..."):
                    input_text = extract_text_from_file(file_bytes, uploaded_file.name)

                if not input_text.strip():
                    st.error("Document parsed successfully, but no readable text was extracted.")
                else:
                    st.success(f"Loaded document: {uploaded_file.name}")
                    words_count = len(input_text.split())
                    st.info(f"Loaded content stats: {words_count} words / {len(input_text)} characters.")

                    with st.expander("👁️ View Extracted Document Content Preview"):
                        st.text(input_text[:1200] + ("..." if len(input_text) > 1200 else ""))
            except Exception as e:
                st.error(f"Failed to read file: {e}")
                logger.error(f"Error loading file {uploaded_file.name}: {e}", exc_info=True)

    # Process and summarize
    if st.button("🚀 Summarize Content"):
        if not input_text.strip():
            st.warning("⚠️ Input is empty. Please enter text or upload a valid document.")
            return

        # Parameter reconciliation (guarantees min <= max)
        min_len = cfg["min_length"]
        max_len = cfg["max_length"]
        if min_len > max_len:
            logger.info("Reconciling parameter min_length > max_length. Adjusting min_length.")
            min_len = max(10, max_len - 20)

        # Validate parameters before execution
        try:
            validate_inference_params(
                min_len,
                max_len,
                cfg["num_beams"],
                cfg["length_penalty"]
            )
        except Exception as e:
            st.error(f"Inference Parameter Error: {e}")
            return

        # Summarization execution
        service = SummarizationService()
        try:
            with st.spinner("AI model is processing summarization..."):
                with time_execution("Summarization Service Run") as timer:
                    results = service.summarize(
                        text=input_text,
                        backend=cfg["backend"],
                        model_name=cfg["model_name"],
                        device=cfg["device"],
                        style=cfg["style"],
                        target_lang=cfg["target_lang"],
                        min_length=min_len,
                        max_length=max_len,
                        num_beams=cfg["num_beams"],
                        length_penalty=cfg["length_penalty"],
                        gemini_api_key=cfg["gemini_api_key"]
                    )

            # Collect results metadata
            summary_str = results["summary"]
            elapsed = timer["elapsed_time"]

            if not summary_str.strip():
                st.error("Model execution completed, but summary output was empty.")
                return

            # Analytics compilation
            output_stats = calculate_counts(summary_str)
            ratio = calculate_compression_ratio(input_text, summary_str)

            metrics_data = {
                "elapsed_time": elapsed,
                "compression_ratio": ratio,
                "device": results["device"],
                "model": results["model"],
                "output_words": output_stats["words"],
                "output_tokens": estimate_tokens(summary_str)
            }

            # Display summaries and analytical dashboards
            render_summary_box(summary_str, results)
            render_metrics_dashboard(metrics_data)

            # Export Summaries
            st.markdown("<br/>", unsafe_allow_html=True)
            col_txt, col_pdf, _ = st.columns([2, 2, 6])

            # Export as TXT
            col_txt.download_button(
                label="📥 Download TXT Summary",
                data=summary_str,
                file_name=f"summary_{source_name.split('.')[0]}.txt",
                mime="text/plain"
            )

            # Export as PDF
            with st.spinner("Compiling PDF summary doc..."):
                pdf_bytes = generate_pdf_bytes(f"Summary of {source_name}", summary_str)

            col_pdf.download_button(
                label="📥 Download PDF Summary",
                data=pdf_bytes,
                file_name=f"summary_{source_name.split('.')[0]}.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"An error occurred during summarization: {e}")
            logger.error(f"Execution failure: {e}", exc_info=True)

if __name__ == "__main__":
    import sys
    # If not running inside Streamlit, launch the Streamlit CLI wrapper programmatically
    if not st.runtime.exists():
        import streamlit.web.cli as stcli
        # Programmatic launch configuration
        sys.argv = ["streamlit", "run", __file__, "--server.port", "8501", "--server.address", "0.0.0.0"]
        sys.exit(stcli.main())
    else:
        run_app()
