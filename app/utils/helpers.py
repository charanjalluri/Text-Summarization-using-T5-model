import time
from contextlib import contextmanager
from typing import Any, Dict, Generator

from app.core.logging_config import get_logger

logger = get_logger(__name__)

@contextmanager
def time_execution(label: str) -> Generator[Dict[str, Any], None, None]:
    """Helper context manager to profile code blocks execution times."""
    result = {"elapsed_time": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_time"] = time.perf_counter() - start
        logger.info(f"{label} execution took {result['elapsed_time']:.4f} seconds.")

def generate_pdf_bytes(title: str, content: str) -> bytes:
    """Generates PDF bytes using ReportLab. Falls back to plain text if failed."""
    try:
        import io

        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        pdf_buffer = io.BytesIO()
        # Clean margins
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=54,
            leftMargin=54,
            topMargin=54,
            bottomMargin=54
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=20,
            leading=24,
            spaceAfter=15
        )
        body_style = ParagraphStyle(
            'BodyStyle',
            parent=styles['BodyText'],
            fontSize=11,
            leading=16,
            spaceAfter=10
        )

        story = [
            Paragraph(title, title_style),
            Spacer(1, 12)
        ]

        # Process text by blocks
        paragraphs = content.split("\n\n")
        for p in paragraphs:
            if p.strip():
                # Escape xml characters for ReportLab Paragraph parser
                escaped = p.strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                escaped_newlines = escaped.replace("\n", "<br/>")
                story.append(Paragraph(escaped_newlines, body_style))
                story.append(Spacer(1, 6))

        doc.build(story)
        return pdf_buffer.getvalue()
    except Exception as e:
        logger.warning(f"ReportLab PDF generation failed: {e}. Falling back to UTF-8 bytes.")
        return content.encode("utf-8")
