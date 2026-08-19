import sys
from unittest.mock import MagicMock

# Secure sys.modules to mock fitz and avoid DLL load failed issues during test collection
sys.modules["fitz"] = MagicMock()

from unittest.mock import patch

import pytest

from app.core.exceptions import FileExtractionError
from app.services.document_processor import (
    extract_audio,
    extract_docx,
    extract_image,
    extract_pdf,
    extract_text_from_file,
    extract_txt,
)


def test_extract_txt_utf8():
    content = "Hello, this is standard UTF-8 text.".encode("utf-8")
    assert extract_txt(content) == "Hello, this is standard UTF-8 text."

def test_extract_txt_latin1():
    content = b"Caf\xe9 text"
    assert extract_txt(content) == "Café text"

@patch("app.services.document_processor.fitz")
def test_extract_pdf_pymupdf_success(mock_fitz):
    # Mock PDF document structure
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Page content text"
    mock_doc.__iter__.return_value = [mock_page]

    # Configure the context manager mock return value
    mock_fitz.open.return_value.__enter__.return_value = mock_doc

    result = extract_pdf(b"dummy pdf bytes")
    assert result == "Page content text"
    mock_fitz.open.assert_called_once()

@patch("app.services.document_processor.fitz")
@patch("pdfplumber.open")
def test_extract_pdf_pymupdf_empty_fallback_plumber(mock_pdfplumber_open, mock_fitz):
    # PyMuPDF returns empty text
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = ""
    mock_doc.__iter__.return_value = [mock_page]

    # Configure the context manager mock return value
    mock_fitz.open.return_value.__enter__.return_value = mock_doc

    # pdfplumber returns text content
    mock_plumber_doc = MagicMock()
    mock_plumber_page = MagicMock()
    mock_plumber_page.extract_text.return_value = "Plumber extracted text"
    mock_plumber_doc.pages = [mock_plumber_page]
    mock_pdfplumber_open.return_value.__enter__.return_value = mock_plumber_doc

    result = extract_pdf(b"dummy pdf bytes")
    assert result == "Plumber extracted text"

@patch("app.services.document_processor.Document")
def test_extract_docx_paragraphs(mock_docx_document):
    mock_doc = MagicMock()
    mock_para1 = MagicMock()
    mock_para1.text = "Paragraph 1"
    mock_para2 = MagicMock()
    mock_para2.text = "Paragraph 2"
    mock_doc.paragraphs = [mock_para1, mock_para2]
    mock_doc.tables = []
    mock_docx_document.return_value = mock_doc

    result = extract_docx(b"dummy docx bytes")
    assert result == "Paragraph 1\nParagraph 2"

@patch("app.services.document_processor.AudioSegment")
@patch("app.services.document_processor.sr")
def test_extract_audio_success(mock_sr, mock_audiosegment):
    mock_audio = MagicMock()
    mock_audiosegment.from_file.return_value = mock_audio

    mock_recognizer = MagicMock()
    mock_recognizer.recognize_google.return_value = "transcribed speech from audio file"
    mock_sr.Recognizer.return_value = mock_recognizer

    result = extract_audio(b"dummy audio bytes", "mp3")
    assert result == "transcribed speech from audio file"

@patch("app.services.document_processor.pytesseract")
@patch("PIL.Image.open")
def test_extract_image_success(mock_image_open, mock_pytesseract):
    mock_pytesseract.image_to_string.return_value = "OCR text extracted from image"
    result = extract_image(b"dummy image bytes")
    assert result == "OCR text extracted from image"

def test_extract_text_from_file_routing():
    content = b"plain text"
    assert extract_text_from_file(content, "test.txt") == "plain text"

    with pytest.raises(FileExtractionError, match="Unsupported file format"):
        extract_text_from_file(content, "test.exe")
