import re
import unicodedata
from typing import Any, List

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

def clean_text(text: str) -> str:
    """Normalizes whitespace, removes control characters, and cleans malformed text

    while preserving standard punctuation and paragraphs.
    """
    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters (except tabs and newlines)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\r\t")

    # Normalize newlines and strip whitespace surrounding newlines
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize horizontal spacing
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()

def chunk_text_by_words(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Splits text into chunks based on word boundaries."""
    chunk_sz = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    ovlp = overlap if overlap is not None else settings.CHUNK_OVERLAP

    if chunk_sz <= 0:
        raise ValueError("chunk_size must be positive")
    if ovlp < 0:
        raise ValueError("overlap must be non-negative")
    if ovlp >= chunk_sz:
        raise ValueError("overlap must be less than chunk_size")

    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_sz:
        return [" ".join(words)]

    chunks = []
    i = 0
    step = max(1, chunk_sz - ovlp)
    while i < len(words):
        # Stop if we have already covered the remaining words in the previous overlap step
        if i > 0 and i + ovlp >= len(words):
            break

        chunk_words = words[i:i + chunk_sz]
        chunks.append(" ".join(chunk_words))
        i += step

    return chunks

def chunk_text_by_tokens(text: str, tokenizer: Any, max_tokens: int = 400, overlap: int = 50) -> List[str]:
    """Token-aware text chunking. Respects tokenizer boundaries to avoid token limit errors."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= max_tokens:
        raise ValueError("overlap must be less than max_tokens")

    if not text.strip():
        return []

    try:
        # Tokenize the text into token IDs without special tokens
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            return []

        if len(tokens) <= max_tokens:
            return [text.strip()]

        chunks = []
        i = 0
        step = max(1, max_tokens - overlap)
        while i < len(tokens):
            # Stop if remaining tokens are already fully covered by the previous step's overlap
            if i > 0 and i + overlap >= len(tokens):
                break

            chunk_ids = tokens[i:i + max_tokens]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            i += step

        return chunks
    except Exception as e:
        logger.warning(f"Token-aware chunking failed, falling back to word-based chunking. Error: {e}")
        word_size = int(max_tokens * 0.75)
        word_overlap = int(overlap * 0.75)
        return chunk_text_by_words(text, chunk_size=word_size, overlap=word_overlap)
