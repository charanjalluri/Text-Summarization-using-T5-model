import langdetect
from deep_translator import GoogleTranslator

from app.core.exceptions import TranslationError
from app.core.logging_config import get_logger

logger = get_logger(__name__)

def detect_language(text: str) -> str:
    """Detects the language of the input text (ISO-639-1 code)."""
    if not text or not text.strip():
        return "unknown"
    try:
        lang = langdetect.detect(text)
        logger.info(f"Detected text language: {lang}")
        return lang
    except Exception as e:
        logger.warning(f"Language detection failed: {e}. Defaulting to 'unknown'.")
        return "unknown"

def translate_text(text: str, target_lang: str = "en", source_lang: str = "auto") -> str:
    """Translates the input text to the target language code."""
    if not text or not text.strip():
        return ""

    if source_lang == target_lang:
        return text

    try:
        logger.info(f"Translating text from source '{source_lang}' to target '{target_lang}'")
        # GoogleTranslator handles 'auto' source language detection
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
        if not translated:
            raise TranslationError("Translator returned empty string.")
        return translated.strip()
    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        raise TranslationError(f"Failed to translate text from '{source_lang}' to '{target_lang}': {e}") from e
