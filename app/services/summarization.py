from typing import Any, Dict, Optional

from app.core.config import settings
from app.core.exceptions import SummarizerException
from app.core.logging_config import get_logger
from app.models.gemini_model import GeminiModelWrapper
from app.models.t5_model import get_t5_model_cached
from app.services.text_processing import chunk_text_by_tokens, clean_text
from app.services.translation import detect_language, translate_text

logger = get_logger(__name__)

class SummarizationService:
    def __init__(self):
        pass

    def summarize(
        self,
        text: str,
        backend: str = "t5",  # "t5" or "gemini"
        model_name: str = None,
        device: str = "auto",
        style: str = "concise",
        target_lang: str = "en",
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        length_penalty: Optional[float] = None,
        gemini_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Summarizes text using T5 or Gemini backends.

        Coordinates text preprocessing, token limits validation, language translation,
        and recursive chunk-summarization.
        """
        if not text or not text.strip():
            return {
                "summary": "",
                "backend": backend,
                "model": model_name or ("Gemini" if backend == "gemini" else "T5"),
                "device": "cpu",
                "source_lang": "unknown",
                "target_lang": target_lang,
                "chunks_processed": 0
            }

        # Step 1: Clean text
        cleaned_text = clean_text(text)

        # Step 2: Detect language
        source_lang = detect_language(cleaned_text)

        # Translate non-English texts for T5 (which is an English-trained model)
        text_to_summarize = cleaned_text
        if backend == "t5" and source_lang != "en" and source_lang != "unknown":
            logger.info("Non-English source detected for local T5. Translating to English first.")
            try:
                text_to_summarize = translate_text(cleaned_text, target_lang="en", source_lang=source_lang)
            except Exception as e:
                logger.warning(f"Source translation to English failed: {e}. Summarizing raw text.")

        summary_text = ""
        actual_device = "cpu"
        actual_model = model_name
        chunks_count = 1

        # Step 3: Inference Backend Execution
        if backend == "t5":
            actual_model = model_name or settings.MODEL_NAME
            # Fetch cached model instance
            t5_wrapper = get_t5_model_cached(actual_model, device)
            actual_device = t5_wrapper.device

            # Check length in tokens
            tokenizer = t5_wrapper.tokenizer
            tokens = tokenizer.encode(text_to_summarize, add_special_tokens=False)

            if len(tokens) <= settings.MAX_INPUT_TOKENS:
                # Direct summary
                logger.info(f"Input is {len(tokens)} tokens; summarizing directly.")
                summary_text = t5_wrapper.generate(
                    text_to_summarize,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty
                )
            else:
                # Map-reduce chunk summarization
                logger.info(f"Input is {len(tokens)} tokens (exceeds T5 {settings.MAX_INPUT_TOKENS} limit). Splitting.")
                # We use max_tokens=400 to allow room for summarization prefixes and parameters
                chunks = chunk_text_by_tokens(text_to_summarize, tokenizer, max_tokens=400, overlap=50)
                chunks_count = len(chunks)
                logger.info(f"Document split into {chunks_count} chunks.")

                chunk_summaries = []
                for idx, chunk in enumerate(chunks):
                    logger.info(f"Summarizing chunk {idx + 1}/{chunks_count}")
                    # For chunks, generate a slightly smaller summary to fit the final combine phase
                    min_len_to_use = min_length if min_length is not None else settings.MIN_SUMMARY_TOKENS
                    max_len_to_use = max_length if max_length is not None else settings.MAX_SUMMARY_TOKENS
                    chunk_max_len = max(min_len_to_use, int(max_len_to_use / 2))
                    chunk_min_len = min(min_len_to_use, chunk_max_len)

                    chunk_sum = t5_wrapper.generate(
                        chunk,
                        max_length=chunk_max_len,
                        min_length=chunk_min_len,
                        num_beams=num_beams,
                        length_penalty=length_penalty
                    )
                    if chunk_sum.strip():
                        chunk_summaries.append(chunk_sum)

                combined_text = " ".join(chunk_summaries)
                combined_tokens = tokenizer.encode(combined_text, add_special_tokens=False)

                # Summarize combined chunks if they exceed input limits
                if len(combined_tokens) > settings.MAX_INPUT_TOKENS:
                    logger.info(f"Combined chunk summaries equal {len(combined_tokens)} tokens. Performing final reduce pass.")
                    summary_text = t5_wrapper.generate(
                        combined_text,
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=num_beams,
                        length_penalty=length_penalty
                    )
                else:
                    summary_text = combined_text

        elif backend == "gemini":
            actual_model = model_name or settings.GEMINI_MODEL
            gemini_wrapper = GeminiModelWrapper(api_key=gemini_api_key, model_name=actual_model)
            actual_device = "cloud"

            # Gemini has 1M+ token limit; chunking is not required. Run direct summarization.
            summary_text = gemini_wrapper.generate(text_to_summarize, style=style)
        else:
            raise SummarizerException(f"Unsupported backend provider: {backend}")

        # Step 4: Translate summary back to target language if needed
        final_summary = summary_text
        if target_lang != "en" and backend == "t5":
            logger.info(f"Translating T5 English summary back to target language '{target_lang}'")
            try:
                final_summary = translate_text(summary_text, target_lang=target_lang, source_lang="en")
            except Exception as e:
                logger.error(f"Failed to translate T5 summary to '{target_lang}': {e}")
        elif target_lang != "en" and backend == "gemini":
            # Translate Gemini summary to target language if it wasn't English
            logger.info(f"Translating Gemini summary to target language '{target_lang}'")
            try:
                final_summary = translate_text(summary_text, target_lang=target_lang, source_lang="auto")
            except Exception as e:
                logger.error(f"Failed to translate Gemini summary to '{target_lang}': {e}")

        return {
            "summary": final_summary,
            "backend": backend,
            "model": actual_model,
            "device": actual_device,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "chunks_processed": chunks_count
        }
