import os
from typing import Optional

import google.generativeai as genai

from app.core.config import settings
from app.core.exceptions import GeminiAPIError
from app.core.logging_config import get_logger

logger = get_logger(__name__)

class GeminiModelWrapper:
    def __init__(self, api_key: Optional[str] = None, model_name: str = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.GEMINI_MODEL
        self._configured = False

    def configure(self) -> None:
        """Configures the google-generativeai SDK with the API key."""
        if self._configured:
            return

        key = self.api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise GeminiAPIError("Gemini API Key is not configured. Please supply an API key.")

        try:
            logger.info("Configuring google-generativeai SDK.")
            genai.configure(api_key=key)
            self._configured = True
            logger.info("google-generativeai SDK configured successfully.")
        except Exception as e:
            logger.error(f"Failed to configure Gemini SDK: {e}", exc_info=True)
            raise GeminiAPIError(f"Failed to configure Gemini SDK: {e}") from e

    def generate(self, text: str, style: str = "concise") -> str:
        """Generates a summary using Gemini with formatting style instruction."""
        if not text.strip():
            return ""

        self.configure()

        try:
            logger.info(f"Generating summary with Gemini model {self.model_name} (style={style})")

            # Style prompts mapping
            style_prompts = {
                "concise": "Summarize this text in a concise, high-level summary.",
                "bullet-heavy": "Summarize this text using detailed, structured bullet points.",
                "academic": "Provide a formal, academic summary of this text highlighting key hypotheses and conclusions.",
                "narrative": "Summarize this text in a narrative story-driven format."
            }
            instruction = style_prompts.get(style, "Summarize this text.")
            prompt = f"{instruction}\n\nText:\n{text}"

            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)

            if not response or not response.text:
                raise GeminiAPIError("Gemini returned an empty response.")

            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API generation error: {e}", exc_info=True)
            raise GeminiAPIError(f"Gemini API error: {e}") from e
