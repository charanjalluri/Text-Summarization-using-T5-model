import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from app.core.config import settings
from app.core.exceptions import InferenceError, ModelLoadError
from app.core.logging_config import get_logger

logger = get_logger(__name__)

class T5ModelWrapper:
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or settings.MODEL_NAME
        self._req_device = device or settings.DEVICE
        self.tokenizer = None
        self.model = None
        self.device = "cpu"

    def load(self) -> None:
        """Loads tokenizer and model, mapping to the appropriate device."""
        try:
            logger.info(f"Loading T5 tokenizer: {self.model_name}")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)

            logger.info(f"Loading T5 model: {self.model_name}")
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

            # Device selection logic
            if self._req_device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self._req_device
                if self.device == "cuda" and not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available. Falling back to CPU.")
                    self.device = "cpu"

            logger.info(f"Moving T5 model to device: {self.device}")
            self.model.to(self.device)
            self.model.eval()  # Put model in evaluation mode
            logger.info("T5 model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load T5 model {self.model_name}: {e}", exc_info=True)
            raise ModelLoadError(f"Could not load T5 model '{self.model_name}': {e}") from e

    def generate(
        self,
        text: str,
        max_length: int = None,
        min_length: int = None,
        num_beams: int = None,
        length_penalty: float = None
    ) -> str:
        """Generates a summary using T5 model with token constraints and beam search."""
        if not self.model or not self.tokenizer:
            self.load()

        max_len = max_length or settings.MAX_SUMMARY_TOKENS
        min_len = min_length or settings.MIN_SUMMARY_TOKENS
        beams = num_beams or settings.NUM_BEAMS
        penalty = length_penalty or settings.LENGTH_PENALTY

        if min_len > max_len:
            raise InferenceError(f"Minimum length ({min_len}) cannot be greater than maximum length ({max_len})")

        try:
            # Prefix required for standard T5 models
            input_text = "summarize: " + text

            inputs = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=settings.MAX_INPUT_TOKENS,
                truncation=True
            ).to(self.device)

            with torch.inference_mode():
                out_ids = self.model.generate(
                    inputs,
                    max_length=max_len,
                    min_length=min_len,
                    num_beams=beams,
                    length_penalty=penalty,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )

            summary = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            return summary.strip()
        except Exception as e:
            logger.error(f"T5 generation error: {e}", exc_info=True)
            raise InferenceError(f"Error during local T5 inference: {e}") from e

@st.cache_resource
def get_t5_model_cached(model_name: str, device: str) -> T5ModelWrapper:
    """Loads and caches the T5 model instance for Streamlit resource efficiency."""
    wrapper = T5ModelWrapper(model_name=model_name, device=device)
    wrapper.load()
    return wrapper
