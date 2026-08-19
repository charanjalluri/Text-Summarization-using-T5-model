from unittest.mock import MagicMock, patch

from app.services.summarization import SummarizationService


@patch("app.services.summarization.get_t5_model_cached")
@patch("app.services.summarization.detect_language")
def test_summarize_t5_direct(mock_detect_lang, mock_get_t5):
    # Setup mocks
    mock_detect_lang.return_value = "en"

    mock_t5_wrapper = MagicMock()
    mock_t5_wrapper.device = "cpu"
    # Token length < 512 limit
    mock_t5_wrapper.tokenizer.encode.return_value = [1] * 100
    mock_t5_wrapper.generate.return_value = "Direct T5 Summary Text"

    mock_get_t5.return_value = mock_t5_wrapper

    service = SummarizationService()
    results = service.summarize("some document content to summarize", backend="t5")

    assert results["summary"] == "Direct T5 Summary Text"
    assert results["backend"] == "t5"
    assert results["device"] == "cpu"
    assert results["chunks_processed"] == 1
    mock_t5_wrapper.generate.assert_called_once()

@patch("app.services.summarization.get_t5_model_cached")
@patch("app.services.summarization.detect_language")
@patch("app.services.summarization.chunk_text_by_tokens")
def test_summarize_t5_map_reduce(mock_chunk_tokens, mock_detect_lang, mock_get_t5):
    # Setup mocks
    mock_detect_lang.return_value = "en"

    mock_t5_wrapper = MagicMock()
    mock_t5_wrapper.device = "cuda"
    # Simulate long input: length check return 600 tokens (limit is 512), combined summary is 100 tokens
    mock_t5_wrapper.tokenizer.encode.side_effect = [
        [1] * 600,  # first length check
        [1] * 100,  # combined summaries length check
    ]
    mock_t5_wrapper.generate.side_effect = [
        "Summary of chunk 1.",
        "Summary of chunk 2."
    ]
    mock_get_t5.return_value = mock_t5_wrapper

    # 2 mock chunks
    mock_chunk_tokens.return_value = ["chunk1 text", "chunk2 text"]

    service = SummarizationService()
    results = service.summarize("a very long document content", backend="t5")

    # Joined chunks text: "Summary of chunk 1. Summary of chunk 2."
    assert results["summary"] == "Summary of chunk 1. Summary of chunk 2."
    assert results["backend"] == "t5"
    assert results["device"] == "cuda"
    assert results["chunks_processed"] == 2
    assert mock_t5_wrapper.generate.call_count == 2

@patch("app.services.summarization.GeminiModelWrapper")
@patch("app.services.summarization.detect_language")
def test_summarize_gemini(mock_detect_lang, mock_gemini_wrapper):
    mock_detect_lang.return_value = "en"

    mock_gemini = MagicMock()
    mock_gemini.generate.return_value = "Gemini generated summary text"
    mock_gemini_wrapper.return_value = mock_gemini

    service = SummarizationService()
    results = service.summarize("document to summarize using gemini", backend="gemini")

    assert results["summary"] == "Gemini generated summary text"
    assert results["backend"] == "gemini"
    assert results["device"] == "cloud"
    mock_gemini.generate.assert_called_once()

@patch("app.services.summarization.get_t5_model_cached")
@patch("app.services.summarization.detect_language")
@patch("app.services.summarization.chunk_text_by_tokens")
def test_summarize_t5_map_reduce_length_clash(mock_chunk_tokens, mock_detect_lang, mock_get_t5):
    mock_detect_lang.return_value = "en"

    mock_t5_wrapper = MagicMock()
    mock_t5_wrapper.device = "cpu"
    mock_t5_wrapper.tokenizer.encode.side_effect = [
        [1] * 600,  # first length check
        [1] * 100,  # combined summaries length check
    ]
    mock_t5_wrapper.generate.side_effect = [
        "Summary of chunk 1.",
        "Summary of chunk 2."
    ]
    mock_get_t5.return_value = mock_t5_wrapper
    mock_chunk_tokens.return_value = ["chunk1 text", "chunk2 text"]

    service = SummarizationService()
    results = service.summarize(
        "a very long document content",
        backend="t5",
        min_length=60,
        max_length=100
    )

    assert results["chunks_processed"] == 2
    mock_t5_wrapper.generate.assert_any_call(
        "chunk1 text",
        max_length=60,
        min_length=60,
        num_beams=None,
        length_penalty=None
    )

import pytest

from app.core.exceptions import ModelLoadError
from app.models.t5_model import T5ModelWrapper


@patch("app.models.t5_model.torch.cuda.is_available")
@patch("app.models.t5_model.T5Tokenizer.from_pretrained")
@patch("app.models.t5_model.T5ForConditionalGeneration.from_pretrained")
def test_t5_model_explicit_cuda_unavailable(mock_model_from_pretrained, mock_tokenizer_from_pretrained, mock_cuda_available):
    mock_cuda_available.return_value = False

    wrapper = T5ModelWrapper(model_name="t5-small", device="cuda")

    with pytest.raises(ModelLoadError, match="CUDA requested but CUDA is not available"):
        wrapper.load()
