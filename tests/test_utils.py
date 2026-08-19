import pytest

from app.core.exceptions import ValidationError
from app.utils.metrics import calculate_compression_ratio, calculate_counts, estimate_tokens
from app.utils.validators import validate_inference_params, validate_uploaded_file


def test_validate_inference_params_valid():
    # Should run without raising any error
    validate_inference_params(30, 150, 4, 2.0)

def test_validate_inference_params_invalid():
    # min_length < 0
    with pytest.raises(ValidationError, match="cannot be negative"):
        validate_inference_params(-5, 100, 4, 1.0)

    # max_length <= 0
    with pytest.raises(ValidationError, match="must be positive"):
        validate_inference_params(10, 0, 4, 1.0)

    # min_length > max_length
    with pytest.raises(ValidationError, match="cannot be greater than maximum"):
        validate_inference_params(150, 100, 4, 1.0)

    # beam count <= 0
    with pytest.raises(ValidationError, match="must be a positive integer"):
        validate_inference_params(10, 100, 0, 1.0)

    # length_penalty < 0
    with pytest.raises(ValidationError, match="cannot be negative"):
        validate_inference_params(10, 100, 4, -0.5)

def test_validate_uploaded_file_valid():
    # Validate PDF and Word files within sizes limit
    validate_uploaded_file("document.pdf", 5 * 1024 * 1024)
    validate_uploaded_file("report.docx", 1024)
    validate_uploaded_file("notes.txt", 100)

def test_validate_uploaded_file_invalid_extension():
    with pytest.raises(ValidationError, match="format.*is not supported"):
        validate_uploaded_file("virus.exe", 1024)

    with pytest.raises(ValidationError, match="lacks a valid file extension"):
        validate_uploaded_file("no_extension_file", 1024)

def test_validate_uploaded_file_excessive_size():
    # 11MB file (limit is 10MB)
    with pytest.raises(ValidationError, match="exceeds the safety threshold"):
        validate_uploaded_file("large.pdf", 11 * 1024 * 1024)

def test_metrics_calculations():
    # Word & Char counts
    text = "Hello world!"
    assert calculate_counts(text) == {"words": 2, "characters": 12}
    assert calculate_counts("") == {"words": 0, "characters": 0}

    # Estimate tokens (2 words * 1.33 = 2.66 -> 2)
    assert estimate_tokens(text) == 2
    assert estimate_tokens("") == 0

    # Compression ratio
    assert calculate_compression_ratio("one two three four", "one two") == 0.5
    assert calculate_compression_ratio("", "") == 0.0
    assert calculate_compression_ratio("one", "one two three") == 1.0  # maxes out at 1.0
