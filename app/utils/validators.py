from app.core.exceptions import ValidationError

ALLOWED_EXTENSIONS = {"txt", "text", "md", "pdf", "docx", "wav", "mp3", "m4a", "flac", "png", "jpg", "jpeg", "tiff", "bmp"}
MAX_FILE_SIZE_MB = 10.0  # 10 MB upload threshold

def validate_inference_params(min_len: int, max_len: int, num_beams: int, length_penalty: float) -> None:
    """Validates that text generation configuration is mathematically sound."""
    if min_len < 0:
        raise ValidationError(f"Minimum length ({min_len}) cannot be negative.")
    if max_len <= 0:
        raise ValidationError(f"Maximum length ({max_len}) must be positive.")
    if min_len > max_len:
        raise ValidationError(f"Minimum length ({min_len}) cannot be greater than maximum length ({max_len}).")
    if num_beams <= 0:
        raise ValidationError(f"Beam count ({num_beams}) must be a positive integer.")
    if length_penalty < 0:
        raise ValidationError(f"Length penalty ({length_penalty}) cannot be negative.")

def validate_uploaded_file(filename: str, file_size_bytes: int) -> None:
    """Secures applications against unsafe extensions or memory exhaustion by verifying uploads."""
    if not filename:
        raise ValidationError("Uploaded file lacks a valid filename.")

    parts = filename.split(".")
    if len(parts) < 2:
        raise ValidationError(f"Uploaded file '{filename}' lacks a valid file extension.")

    ext = parts[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"File format '.{ext}' is not supported. "
            f"Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValidationError(
            f"Uploaded file size ({file_size_mb:.2f}MB) exceeds the safety threshold of {MAX_FILE_SIZE_MB}MB."
        )
