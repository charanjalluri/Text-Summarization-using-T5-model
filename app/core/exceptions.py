class SummarizerException(Exception):
    """Base exception class for all custom summarizer exceptions."""
    pass

class ModelLoadError(SummarizerException):
    """Exception raised when loading local T5 or Gemini models fails."""
    pass

class InferenceError(SummarizerException):
    """Exception raised when running inference fails."""
    pass

class FileExtractionError(SummarizerException):
    """Exception raised when extracting text from a file fails."""
    pass

class TranslationError(SummarizerException):
    """Exception raised when language detection or translation fails."""
    pass

class ValidationError(SummarizerException):
    """Exception raised when parameter validation fails."""
    pass

class GeminiAPIError(SummarizerException):
    """Exception raised when calling the Gemini API fails."""
    pass
