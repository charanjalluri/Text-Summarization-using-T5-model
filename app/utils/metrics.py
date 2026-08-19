from typing import Dict


def calculate_counts(text: str) -> Dict[str, int]:
    """Calculates word and character counts in a text block."""
    if not text or not text.strip():
        return {"words": 0, "characters": 0}

    words = len(text.split())
    chars = len(text)
    return {"words": words, "characters": chars}

def estimate_tokens(text: str) -> int:
    """Approximates token count based on typical English language ratios (1 word ~ 1.33 tokens)."""
    if not text or not text.strip():
        return 0
    words = len(text.split())
    return int(words * 1.33)

def calculate_compression_ratio(input_text: str, summary_text: str) -> float:
    """Calculates the compression ratio (summary word count / input word count)."""
    if not input_text or not input_text.strip() or not summary_text or not summary_text.strip():
        return 0.0

    input_words = len(input_text.split())
    summary_words = len(summary_text.split())

    if input_words == 0:
        return 0.0

    return min(1.0, summary_words / input_words)
