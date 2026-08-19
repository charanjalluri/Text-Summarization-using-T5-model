from app.services.text_processing import chunk_text_by_tokens, chunk_text_by_words, clean_text


def test_clean_text_normal():
    text = "Hello world, this is a clean sentence."
    assert clean_text(text) == text

def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""

def test_clean_text_excessive_whitespace():
    dirty_text = "   Hello   world! \t  This \n\n\n has newlines.   "
    expected = "Hello world! This\n\nhas newlines."
    assert clean_text(dirty_text) == expected

def test_clean_text_unicode():
    unicode_text = "Thîs ĩs ûnicodê text with café."
    assert clean_text(unicode_text) == unicode_text

def test_chunk_text_by_words_basic():
    text = "one two three four five six"
    # Chunk size 2, overlap 1 -> expected: ["one two", "two three", "three four", "four five", "five six"]
    chunks = chunk_text_by_words(text, chunk_size=2, overlap=1)
    assert len(chunks) == 5
    assert chunks[0] == "one two"
    assert chunks[-1] == "five six"

def test_chunk_text_by_words_no_overlap():
    text = "one two three four five"
    # Chunk size 2, overlap 0 -> expected: ["one two", "three four", "five"]
    chunks = chunk_text_by_words(text, chunk_size=2, overlap=0)
    assert chunks == ["one two", "three four", "five"]

def test_chunk_text_by_words_very_short():
    text = "short text"
    chunks = chunk_text_by_words(text, chunk_size=10, overlap=2)
    assert chunks == ["short text"]

class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        # Return list of lengths of words as mock token ids
        return [len(w) for w in text.split()]

    def decode(self, ids, skip_special_tokens=True):
        # Decode back dummy string
        return " ".join([str(i) for i in ids])

def test_chunk_text_by_tokens_fallback():
    # If tokenizer fails, chunk_text_by_tokens falls back to word chunking
    text = "one two three four five"
    chunks = chunk_text_by_tokens(text, tokenizer=None, max_tokens=2, overlap=0)
    # Expected word fallback (max_tokens * 0.75 = 1 word chunks)
    assert len(chunks) > 0

def test_chunk_text_by_tokens_success():
    tokenizer = DummyTokenizer()
    text = "one two three"
    # encoded: [3, 3, 5] (len of words)
    # chunk max_tokens = 2, overlap = 1 -> indices: 0:2, 1:3 -> [3, 3], [3, 5]
    # decoded: "3 3", "3 5"
    chunks = chunk_text_by_tokens(text, tokenizer=tokenizer, max_tokens=2, overlap=1)
    assert chunks == ["3 3", "3 5"]
