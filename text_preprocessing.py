"""
text_preprocessing.py
---------------------
NLP text preprocessing utilities for resume and job description text.
Pipeline: lowercase → remove punctuation → remove stopwords → strip whitespace.
"""

import re
import string
from typing import Set

# ---------------------------------------------------------------------------
# Stopword list (built-in fallback — no NLTK download required)
# ---------------------------------------------------------------------------
_BUILTIN_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "ought", "used", "i", "me", "my", "myself", "we", "our", "ours",
    "ourselves", "you", "your", "yours", "yourself", "yourselves", "he",
    "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "they", "them", "their", "theirs", "themselves", "what",
    "which", "who", "whom", "this", "that", "these", "those", "am", "not",
    "no", "nor", "so", "yet", "both", "either", "neither", "as", "if",
    "than", "too", "very", "just", "s", "t", "don", "now", "about",
    "above", "after", "before", "between", "into", "through", "during",
    "also", "each", "more", "most", "other", "some", "such", "only",
    "own", "same", "then", "there", "when", "where", "how", "all", "any",
    "because", "up", "out", "over", "under", "again", "further", "while",
    "here", "why", "both", "few", "against", "once",
}


def _load_stopwords() -> Set[str]:
    """
    Attempt to load NLTK stopwords; fall back to built-in set if unavailable.
    """
    try:
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception:
        return _BUILTIN_STOPWORDS


# Load once at module import time
STOPWORDS: Set[str] = _load_stopwords()


# ---------------------------------------------------------------------------
# Core preprocessing function
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline.

    Steps:
      1. Lowercase all characters
      2. Remove punctuation
      3. Remove numeric tokens (optional but reduces noise)
      4. Remove stopwords
      5. Collapse extra whitespace

    Args:
        text: Raw input string.

    Returns:
        Clean, preprocessed string ready for vectorization.
    """
    if not text or not text.strip():
        return ""

    # Step 1 – Lowercase
    text = text.lower()

    # Step 2 – Remove punctuation (replace with space to avoid word-merging)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    # Step 3 – Remove digits (numbers rarely aid keyword matching for resumes)
    text = re.sub(r"\d+", " ", text)

    # Step 4 – Tokenise and remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    # Step 5 – Rejoin and strip extra whitespace
    clean_text = " ".join(tokens)
    return clean_text


def extract_keywords(text: str, top_n: int = 30) -> Set[str]:
    """
    Return a set of meaningful keywords from preprocessed text.

    Preprocessing is applied internally so raw text can be passed directly.

    Args:
        text:  Raw or preprocessed text.
        top_n: Maximum number of keywords to return (most frequent).

    Returns:
        A set of keyword strings (already preprocessed).
    """
    clean = preprocess_text(text)
    tokens = clean.split()

    if not tokens:
        return set()

    # Count frequency and return the top_n most common tokens
    from collections import Counter
    freq = Counter(tokens)
    top_keywords = {word for word, _ in freq.most_common(top_n)}
    return top_keywords


def get_token_set(text: str) -> Set[str]:
    """
    Return the full set of unique preprocessed tokens from the text.
    Used for keyword intersection/difference analysis.
    """
    clean = preprocess_text(text)
    return set(clean.split())
