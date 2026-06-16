"""
feature_extractor.py — Stylometric feature extraction.

No external NLP libraries — every feature below is computed from raw text
using only stdlib (re, string, collections) and numpy for array packing.

Feature groups
--------------
Char-level     : counts of chars/digits/uppercase/punctuation marks,
                 frequency of common marks ( . , : ; ' " ? ! ), letter/digit ratio.
Word-level     : total words, average word length, long/short-word ratios,
                 function-word frequency.
Lexical        : type-token ratio, hapax-legomena ratio, Simpson's diversity index.
Position       : a sentence's relative location within its document.

Excluded by design (too compute-heavy or need careful tuning for this project):
word n-grams, POS-tag frequencies, TF-IDF, word embeddings, entropy, NMF.
"""

import re
import string
from collections import Counter
from typing import List

import numpy as np

from src.features.feature_vector import FeatureVector

# Function words: top-200 of the Google 10k English list
# https://github.com/first20hours/google-10000-english/blob/master/google-10000-english.txt
_TOP_200 = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "great", "man", "here", "between", "need", "large", "long", "little", "very", "still",
    "own", "big", "same", "right", "house", "world", "old", "too", "small", "place",
    "again", "without", "while", "begin", "might", "end", "against", "another", "government", "system",
    "each", "something", "hand", "group", "during", "much", "many", "such", "part", "number",
    "both", "few", "while", "under", "high", "life", "down", "really", "family", "point",
    "play", "lead", "state", "home", "water", "room", "mother", "area", "national", "money",
    "story", "fact", "month", "lot", "right", "study", "book", "eye", "job", "word",
    "business", "issue", "side", "kind", "head", "house", "service", "friend", "father", "power",
    "hour", "game", "line", "end", "member", "law", "car", "city", "community", "name",
    "president", "team", "minute", "idea", "kid", "body", "information", "back", "parent", "face",
    "others", "level", "office", "door", "health", "person", "art", "war", "history", "party",
    "result", "change", "morning", "reason", "research", "girl", "guy", "moment", "air", "teacher",
]

_FUNCTION_WORDS = frozenset(_TOP_200)

FEATURE_NAMES: List[str] = [
    # Group 1 — char-level basic counts & punctuation
    "n_chars", "n_words", "avg_word_len",
    "n_commas", "n_periods", "n_semicolons", "n_colons",
    "n_excl", "n_quest", "n_quotes", "n_parens", "n_hyphens",
    "punct_density",
    # Group 2 — char-level ratios
    "n_digits", "digit_ratio", "upper_ratio",
    # Group 3 — word-level function words
    "fw_count", "fw_ratio",
    # Group 4 — lexical richness
    "ttr", "hapax_ratio", "simpson_d",
    # Group 5 — word-level length bins
    "wl_short", "wl_medium", "wl_long", "wl_very_long",
    # Group 6 — word-level readability
    "avg_syllables", "long_word_ratio",
    # Group 7 — position
    "norm_position", "is_first", "is_last",
]


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _syllables(word: str) -> int:
    """Estimate syllable count by counting vowel-letter groups."""
    word = word.lower()
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def extract(sentence: str, position: int, n_total: int) -> np.ndarray:
    """Build the full feature vector for one sentence (all groups, in order)."""
    words = sentence.split()
    alpha = re.findall(r"[a-zA-Z]+", sentence)
    lower = re.findall(r"[a-z]+", sentence.lower())
    n_words = max(1, len(words))
    n_alpha = max(1, len(alpha))
    n_chars = max(1, len(sentence))

    # Group 1 — char-level basic counts & punctuation
    avg_wl = sum(len(w) for w in alpha) / n_alpha
    n_punct = sum(1 for c in sentence if c in string.punctuation)
    g1 = [
        len(sentence), len(re.findall(r"\S+", sentence)), len(words), avg_wl,
        sentence.count(","), sentence.count("."), sentence.count(";"),
        sentence.count(":"), sentence.count("!"), sentence.count("?"),
        sentence.count('"') + sentence.count("'"),
        sentence.count("(") + sentence.count(")"),
        sentence.count("-"),
        n_punct / n_words,
    ]

    # Group 2 — char-level ratios
    n_dig = sum(c.isdigit() for c in sentence)
    n_upp = sum(c.isupper() for c in sentence)
    g2 = [n_dig, n_dig / n_chars, n_upp / n_chars]

    # Group 3 — word-level function words
    fw = sum(1 for w in lower if w in _FUNCTION_WORDS)
    n_lower = max(1, len(lower))
    g3 = [fw, fw / n_lower]

    # Group 4 — lexical richness (TTR, hapax ratio, Simpson's D)
    if not lower:
        g4 = [0.0, 0.0, 0.0]
    else:
        counts = Counter(lower)
        n = len(lower)
        ttr = len(counts) / n
        hapax = sum(1 for c in counts.values() if c == 1) / n
        simp = sum(c * (c - 1) for c in counts.values()) / max(1, n * (n - 1))
        g4 = [ttr, hapax, simp]

    # Group 5 — word-level length bins
    bins = [0, 0, 0, 0]
    for w in alpha:
        wl = len(w)
        if wl <= 3: bins[0] += 1
        elif wl <= 6: bins[1] += 1
        elif wl <= 9: bins[2] += 1
        else: bins[3] += 1
    g5 = [b / n_alpha for b in bins]

    # Group 6 — word-level readability
    avg_syl = sum(_syllables(w) for w in alpha) / n_alpha
    long_r = sum(1 for w in alpha if len(w) >= 7) / n_alpha
    g6 = [avg_syl, long_r]

    # Group 7 — position within the document
    norm = position / max(1, n_total - 1)
    g7 = [norm, float(position == 0), float(position == n_total - 1)]

    return np.array(g1 + g2 + g3 + g4 + g5 + g6 + g7, dtype=np.float32)


# ── FeatureExtractor ──────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Computes stylometric feature vectors for sentences.

    Attributes
    ----------
    featureNames : List[str]  — names of each feature dimension
    numFeatures  : int        — total number of features

    collectFeatures(document) -> FeatureVector
        Primary API. Accepts any object with a `.sentences` attribute
        (e.g. data.document_loader.Document) and returns a FeatureVector
        whose `.values` has shape [n_sentences, numFeatures].
    """

    def __init__(self, **kwargs):
        self.featureNames: List[str] = FEATURE_NAMES
        self.numFeatures: int = len(FEATURE_NAMES)

        # legacy aliases used elsewhere in the codebase
        self.feature_names = self.featureNames
        self.n_features = self.numFeatures

    # ── main API ──────────────────────────────────────────────────

    def collectFeatures(self, document) -> FeatureVector:
        """Extract features for every sentence in *document* → FeatureVector."""
        sentences = document.sentences
        matrix = self.transform_document(sentences)
        return FeatureVector(values=matrix, names=list(self.featureNames))

    # ── per-group helpers (one method per feature group) ───────────

    def _basic_counts(self, sentence: str):
        """Group 1 — char/word counts and punctuation."""
        alpha = re.findall(r"[a-zA-Z]+", sentence)
        n_alpha = max(1, len(alpha))
        n_words = max(1, len(sentence.split()))
        avg_wl = sum(len(w) for w in alpha) / n_alpha
        n_punct = sum(1 for c in sentence if c in string.punctuation)
        return [
            len(sentence), len(re.findall(r"\S+", sentence)), n_words, avg_wl,
            sentence.count(","), sentence.count("."), sentence.count(";"),
            sentence.count(":"), sentence.count("!"), sentence.count("?"),
            sentence.count('"') + sentence.count("'"),
            sentence.count("(") + sentence.count(")"),
            sentence.count("-"),
            n_punct / n_words,
        ]

    def _char_ratios(self, sentence: str):
        """Group 2 — digit and uppercase ratios."""
        n_chars = max(1, len(sentence))
        n_dig = sum(c.isdigit() for c in sentence)
        n_upp = sum(c.isupper() for c in sentence)
        return [n_dig, n_dig / n_chars, n_upp / n_chars]

    def _function_words(self, sentence: str):
        """Group 3 — function-word frequency."""
        lower = re.findall(r"[a-z]+", sentence.lower())
        n_lower = max(1, len(lower))
        fw = sum(1 for w in lower if w in _FUNCTION_WORDS)
        return [fw, fw / n_lower]

    def _vocab_richness(self, sentence: str):
        """Group 4 — TTR, hapax ratio, Simpson's D."""
        lower = re.findall(r"[a-z]+", sentence.lower())
        if not lower:
            return [0.0, 0.0, 0.0]
        counts = Counter(lower)
        n = len(lower)
        ttr = len(counts) / n
        hapax = sum(1 for c in counts.values() if c == 1) / n
        simp = sum(c * (c - 1) for c in counts.values()) / max(1, n * (n - 1))
        return [ttr, hapax, simp]

    def _word_length_bins(self, sentence: str):
        """Group 5 — proportion of words in each length bucket."""
        alpha = re.findall(r"[a-zA-Z]+", sentence)
        n_alpha = max(1, len(alpha))
        bins = [0, 0, 0, 0]
        for w in alpha:
            wl = len(w)
            if wl <= 3: bins[0] += 1
            elif wl <= 6: bins[1] += 1
            elif wl <= 9: bins[2] += 1
            else: bins[3] += 1
        return [b / n_alpha for b in bins]

    def _readability(self, sentence: str):
        """Group 6 — average syllables and long-word ratio."""
        alpha = re.findall(r"[a-zA-Z]+", sentence)
        n_alpha = max(1, len(alpha))
        avg_syl = sum(_syllables(w) for w in alpha) / n_alpha
        long_r = sum(1 for w in alpha if len(w) >= 7) / n_alpha
        return [avg_syl, long_r]

    def _position_features(self, position: int, n_total: int):
        """Group 7 — relative position in document."""
        norm = position / max(1, n_total - 1)
        return [norm, float(position == 0), float(position == n_total - 1)]

    # ── sklearn-compatible interface (used by app.py / training) ──

    def fit(self, sentences):
        """No-op — extractor is stateless, kept for pipeline compatibility."""
        return self

    def transform(self, sentence: str, position: int, n_total: int) -> np.ndarray:
        """Feature vector for a single sentence (1-D float32 array)."""
        return extract(sentence, position, n_total)

    def transform_document(self, sentences: List[str]) -> np.ndarray:
        """Feature matrix for all sentences (2-D float32, shape [n, numFeatures])."""
        n = len(sentences)
        return np.stack(
            [extract(s, i, n) for i, s in enumerate(sentences)],
            axis=0,
        )


# ── Entry-point example ───────────────────────────────────────────────────────

if __name__ == "__main__":
    sentence = "This is an example sentence, with punctuation!"
    features = extract(sentence, position=0, n_total=1)
    print("Raw feature array:", features)

    fe = FeatureExtractor()

    class _MockDoc:
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
        ]

    fv = fe.collectFeatures(_MockDoc())
    print(f"\nFeatureVector: {fv}")
    print(f"Shape: {fv.values.shape}")
