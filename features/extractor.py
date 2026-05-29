""" Stylometric features.
"""

import re
import string
from collections import Counter

import numpy as np

# Function words are taken from the top-200 words of
# https://github.com/first20hours/google-10000-english/blob/master/google-10000-english.txt English words (Google 10k list)

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

FEATURE_NAMES = [
    # Group 1 — basic counts (always keep these)
    "n_chars", "n_tokens", "n_words", "avg_word_len",
    "n_commas", "n_periods", "n_semicolons", "n_colons",
    "n_excl", "n_quest", "n_quotes", "n_parens", "n_hyphens",
    "punct_density",
    # Group 2 — character ratios
    "n_digits", "digit_ratio", "upper_ratio",
    # Group 3 — function words
    "fw_count", "fw_ratio",
    # Group 4 — vocabulary richness
    "ttr", "hapax_ratio", "simpson_d",
    # Group 5 — word length bins
    "wl_short", "wl_medium", "wl_long", "wl_very_long",
    # Group 6 — readability
    "avg_syllables", "long_word_ratio",
    # Group 7 — position
    "norm_position", "is_first", "is_last",
]


def _syllables(word):
    """Rough syllable count using vowel groups."""
    word = word.lower()
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def extract(sentence, position, n_total):
    """Return a 1-D float32 feature vector for one sentence."""
    words = sentence.split()
    alpha = re.findall(r"[a-zA-Z]+", sentence)
    lower = re.findall(r"[a-z]+", sentence.lower())
    n_words = max(1, len(words))
    n_alpha = max(1, len(alpha))
    n_chars = max(1, len(sentence))

    # Group 1 — basic counts
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

    # Group 2 — character ratios (comment out if too much)
    n_dig = sum(c.isdigit() for c in sentence)
    n_upp = sum(c.isupper() for c in sentence)
    g2 = [n_dig, n_dig / n_chars, n_upp / n_chars]

    # Group 3 — function words (based on Google top-200)
    fw = sum(1 for w in lower if w in _FUNCTION_WORDS)
    n_lower = max(1, len(lower))
    g3 = [fw, fw / n_lower]

    # Group 4 — vocabulary richness (drop for simplicity)
    if not lower:
        g4 = [0.0, 0.0, 0.0]
    else:
        counts = Counter(lower)
        n = len(lower)
        ttr = len(counts) / n
        hapax = sum(1 for c in counts.values() if c == 1) / n
        simp = sum(c * (c - 1) for c in counts.values()) / max(1, n * (n - 1))
        g4 = [ttr, hapax, simp]

    # Group 5 — word length bins (drop for simplicity)
    bins = [0, 0, 0, 0]
    for w in alpha:
        wl = len(w)
        if wl <= 3:
            bins[0] += 1
        elif wl <= 6:
            bins[1] += 1
        elif wl <= 9:
            bins[2] += 1
        else:
            bins[3] += 1
    g5 = [b / n_alpha for b in bins]

    # Group 6 — readability (drop for simplicity)
    avg_syl = sum(_syllables(w) for w in alpha) / n_alpha
    long_r = sum(1 for w in alpha if len(w) >= 7) / n_alpha
    g6 = [avg_syl, long_r]

    # Group 7 — position (keep – very useful for text structure)
    norm = position / max(1, n_total - 1)
    g7 = [norm, float(position == 0), float(position == n_total - 1)]

    return np.array(g1 + g2 + g3 + g4 + g5 + g6 + g7, dtype=np.float32)


class FeatureExtractor:
    """
    Thin wrapper so the rest of the codebase can call fe.fit() and
    fe.transform() without knowing about the implementation details.

    """

    def __init__(self, **kwargs):
        self.n_features = len(FEATURE_NAMES)
        self.feature_names = FEATURE_NAMES

    def fit(self, sentences):
        return self

    def transform(self, sentence, position, n_total):
        return extract(sentence, position, n_total)

    def transform_document(self, sentences):
        n = len(sentences)
        return np.stack(
            [extract(s, i, n) for i, s in enumerate(sentences)],
            axis=0,
        )

if __name__ == "__main__":
    # usage example
    sentence = "This is an example sentence, with punctuation!"
    features = extract(sentence, position=0, n_total=1)
    print("Features:", features)
    