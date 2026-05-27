""" Utility functions for text processing, dataset construction, and metadata extraction.
"""

from flask import request
import numpy as np
import numpy as np

from features.extractor import extract


def build_pairwise_dataset(problems, feature_extractor):
    """
    Build feature matrix X, label vector y, and metadata for all problems.
    """
    X_rows = []
    y_rows = []
    meta = []

    for problem in problems:
        sentences = problem["sentences"]
        changes = problem["changes"]

        # Skip problems with fewer than 5 sentences to ensure enough data for training
        if len(sentences) < 5:
            continue

        vecs = feature_extractor.transform_document(sentences)

        for i, label in enumerate(changes):

            # Absolute difference of feature vectors
            diff = np.abs(vecs[i] - vecs[i + 1])
            X_rows.append(diff)
            y_rows.append(label)
            meta.append(
                {
                    "problem_id": problem["problem_id"],
                    "difficulty": problem["difficulty"],
                    "pair_index": i,
                }
            )

    if not X_rows:
        raise ValueError(
            "No pairs found. Check that the dataset loaded correctly.")

    X = np.stack(X_rows, axis=0).astype(np.float32)
    y = np.array(y_rows, dtype=np.int32)
    return X, y, meta


def split_sentences(text):
    """Split raw text into a list of non-empty sentences (one per line)."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def text_from_request(text_field="text", file_field="file"):
    """ 
        Pull raw text from either a textarea or an uploaded .txt file.
    """
    uploaded = request.files.get(file_field)
    if uploaded and uploaded.filename:
        if not uploaded.filename.lower().endswith(".txt"):
            return None, "Only plain .txt files are supported for upload."
        try:
            return uploaded.read().decode("utf-8"), None
        except Exception as e:
            return None, f"Could not read file: {e}"

    text = request.form.get(text_field, "").strip()
    if text:
        return text, None

    return None, "Please paste some text or upload a .txt file."


def text_stats(sentences):
    """Return a simple summary dict for a list of sentences."""
    if not sentences:
        return {}

    all_words = [w for s in sentences for w in s.split()]
    word_counts = [len(s.split()) for s in sentences]
    char_counts = [len(s) for s in sentences]
    unique_words = len(set(w.lower() for w in all_words))

    return {
        "n_sentences": len(sentences),
        "n_words": len(all_words),
        "unique_words": unique_words,
        "vocabulary_richness": round(unique_words / max(1, len(all_words)) * 100, 1),
        "avg_words_per_sent": round(sum(word_counts) / len(word_counts), 1),
        "shortest_sent": min(word_counts),
        "longest_sent": max(word_counts),
        "avg_chars_per_sent": round(sum(char_counts) / len(char_counts), 1),
    }


def sentence_details(sentences, max_display=60):
    """
    Return a list of per-sentence dicts with key stylometric values,
    plus a style-distance score between each pair of consecutive sentences.
    Capped at max_display sentences for readability.
    """
    display = sentences[:max_display]
    n = len(display)
    details = []

    for i, s in enumerate(display):
        vec = extract(s, i, n)
        # Pick a handful of readable feature values for display
        details.append({
            "index": i + 1,
            "text": s,
            "words": len(s.split()),
            "avg_word_len": round(float(vec[3]), 1),
            "punct_density": round(float(vec[13]), 2),
            "fw_ratio": round(float(vec[18]), 2),
            "ttr": round(float(vec[19]), 2),
            "avg_syllables": round(float(vec[26]), 1),
        })

    # Style distance between consecutive sentences (sum of abs feature diffs)
    # Normalised to 0–100 for the progress-bar display
    if n >= 2:
        vecs = np.stack([extract(s, i, n) for i, s in enumerate(display)])
        diffs = np.abs(vecs[:-1] - vecs[1:]).sum(axis=1)
        max_diff = diffs.max() if diffs.max() > 0 else 1.0
        for i, d in enumerate(details[:-1]):
            d["style_distance"] = round(float(diffs[i] / max_diff * 100), 1)
        details[-1]["style_distance"] = None
    else:
        for d in details:
            d["style_distance"] = None

    return details, len(sentences) > max_display
