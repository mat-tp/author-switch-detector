"""
    data_processing.py — DocumentLoader and Document classes
    Loads PAN 2026 MAWSA problems from disk, validates files,
    converts PDFs, and collects file statistics.
"""

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

# Regular expression to match problem files.
PROBLEM_FILE = re.compile(r"^problem-(\d+)\.txt$")


# ── Enums ─────────────────────────────────────────────────────────────────────

class FileType(Enum):
    PDF = "PDF"
    TXT = "TXT"


class DocLevel(Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ── Document ──────────────────────────────────────────────────────────────────

@dataclass
class Document:
    """Represents a single loaded document."""
    id: str                                       # problem_id or filename stem
    rawText: str                                  # original file content
    cleanedText: str                              # whitespace-normalised text
    sentences: List[str]                          # one sentence per element
    metadata: Dict[str, object] = field(default_factory=dict)  # difficulty, path, …
    fileType: FileType = FileType.TXT


# ── FeatureVector ─────────────────────────────────────────────────────────────

@dataclass
class FeatureVector:
    """Thin wrapper around a numpy array + feature name list."""
    values: list          # float[] / numpy array
    names: List[str]


# ── DocumentLoader ────────────────────────────────────────────────────────────

class DocumentLoader:
    """
    Responsibilities:
        - load files  (TXT and PDF)
        - validate file types
        - convert PDF to plain text
        - collect file statistics

    Attributes
    ----------
    fileCount     : int   – number of files loaded in this session
    sentenceCount : int   – total sentences across all loaded documents
    docLevel      : DocLevel – difficulty level applied to the next load call
    """

    SUPPORTED_EXTENSIONS = {".txt", ".pdf"}

    def __init__(self, docLevel: DocLevel = DocLevel.EASY):
        self.fileCount: int       = 0
        self.sentenceCount: int   = 0
        self.docLevel: DocLevel   = docLevel
        self._loaded: List[Document] = []

    # ── public API ──────────────────────────────────────────────

    def loadFile(self, path: str) -> Optional[Document]:
        """
        Load a single .txt or .pdf file and return a Document.
        Returns None if the file cannot be loaded.
        """
        if not self.validateFile(path):
            return None

        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            raw = self.convertPdfToText(path)
            ftype = FileType.PDF
        else:
            try:
                with open(path, encoding="utf-8") as fh:
                    raw = fh.read()
            except Exception as exc:
                print(f"[DocumentLoader] Could not read {path}: {exc}")
                return None
            ftype = FileType.TXT

        text = self.extractText(raw)
        sents = _get_sentences(text)
        doc_id = os.path.splitext(os.path.basename(path))[0]

        doc = Document(
            id=doc_id,
            rawText=raw,
            cleanedText=text,
            sentences=sents,
            metadata={"path": path, "docLevel": self.docLevel.value},
            fileType=ftype,
        )
        self.fileCount     += 1
        self.sentenceCount += len(sents)
        self._loaded.append(doc)
        return doc

    def validateFile(self, path: str) -> bool:
        """Return True iff the file exists and has a supported extension."""
        if not os.path.isfile(path):
            print(f"[DocumentLoader] File not found: {path}")
            return False
        ext = os.path.splitext(path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            print(f"[DocumentLoader] Unsupported type '{ext}' for {path}")
            return False
        return True

    def convertPdfToText(self, pdfPath: str) -> str:
        """
        Extract plain text from a PDF using pdfminer.six.
        Returns an empty string if extraction fails.
        """
        try:
            from pdfminer.high_level import extract_text as _extract
            return _extract(pdfPath) or ""
        except Exception as exc:
            print(f"[DocumentLoader] PDF conversion failed for {pdfPath}: {exc}")
            return ""

    def extractText(self, raw: str) -> str:
        """Normalise whitespace in raw text."""
        lines = [l.strip() for l in raw.splitlines()]
        return "\n".join(l for l in lines if l)

    def getStats(self) -> dict:
        """Return a summary of all documents loaded so far."""
        if not self._loaded:
            return {}
        all_sents = [s for doc in self._loaded for s in doc.sentences]
        word_lens = [len(s.split()) for s in all_sents]
        return {
            "fileCount":       self.fileCount,
            "sentenceCount":   self.sentenceCount,
            "docLevel":        self.docLevel.value,
            "n_documents":     len(self._loaded),
            "avg_sents_per_doc":  round(self.sentenceCount / max(1, len(self._loaded)), 1),
            "avg_words_per_sent": round(sum(word_lens) / max(1, len(word_lens)), 1),
        }


# ── Legacy helpers (kept for backward-compatibility with app.py) ──────────────

def _get_sentences(text: str) -> List[str]:
    """Split normalised text into non-empty lines (one sentence per line)."""
    return [l.strip() for l in text.splitlines() if l.strip()]


# public alias used by the rest of the codebase
def get_sentences(text: str) -> List[str]:
    return _get_sentences(text)


def load_problem(txt_path: str, truth_path: str, difficulty: str) -> Optional[dict]:
    """Load one problem (text) and truth .json files."""
    if not os.path.exists(truth_path):
        print(f"[loader] Missing truth file: {truth_path} — skipping.")
        return None

    try:
        with open(txt_path, encoding="utf-8") as f:
            sentences = get_sentences(f.read())
        with open(truth_path, encoding="utf-8") as f:
            changes = [int(c) for c in json.load(f)["changes"]]
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"[loader] Error reading {txt_path}: {e} — skipping.")
        return None

    if len(sentences) - 1 != len(changes):
        print(
            f"[loader] Skipping {os.path.basename(txt_path)}: "
            f"{len(sentences)} sentences but {len(changes)} labels."
        )
        return None

    problem_id = re.search(r"problem-(\d+)\.txt$", txt_path).group(1)
    return {
        "problem_id": problem_id,
        "difficulty": difficulty,
        "sentences":  sentences,
        "changes":    changes,
    }


def load_split(path: str, difficulty: str) -> List[dict]:
    """Load all problems from one directory."""
    problems = []
    if not os.path.isdir(path):
        print(f"[loader] Directory not found: {path}")
        return problems
    for fname in sorted(os.listdir(path)):
        match = PROBLEM_FILE.match(fname)
        if not match:
            continue
        pid        = match.group(1)
        txt_path   = os.path.join(path, fname)
        truth_path = os.path.join(path, f"truth-problem-{pid}.json")
        problem    = load_problem(txt_path, truth_path, difficulty)
        if problem:
            problems.append(problem)
    return problems


def load_all(root: str, difficulties=("easy", "medium", "hard"), splits=("train",)) -> dict:
    """Load all difficulty/split combinations."""
    data = {}
    for diff in difficulties:
        for split in splits:
            key        = f"{diff}_{split}"
            path       = os.path.join(root, diff, split)
            data[key]  = load_split(path, diff)
            print(f"[loader] {key}: {len(data[key])} problems loaded.")
    return data


def dataset_stats(problems: List[dict]) -> dict:
    """Return a summary dict for a list of problems."""
    if not problems:
        return {}
    n_probs    = len(problems)
    n_sents    = sum(len(p["sentences"]) for p in problems)
    n_pairs    = sum(len(p["changes"])   for p in problems)
    n_switches = sum(sum(p["changes"])   for p in problems)
    word_lens  = [len(s.split()) for p in problems for s in p["sentences"]]
    avg_words  = round(sum(word_lens) / len(word_lens), 1) if word_lens else 0.0
    return {
        "n_problems":           n_probs,
        "n_sentences":          n_sents,
        "n_pairs":              n_pairs,
        "n_switches":           n_switches,
        "switch_rate":          round(n_switches / n_pairs * 100, 2) if n_pairs else 0.0,
        "avg_sents_per_problem": round(n_sents / n_probs, 1),
        "avg_words_per_sent":   avg_words,
    }


# ── Entry-point example ───────────────────────────────────────────────────────

if __name__ == "__main__":
    root_dir = "mawsa26-pan-zenodo-DATA"
    data = load_all(root_dir, difficulties=["easy", "medium", "hard"], splits=["train"])
    for key, problems in data.items():
        stats = dataset_stats(problems)
        print(f"{key}: {stats}")
