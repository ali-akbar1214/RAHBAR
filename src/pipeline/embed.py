"""Deterministic Versioned Lexical Projection Embeddings.

Uses pure NumPy + standard-library hashing for zero-dependency, reproducible
embeddings across local dev and Kubernetes containers.

Versions:
  - lexical-proj-v1 (dim=256): word unigrams + bigrams with sublinear log-TF
    weighting and QR-orthogonalized Gaussian projection.
  - lexical-proj-v2 (dim=384): word unigrams + bigrams + character 4-grams
    with sublinear log-TF weighting and QR-orthogonalized Gaussian projection.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

EMBEDDING_MODELS: dict[str, dict[str, object]] = {
    "lexical-proj-v1": {
        "dim": 256,
        "vocab_buckets": 2048,
        "use_char_ngrams": False,
        "description": "Lexical projection v1 (word unigrams + bigrams, dim=256)",
    },
    "lexical-proj-v2": {
        "dim": 384,
        "vocab_buckets": 4096,
        "use_char_ngrams": True,
        "description": "Lexical projection v2 (word unigrams + bigrams + char 4-grams, dim=384)",
    },
}

DEFAULT_EMBEDDING_VERSION = "lexical-proj-v1"

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
    "of", "and", "or", "if", "it", "its", "be", "by", "as", "with", "that", "this",
    "from", "can", "may", "shall", "will", "do", "does", "did", "have", "has", "had",
}

_PROJ_MATRICES: dict[str, np.ndarray] = {}


def _get_projection_matrix(version: str) -> np.ndarray:
    if version not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown embedding version '{version}'. Supported: {list(EMBEDDING_MODELS.keys())}")
    if version not in _PROJ_MATRICES:
        spec = EMBEDDING_MODELS[version]
        vocab_buckets = int(spec["vocab_buckets"])
        dim = int(spec["dim"])
        seed = 42 if version == "lexical-proj-v1" else 1337
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((vocab_buckets, dim)).astype(np.float64)
        q, _ = np.linalg.qr(raw)
        _PROJ_MATRICES[version] = q
    return _PROJ_MATRICES[version]


def get_embedding_spec(version: str = DEFAULT_EMBEDDING_VERSION) -> dict[str, object]:
    if version not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown embedding version '{version}'")
    return dict(EMBEDDING_MODELS[version])


def _tokenize_and_extract_features(text: str, use_char_ngrams: bool) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9\s\.\-]", " ", text.lower())
    raw_tokens = [t.strip(".-") for t in cleaned.split() if len(t.strip(".-")) > 1]
    tokens = [t for t in raw_tokens if t not in _STOPWORDS]
    if not tokens:
        tokens = raw_tokens

    features = list(tokens)
    for i in range(len(tokens) - 1):
        features.append(f"{tokens[i]}_{tokens[i + 1]}")

    if use_char_ngrams:
        joined = " ".join(tokens)
        for i in range(max(0, len(joined) - 3)):
            gram = joined[i : i + 4]
            if " " not in gram:
                features.append(f"c4:{gram}")

    return features


def _stable_bucket(token: str, vocab_buckets: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % vocab_buckets


def embed_text(text: str, version: str = DEFAULT_EMBEDDING_VERSION) -> list[float]:
    """Return a deterministic L2-normalized embedding vector for text using the given version."""
    spec = get_embedding_spec(version)
    vocab_buckets = int(spec["vocab_buckets"])
    dim = int(spec["dim"])
    use_char_ngrams = bool(spec["use_char_ngrams"])
    proj = _get_projection_matrix(version)

    features = _tokenize_and_extract_features(text, use_char_ngrams=use_char_ngrams)
    if not features:
        return [0.0] * dim

    counts = Counter(features)
    sparse_vec = np.zeros(vocab_buckets, dtype=np.float64)
    for feat, freq in counts.items():
        b = _stable_bucket(feat, vocab_buckets)
        sparse_vec[b] += 1.0 + math.log(freq)

    dense = sparse_vec @ proj
    norm = float(np.linalg.norm(dense))
    if norm > 0.0:
        dense = dense / norm
    return [round(float(x), 6) for x in dense]


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
