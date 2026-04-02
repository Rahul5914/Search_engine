"""
search_engine.py - Semantic + keyword hybrid search
"""

import re

import numpy as np
import pandas as pd

SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z][a-z0-9+#.]*\b", text.lower()))


def keyword_overlap(query: str, profile_text: str) -> float:
    """Jaccard-style overlap between query tokens and profile tokens."""
    q_tokens = _tokenize(query)
    p_tokens = _tokenize(profile_text)
    if not q_tokens:
        return 0.0
    return len(q_tokens & p_tokens) / len(q_tokens)


def search(
    query: str,
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    df: pd.DataFrame,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Rank candidates with a hybrid score:
        score = 0.7 * cosine_similarity + 0.3 * keyword_overlap
    Returns a DataFrame slice sorted by score (descending).
    """
    # Cosine similarity (embeddings are already L2-normalised → dot product = cosine)
    semantic_scores = candidate_embeddings @ query_embedding       # shape (N,)

    # Keyword overlap per candidate
    kw_scores = np.array(
        [keyword_overlap(query, pt) for pt in df["profile_text"]]
    )

    hybrid = SEMANTIC_WEIGHT * semantic_scores + KEYWORD_WEIGHT * kw_scores

    top_indices = np.argsort(hybrid)[::-1][:top_k]
    results = df.iloc[top_indices].copy()
    results["semantic_score"] = semantic_scores[top_indices]
    results["keyword_score"] = kw_scores[top_indices]
    results["score"] = hybrid[top_indices]
    results["rank"] = range(1, len(results) + 1)
    return results.reset_index(drop=True)
