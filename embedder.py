"""
embedder.py - Generate and cache sentence embeddings
"""

from pathlib import Path

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = Path("data/embeddings.npy")


@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    """Load (and cache) the sentence-transformer model."""
    return SentenceTransformer(MODEL_NAME)


@st.cache_data(show_spinner=False)
def get_embeddings(profile_texts: tuple[str, ...]) -> np.ndarray:
    """
    Return embeddings for profile_texts.
    Tries to load from disk first; falls back to computing and saving.
    """
    if EMBEDDINGS_PATH.exists():
        stored = np.load(EMBEDDINGS_PATH)
        if stored.shape[0] == len(profile_texts):
            return stored

    model = load_model()
    embeddings = model.encode(
        list(profile_texts),
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Return a normalised embedding for a single query string."""
    model = load_model()
    return model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
