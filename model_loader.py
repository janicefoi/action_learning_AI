import logging
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"

_model: Optional[SentenceTransformer] = None
_reference_embeddings: Dict[str, np.ndarray] = {}


def load_model() -> SentenceTransformer:
    """Load the model and pre-compute reference embeddings once at startup."""
    global _model
    if _model is None:
        logger.info("Loading sentence-transformer model: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        dim = _model.get_embedding_dimension()
        logger.info("Model ready — embedding dimension: %d", dim)
        _precompute_reference_embeddings()
    return _model


def _precompute_reference_embeddings() -> None:
    from rubric import ALL_REFERENCES
    for key, sentences in ALL_REFERENCES.items():
        _reference_embeddings[key] = _model.encode(sentences, convert_to_numpy=True)
        logger.info("Reference embeddings ready — %s (%d sentences)", key, len(sentences))


def get_model() -> SentenceTransformer:
    if _model is None:
        raise RuntimeError("Model not initialised. Call load_model() at startup.")
    return _model


def get_reference_embeddings(key: str) -> np.ndarray:
    if key not in _reference_embeddings:
        raise RuntimeError(f"Reference embeddings not loaded for key: {key}. Call load_model() at startup.")
    return _reference_embeddings[key]


def embed(texts: List[str]) -> np.ndarray:
    """Return a 2-D numpy array of embeddings, one row per text."""
    return get_model().encode(texts, convert_to_numpy=True)
