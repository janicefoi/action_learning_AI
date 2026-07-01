import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"

_model: Optional[SentenceTransformer] = None


def load_model() -> SentenceTransformer:
    """Load the model once; subsequent calls return the cached instance."""
    global _model
    if _model is None:
        logger.info("Loading sentence-transformer model: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        dim = _model.get_sentence_embedding_dimension()
        logger.info("Model ready — embedding dimension: %d", dim)
    return _model


def get_model() -> SentenceTransformer:
    if _model is None:
        raise RuntimeError("Model not initialised. Call load_model() at startup.")
    return _model


def embed(texts: List[str]) -> np.ndarray:
    """Return a 2-D numpy array of embeddings, one row per text."""
    return get_model().encode(texts, convert_to_numpy=True)
