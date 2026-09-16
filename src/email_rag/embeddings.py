"""A local sentence-transformers adapter."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, trust_remote_code=False)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return unit vectors; dot product is therefore cosine similarity."""
        return np.asarray(
            self._model.encode(
                list(texts), normalize_embeddings=True, show_progress_bar=len(texts) > 20
            ),
            dtype=np.float32,
        )

