import os
from functools import lru_cache
from typing import List


@lru_cache(maxsize=2)
def _model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def get_embedding_model():
    model_name = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    return _model(model_name)


def encode_texts(texts: List[str]) -> List[list]:
    """Return list of embedding vectors as Python lists."""
    model = get_embedding_model()
    vecs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [list(map(float, v)) for v in vecs]


def embedding_dim() -> int:
    """Dimension for schema.sql vector(N)."""
    m = get_embedding_model()
    # sentence-transformers first linear dim
    return int(m.get_sentence_embedding_dimension())
