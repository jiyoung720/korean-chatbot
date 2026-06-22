from sentence_transformers import SentenceTransformer
import numpy as np

EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"

_embedder = None  # 모델을 한 번만 로드해서 재사용 (lazy loading)


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def embed_texts(texts: list[str]) -> np.ndarray:
    """텍스트 리스트를 임베딩 벡터 배열로 변환. shape: (N, 768)"""
    embedder = get_embedder()
    return embedder.encode(texts, convert_to_numpy=True)


def embed_query(text: str) -> np.ndarray:
    """단일 쿼리를 임베딩 벡터로 변환. shape: (768,)"""
    embedder = get_embedder()
    return embedder.encode(text, convert_to_numpy=True)