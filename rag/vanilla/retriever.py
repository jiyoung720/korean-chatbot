import os
import numpy as np
import pickle

from rag.shared.embedding import embed_texts, embed_query
from rag.shared.corpus import load_wiki_corpus
from rag.shared.chunking import chunk_documents

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "artifacts", "rag")
CACHE_PATH = os.path.join(CACHE_DIR, "vanilla_chunk_embeddings.pkl")


def build_or_load_index():
    """청크와 임베딩을 미리 계산해서 캐싱. 이미 있으면 그대로 로드."""
    if os.path.exists(CACHE_PATH):
        print("캐시된 인덱스 로드 중...")
        with open(CACHE_PATH, "rb") as f:
            data = pickle.load(f)
        return data["chunks"], data["embeddings"]

    print("인덱스 새로 구축 중...")
    docs = load_wiki_corpus()
    chunks = chunk_documents(docs)
    texts = [c["text"] for c in chunks]

    embeddings = embed_texts(texts)  # shape: (N, 768)

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)

    print(f"인덱스 구축 완료: {len(chunks)}개 청크")
    return chunks, embeddings


def search(query: str, top_k: int = 3):
    """쿼리와 가장 유사한 청크 top_k개를 반환."""
    chunks, embeddings = build_or_load_index()

    query_vec = embed_query(query)

    # 코사인 유사도 계산: (A · B) / (|A| * |B|)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    similarities = (embeddings @ query_vec) / norms

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "text": chunks[idx]["text"],
            "doc_idx": chunks[idx]["doc_idx"],
            "score": float(similarities[idx]),
        })
    return results