from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.shared.embedding import EMBEDDING_MODEL_NAME
from rag.shared.corpus import load_wiki_corpus
from rag.shared.chunking import chunk_documents

import os

VECTORSTORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "artifacts", "rag", "langchain_chroma_index"
)


def get_embeddings():
    """LangChain의 Embeddings 인터페이스로 감싼 임베딩 모델.
    Vanilla 버전과 동일한 모델(EMBEDDING_MODEL_NAME)을 사용해 공정하게 비교."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_or_load_vectorstore():
    """Chroma 벡터스토어를 구축하거나, 이미 있으면 로드.

    FAISS(faiss-cpu)는 macOS ARM + Python 3.12 환경에서
    index.search() 호출 시 segmentation fault가 발생해 Chroma로 전환했다.
    (원인 추적 과정은 docs/training_log.md 참고)
    """
    embeddings = get_embeddings()

    if os.path.exists(VECTORSTORE_DIR):
        print("저장된 Chroma 인덱스 로드 중...")
        return Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=embeddings,
        )

    print("Chroma 인덱스 새로 구축 중...")
    docs = load_wiki_corpus()
    chunks = chunk_documents(docs)  # Vanilla와 동일한 청킹 함수 재사용

    documents = [
        Document(page_content=c["text"], metadata={"doc_idx": c["doc_idx"]})
        for c in chunks
    ]

    vectorstore = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=VECTORSTORE_DIR,
        collection_metadata={"hnsw:space": "cosine"},  # L2 대신 코사인 유사도 사용
    )
    print(f"Chroma 인덱스 저장 완료: {VECTORSTORE_DIR}")

    return vectorstore