def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """텍스트를 chunk_size 글자 단위로 자르되, overlap만큼 겹치게 해서
    문맥이 청크 경계에서 끊기는 것을 완화."""
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def chunk_documents(docs: list[str], chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """문서 리스트를 청크 리스트로 변환. 각 청크는 원본 문서 인덱스를 함께 가짐."""
    all_chunks = []
    for doc_idx, doc in enumerate(docs):
        chunks = chunk_text(doc, chunk_size, overlap)
        for chunk in chunks:
            all_chunks.append({"text": chunk, "doc_idx": doc_idx})
    return all_chunks