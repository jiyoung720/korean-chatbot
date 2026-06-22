import os
from datasets import load_dataset

CORPUS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "rag_corpus.txt")
NUM_DOCS = 1000  # RAG 검증용으로는 1000개면 충분 (학습용 5000개보다 적게)


def build_wiki_corpus():
    """RAG 검색 대상이 될 위키 문서를 받아서 저장.
    문서 단위 구분을 위해 각 문서 사이에 구분자(===DOC===)를 넣는다."""
    print(f"위키백과 {NUM_DOCS}개 문서 로딩 중...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)

    docs = []
    for i, example in enumerate(dataset):
        docs.append(example["text"])
        if i >= NUM_DOCS:
            break

    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        f.write("\n===DOC===\n".join(docs))

    print(f"저장 완료: {CORPUS_PATH} ({len(docs)}개 문서)")
    return docs


def load_wiki_corpus() -> list[str]:
    """저장된 코퍼스를 문서 리스트로 로드."""
    if not os.path.exists(CORPUS_PATH):
        return build_wiki_corpus()

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    return content.split("\n===DOC===\n")


if __name__ == "__main__":
    docs = build_wiki_corpus()
    print("샘플:", docs[0][:200])