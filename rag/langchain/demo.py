"""
LangChain RAG 데모

실행 방법 (프로젝트 루트에서):
    python3 -m rag.langchain.demo

Vanilla(rag/vanilla/demo.py)와 동일한 임베딩·코퍼스·생성 모델을 사용해
LangChain으로 마이그레이션한 버전을 시연합니다. FAISS->Chroma 전환,
거리 함수 설정 등 디버깅 과정은 docs/training_log.md 참고.
"""

from rag.langchain.chain import build_rag_chain


if __name__ == "__main__":
    questions = ["세종대왕은 누구야?", "지미 카터는 누구야?"]

    chain = build_rag_chain(model_key="wiki", top_k=1)

    for q in questions:
        print(f"\n{'=' * 50}\n질문: {q}\n{'=' * 50}")
        answer = chain.invoke(q)
        print(answer)