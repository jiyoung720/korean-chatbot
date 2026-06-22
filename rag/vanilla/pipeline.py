# from rag.vanilla.retriever import search


# def build_rag_prompt(query: str, top_k: int = 3) -> str:
#     """검색된 청크를 컨텍스트로 추가한 프롬프트를 만든다."""
#     results = search(query, top_k=top_k)
#     context = "\n\n".join([r["text"] for r in results])

#     prompt = f"""다음 정보를 참고하여 질문에 답하세요.

# 정보:
# {context}

# 질문: {query}
# 답변:"""
#     return prompt

from rag.vanilla.retriever import search
import re

def build_rag_prompt(query: str, top_k: int = 3) -> str:
    """검색된 청크를 컨텍스트로 추가한 프롬프트를 만든다 (QA 형식)."""
    results = search(query, top_k=top_k)
    context = "\n\n".join([r["text"] for r in results])

    prompt = f"""다음 정보를 참고하여 질문에 답하세요.

정보:
{context}

질문: {query}
답변:"""
    return prompt


def build_rag_prompt_v2(query: str, top_k: int = 1) -> str:
    """이어쓰기 형식으로 조립 — wiki 모델이 학습한 서술체에 맞춤."""
    results = search(query, top_k=top_k)
    context = "\n\n".join([r["text"] for r in results])

    # 질문에서 "~은/는/이/가 누구야?/뭐야?" 패턴을 제거해 주어만 추출
    subject = re.sub(r"[은는이가]?\s*(누구|뭐|무엇)(야|예요|입니까|인가요)?\??$", "", query).strip()

    prompt = f"""{context}

{subject}는"""
    return prompt

def build_rag_prompt_dialog(query: str, top_k: int = 1) -> str:
    """dialog 모델(화자 토큰 기반)에 맞춘 RAG 프롬프트 조립."""
    results = search(query, top_k=top_k)
    context = "\n\n".join([r["text"] for r in results])

    prompt = f"""<sp1> {query}
<sp2> {context[:200]}"""
    return prompt