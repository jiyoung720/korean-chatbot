from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from rag.langchain.vectorstore import build_or_load_vectorstore
from rag.langchain.llm_wrapper import KoreanGPTWrapper

import re

def extract_subject(question: str) -> str:
    """Vanilla의 정규식과 동일한 주어 추출 로직."""
    return re.sub(r"[은는이가]?\s*(누구|뭐|무엇)(야|예요|입니까|인가요)?\??$", "", question).strip()

def format_docs(docs):
    """검색된 Document 리스트를 텍스트로 합치기 (Vanilla의 컨텍스트 조립과 동일한 역할)."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(model_key: str = "wiki", top_k: int = 1):
    """검색 -> 프롬프트 조립 -> 생성을 하나로 엮은 LangChain 체인.
    Vanilla의 build_rag_prompt_v2() + generate() 호출과 동일한 역할을
    LangChain의 LCEL(체인 연산자 |)로 구현."""
    vectorstore = build_or_load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    llm = KoreanGPTWrapper(model_key=model_key)

    # Vanilla의 build_rag_prompt_v2와 동일한 "이어쓰기" 형식
    prompt = PromptTemplate.from_template("{context}\n\n{question}는")

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough() | extract_subject}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain