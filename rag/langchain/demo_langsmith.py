"""
LangSmith Tracing 적용 데모.

rag/langchain/chain.py의 체인은 코드를 전혀 바꾸지 않고, 아래 세
환경변수만 설정하면 모든 실행이 LangSmith에 자동으로 추적된다.

    export LANGCHAIN_TRACING_V2="true"
    export LANGCHAIN_API_KEY="..."
    export LANGCHAIN_PROJECT="korean-chatbot-rag"

실행 후 https://smith.langchain.com 에서 LANGCHAIN_PROJECT로 지정한
프로젝트를 열면 트레이스를 확인할 수 있다. 검색(Retriever)·프롬프트
조립(extract_subject 등)·생성(KoreanGPTWrapper) 각 단계가 분리되어
보이고, 단계별 소요 시간과 중간 입출력을 확인할 수 있다.

실행: python3 -m rag.langchain.demo_langsmith
"""

import os

from rag.langchain.chain import build_rag_chain

if __name__ == "__main__":
    if os.environ.get("LANGCHAIN_TRACING_V2") != "true":
        print("경고: LANGCHAIN_TRACING_V2가 설정되지 않았습니다. "
              "LangSmith에 기록되지 않고 일반 실행만 됩니다.")

    chain = build_rag_chain(model_key="wiki", top_k=1)
    answer = chain.invoke("세종대왕은 누구야?")
    print(answer)