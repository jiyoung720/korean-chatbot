"""
korean-chatbot-evaluation-v1 Dataset으로 RAG 체인(/rag-chat과 동일한
rag/langchain/chain.py)을 LangSmith Evaluation으로 평가하는 스크립트.

사용법:
    export LANGCHAIN_API_KEY="..."
    export GEMINI_API_KEY="..."
    python -m rag.langchain.run_evaluation

평가 대상:
    rag/langchain/chain.py의 build_rag_chain() — api/main.py의
    /rag-chat 엔드포인트가 호출하는 것과 동일한 체인입니다.

평가 기준 (evaluation_dataset_v1.md "평가 기준" 절 그대로 사용):
    - 정확성: 모범답안과 사실 내용이 일치하는가
    - 질문 관련성: 질문에서 요구한 정보에 정확히 답하는가
    - 자연스러움: 문법적으로 통하는 완성된 문장인가
    각 0~5점, Judge는 Gemini 2.5 Flash.

Q10 채점 특례:
    검색된 컨텍스트에 답을 찾을 수 없는 경우, 모델이 솔직하게 정보
    부족을 인정하면 정확성 항목에서 만점을 준다 (hallucination보다
    우수한 대응으로 간주). build_dataset.py에서 등록한 메타데이터
    (known_retrieval_issue)를 judge 프롬프트에 함께 전달해 이 기준이
    적용되도록 한다.
"""

import os
import re

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client
from langsmith.evaluation import evaluate
from google import genai

from rag.langchain.chain import build_rag_chain

DATASET_NAME = "korean-chatbot-evaluation-v1"
JUDGE_MODEL = "gemini-2.5-flash"

genai_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# 평가 대상 체인 — api/main.py의 /rag-chat과 동일한 구성
# (model_key="wiki", top_k=1)
rag_chain = build_rag_chain(model_key="wiki", top_k=1)


def target(inputs: dict) -> dict:
    """LangSmith가 Dataset의 각 example.inputs를 넣어 호출하는 함수.
    체인의 invoke 결과를 outputs로 반환한다."""
    answer = rag_chain.invoke(inputs["question"])
    return {"answer": answer}


JUDGE_PROMPT_TEMPLATE = """당신은 한국어 RAG 챗봇의 응답을 평가하는 전문가입니다.
아래 질문, 모범답안, 모델 응답을 보고 세 가지 기준으로 각각 0~5점을 매기세요.

질문: {question}
모범답안: {expected_answer}
모델 응답: {actual_answer}

평가 기준:
1. 정확성 (0~5점): 모범답안과 사실 내용이 일치하는가.
   예외: 만약 이 질문이 알려진 검색 실패 사례(아래 참고)이고, 모델이
   "정보가 없다/모르겠다"는 취지로 솔직하게 답했다면, 무관한 내용을
   지어내는 hallucination보다 우수한 대응이므로 정확성 5점을 준다.
2. 질문 관련성 (0~5점): 질문에서 요구한 정보에 정확히 답하는가, 아니면
   무관한 내용이 섞여 있는가.
3. 자연스러움 (0~5점): 문법적으로 통하는 완성된 문장인가.

참고 (이 질문에 대한 알려진 이슈): {retrieval_note}

다음 JSON 형식으로만 답하세요. 다른 설명은 추가하지 마세요.
{{"accuracy": 0, "relevance": 0, "fluency": 0, "reasoning": "한 줄 평가 이유"}}
"""


def llm_judge(run, example) -> dict:
    """LangSmith evaluator 함수. run(체인 실행 결과)과 example(Dataset
    항목)을 받아 Gemini judge로 채점하고 LangSmith가 인식하는 형식으로
    반환한다."""
    question = example.inputs["question"]
    expected_answer = example.outputs["expected_answer"]
    actual_answer = run.outputs.get("answer", "")

    metadata = example.metadata or {}
    if metadata.get("known_retrieval_issue"):
        retrieval_note = metadata.get("note", "이 질문은 알려진 검색 실패 사례입니다.")
    else:
        retrieval_note = "알려진 이슈 없음 (정상적으로 검색 가능한 질문)."

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        expected_answer=expected_answer,
        actual_answer=actual_answer,
        retrieval_note=retrieval_note,
    )

    response = genai_client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
    )
    text = response.text.strip()

    # judge가 코드블록(```json ... ```)으로 감싸 응답하는 경우 제거
    text = re.sub(r"^```json\s*|\s*```$", "", text).strip()

    import json
    try:
        scores = json.loads(text)
    except json.JSONDecodeError:
        # judge 응답이 JSON이 아니면 평가 실패로 기록하고 0점 처리
        return [
            {"key": "accuracy", "score": 0, "comment": f"judge 응답 파싱 실패: {text[:200]}"},
            {"key": "relevance", "score": 0, "comment": "judge 응답 파싱 실패"},
            {"key": "fluency", "score": 0, "comment": "judge 응답 파싱 실패"},
        ]

    reasoning = scores.get("reasoning", "")
    return [
        {"key": "accuracy", "score": scores.get("accuracy", 0), "comment": reasoning},
        {"key": "relevance", "score": scores.get("relevance", 0), "comment": reasoning},
        {"key": "fluency", "score": scores.get("fluency", 0), "comment": reasoning},
    ]


def main():
    client = Client()

    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[llm_judge],
        experiment_prefix="ella-wiki-rag-v1",
        client=client,
        metadata={
            "model": "ella-23M",
            "model_key": "wiki",
            "top_k": 1,
            "chain": "rag/langchain/chain.py:build_rag_chain",
        },
    )

    print("\n평가 완료. LangSmith 대시보드에서 experiment 결과를 확인하세요.")
    print(results)


if __name__ == "__main__":
    main()