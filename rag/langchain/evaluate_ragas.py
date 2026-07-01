"""
RAGAS를 사용한 Ella 모델 RAG 파이프라인 평가

평가 메트릭:
- Faithfulness: 생성 답변이 검색된 컨텍스트에 충실한가
- Context Precision: 검색된 문서 중 관련 있는 부분의 비율
- Context Recall: 답변에 필요한 정보를 검색이 모두 담았는가
- Answer Relevancy: 답변이 질문과 관련 있는가

사용법:
    export GEMINI_API_KEY="..."
    export LANGCHAIN_API_KEY="..."
    python -m rag.langchain.evaluate_ragas
"""

import os
from dotenv import load_dotenv

load_dotenv()

from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from rag.langchain.chain import build_rag_chain
from rag.langchain.vectorstore import build_or_load_vectorstore

# RAGAS용 judge LLM (LangChain ChatModel 인터페이스로 감싼 Gemini)
judge_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=0,
)

# RAGAS의 context_precision/answer_relevancy는 임베딩 기반 유사도 계산이
# 필요하다. embeddings를 명시하지 않으면 RAGAS가 기본값으로 OpenAI
# 임베딩을 시도해 OPENAI_API_KEY를 요구하므로, Gemini 임베딩으로 명시한다.
judge_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ["GEMINI_API_KEY"],
)

# 평가 데이터셋
EVALUATION_DATA = [
    {
        "question": "지미 카터는 누구인가?",
        "ground_truth": "미국의 제39대 대통령(1977년~1981년)이며, 민주당 출신이다.",
    },
    {
        "question": "백남준은 어떤 예술 분야로 유명한가?",
        "ground_truth": "한국 태생의 세계적인 비디오 아트 예술가이자 작곡가, 전위 예술가로 유명하다.",
    },
    {
        "question": "데니스 리치는 어떤 프로그래밍 언어 개발에 기여했는가?",
        "ground_truth": "C 언어를 만들었으며, 유닉스 운영체제 개발에도 기여한 컴퓨터과학자이다.",
    },
    {
        "question": "우크라이나는 어디에 위치한 국가인가?",
        "ground_truth": "동유럽에 위치한 국가로, 남쪽은 흑해와 아조프해, 동쪽은 러시아, 북쪽은 벨라루스와 접한다.",
    },
    {
        "question": "체첸 공화국은 어느 지역에 위치하는가?",
        "ground_truth": "러시아를 이루는 공화국 중 하나로, 북캅카스 지역에 위치한다.",
    },
    {
        "question": "수학은 무엇을 연구하는 학문인가?",
        "ground_truth": "수, 양, 구조, 공간, 변화 등의 개념을 다루는 학문이다.",
    },
    {
        "question": "화학은 무엇을 연구하는 학문인가?",
        "ground_truth": "물질의 성질, 조성, 구조, 변화 및 그에 수반하는 에너지의 변화를 연구하는 자연과학의 한 분야이다.",
    },
    {
        "question": "초월수란 무엇인가?",
        "ground_truth": "유리수 계수를 가지는 0이 아닌 유한 차수 다항 방정식의 해가 될 수 없는 수, 즉 대수학적이지 않은 수를 의미한다.",
    },
    {
        "question": "맥스웰 방정식은 무엇인가?",
        "ground_truth": "전기와 자기의 발생, 전기장과 자기장, 전하 밀도와 전류 밀도의 형성을 나타내는 4개의 편미분 방정식이다.",
    },
    {
        "question": "세종대왕은 어떤 업적으로 유명한가?",
        "ground_truth": "1443년 한글(훈민정음)을 창제하기 시작하여 1446년 반포했다.",
    },
]

# RAG 체인과 vectorstore를 모듈 레벨에서 한 번만 생성 (10문항 평가 내내 재사용)
vectorstore = build_or_load_vectorstore()
rag_chain = build_rag_chain(model_key="wiki", top_k=1)


def get_contexts_from_chain(question: str) -> list[str]:
    """RAG 체인에서 검색된 컨텍스트 추출 (top_k=1은 README에 정의된 Ella 설정과 동일)"""
    docs = vectorstore.similarity_search(question, k=1)
    return [doc.page_content for doc in docs]


def get_answer_from_chain(question: str) -> str:
    """RAG 체인에서 답변 생성"""
    answer = rag_chain.invoke(question)
    return answer


def main():
    print("RAGAS 평가 시작...")
    
    # 평가 데이터 준비
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    
    for i, item in enumerate(EVALUATION_DATA):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        print(f"\n[{i+1}/10] {question}")
        
        # 답변 생성
        answer = get_answer_from_chain(question)
        print(f"  생성된 답변: {answer[:100]}...")
        
        # 컨텍스트 추출
        contexts = get_contexts_from_chain(question)
        print(f"  검색된 컨텍스트 수: {len(contexts)}")
        
        eval_data["question"].append(question)
        eval_data["answer"].append(answer)
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(ground_truth)
    
    # Dataset 변환
    dataset = Dataset.from_dict(eval_data)
    
    # RAGAS 평가 실행
    print("\n\nRAGAS 평가 실행 중...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    
    # 결과 저장
    # RAGAS의 evaluate()는 EvaluationResult 객체를 반환하며 json.dump로
    # 바로 직렬화할 수 없다. to_pandas()로 변환한 뒤 저장한다.
    results_df = results.to_pandas()

    csv_path = "docs/evaluation_ragas_results.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    json_path = "docs/evaluation_ragas_results.json"
    results_df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    print(f"\n평가 완료. 결과 저장: {csv_path}, {json_path}")
    print(f"\n평균 점수:")
    for metric_name in ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]:
        avg_score = results_df[metric_name].mean()
        print(f"  {metric_name}: {avg_score:.3f}")


if __name__ == "__main__":
    main()