"""
evaluation_dataset_v1.md의 10문항을 LangSmith Dataset으로 등록하는 스크립트.

사용법:
    export LANGCHAIN_API_KEY="..."
    python -m rag.langchain.build_dataset

실행 전 확인:
    - LANGCHAIN_API_KEY 환경변수가 설정되어 있어야 합니다.
    - 같은 이름의 Dataset이 이미 있으면 새로 만들지 않고 기존 Dataset에
      예제를 추가합니다 (client.create_dataset이 멱등은 아니므로,
      재실행 시 중복 등록을 막기 위해 존재 여부를 먼저 확인합니다).
"""

from langsmith import Client

DATASET_NAME = "korean-chatbot-evaluation-v1"
DATASET_DESCRIPTION = (
    "RAG 코퍼스(한국어 위키백과 1,000문서) 기반 질문 10개. "
    "엘라/Gemma/Gemini 세 모델을 평가셋 v1으로 비교한 데이터를 "
    "LangSmith Dataset Evaluation용으로 재구성했습니다. "
    "원본: docs/evaluation_dataset_v1.md"
)

# evaluation_dataset_v1.md 원문을 그대로 옮긴 예제 목록.
# Q10은 known_retrieval_issue=True로 표시 — 실제 검색 실패가 발생하는
# 케이스이므로(주어 추출 정규식이 "어떤 ~로 유명한가" 패턴을 처리하지
# 못함), 평가 결과 해석 시 이 메타데이터로 구분합니다.
# 자세한 내용: docs/evaluation_v1_results_ella.md "발견한 추가 한계" 절
EXAMPLES = [
    {
        "question": "지미 카터는 누구인가?",
        "expected_answer": "미국의 제39대 대통령(1977년~1981년)이며, 민주당 출신이다.",
        "category": "인물 / 정의형",
        "known_retrieval_issue": False,
    },
    {
        "question": "백남준은 어떤 예술 분야로 유명한가?",
        "expected_answer": "한국 태생의 세계적인 비디오 아트 예술가이자 작곡가, 전위 예술가로 유명하다.",
        "category": "인물 / 특징형",
        "known_retrieval_issue": False,
    },
    {
        "question": "데니스 리치는 어떤 프로그래밍 언어 개발에 기여했는가?",
        "expected_answer": "C 언어를 만들었으며, 유닉스 운영체제 개발에도 기여한 컴퓨터과학자이다.",
        "category": "인물 / 관계형",
        "known_retrieval_issue": False,
    },
    {
        "question": "우크라이나는 어디에 위치한 국가인가?",
        "expected_answer": "동유럽에 위치한 국가로, 남쪽은 흑해와 아조프해, 동쪽은 러시아, 북쪽은 벨라루스와 접한다.",
        "category": "국가/지역 / 위치형",
        "known_retrieval_issue": False,
    },
    {
        "question": "체첸 공화국은 어느 지역에 위치하는가?",
        "expected_answer": "러시아를 이루는 공화국 중 하나로, 북캅카스 지역에 위치한다.",
        "category": "국가/지역 / 위치형",
        "known_retrieval_issue": False,
    },
    {
        "question": "수학은 무엇을 연구하는 학문인가?",
        "expected_answer": "수, 양, 구조, 공간, 변화 등의 개념을 다루는 학문이다.",
        "category": "개념 / 정의형",
        "known_retrieval_issue": False,
    },
    {
        "question": "화학은 무엇을 연구하는 학문인가?",
        "expected_answer": "물질의 성질, 조성, 구조, 변화 및 그에 수반하는 에너지의 변화를 연구하는 자연과학의 한 분야이다.",
        "category": "개념 / 정의형",
        "known_retrieval_issue": False,
    },
    {
        "question": "초월수란 무엇인가?",
        "expected_answer": "유리수 계수를 가지는 0이 아닌 유한 차수 다항 방정식의 해가 될 수 없는 수, 즉 대수학적이지 않은 수를 의미한다.",
        "category": "개념 / 정의형 (다소 어려움)",
        "known_retrieval_issue": False,
    },
    {
        "question": "맥스웰 방정식은 무엇인가?",
        "expected_answer": "전기와 자기의 발생, 전기장과 자기장, 전하 밀도와 전류 밀도의 형성을 나타내는 4개의 편미분 방정식이다.",
        "category": "과학 / 정의형",
        "known_retrieval_issue": False,
    },
    {
        "question": "세종대왕은 어떤 업적으로 유명한가?",
        "expected_answer": "1443년 한글(훈민정음)을 창제하기 시작하여 1446년 반포했다.",
        "category": "프로젝트 연계 / 업적형",
        "known_retrieval_issue": True,
        "note": (
            "주어 추출 정규식이 '어떤 ~로 유명한가' 패턴을 처리하지 못해 "
            "검색 쿼리에 질문 전체가 그대로 들어가고, 그 결과 무관한 청크가 "
            "검색됨 (docs/evaluation_v1_results_ella.md 참고). 세 모델 "
            "(엘라/Gemma/Gemini) 모두 동일한 검색 실패를 겪었으나 생성 단계 "
            "대응은 모델마다 달랐음 — 평가 결과 해석 시 이 메타데이터를 "
            "참고해 Q10을 포함/제외한 평균을 모두 검토할 것."
        ),
    },
]


def build_dataset():
    client = Client()

    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        dataset = existing[0]
        print(f"기존 Dataset 재사용: {dataset.name} (id={dataset.id})")
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=DATASET_DESCRIPTION,
        )
        print(f"Dataset 생성: {dataset.name} (id={dataset.id})")

    # 이미 등록된 example과 중복되지 않도록, 질문 텍스트 기준으로
    # 기존 example을 먼저 조회합니다.
    existing_questions = {
        ex.inputs.get("question")
        for ex in client.list_examples(dataset_id=dataset.id)
    }

    added = 0
    for item in EXAMPLES:
        if item["question"] in existing_questions:
            print(f"  스킵 (이미 존재): {item['question']}")
            continue

        client.create_example(
            inputs={"question": item["question"]},
            outputs={"expected_answer": item["expected_answer"]},
            metadata={
                "category": item["category"],
                "known_retrieval_issue": item["known_retrieval_issue"],
                **({"note": item["note"]} if "note" in item else {}),
            },
            dataset_id=dataset.id,
        )
        added += 1
        print(f"  추가: {item['question']}")

    print(f"\n완료: {added}개 example 추가됨 (총 {len(existing_questions) + added}개)")


if __name__ == "__main__":
    build_dataset()