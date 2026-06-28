"""
Evaluation v1의 세 모델(엘라/Gemma/Gemini) 응답을 Gemini를 Judge로
사용해 0~5점으로 채점.

v3 변경점 (v2 대비):
- 예외 타입을 문자열 검색이 아닌 google.api_core.exceptions.ResourceExhausted로 처리
- API가 알려주는 retry_delay를 읽어서 그만큼만 대기 (고정 60초 대신)
- 점수 기준(0~5 각 단계 의미)을 프롬프트에 명시해 채점 일관성 강화
- 정규식이 "정확성: 5" / "정확성:5" / "정확성 - 5" 등 변형 표기를 허용하도록 완화
- 각 모델 점수에 짧은 채점 이유(reason)를 추가로 받음
- 결과 파일을 docs/evaluation_v1_judge_scores.json으로 저장

실행: python3 rag/eval_external/judge.py

주의: Judge로 쓰는 모델(Gemini 2.5 Flash)이 평가 대상 중 하나(Gemini)와
동일하다는 자기참조 한계가 있음. docs/evaluation_v1_comparison.md의
"이번 실험의 한계"에 추가로 기록할 것.
"""

import os
import re
import time
import json
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
judge_model = genai.GenerativeModel("gemini-2.5-flash")

EVAL_DATA = [
    {
        "question": "지미 카터는 누구인가?",
        "reference": "미국의 제39대 대통령(1977년~1981년)이며, 민주당 출신이다.",
        "answers": {
            "ella": "제임스 얼 카터 주니어(, 1924년 10월 1일~)는 민주당 출신 미국의 제39대 대통령(1977년~1981년)이다. ... 지미 카터는 누구인가?는 현재 미국 시민학교(Feeo Merastish Sonst Macquest)를 신설하여 \"Prunge in Agcam Sunet\"를 만들어냈다. 미국 대통령은 미국, 영국, 미국, 미국, 프랑스 등에서도 독립적으로 영국에게",
            "gemma": "지미 카터는 미국 민주당 출신의 제39대 대통령입니다. 그는 1977년부터 1981년까지 미국 대통령을 역임했으며, 캠프 데이비드 협정을 통해 중동 평화에 기여한 공로로 노벨 평화상을 수상했습니다. 주요 이력으로는 조지아 공과대학교를 졸업하고 미국 해군 장교로 복무한 경력, 농부(땅콩 농부)로서의 삶, 그리고 대통령 재임 시절의 정치 경력이 있습니다.",
            "gemini": "지미 카터는 민주당 출신 미국의 제39대 대통령(1977년~1981년)이며, 전직 미국 해군 장교이자 땅콩 농부이다.",
        },
    },
    {
        "question": "백남준은 어떤 예술 분야로 유명한가?",
        "reference": "한국 태생의 세계적인 비디오 아트 예술가이자 작곡가, 전위 예술가로 유명하다.",
        "answers": {
            "ella": "백남준(白南準, 1932년 7월 20일 ~ 2006년 1월 29일, 서울 출생, )은 한국 태생의 세계적인 비디오 아트 예술가, 작곡가, 전위 예술가이다. ... 백남준은 어떤 예술 분야로 유명한가?는 전혀 전하지 않는가 아니라는 것을 보여주지 않은 것을 본성명을 제공해야 한다는 점에서, 일렉터리이다.",
            "gemma": "비디오 아트(Video Art) 예술가로 가장 유명하며, '비디오 아트의 창시자'로 알려져 있습니다. 또한, 작곡가, 전위 예술가, 그리고 미디어 아티스트로 활동했습니다.",
            "gemini": "비디오 아트",
        },
    },
    {
        "question": "데니스 리치는 어떤 프로그래밍 언어 개발에 기여했는가?",
        "reference": "C 언어를 만들었으며, 유닉스 운영체제 개발에도 기여한 컴퓨터과학자이다.",
        "answers": {
            "ella": "데니스 리치의 업적에 기반하고 있다'이라며 그의 업적을 평가했다. ... 데니스 리치는 어떤 프로그래밍 언어 개발에 기여했는가?는 그릴러, 이러한 정보 플랫폼 소프트웨어 개발, 언어, 기업 운영 체제, 웹 서버 소프트웨어 개발자, 컴퓨터 하드웨어 등 다양한 언어들을 개발하고 개발하고 개발하는 언어이다.",
            "gemma": "데니스 리치는 C 언어와 유닉스(Unix) 운영체제 개발에 기여했습니다. - C 언어: 기존의 B 언어에 데이터 유형 개념을 추가하여 C 언어를 만들었습니다. - 유닉스: C 언어가 유닉스의 구현 언어로 채택되면서 유닉스 운영체제가 탄생하게 했습니다.",
            "gemini": "데니스 리치는 C 프로그래밍 언어 개발에 기여했다.",
        },
    },
    {
        "question": "우크라이나는 어디에 위치한 국가인가?",
        "reference": "동유럽에 위치한 국가로, 남쪽은 흑해와 아조프해, 동쪽은 러시아, 북쪽은 벨라루스와 접한다.",
        "answers": {
            "ella": "우크라이나()는 동유럽 국가다. 남쪽과 남동쪽으로는 흑해와 아조프해, 동쪽과 북동쪽으로는 러시아, 북쪽과 북서쪽으로는 벨라루스 ... 우크라이나는 어디에 위치한 국가인가?는 크리올어이다. 또한 이 나라에서 가장 강력한 도시이다. 프랑스어가 아닌 인구수를 차지하는 도시이다.",
            "gemma": "동유럽에 위치한 국가입니다. (주변국 및 해역 접경 국가: 남쪽/남동쪽으로는 흑해와 아조프해, 동쪽/북동쪽으로는 러시아, 북쪽/북서쪽으로는 벨라루스, 서쪽으로는 폴란드, 슬로바키아, 헝가리, 남서쪽으로는 루마니아, 몰도바와 접합니다.)",
            "gemini": "우크라이나는 동유럽 국가이다. 남쪽과 남동쪽으로는 흑해와 아조프해, 동쪽과 북동쪽으로는 러시아, 북쪽과 북서쪽으로는 벨라루스, 서쪽으로는 폴란드, 슬로바키아, 헝가리, 남서쪽으로는 루마니아, 몰도바와 접한다.",
        },
    },
    {
        "question": "체첸 공화국은 어느 지역에 위치하는가?",
        "reference": "러시아를 이루는 공화국 중 하나로, 북캅카스 지역에 위치한다.",
        "answers": {
            "ella": "체첸 공화국(, , ), 또는 줄여서 체첸(, , )은 연방국가인 러시아를 이루는 러시아의 공화국이다. 북캅카스 지역에 위치하여 있으며 ... 체첸 공화국은 어느 지역에 위치하는가?는 샬럿이다. 우즈베키스탄 인구는 1950년 인구밀도가 13.6%가 넘지 않다.",
            "gemma": "북캅카스 지역",
            "gemini": "북캅카스 지역에 위치한다.",
        },
    },
    {
        "question": "수학은 무엇을 연구하는 학문인가?",
        "reference": "수, 양, 구조, 공간, 변화 등의 개념을 다루는 학문이다.",
        "answers": {
            "ella": "수학(數學, , 줄여서 math)은 수, 양, 구조, 공간, 변화 등의 개념을 다루는 학문이다. ... 수학은 무엇을 연구하는 학문인가?는 과학적인 의미론이다. 실증철학 과학적 방법론 그는 과학의 발전과 지도를 연구한다.",
            "gemma": "수, 양, 구조, 공간, 변화 등의 개념을 다루는 학문입니다. 현대 수학은 일반적으로 엄밀한 논리에 근거하여 추상적인 대상을 탐구하며, 특히 자연계에서 관측되지 않는 개념에 대해서까지 이론을 추상화시키는 특징을 보입니다.",
            "gemini": "수학은 수, 양, 구조, 공간, 변화 등의 개념을 다루며, 엄밀한 논리에 근거하여 추상적 대상을 탐구하는 학문이다.",
        },
    },
    {
        "question": "화학은 무엇을 연구하는 학문인가?",
        "reference": "물질의 성질, 조성, 구조, 변화 및 그에 수반하는 에너지의 변화를 연구하는 자연과학의 한 분야이다.",
        "answers": {
            "ella": "화학(化學)은 물질의 성질, 조성, 구조, 변화 및 그에 수반하는 에너지의 변화를 연구하는 자연과학(自然科學)의 한 분야이다. ... 화학은 무엇을 연구하는 학문인가?는 자연철학적 방법과 관련된 과정을 통해, 물질에 의해 발생하는 데 필요한, 따라서 이 과정을 연구하는 자연과학적인 분야이다.",
            "gemma": "물질의 성질, 조성, 구조, 변화 및 그에 수반하는 에너지의 변화를 연구하는 자연과학의 한 분야입니다.",
            "gemini": "화학은 물질의 성질, 조성, 구조, 변화 및 그에 수반하는 에너지의 변화를 연구하는 자연과학의 한 분야이다.",
        },
    },
    {
        "question": "초월수란 무엇인가?",
        "reference": "유리수 계수를 가지는 0이 아닌 유한 차수 다항 방정식의 해가 될 수 없는 수, 즉 대수학적이지 않은 수를 의미한다.",
        "answers": {
            "ella": "대수함수에서 초월수는 다른 초월수에 대응된다. ... 초월수란 무엇인가?는 의 순서수이며, 자연수계수 다항식수 다항식도 적용한다. 역사 고대 그리스 신화에서 출발한 한사람 기원전 3세기 말까지",
            "gemma": "초월수는 대수적인 다항식의 근이 될 수 없는 수입니다. 주요 특징: 대수적 폐쇄성과의 관계, 계층적 포함 관계, 예시: 원주율(π), 자연상수(e) 등.",
            "gemini": "정보에 따르면 초월수에 대한 직접적인 정의는 나와 있지 않습니다.",
        },
    },
    {
        "question": "맥스웰 방정식은 무엇인가?",
        "reference": "전기와 자기의 발생, 전기장과 자기장, 전하 밀도와 전류 밀도의 형성을 나타내는 4개의 편미분 방정식이다.",
        "answers": {
            "ella": "맥스웰 방정식(-方程式, s)은 전기와 자기의 발생, 전기장과 자기장, 전하 밀도와 전류 밀도의 형성을 나타내는 4개의 편미분 방정식이다. ... 맥스웰 방정식은 무엇인가?는 것을 설명하기 위한 딜의 한 원리가 있다. 맥스웰은 맥스웰이 맥스웰 방정식에 대한 전자기파의 전자기파를 통해 설명될 수 있다.",
            "gemma": "맥스웰 방정식은 전기와 자기의 발생, 전기장과 자기장, 전하 밀도와 전류 밀도의 형성을 나타내는 4개의 편미분 방정식입니다. 가우스 법칙, 가우스 자기 법칙, 패러데이 전자기 유도 법칙, 앙페르 회로 법칙으로 구성됩니다. 빛 역시 전자기파의 한 형태임을 보여주었습니다.",
            "gemini": "맥스웰 방정식은 전기와 자기의 발생, 전기장과 자기장, 전하 밀도와 전류 밀도의 형성을 나타내는 4개의 편미분 방정식이다.",
        },
    },
    {
        "question": "세종대왕은 어떤 업적으로 유명한가?",
        "reference": "1443년 한글(훈민정음)을 창제하기 시작하여 1446년 반포했다.",
        "answers": {
            "ella": "왕. (897년~) 1573년 - 일본 센고쿠 시대의 장수, 다이묘 다케다 신겐. ... 세종대왕은 어떤 업적으로 유명한가?는 미국의 배우, 영화 제작자, 영화 각본가, 영화 감독, 연극부문 작가, 영화 배우자, 영화음악가, 각본가, 영화 제작자, 감독상 수상자",
            "gemma": "제공된 정보에는 세종대왕에 대한 내용이 없습니다.",
            "gemini": "모른다.",
        },
    },
]

SCORE_RUBRIC = """
점수 기준 (각 항목 공통):
5 = 완벽함
4 = 거의 맞음, 미미한 누락/표현 차이만 있음
3 = 부분적으로 맞음, 핵심 일부 누락
2 = 핵심 정보가 대부분 누락됨
1 = 대부분 틀리거나 무관함
0 = 완전히 틀리거나 질문과 무관함
"""


def build_judge_prompt(question, reference, answers: dict):
    answers_block = "\n\n".join(
        f"[{name}]\n{text}" for name, text in answers.items()
    )
    model_names = list(answers.keys())

    return f"""당신은 한국어 위키백과 지식에 대한 질의응답 품질을 평가하는 전문가입니다.
아래 질문과 모범답안을 기준으로, 세 모델의 답변을 각각 0~5점씩
정확성/관련성/자연스러움 세 기준으로 채점하세요.
{SCORE_RUBRIC}
답변이 정보 부족을 인정하며 거부한 경우, 모범답안에 필요한 정보가
실제로 주어지지 않았다면 정직한 대응으로 보고 관련성에서 높은 점수를
줄 수 있습니다.

기준:
- 정확성: 모범답안과 사실 내용이 일치하는가
- 관련성: 질문에서 요구한 정보에 정확히 답하는가, 무관한 내용이 섞여있지 않은가
- 자연스러움: 문법적으로 통하는 완성된 문장인가

질문: {question}
모범답안: {reference}

{answers_block}

각 모델에 대해 정확히 다음 형식으로만 출력하세요(다른 설명 없이):

[{model_names[0]}] 정확성: X, 관련성: X, 자연스러움: X
이유: (한 문장으로 짧게)
[{model_names[1]}] 정확성: X, 관련성: X, 자연스러움: X
이유: (한 문장으로 짧게)
[{model_names[2]}] 정확성: X, 관련성: X, 자연스러움: X
이유: (한 문장으로 짧게)
"""


def parse_scores(text, model_names):
    result = {}
    for name in model_names:
        # "정확성: 5", "정확성:5", "정확성 - 5" 등 표기 변형을 허용.
        # 이유는 non-greedy + 다음 모델 태그/문자열 끝 전까지로 한정해
        # 다음 모델 블록을 침범하지 않게 함.
        pattern = (
            rf"\[{name}\].*?정확성\s*[:\-]\s*(\d+).*?"
            rf"관련성\s*[:\-]\s*(\d+).*?"
            rf"자연스러움\s*[:\-]\s*(\d+)"
            rf"(?:\s*\n\s*이유[:\-]?\s*(.+?))?(?=\n\[|\Z)"
        )
        match = re.search(pattern, text, re.DOTALL)
        if match:
            result[name] = {
                "정확성": int(match.group(1)),
                "관련성": int(match.group(2)),
                "자연스러움": int(match.group(3)),
                "이유": match.group(4).strip() if match.group(4) else None,
            }
        else:
            result[name] = None
    return result


def judge_question(question, reference, answers, retries=3):
    model_names = list(answers.keys())
    prompt = build_judge_prompt(question, reference, answers)
    for attempt in range(retries):
        try:
            response = judge_model.generate_content(prompt)
            return parse_scores(response.text, model_names)
        except ResourceExhausted as e:
            if attempt >= retries - 1:
                raise
            # API가 알려주는 정확한 재시도 대기 시간을 사용 (없으면 60초)
            retry_delay = 60
            match = re.search(r"retry in (\d+(?:\.\d+)?)", str(e))
            if match:
                retry_delay = float(match.group(1)) + 2  # 여유 2초 추가
            print(f"  (요청 제한, {retry_delay:.0f}초 대기 후 재시도 {attempt + 1}/{retries})")
            time.sleep(retry_delay)


if __name__ == "__main__":
    all_results = []
    for item in EVAL_DATA:
        q = item["question"]
        ref = item["reference"]
        print(f"=== {q} ===")

        scores = judge_question(q, ref, item["answers"])
        print(f"  {scores}")

        all_results.append({"question": q, "scores": scores})
        time.sleep(15)  # 무료 등급 분당 제한 회피용 여유

    output_path = os.path.join("docs", "evaluation_v1_judge_scores.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {output_path}")