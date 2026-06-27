import os
import time
import google.generativeai as genai

from rag.vanilla.retriever import search

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

QUESTIONS = [
    "지미 카터는 누구인가?",
    "백남준은 어떤 예술 분야로 유명한가?",
    "데니스 리치는 어떤 프로그래밍 언어 개발에 기여했는가?",
    "우크라이나는 어디에 위치한 국가인가?",
    "체첸 공화국은 어느 지역에 위치하는가?",
    "수학은 무엇을 연구하는 학문인가?",
    "화학은 무엇을 연구하는 학문인가?",
    "초월수란 무엇인가?",
    "맥스웰 방정식은 무엇인가?",
    "세종대왕은 어떤 업적으로 유명한가?",
]


def ask_gemini(question: str, context: str, retries: int = 3) -> str:
    prompt = f"""다음 정보를 참고하여 질문에 간결하고 정확하게 답하세요.
정보에 답이 없으면 모른다고 답하세요.

정보:
{context}

질문: {question}
답변:"""

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) and attempt < retries - 1:
                print(f"  (요청 제한, 60초 대기 후 재시도 {attempt + 1}/{retries})")
                time.sleep(60)
            else:
                raise


if __name__ == "__main__":
    for i, q in enumerate(QUESTIONS, start=1):
        results = search(q, top_k=3)
        context = "\n\n".join(r["text"] for r in results)

        answer = ask_gemini(q, context)
        print(f"=== Q{i}: {q} ===")
        print(answer)
        print()

        time.sleep(15)  # 무료 등급 분당 5회 제한을 피하기 위해 요청 사이 대기