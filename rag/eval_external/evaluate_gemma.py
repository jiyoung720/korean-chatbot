import ollama

from rag.vanilla.retriever import search

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


def ask_gemma(question: str, context: str) -> str:
    prompt = f"""다음 정보를 참고하여 질문에 간결하고 정확하게 답하세요.

정보:
{context}

질문: {question}
답변:"""

    response = ollama.chat(
        model="gemma4",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    for i, q in enumerate(QUESTIONS, start=1):
        results = search(q, top_k=3)
        context = "\n\n".join(r["text"] for r in results)

        answer = ask_gemma(q, context)
        print(f"=== Q{i}: {q} ===")
        print(answer)
        print()
