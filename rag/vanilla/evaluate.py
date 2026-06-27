"""
Evaluation Dataset v1의 10개 질문을 엘라 모델(Wiki RAG)로 실행해
docs/evaluation_dataset_v1.md의 비교표에 채울 응답을 수집.

실행: python3 -m rag.vanilla.evaluate
"""

from rag.vanilla.pipeline import build_rag_prompt_v2
from config.config import MODELS
from tokenizer.tokenizer import KoreanTokenizer
from model.transformer import GPTMini
from inference.generate import generate

import torch

DEVICE = "cpu"

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


def load_wiki_model():
    cfg = MODELS["wiki"]
    tokenizer = KoreanTokenizer(cfg["tokenizer_path"], eos_token=cfg["eos_token"])
    model = GPTMini(
        vocab_size=tokenizer.vocab_size,
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        ff_dim=cfg["ff_dim"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
    )
    model.load_state_dict(torch.load(cfg["model_path"], map_location=DEVICE))
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_wiki_model()

    for i, q in enumerate(QUESTIONS, start=1):
        prompt = build_rag_prompt_v2(q, top_k=1)
        answer = generate(model, tokenizer, prompt, max_new_tokens=60,
                           temperature=0.7, top_k=50, device=DEVICE)
        print(f"=== Q{i}: {q} ===")
        print(answer)
        print()