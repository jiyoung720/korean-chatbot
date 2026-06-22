"""
Vanilla RAG 데모

실행 방법 (프로젝트 루트에서):
    python3 -m rag.vanilla.demo

검색(청킹 -> 임베딩 -> 코사인 유사도 검색) -> 프롬프트 조립 -> 생성까지
전체 RAG 파이프라인을 직접 구현한 버전으로 시연합니다.

실험 결과 요약(자세한 내용은 docs/training_log.md 참고):
- 검색은 정상 동작하며 관련 청크를 정확히 찾아옴
- 생성 모델(23M, from scratch)의 표현력 한계로 검색된 정보를 의미 있는
  문장으로 정리하는 데는 한계가 있음 (키워드 수준의 반영은 확인됨)
- Wiki 모델: 이어쓰기 형식 + top_k=1이 QA 형식 + top_k=3보다 더 나음
- Dialog 모델: 도메인이 다른(위키) 정보를 넣으면 더 빠르게 무너짐
"""

import torch

from rag.vanilla.pipeline import build_rag_prompt_v2, build_rag_prompt_dialog
from config.config import MODELS
from tokenizer.tokenizer import KoreanTokenizer
from model.transformer import GPTMini
from inference.generate import generate

DEVICE = "cpu"


def load_model(model_key: str):
    cfg = MODELS[model_key]
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


def run_wiki_rag(query: str):
    model, tokenizer = load_model("wiki")
    prompt = build_rag_prompt_v2(query, top_k=1)
    answer = generate(model, tokenizer, prompt, max_new_tokens=60,
                       temperature=0.7, top_k=50, device=DEVICE)
    return prompt, answer


def run_dialog_rag(query: str):
    model, tokenizer = load_model("dialog")
    prompt = build_rag_prompt_dialog(query, top_k=1)
    answer = generate(model, tokenizer, prompt, max_new_tokens=50,
                       temperature=0.7, top_k=50, device=DEVICE)
    return prompt, answer


if __name__ == "__main__":
    questions = ["세종대왕은 누구야?", "지미 카터는 누구야?"]

    for q in questions:
        print(f"\n{'=' * 50}\n질문: {q}\n{'=' * 50}")

        print("\n[Wiki 모델 + RAG]")
        _, answer = run_wiki_rag(q)
        print(answer)

        print("\n[Dialog 모델 + RAG]")
        _, answer = run_dialog_rag(q)
        print(answer)