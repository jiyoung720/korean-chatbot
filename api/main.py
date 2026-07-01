from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

from config.config import (
    MODELS, DEFAULT_MODEL,
    DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_K,
)
from tokenizer.tokenizer import KoreanTokenizer
from model.transformer import GPTMini
from inference.generate import generate
from rag.langchain.chain import build_rag_chain

import torch

app = FastAPI()

# ── 모델 로딩과 추론 로직 분리 ──
# 앱이 시작될 때 두 모델을 한 번만 메모리에 올려둠
device = "cuda" if torch.cuda.is_available() else "cpu"

loaded_models = {}

for name, cfg in MODELS.items():
    tokenizer = KoreanTokenizer(cfg["tokenizer_path"], eos_token=cfg["eos_token"])

    model = GPTMini(
        vocab_size=tokenizer.vocab_size,
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        ff_dim=cfg["ff_dim"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
    )
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device))
    model.to(device)
    model.eval()

    loaded_models[name] = {"model": model, "tokenizer": tokenizer}

print(f"로드된 모델: {list(loaded_models.keys())}")

# ── RAG 체인도 앱 시작 시 한 번만 빌드 ──
# 매 요청마다 vectorstore를 새로 로드하지 않도록 모듈 레벨에서 한 번만 생성
rag_chain = build_rag_chain(model_key="wiki", top_k=1)

print("RAG 체인 빌드 완료")


class ChatRequest(BaseModel):
    message: str
    model: Literal["wiki", "dialog"] = DEFAULT_MODEL


class ChatResponse(BaseModel):
    answer: str
    model_used: str


class RagChatRequest(BaseModel):
    message: str


class RagChatResponse(BaseModel):
    answer: str
    model_used: str = "wiki+rag"


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if request.model not in loaded_models:
        raise HTTPException(status_code=400, detail=f"알 수 없는 모델: {request.model}")

    selected = loaded_models[request.model]

    answer = generate(
        model=selected["model"],
        tokenizer=selected["tokenizer"],
        prompt=request.message,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_k=DEFAULT_TOP_K,
        device=device,
    )
    return ChatResponse(answer=answer, model_used=request.model)


@app.post("/rag-chat", response_model=RagChatResponse)
def rag_chat(request: RagChatRequest):
    """검색(Retriever) -> 프롬프트 조립 -> KoreanGPTWrapper 생성까지
    LangChain RAG 체인을 그대로 거쳐 응답을 생성하는 엔드포인트.

    /chat과 달리 RAG가 적용되어, 질문과 관련된 위키 문서를 검색해
    컨텍스트로 함께 넣은 뒤 생성한다 (docs/training_log.md 5~6절 참고).
    """
    try:
        answer = rag_chain.invoke(request.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG 체인 실행 중 오류: {e}")

    return RagChatResponse(answer=answer)