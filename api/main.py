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


class ChatRequest(BaseModel):
    message: str
    model: Literal["wiki", "dialog"] = DEFAULT_MODEL


class ChatResponse(BaseModel):
    answer: str
    model_used: str


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