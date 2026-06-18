from fastapi import FastAPI
from pydantic import BaseModel

from config.config import (
    MODEL_PATH, TOKENIZER_PATH,
    EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS, MAX_SEQ_LEN,
    DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_K,
)
from tokenizer.tokenizer import KoreanTokenizer
from model.transformer import GPTMini
from inference.generate import generate

import torch

app = FastAPI()

# ── 모델 로딩과 추론 로직 분리 ──
# 앱이 시작될 때 한 번만 모델/토크나이저를 메모리에 올려둠
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = KoreanTokenizer(TOKENIZER_PATH)

model = GPTMini(
    vocab_size=tokenizer.vocab_size,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    answer = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=request.message,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_k=DEFAULT_TOP_K,
        device=device,
    )
    return ChatResponse(answer=answer)