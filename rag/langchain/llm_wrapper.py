from typing import Optional, List
from langchain_core.language_models.llms import LLM

from model.transformer import GPTMini
from tokenizer.tokenizer import KoreanTokenizer
from inference.generate import generate
from config.config import MODELS

import torch


class KoreanGPTWrapper(LLM):
    """우리가 직접 만든 GPTMini 모델을 LangChain의 LLM 인터페이스로 감싼 래퍼.
    LangChain은 보통 OpenAI 등 외부 API를 가정하지만, 이 클래스를 통해
    직접 학습한 모델도 LangChain 체인 안에서 동일하게 사용할 수 있다."""

    model_key: str = "wiki"
    max_new_tokens: int = 60
    temperature: float = 0.7
    top_k_sampling: int = 50

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cfg = MODELS[self.model_key]
        self._tokenizer = KoreanTokenizer(cfg["tokenizer_path"], eos_token=cfg["eos_token"])
        self._model = GPTMini(
            vocab_size=self._tokenizer.vocab_size,
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            ff_dim=cfg["ff_dim"],
            num_layers=cfg["num_layers"],
            max_seq_len=cfg["max_seq_len"],
        )
        self._model.load_state_dict(torch.load(cfg["model_path"], map_location="cpu"))
        self._model.eval()

    @property
    def _llm_type(self) -> str:
        return "korean-gpt-from-scratch"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return generate(
            self._model, self._tokenizer, prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k_sampling,
            device="cpu",
        )