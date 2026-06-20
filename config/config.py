import os

# 프로젝트 루트 기준 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
TOKENIZER_DIR = os.path.join(ARTIFACTS_DIR, "tokenizer")

# 모델별 설정 (학습 시 사용한 값과 반드시 동일해야 함)
MODELS = {
    "wiki": {
        "model_path": os.path.join(ARTIFACTS_DIR, "gpt_wiki_latest.pt"),
        "tokenizer_path": os.path.join(TOKENIZER_DIR, "ko_sp_wiki_16k.model"),
        "embed_dim": 384,
        "num_heads": 6,
        "ff_dim": 1536,
        "num_layers": 6,
        "max_seq_len": 256,
        "eos_token": "<eos>",  # 위키 모델은 표준 EOS 사용
    },
    "dialog": {
        "model_path": os.path.join(ARTIFACTS_DIR, "gpt_dialog_latest.pt"),
        "tokenizer_path": os.path.join(TOKENIZER_DIR, "ko_sp_dialog_16k.model"),
        "embed_dim": 384,
        "num_heads": 6,
        "ff_dim": 1536,
        "num_layers": 6,
        "max_seq_len": 128,
        "eos_token": "<eot>",  # 대화 모델은 대화 종료 토큰을 생성 종료 신호로 사용
    },
}

DEFAULT_MODEL = "dialog"  # /chat 요청에서 model을 지정하지 않으면 사용할 기본값

# 생성 옵션 기본값
DEFAULT_MAX_NEW_TOKENS = 50
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 50