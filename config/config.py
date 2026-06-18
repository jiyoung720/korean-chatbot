import os

# 프로젝트 루트 기준 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# 모델 가중치 / 토크나이저 파일 경로
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "gpt_mini_latest.pt")
TOKENIZER_PATH = os.path.join(ARTIFACTS_DIR, "ko_sp_16k.model")

# 모델 하이퍼파라미터 (학습 시 사용한 값과 반드시 동일해야 함)
EMBED_DIM = 384
NUM_HEADS = 6
FF_DIM = 1536
NUM_LAYERS = 6
MAX_SEQ_LEN = 256

# 생성 옵션 기본값
DEFAULT_MAX_NEW_TOKENS = 50
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 50