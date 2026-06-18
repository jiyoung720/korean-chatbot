"""
한국어 다음 단어 예측 모델 학습 스크립트

실행 환경: Google Colab (GPU 필요)

Colab에서 실행하는 방법:
    1. 런타임 유형을 GPU(T4 권장)로 설정
    2. 아래 명령으로 이 레포를 clone
        !git clone https://github.com/jiyoung720/korean-chatbot.git
        %cd korean-chatbot
        !pip install -q datasets sentencepiece torch
    3. 이 파일을 실행하거나, 셀 단위로 나눠서 실행
        !python train/train.py

학습 결과물(gpt_mini_latest.pt, ko_sp_16k.model/.vocab)은 artifacts/ 에 저장되며,
이후 Google Drive 등을 통해 로컬(api/, inference/가 동작하는 환경)로 옮겨야 합니다.
"""

import os
import re
import shutil

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from model.transformer import GPTMini
from inference.generate import generate


# ── 0. 경로 설정 ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

CORPUS_PATH = os.path.join(DATA_DIR, "wiki_corpus.txt")
TOKENIZER_PREFIX = os.path.join(ARTIFACTS_DIR, "ko_sp_16k")
MODEL_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "gpt_mini_latest.pt")

# ── 모델/학습 하이퍼파라미터 ──
# (model/transformer.py 구조에 맞춰 config/config.py와 동일한 값 유지)
VOCAB_SIZE_TARGET = 16000
EMBED_DIM = 384
NUM_HEADS = 6
FF_DIM = 1536
NUM_LAYERS = 6
MAX_SEQ_LEN = 256
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 5
NUM_WIKI_DOCS = 5000  # 검증용 샘플 수. 본 학습 시 늘릴 수 있음


def clean_text(text: str) -> str:
    """위키백과 특유의 빈 괄호/쉼표 패턴과 공백/줄바꿈 노이즈만 가볍게 정리."""
    text = re.sub(r"\(\s*,\s*", "(", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


class NextTokenDataset(Dataset):
    """전체 토큰 시퀀스를 겹치지 않게 (max_seq_len+1) 단위로 잘라
    (입력, 한 칸씩 밀린 정답) 쌍을 만드는 Dataset.

    슬라이딩 윈도우(1토큰씩 이동) 대신 이 방식을 쓰는 이유:
    슬라이딩 윈도우는 거의 동일한 샘플을 토큰 수만큼 만들어내 1 epoch에
    25만 스텝 이상이 필요했음. 겹치지 않게 자르면 같은 데이터로 998 스텝까지 축소됨.
    """

    def __init__(self, token_ids: torch.Tensor, max_seq_len: int):
        self.max_seq_len = max_seq_len
        chunk_size = max_seq_len + 1
        num_chunks = len(token_ids) // chunk_size
        usable_len = num_chunks * chunk_size
        self.chunks = token_ids[:usable_len].view(num_chunks, chunk_size)

    def __len__(self):
        return self.chunks.size(0)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]


def prepare_corpus() -> list[str]:
    """한국어 위키백과에서 일부 문서를 받아 가볍게 정제."""
    print(f"위키백과 {NUM_WIKI_DOCS}개 문서 로딩 중...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)

    texts = []
    for i, example in enumerate(dataset):
        texts.append(example["text"])
        if i >= NUM_WIKI_DOCS:
            break

    cleaned = [clean_text(t) for t in texts]
    print(f"문서 {len(cleaned)}개 정제 완료")
    return cleaned


def train_tokenizer(cleaned_texts: list[str]) -> spm.SentencePieceProcessor:
    """정제된 텍스트로 SentencePiece(BPE) 토크나이저 학습."""
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for text in cleaned_texts:
            f.write(text + "\n")

    spm.SentencePieceTrainer.train(
        input=CORPUS_PATH,
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=VOCAB_SIZE_TARGET,
        character_coverage=0.9995,  # 한국어는 문자 종류가 많아 1.0 대신 권장값 사용
        model_type="bpe",
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
    )

    sp = spm.SentencePieceProcessor()
    sp.load(f"{TOKENIZER_PREFIX}.model")
    print(f"토크나이저 학습 완료 (vocab_size={sp.get_piece_size()})")
    return sp


def build_dataloader(cleaned_texts: list[str], sp: spm.SentencePieceProcessor) -> DataLoader:
    """문서들을 EOS로 구분해 하나의 토큰 시퀀스로 합치고 DataLoader로 변환."""
    all_token_ids = []
    for text in cleaned_texts:
        all_token_ids.extend(sp.encode(text, out_type=int))
        all_token_ids.append(sp.eos_id())  # 문서 경계 표시 → 모델이 "여기서 한 주제가 끝난다"를 학습

    all_token_ids = torch.tensor(all_token_ids, dtype=torch.long)
    print(f"전체 토큰 수: {len(all_token_ids):,}")

    dataset = NextTokenDataset(all_token_ids, max_seq_len=MAX_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"1 epoch 스텝 수: {len(loader)}")
    return loader


def train(model, loader, device, sp):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)}  loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"=== epoch {epoch} 평균 loss: {avg_loss:.4f} ===")

        # 학습 중간에도 생성 품질을 눈으로 확인
        sample = generate(model, _SPWrapper(sp), "오늘 날씨가",
                           max_new_tokens=30, temperature=0.8, top_k=50, device=device)
        print(f"[샘플 생성] {sample}\n")

        # 매 epoch 끝날 때마다 같은 파일에 덮어쓰기
        # (epoch마다 다른 파일명으로 쌓으면 Drive 용량이 계속 늘어나므로 최신 상태만 유지)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"epoch {epoch + 1} 모델 저장 완료\n")


class _SPWrapper:
    """학습 스크립트 안에서 raw SentencePieceProcessor를
    inference.generate()가 기대하는 인터페이스(encode/decode/eos_id)에 맞추기 위한 어댑터.
    실제 서빙(api/main.py)에서는 tokenizer/tokenizer.py의 KoreanTokenizer를 사용한다.
    """

    def __init__(self, sp):
        self.sp = sp

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(ids)

    @property
    def eos_id(self):
        return self.sp.eos_id()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("사용 디바이스:", device)

    cleaned_texts = prepare_corpus()
    sp = train_tokenizer(cleaned_texts)
    loader = build_dataloader(cleaned_texts, sp)

    model = GPTMini(
        vocab_size=sp.get_piece_size(),
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"전체 파라미터 수: {num_params:,}")

    train(model, loader, device, sp)

    # Colab 환경이면 Google Drive에도 백업 (선택)
    try:
        from google.colab import drive  # noqa
        drive.mount("/content/drive")
        drive_dir = "/content/drive/MyDrive/korean_chatbot_project"
        os.makedirs(drive_dir, exist_ok=True)
        shutil.copy(MODEL_SAVE_PATH, drive_dir)
        shutil.copy(f"{TOKENIZER_PREFIX}.model", drive_dir)
        shutil.copy(f"{TOKENIZER_PREFIX}.vocab", drive_dir)
        print("Google Drive 백업 완료:", drive_dir)
    except ImportError:
        print("Colab 환경이 아니므로 Drive 백업을 건너뜁니다.")


if __name__ == "__main__":
    main()