import os
import sentencepiece as spm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tokenizer/ 의 부모 = 프로젝트 루트

spm.SentencePieceTrainer.train(
    input=os.path.join(BASE_DIR, "data/processed/dialog_corpus_speaker.txt"),
    model_prefix=os.path.join(BASE_DIR, "artifacts/tokenizer/ko_sp_dialog_16k"),
    vocab_size=16000,
    character_coverage=0.9995,
    model_type="bpe",
    user_defined_symbols=[
        "<sp1>", "<sp2>", "<sp3>", "<sp4>", "<sp5>",
        "<eot>",
        "<mask1>", "<mask2>", "<mask3>", "<mask4>",
        "<mask5>", "<mask6>", "<mask7>", "<mask8>",
    ],
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    pad_piece="<pad>", unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
)
print("대화용 토크나이저 학습 완료")

sp_dialog = spm.SentencePieceProcessor()
sp_dialog.load(os.path.join(BASE_DIR, "artifacts/tokenizer/ko_sp_dialog_16k.model"))

test = "<sp1> 오늘 뭐 먹을까?\n<sp2> 치킨 먹자\n<eot>"
pieces = sp_dialog.encode(test, out_type=str)
print(pieces)