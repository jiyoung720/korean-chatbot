import sentencepiece as spm


class KoreanTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()