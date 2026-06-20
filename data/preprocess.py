import re
from pathlib import Path

MAX_SPEAKER_TOKENS = 5  # <sp1> ~ <sp5>

def replace_masks(text: str) -> str:
    """AI Hub 익명화 토큰(*, **, ***)을 <maskN>으로 변환."""
    return re.sub(
        r"\*+",
        lambda m: f"<mask{min(len(m.group()), 8)}>",
        text
    )

def parse_dialog(txt_path: Path) -> list[tuple[str, str]]:
    """원본 txt 파일을 읽어 (화자ID, 발화) 튜플 리스트로 변환.
    화자ID는 원본 그대로(예: '12', '45') — 아직 정규화 전.
    """
    lines = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = re.match(r"^\s*(\d+)\s*:\s*(.*)$", line)
            if m:
                speaker_id, utterance = m.group(1), m.group(2).strip()

                utterance = replace_masks(utterance)

                if utterance:
                    lines.append((speaker_id, utterance))
                    return lines


def normalize_speakers(dialog: list[tuple[str, str]]) -> list[tuple[int, str]]:
    """대화 내 첫 등장 순서대로 화자ID를 1, 2, 3...으로 재매핑.
    예: ('12', '안녕'), ('45', '반가워'), ('12', '뭐해')
        -> (1, '안녕'), (2, '반가워'), (1, '뭐해')

    원본 화자 ID가 몇 번이든(3, 7, 12...) 대화 내 등장 순서 기준으로
    1부터 다시 번호를 매기기 때문에, 화자 수가 가변적이어도
    <sp1>~<sp5>라는 고정된 작은 vocab으로 항상 표현 가능하다.
    """
    speaker_map = {}
    result = []
    for speaker_id, utterance in dialog:
        if speaker_id not in speaker_map:
            speaker_map[speaker_id] = len(speaker_map) + 1
        normalized_id = speaker_map[speaker_id]
        # MAX_SPEAKER_TOKENS를 초과하는 화자는 마지막 토큰으로 합침 (안전장치)
        normalized_id = min(normalized_id, MAX_SPEAKER_TOKENS)
        result.append((normalized_id, utterance))
    return result


def format_with_speaker_tokens(dialog: list[tuple[str, str]]) -> str:
    """화자 토큰을 포함한 학습용 텍스트로 변환.
    <sp1> 발화\n<sp2> 발화\n...\n<eot>
    """
    normalized = normalize_speakers(dialog)
    lines = [f"<sp{speaker_id}> {utterance}" for speaker_id, utterance in normalized]
    return "\n".join(lines) + "\n<eot>"


def format_plain(dialog: list[tuple[str, str]]) -> str:
    """화자 토큰 없이 발화만 이어붙인 텍스트로 변환 (A/B 비교용 baseline)."""
    utterances = [utterance for _, utterance in dialog]
    return "\n".join(utterances)


if __name__ == "__main__":
    # 동작 확인용 간단 테스트
    sample = [("1", "오늘 뭐 먹을까"), ("2", "치킨 먹자"), ("1", "좋다")]

    print("=== format_with_speaker_tokens ===")
    print(format_with_speaker_tokens(sample))
    print()
    print("=== format_plain ===")
    print(format_plain(sample))