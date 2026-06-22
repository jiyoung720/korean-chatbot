"""
AI Hub '주제별 텍스트 일상 대화 데이터' (dataSetSn=543) 전용 로더.

이 파일은 AI Hub 543 데이터셋의 폴더 구조와 파일 포맷에 종속적인 코드만 담당한다.
실제 화자 정규화/포맷 변환 같은 재사용 가능한 로직은 preprocess.py에 있다.

나중에 다른 대화 데이터셋(예: 오픈카카오, 자체 수집 데이터)을 추가할 때는
이 파일과 같은 구조로 load_xxx_dataset() 함수를 새로 만들면 되고,
preprocess.py의 함수들은 그대로 재사용한다.
"""

import os
from pathlib import Path

from preprocess import parse_dialog, format_with_speaker_tokens, format_plain

# 이 파일(data/build_corpus.py)이 어디서 실행되든 항상 프로젝트 루트를 정확히 찾기 위함
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../korean-chatbot/data
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                 # .../korean-chatbot


def load_aihub_dialogs(raw_dir: Path, exclude_dirs: set[str] | None = None):
    """raw_dir 아래의 모든 .txt 파일을 순회하며 (파일경로, 대화) 쌍을 생성.

    AI Hub 543 데이터셋은 zip을 풀면 플랫폼별 폴더(KAKAO, FACEBOOK, ...) 아래에
    수만 개의 .txt 파일이 흩어져 있는 구조다.
    """
    exclude_dirs = exclude_dirs or set()

    for txt_file in raw_dir.rglob("*.txt"):
        if any(excluded in txt_file.parts for excluded in exclude_dirs):
            continue
        dialog = parse_dialog(txt_file)
        if dialog:  # 빈 파일은 건너뜀
            yield txt_file, dialog


def build_corpus(raw_dir: str, output_path: str, use_speaker_tokens: bool = True,
                  exclude_dirs: set[str] | None = None):
    """raw_dir의 모든 대화를 전처리해 output_path에 한 줄씩(대화 단위) 저장.

    use_speaker_tokens=True  -> <sp1>, <sp2>, <eot> 포함 (메인 학습용)
    use_speaker_tokens=False -> 발화만 이어붙임 (A/B 비교 baseline)
    """
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    format_fn = format_with_speaker_tokens if use_speaker_tokens else format_plain

    num_dialogs = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for _, dialog in load_aihub_dialogs(raw_dir, exclude_dirs=exclude_dirs):
            formatted = format_fn(dialog)
            out.write(formatted + "\n\n")  # 대화 사이는 빈 줄로 구분
            num_dialogs += 1

    print(f"corpus 생성 완료: {output_path} (대화 {num_dialogs:,}개, "
          f"{'speaker_tokens' if use_speaker_tokens else 'plain'} 포맷)")


if __name__ == "__main__":
    RAW_DIR = os.path.join(
        PROJECT_ROOT,
        "data/raw/020.주제별 텍스트 일상 대화 데이터/01.데이터/1.Training/원천데이터",
    )

    EXCLUDE = {"kakao1_test"}  # 테스트용 폴더 제외

    raw_path = Path(RAW_DIR)

    print(f"[DEBUG] raw_dir: {raw_path}")
    print(f"[DEBUG] exists: {raw_path.exists()}")

    txt_files = list(raw_path.rglob("*.txt"))
    print(f"[DEBUG] txt files: {len(txt_files)}")

    if txt_files:
        print(f"[DEBUG] first file: {txt_files[0]}")

    build_corpus(
        raw_dir=RAW_DIR,
        output_path=os.path.join(PROJECT_ROOT, "data/processed/dialog_corpus_speaker.txt"),
        use_speaker_tokens=True,
        exclude_dirs=EXCLUDE,
    )

    build_corpus(
        raw_dir=RAW_DIR,
        output_path=os.path.join(PROJECT_ROOT, "data/processed/dialog_corpus_plain.txt"),
        use_speaker_tokens=False,
        exclude_dirs=EXCLUDE,
    )