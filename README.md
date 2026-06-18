# Korean Next Word Prediction Chatbot

한국어 문장을 입력받아 다음 토큰을 예측하고, 이를 반복 생성하여 완전한 문장을 만드는 챗봇입니다.
GPT 스타일 Decoder Transformer를 직접 설계·구현했습니다. (AI 페어 프로그래밍으로 구현 속도를 높였고,
설계 결정과 트레이드오프는 직접 검토하고 선택했습니다 — 자세한 내용은 아래 "설계 결정과 이유" 참고)FastAPI로 감싸 웹에서 호출할 수 있게 만들었습니다.

> 부트캠프(KTB4) 과제로 시작했지만, 이후 RAG·벡터DB·모델 교체 등을 단계적으로 추가하며
> 지속적으로 발전시킬 개인 포트폴리오 프로젝트입니다.

## 데모

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "오늘 날씨가"}'

# {"answer": "오늘 날씨가 되는 날씨를 날아 날려오는 날렵하다. 같이 보기 전날: 11월 3일 ..."}
```

## 아키텍처

```
                    사용자 요청 (POST /chat)
                            │
                            ▼
                    ┌───────────────┐
                    │   api/main.py  │  FastAPI 엔드포인트
                    └───────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
       ┌───────────┐ ┌────────────┐ ┌──────────┐
       │ tokenizer/ │ │   model/   │ │  config/ │
       │  (인코딩)   │ │ (구조 정의) │ │ (설정값)  │
       └─────┬─────┘ └─────┬──────┘ └──────────┘
             │             │
             └──────┬──────┘
                     ▼
            ┌──────────────────┐
            │ inference/        │  반복 생성
            │ generate.py        │  (temperature, top-k, EOS)
            └──────────────────┘
                     │
                     ▼
              생성된 문장 응답
```

**RAG 확장 시 변경 범위**: 검색기(`retriever`)가 찾은 문서를 `inference/generate.py`의
프롬프트 조립 단계에 추가하는 정도로 끝납니다. `model/transformer.py`(모델 구조 자체)는
변경 대상이 아닙니다 — RAG는 "모델에 무엇을 입력으로 주는지"의 문제이지, 모델 구조의 문제가
아니기 때문입니다.

### 모델 구조 (GPT-style Decoder Transformer)

```
입력 토큰 ID 시퀀스
        │
        ▼
토큰 임베딩 + 위치 임베딩 (embed_dim=384)
        │
        ▼
┌─────────────────────────┐
│  Masked Self-Attention    │
│  Add & LayerNorm          │  × 6 layers
│  Feed-Forward (GELU)      │
│  Add & LayerNorm          │
└─────────────────────────┘
        │
        ▼
Linear + Softmax → vocab(16000)에 대한 확률 분포
        │
        ▼
Temperature / Top-k 샘플링 → 다음 토큰 선택 → 반복(autoregressive)
```

## 설계 결정과 이유

이 프로젝트는 "지금 당장 동작하는 코드"가 아니라 "이해하고 설명할 수 있는 코드"를
목표로 진행했습니다. 주요 결정들을 기록합니다.

| 결정 | 선택 | 이유 |
|---|---|---|
| 모델 아키텍처 | LSTM 대신 Decoder Transformer 직접 구현 | RAG로 확장할 때 self-attention 구조를 이해하고 있어야 동작 원리를 설명할 수 있음 |
| `model/` vs `inference/` 분리 | 분리 | RAG가 건드릴 범위(입력 조립)와 건드리지 않을 범위(모델 구조)의 경계를 명확히 하기 위함 |
| 토크나이저 구조 | 독립 `tokenizer/` 모듈 + 클래스로 래핑 | `model/`, `inference/`, 향후 `rag/`에서 공용으로 쓰이는 자원이라 특정 레이어에 종속시키지 않음 |
| vocab size | 16,000 (BPE) | 8,000과 비교했을 때 토큰 분리 품질 차이가 크지 않아, 데이터 양 대비 적정 수준으로 판단 |
| 학습 데이터 | 한국어 위키백과 5,000문서 | AI Hub 대화 데이터는 승인 대기가 필요해, 파이프라인 검증을 먼저 끝내기 위해 위키로 시작. 추후 데이터만 교체 예정 |
| 학습 시퀀스 분할 | 겹치지 않는(non-overlapping) 청크 | 슬라이딩 윈도우 방식은 거의 동일한 샘플을 8M개 가까이 생성해 1 epoch에 25만 스텝이 필요했음. 같은 데이터로 998 스텝까지 축소 |
| 디바이스 분리 | 학습은 Colab(GPU), 서빙은 로컬(CPU) | 학습은 GPU 연산량이 필요하지만 추론은 CPU로도 충분히 가능해, 비용/구조 모두에서 합리적 |

## 학습 결과

5 epoch, 위키백과 5,000문서(약 820만 토큰) 기준입니다.

![Training Loss](docs/loss_curve.png)

| Epoch | Avg Loss | 샘플 생성 (prompt: "오늘 날씨가") |
|---|---|---|
| 0 | 6.83 | `이다. 는 이다. (go cat) ...` |
| 1 | 5.57 | `이는 중국, 그 내용은 아니다. 같은 해 10월 12일 ...` |
| 4 | 4.47 | `되는 날씨를 날아 날려오는 날렵하다. 같이 보기 전날: 11월 3일 ...` |

문법적 조각(조사, 어미, 위키 특유의 "같이 보기" 구조)은 학습되고 있으나, 위키백과가
서술체 데이터라 자연스러운 대화체 응답에는 한계가 있습니다. AI Hub 대화 데이터 확보 후
재학습이 다음 단계입니다.

## 프로젝트 구조

```
korean-chatbot/
├── api/             # FastAPI 라우터 (POST /chat)
├── config/          # 하이퍼파라미터, 경로 등 설정값
├── data/            # 원본/전처리 텍스트 데이터 (git 비추적)
├── tokenizer/        # SentencePiece 로딩/인코딩 래퍼
├── model/           # GPT 스타일 Decoder Transformer 구조 정의
├── inference/        # 반복 생성 로직 (temperature/top-k/EOS)
├── train/           # Colab에서 실행하는 학습 스크립트
├── artifacts/        # 학습된 가중치(.pt), 토크나이저 파일 (git 비추적)
├── tests/
├── docs/             # 학습 결과 그래프 등 문서 자료
└── README.md
```

`artifacts/`의 모델 가중치와 토크나이저 파일은 용량 문제로 git에 포함하지 않았습니다.
재현 방법은 `train/train.py`를 참고하세요.

## 실행 방법

```bash
# 가상환경 설정
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# artifacts/ 에 학습된 모델(gpt_mini_latest.pt)과
# 토크나이저(ko_sp_16k.model) 파일을 위치시킨 뒤

uvicorn api.main:app --reload
```

## 향후 계획

현재 구현은 1단계이며, 다음 항목들을 단계적으로 추가할 예정입니다. 구조상 RAG 추가 시
`inference/`와 신규 `rag/` 모듈만 영향을 받도록 설계했습니다.

- [ ] AI Hub 대화 데이터로 재학습 (대화체 응답 품질 개선)
- [ ] RAG (Retrieval-Augmented Generation) 추가
- [ ] 벡터DB 연동 (FAISS/Chroma)
- [ ] 대화 히스토리 저장, 세션 관리
- [ ] 모델 교체 가능한 추상화 계층 (`BaseLanguageModel`)
- [ ] Docker 기반 배포

## 기술 스택

Python 3.12 · PyTorch · SentencePiece · FastAPI · Uvicorn · Pydantic