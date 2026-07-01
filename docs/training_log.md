# Training Log

이 문서는 `Korean GPT-style Chatbot from Scratch` 프로젝트의 전체 학습 실험
기록입니다. README의 "주요 실험 과정 요약"에서 이어지는 상세 내용입니다.

## 1. Wiki 모델 학습

5 epoch, 위키백과 5,000문서(약 820만 토큰) 기준입니다.

![Training Loss](loss_curve.png)

| Epoch | Avg Loss | 샘플 생성 (prompt: "오늘 날씨가") |
|---|---|---|
| 0 | 6.83 | `이다. 는 이다. (go cat) ...` |
| 1 | 5.57 | `이는 중국, 그 내용은 아니다. 같은 해 10월 12일 ...` |
| 4 | 4.47 | `되는 날씨를 날아 날려오는 날렵하다. 같이 보기 전날: 11월 3일 ...` |

문법적 조각(조사, 어미, 위키 특유의 "같이 보기" 구조)은 학습되고 있으나, 위키백과가
서술체 데이터라 자연스러운 대화체 응답에는 한계가 있습니다.

## 2. AI Hub 대화 데이터 재학습 (Dialog v1)

AI Hub "주제별 텍스트 일상 대화 데이터"(dataSetSn=543)로 재학습했습니다. 카카오톡 등에서
수집한 일상 대화 87,689건(20개 주제, 평균 3턴/대화)을 화자 정규화(`<sp1>~<sp5>`)와
대화 종료 토큰(`<eot>`), 길이 보존 마스킹 토큰(`<mask1>~<mask8>`)으로 전처리해
새 토크나이저(`ko_sp_dialog_16k`)를 학습하고, 같은 모델 구조(384/6/6)로 10 epoch 학습했습니다.

| 결정 | 선택 | 이유 |
|---|---|---|
| 화자 표현 | `<sp1>~<sp5>` (대화 내 등장 순서로 정규화) | 원본 화자 ID가 가변적이어도 고정된 작은 vocab으로 표현 가능하며, 화자 교대(질문-응답) 구조를 보존하기 위함 |
| 대화 종료 | `<eot>` | 대화 경계를 모델이 학습하도록 함. 추론 시 생성 종료 신호로도 `<eot>`를 사용(`<eos>`와는 별개 역할로 유지) |
| 익명화 토큰 | `<mask1>~<mask8>` | AI Hub의 길이 보존 마스킹(`*`, `**`, `***`...) 정보를 유지해 호칭+조사 패턴 학습에 활용 |
| 시퀀스 길이 (초기 실험) | `max_seq_len=128` (위키 256에서 축소) | 당시 코퍼스가 버그로 1턴만 남아 있어, 대화가 짧다고 판단해 줄임 (버그 수정 후 256으로 환원 — 아래 참고) |

**학습 결과** (87,689개 대화, 약 101만 토큰, 모델 파라미터 약 23M, 10 epoch)

| Epoch | Avg Loss | 샘플 생성 (prompt: `<sp1> 오늘 너무 피곤해`) |
|---|---|---|
| 0 | 5.34 | `나 요즘 다 어떻게 생각해? <eot> 아 오늘 회사 좀 너무 힘들어 ...` |
| 4 | 3.83 | `오늘 진짜 오랜만에 집 매매하고 싶어서? <eot> 요즘은 아파트 값이 너무 좋아 ...` |
| 9 | 2.23 | `우리 강아지 미용 간다~ <eot> 오늘 날씨 추워지니까 벌써 일이다... <eot> 회사 출근하기 너무 좋다~` |

`<eot>` 경계와 "~", "...", "키키", "ㅠㅠ" 같은 구어체 어미가 학습 초반부터 안정적으로
나타났고, loss도 위키 모델보다 더 낮은 수준(2.23)까지 떨어졌습니다.

### 발견한 한계 (당시)

화자 토큰(`<sp1>`, `<sp2>`)이 텍스트로서는 정상 생성되었지만, *화자 전환*이라는
의미적 역할까지는 학습되지 않았습니다.

```
prompt: <sp1> 오늘 너무 피곤해\n<sp2>
output: <sp1> 오늘 너무 피곤해 <sp2>... <eot> ...
```

`<sp2>`가 새로운 응답의 시작이 아니라 `<sp1>` 발화 중간에 등장하는 토큰처럼
처리됩니다. 멀티턴 프롬프트(`<sp1> 오늘 너무 피곤해\n<sp2> 왜?\n<sp1> 야근했거든\n<sp2>`)로
테스트했을 때도 의미 있는 응답 대신 즉시 종료되는 패턴이 나타났습니다.

당시에는 정확한 원인을 알 수 없었고, 학습 방식·데이터 구성·데이터 규모를
후보로 추정했습니다. 화자 전환 패턴이 충분히 학습되지 않은 이유로는 다음이
복합적으로 영향을 준 것으로 보았습니다.

- Next Token Prediction 학습 방식의 한계 (통계적으로 빈번한 패턴인 조사, 어미를
  먼저 학습하고, "직전 발화를 이해하고 응답하는" 패턴은 신호가 희소해 늦게 학습됨)
- 대화 전체를 한 번에 제공한 데이터 구성 방식 ("이전 턴까지 주고 다음 턴을 맞히는"
  형태가 아니었음)
- 상대적으로 작은 데이터 규모

추가로, temperature/top-k 샘플링 특성상 같은 입력에도 호출마다 생성 품질 편차가
큽니다(예: 같은 prompt가 자연스러운 문장과 의미 없는 단어 나열을 번갈아 생성).
소규모 모델·데이터 규모에서 나타나는 자연스러운 현상으로 판단했습니다.

## 3. 데이터 파싱 버그 발견과 재학습 (Dialog v2)

### 코퍼스 규모 변화

| 항목 | 버그 있던 버전 | 수정 후 |
|---|---|---|
| 대화 수 | 87,689개 | 87,689개 (동일) |
| 코퍼스 줄 수 | 263,067줄 | 1,612,484줄 (약 6.1배) |
| 전체 토큰 수 | 약 101만 | 약 1,490만 (약 14.8배) |
| 평균 대화 길이 | 약 3턴 | 약 18.4턴 |

위 한계를 검증하기 위해 "이전 턴까지 주고 다음 턴을 예측하는" 학습 샘플을
직접 만드는 작업을 시작했는데, 그 과정에서 `data/preprocess.py`의 `parse_dialog()`에
**`return` 위치 버그**가 있다는 것을 발견했습니다. 함수 내부 `for` 루프 안에
`return`이 있어, 매 파일의 **첫 발화 한 줄만 읽고 즉시 반환**하고 있었습니다.

```python
# 버그가 있던 코드 (return이 for 루프 내부에 위치)
for line in f:
    ...
    if utterance:
        lines.append((speaker_id, utterance))
        return lines  # 첫 줄에서 즉시 종료됨

# 수정 후 (return을 루프 밖으로)
for line in f:
    ...
    if utterance:
        lines.append((speaker_id, utterance))
return lines
```

실제로 원본 raw 파일은 평균 18턴 이상의 멀티턴 대화였지만, 버그로 인해 학습
코퍼스에는 **거의 모든 대화가 1턴으로 잘려** 들어가 있었습니다. 화자 전환을
학습하지 못한 근본 원인이 데이터 자체에 있었던 것입니다.

버그를 수정하고 같은 학습 방식(전체 시퀀스에 loss 계산)으로 재학습한 결과,
별도의 학습 구조 변경 없이도 화자 전환이 뚜렷하게 개선되었습니다.

**재학습 결과** (87,689개 대화, 약 1,490만 토큰, `max_seq_len=256`으로 환원, 10 epoch)

![Dialog Model Loss Comparison](dialog_loss_comparison.png)

위 그래프에서 버그 있던 데이터(빨강)의 loss가 더 낮게 떨어지는 것은 모델이
더 잘 학습됐다는 뜻이 아니라, 1턴짜리 과제가 더 쉬웠기 때문입니다. 수정된
데이터(파랑)는 멀티턴 맥락을 다뤄야 하는 더 어려운 과제를 풀고 있어 loss가
더 높게 유지되지만, 실제 생성 품질은 명확히 더 낫습니다.

| Epoch | Avg Loss | 샘플 생성 (prompt: `<sp1> 오늘 너무 피곤해`) |
|---|---|---|
| 0 | 5.52 | (학습 시작) |
| 5 | 3.90 | `<sp2> 내일도 반팔 입고 나가지 않을까? <sp1> 맞아 얇고 더운거 같아 ...` |
| 7 | 3.73 | `<sp2> 키키 오늘 날씨 좋아? 키키 <sp3> 오 너무 좋다! ... <sp2> 나는 진짜 퇴사하고 싶다 ᅲᅲ <sp3> 나도 그래!` |
| 9 | 3.60 | `<sp2> 오늘 저녁시간이 4시더라고 ᅲᅲ <sp3> 너무 늦은 시간에 늦었는데 <sp1> 다들 아침 잘 챙겨 먹고 잘 챙겨!` |

최종 loss(3.60)는 버그가 있던 버전(2.23)보다 높지만, 이는 나쁜 신호가 아닙니다.
학습 데이터가 6배 늘고 시퀀스 길이도 늘어 과제 자체가 더 어려워졌기 때문이며,
실제 생성 품질은 명확히 개선되었습니다. `<sp2>`, `<sp3>`가 직전 발화에 대한
공감·맞장구·되묻기로 응답하는 패턴이 여러 epoch에서 일관되게 나타났습니다.

이번 경험으로 얻은 교훈은, 처음 세운 가설(학습 방식 자체의 한계)을 검증하기
전에 **데이터 파이프라인이 정확한지부터 확인해야 한다**는 것이었습니다.
Loss masking("이전 턴은 loss 계산에서 제외하고 다음 턴 생성에만 집중") 같은
학습 방식 개선은 여전히 의미 있는 다음 단계지만, 그 효과를 제대로 측정하려면
먼저 데이터가 온전한 상태여야 합니다.

## 4. Loss Masking 실험

데이터 버그 수정 후, 원래 가설이었던 학습 방식 개선(loss masking)을 마저
검증했습니다. 각 대화를 턴 경계마다 (이전 턴들, 다음 턴) 쌍으로 재구성하고,
`F.cross_entropy`의 `ignore_index=-100`을 이용해 이전 턴 부분은 loss 계산에서
제외했습니다. 정답 부분(다음 턴)에만 학습 신호가 집중되도록 한 것입니다.

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    y.view(-1),
    ignore_index=-100,  # 이전 턴 위치는 loss 계산에서 제외
)
```

134만 개 턴 단위 샘플 전체를 한 번에 텐서로 변환하려다 Colab 메모리 부족으로
런타임이 다운되는 일이 있어, 30만 개를 무작위로 샘플링해 사용했습니다(검증
목적에는 충분한 규모로 판단). 처음 만든 `Dataset`이 매 배치마다 패딩 연산을
반복해 1 epoch에 4시간 이상 걸렸는데, 패딩을 미리 텐서로 만들어두는 방식으로
바꾸고 batch_size를 32→64로 늘려 약 10배 빠르게 개선했습니다.

**실험 결과** (30만 샘플, `max_seq_len=256`, 5 epoch)

| Epoch | Avg Loss | 샘플 생성 (prompt: `<sp1> 오늘 너무 피곤해`) |
|---|---|---|
| 0 | 5.33 | `<sp2> 벌써 여름 겨울도 안 나더라` (맥락 무관) |
| 2 | 4.18 | `<sp2> 무슨일이야? <sp1> 오늘 좀 잤어? ...` (직접 반응 시작) |
| 3 | 3.87 | `<sp2> 갑자기 겨울이 왔다 갔다. <sp1> 맞아 바람도... <sp2> 맞아 기온이 되게 좋았어` |
| 4 | 3.59 | `<sp2> 오늘 너무 화창해 <sp1> 무슨일이야` (epoch 3보다 다소 단조로워짐) |

Epoch 3에서 화자들이 "맞아"로 공감하며 대화를 자연스럽게 이어가는, 지금까지
중 가장 일관된 응답이 나타났습니다. Epoch 4는 오히려 다소 단조로워졌는데,
데이터를 30만 개로 줄인 상태에서 5 epoch을 돈 것이 과적합 초기 단계에
들어선 것으로 추정합니다. 같은 epoch 수 기준으로 비교하면 loss masking
버전이 더 적은 데이터로도 일관된 응답 패턴에 더 빠르게 도달하는 경향을
보였으나, 데이터 양과 epoch 수가 달라 엄밀한 비교는 아닙니다.

이번 실험은 별도 체크포인트(`gpt_dialog_loss_masked.pt`)로 보존하고,
FastAPI가 서빙하는 기본 dialog 모델은 더 많은 데이터(134만 토큰 전체)로
검증된 기존 버전(Dialog v2)을 유지했습니다. RAG를 추가하는 시점에 모델
입력 형태가 다시 바뀌므로, 그 때 어떤 버전을 기반으로 할지 다시 판단할
계획입니다.

## 5. Vanilla RAG 구현

부트캠프 과제 요구사항(Vanilla RAG 구현 → LangChain 마이그레이션 → LangSmith
평가)에 따라, 먼저 라이브러리 없이 RAG 파이프라인 5단계(청킹 → 임베딩 →
검색 → 프롬프트 조립 → 생성)를 직접 구현했습니다.

### 구성

- **코퍼스**: 한국어 위키백과 1,000문서 (검증 목적으로 학습용 5,000문서보다
  적게 사용), 문서 단위 구분자(`===DOC===`)로 저장
- **임베딩**: `jhgan/ko-sroberta-multitask` (사전학습 모델, 한국어 검색
  벤치마크에서 검증된 모델). `intfloat/multilingual-e5-base` 등 query/
  passage를 구분해 학습한 최신 임베딩 모델도 후보였으나, 임베딩 모델 교체는
  코드 한 줄 차이라 우선 Vanilla RAG 구조 완성에 집중하고 추후 비교 실험
  대상으로 보류
- **청킹**: 고정 길이(300자, overlap 50자)로 단순 분할. 문장 경계 인식 등
  정교한 방법은 적용하지 않음 (지금 단계에 필요한 만큼만 구현)
- **검색**: NumPy로 코사인 유사도 직접 계산 (FAISS/Chroma 미사용)
- **구조**: `rag/shared/`(임베딩, 코퍼스, 청킹 — Vanilla/LangChain 공용),
  `rag/vanilla/`(retriever, pipeline)

19,495개 청크에 대해 인덱스(임베딩)를 구축해 캐싱했습니다(`pickle`).

### 검색 결과

질문 "세종대왕은 누구야?"에 대해 상위 결과가 한글 창제(1443년 창제,
1446년 훈민정음 반포) 관련 청크를 정확히 찾아왔습니다(score 0.48, 0.47).
검색기 자체는 의도대로 작동했습니다.

### 프롬프트 형식 실험

Wiki 모델(서술체 학습)에 검색 결과를 그대로 QA 형식(`질문: ... 답변:`)으로
주입했을 때는 의미 없는 명사 나열이 나왔습니다. 원인은 두 가지로 보입니다.

1. `max_seq_len=256`을 초과하는 긴 컨텍스트(top_k=3)가 뒤쪽만 남고 잘려,
   가장 관련도 높은 정보가 모델에 전달되지 못함
2. Wiki 모델은 QA 형식(질문-답변)을 학습한 적이 없어, `답변:` 다음에
   올 패턴에 대한 학습 신호가 전혀 없음

`top_k=1`로 컨텍스트를 줄이고, QA 형식 대신 모델이 학습한 서술체에 맞춘
이어쓰기 형식(`{검색된 텍스트}\n\n{주어}는`)으로 바꾼 결과, "한글",
"훈민정음", "세종실록" 등 관련 키워드를 더 오래 유지하는 출력이 나왔습니다.
다만 이마저도 의미가 통하는 문장은 아니었습니다(`세종대왕는 훈민정음
반포성술의 한자로, '월말'이란 제목의 한글에서 유래된...`).

### Dialog 모델과의 비교

같은 검색 결과를 Dialog 모델(`<sp1>`/`<sp2>` 화자 토큰 기반)에 화자 형식으로
주입했을 때는 더 빠르고 심하게 무너졌습니다. "지미 카터는 누구야?" 질문에서
검색된 인물 정보(조지아주, 해군 대위 등)의 앞부분은 원문을 그대로 따라가다,
이후 `<unk>` 토큰이 섞이며 "면화탄백화원은 북한의 파리에서 사는 거"처럼
완전히 무관한 내용으로 이어졌습니다.

### 결론

RAG 파이프라인은 정상 동작했으며, 검색된 정보가 생성 결과에 반영되는
것을 확인했습니다(키워드 수준). 질문 "세종대왕은 누구야?"에 대해 한글
창제 관련 문서를 검색해 왔고, 생성 결과에도 "한글", "훈민정음",
"세종실록" 등의 관련 키워드가 반복적으로 나타났습니다.

다만 직접 학습한 23M 규모 모델은 검색 결과를 바탕으로 일관되고 의미 있는
QA 응답을 생성하기에는 한계가 있었습니다. Wiki 모델은 검색된 주제와
관련된 키워드와 위키백과 특유의 서술 패턴을 어느 정도 유지했지만, 이를
자연스러운 설명 문장으로 구성하지는 못했습니다. Dialog 모델은 일상 대화
데이터로 학습되어 위키백과 기반 검색 결과와 도메인 차이가 컸고, 검색된
정보를 일부 따라가다가도 빠르게 무관한 내용으로 이탈하는 경향을 보였습니다.

이번 실험을 통해 검색된 정보가 생성 단계까지 전달되는 과정은 확인할 수
있었으나, 검색 품질은 충분히 확보된 반면 그 정보를 활용해 일관된 답변을
생성하는 단계에서 생성 모델의 한계가 더 크게 드러났습니다. 특히 생성
모델의 표현력이 부족하거나 검색된 정보와 학습 도메인이 크게 다를 경우,
검색이 정확하더라도 만족스러운 답변으로 이어지지 않을 수 있었습니다.
이는 실제 RAG 서비스들이 검색 단계 뒤에 대형 사전학습 언어모델을 생성
엔진으로 사용하는 이유를 체감하게 해준 실험이었습니다.

`rag/vanilla/demo.py`(`python3 -m rag.vanilla.demo`)로 재현 가능합니다.
추가 테스트(질문: "지미 카터는 누구야?")에서도 같은 패턴이 반복됐는데,
Wiki 모델이 "박정희", "김영삼" 등 실제 한국 정치인 이름과 날짜
형식("1980년 6월 27일")을 정확한 문법으로 생성했지만, 그 인물들은
검색된 지미 카터 정보와 전혀 무관했습니다. 이는 모델이 위키 문서의
"패턴(형식)"은 학습했지만, 그 패턴에 들어갈 "내용"을 검색된 정보에
맞춰 채우는 능력은 갖추지 못했음을 보여주는 추가 증거입니다.

## 6. LangChain RAG 마이그레이션

Vanilla RAG와 같은 임베딩(`jhgan/ko-sroberta-multitask`, `rag/shared/`
모듈 공유)으로 LangChain 버전을 구현했습니다.

### FAISS segmentation fault 디버깅

벡터스토어로 FAISS를 먼저 선택했습니다(Vanilla의 NumPy 코사인 유사도
방식과 개념적으로 가장 가까워 비교에 적합하다고 판단). 인덱스 생성과
저장까지는 정상 동작했으나, 검색(`similarity_search`) 호출 시
**segmentation fault**로 프로세스가 죽는 문제가 발생했습니다.

원인을 단계적으로 좁혔습니다.

1. 인덱스 생성 — 정상
2. 인덱스 저장 — 정상
3. 인덱스 로드 — 정상
4. 임베딩(쿼리 벡터) 생성 — 정상 (768차원 확인)
5. `vectorstore.similarity_search()` — **segfault**
6. `retriever.invoke()` (LangChain retriever 경로) — **segfault** (동일)
7. LangChain을 완전히 우회하고 `vectorstore.index.search()`를 직접
   호출 — **segfault** (동일 지점)

마지막 단계에서 LangChain 코드를 전혀 거치지 않고도 동일한 segmentation
fault가 재현되었습니다. 이를 통해 문제 원인을 LangChain 계층이 아닌
FAISS 검색 단계 또는 그 주변의 런타임 환경(macOS ARM, Python 3.12,
OpenMP 런타임 등) 수준까지 좁힐 수 있었습니다. 다만 그 안에서 정확한
원인(faiss-cpu 자체의 문제인지, 특정 라이브러리 버전 조합 문제인지)은
확정하지 못했습니다. macOS ARM 환경에서 `faiss-cpu`와 PyTorch가 각각
다른 OpenMP 런타임을 로드해 충돌하는 사례가 보고되어 있어, 이를
검증하기 위해 `KMP_DUPLICATE_LIB_OK=TRUE` 환경변수로 한 차례 더
테스트했으나 동일하게 segfault가 발생해 이 가설은 기각했습니다.

### 판단: Chroma로 전환

원인을 LangChain 계층이 아닌 FAISS 검색 단계 수준까지는 좁혔으나, 이
시점에서 더 깊이 파고드는 것(라이브러리 버전 조합 변경, 별도 환경
구성 등)은 본래 목표(Vanilla RAG를 LangChain으로 마이그레이션해
비교하는 것)에서 벗어난다고 판단했습니다. FAISS 대신 순수 Python
구현인 Chroma로 전환했습니다 — C++ 바인딩 레벨 충돌 위험이 낮고,
LangChain 통합도 동등하게 잘 지원됩니다. 벡터스토어 교체는
`rag/langchain/vectorstore.py` 한 파일, 함수 하나 안에서의 변경으로
끝났습니다.

이 디버깅에서 얻은 것은 "끝까지 원인을 좁히는 것"과 "그 다음 어디서
멈추고 우회할지 판단하는 것"이 별개의 능력이라는 점입니다.

### Chroma 거리 측정 기본값 문제

Chroma로 전환한 뒤, 같은 질문("세종대왕은 누구야?")으로 검색했는데
Vanilla(NumPy 코사인 유사도)와 전혀 다른 결과가 나왔습니다. Vanilla는
한글 창제 관련 청크를 score 0.48로 1위로 찾았는데, Chroma는 관련
없는 인물 출생/사망 날짜 목록을 반환했습니다.

같은 임베딩 모델, 같은 코퍼스, 같은 질문인데도 검색 결과가 달랐던
원인은 임베딩이 아니라 **벡터스토어의 기본 거리 함수**였습니다.
`vectorstore._collection.metadata`를 확인하니 `None`이 나왔는데,
이는 거리 측정 방식이 명시적으로 설정되지 않아 Chroma가 기본값(L2
유클리드 거리)을 쓰고 있었다는 뜻이었습니다. Vanilla는 코사인
유사도를 직접 계산했으니, 둘이 "가장 가까운" 청크를 다른 기준으로
판단하고 있었던 것입니다.

```python
# 기본값(L2)이 아니라 코사인 유사도를 명시적으로 지정
vectorstore = Chroma.from_documents(
    documents, embeddings,
    persist_directory=VECTORSTORE_DIR,
    collection_metadata={"hnsw:space": "cosine"},
)
```

이 설정을 추가하고 인덱스를 재구축하자, Vanilla와 1~3위 검색 결과가
정확히 일치했습니다. `Chroma.from_documents(...)`만 쓰고 거리 함수를
별도로 확인하지 않으면 이런 차이를 놓치기 쉽다는 것을 직접 겪은
사례입니다.

### LLM 래퍼와 Chain 구성

LangChain의 `LLM` 베이스 클래스를 상속해 우리가 직접 만든 GPTMini
모델(23M, PyTorch)을 LangChain 인터페이스로 감싼 래퍼
(`rag/langchain/llm_wrapper.py`)를 만들고, LCEL(`retriever |
format_docs`, `RunnablePassthrough`)로 검색·프롬프트 조립·생성을
하나의 체인(`rag/langchain/chain.py`)으로 엮었습니다.

같은 질문으로 실행한 결과는 Vanilla와 거의 동일한 패턴을 보였습니다.
"세종", "세종대왕", "세종대학교" 등 관련 키워드는 반복되지만 의미
있는 문장으로는 이어지지 않았습니다(`세종대왕는 조선 초기부터, 조선
세종대왕제, 세종대학교의 세종대학교의 세종대학교의 전신으로...`).
구현 방식(직접 구현 vs LangChain)과 무관하게 23M 모델의 생성
한계가 동일하게 나타난 것입니다.

### Vanilla vs LangChain 비교

| 비교 축 | Vanilla | LangChain |
|---|---|---|
| 코드량 | retriever.py + pipeline.py, 함수 단위로 직접 구현 | vectorstore.py + llm_wrapper.py + chain.py, 컴포넌트 조합 |
| 개념적 학습 비용 | NumPy 코사인 유사도 계산 — 수식 그대로 코드로 옮김 | LCEL(`\|` 연산자), `RunnablePassthrough` 등 LangChain 고유 문법 학습 필요 |
| 디버깅 가능성 | 검색 점수, 거리 계산까지 매 단계가 코드에 그대로 드러남 | Retriever→VectorStore→Embeddings→Chain으로 여러 계층이 추상화되어, 결과가 다를 때 원인을 한 계층씩 벗겨봐야 함(Chroma 거리 함수 문제가 그 예) |
| 설정의 명시성 | `cosine_similarity(...)`처럼 동작이 코드에 명시적으로 드러남 | `Chroma.from_documents(...)`의 기본값(L2)에 의존하면 의도와 다른 동작이 조용히 발생할 수 있음 |
| 확장성 | 모델/벡터DB를 바꾸려면 해당 함수를 직접 다시 작성 | LLM이나 벡터스토어 교체가 설정 변경에 가까움(다만 우리 모델처럼 LangChain이 기본 지원하지 않는 모델은 래퍼 작성이 필요) |
| 버그 발생 | 없음 | FAISS segmentation fault(환경 문제), Chroma 거리 함수 기본값(설정 누락) 두 차례 발생 |

Vanilla와 LangChain은 구현 방식과 추상화 수준은 달랐지만, 동일한
임베딩·코퍼스·생성 모델을 사용했을 때 검색 결과와 최종 생성 품질은
본질적으로 동일했습니다. 이번 실험을 통해 RAG 품질을 결정하는 요소를
검색기(retriever), 프레임워크(LangChain), 생성 모델(LLM)로 분리해
볼 수 있었고, 현재 프로젝트에서는 생성 모델의 표현 능력이 가장 큰
병목임을 다시 한 번 확인했습니다.

## 7. LangSmith Tracing 적용

부트캠프 진도에 맞춰 LangSmith를 연동했습니다(`docs/mentoring_notes.md`
기록대로, 멘토링에서 "관찰(Tracing)은 필수, 평가/프롬프트 관리는
선택"이라는 기준을 받았습니다).

### 적용 방법

기존 `rag/langchain/chain.py`(어제 만든 LCEL 체인)를 코드 한 줄도
바꾸지 않고 그대로 사용했습니다. 환경변수 세 개만 설정하면 LangChain이
자동으로 모든 실행을 LangSmith에 기록합니다.

```bash
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="..."
export LANGCHAIN_PROJECT="korean-chatbot-rag"
```

### 확인한 것

같은 질문("세종대왕은 누구야?")으로 체인을 한 번 실행하고, LangSmith
대시보드에서 trace를 확인했습니다. 우리 LCEL 코드 구조
(`{"context": retriever | format_docs, "question": RunnablePassthrough()
| extract_subject} | prompt | llm`)가 그대로 트레이스 트리로 펼쳐져
보였습니다.

- `map:key:context` → `VectorStoreRetriever`(0.32s) → `format_docs` —
  검색 단계
- `map:key:question` → `extract_subject`(0.00s) — 질문에서 주어를
  추출하는 우리 함수
- `KoreanGPTWrapper`(0.87s) — 우리가 만든 LLM 래퍼, 생성 단계

전체 1.25초 중 검색 0.32초, 생성 0.87초로 단계별 소요 시간이 바로
보였습니다. 또한 `KoreanGPTWrapper`의 Input을 클릭하니, `extract_subject`
가 "세종대왕은 누구야?"에서 "세종대왕"만 정확히 추출했고, 그 결과가
검색된 청크 뒤에 자연스럽게 이어붙어 최종 프롬프트가 조립되는 과정을
그대로 확인할 수 있었습니다.

이는 `docs/evaluation_v1_results_ella.md`에서 "정규식이 원인일 가능성이
있다"고 가설로만 적어뒀던 부분을, 실제 실행 결과로 직접 검증할 수
있는 도구를 갖췄다는 뜻입니다. 코드에 `print()` 문을 추가하고 다시
실행하는 대신, 이미 실행된 trace를 클릭해서 어느 단계든 들여다볼 수
있습니다.

## 8. LangSmith Dataset 생성
 
LangChain RAG 체인을 FastAPI(`/rag-chat`)에 연결한 뒤, 
`docs/evaluation_dataset_v1.md`의 10문항을 LangSmith Dataset으로 등록했습니다.
 
### 엔드포인트 네이밍
 
`/chat/rag`와 `/rag-chat`을 비교했으며, 두 엔드포인트(RAG 없음/있음)가
계층 관계가 아니라 독립적인 기능이라는 점을 고려해 `/rag-chat`을
선택했습니다.
 
### Q10 메타데이터 설계
 
`evaluation_v1_results_ella.md`(위 5절)에서 이미 확인했듯, Q10
("세종대왕은 어떤 업적으로 유명한가?")은 주어 추출 정규식이
"어떤 ~로 유명한가" 패턴을 처리하지 못해 검색이 실패하는 케이스입니다.
Dataset에서 이 문항을 제외할지 고민했으나, 검색 실패에 대한 세
모델(엘라/Gemma/Gemini)의 대응 차이 자체가 의미 있는 평가 결과이므로
포함하되 메타데이터로 표시하기로 했습니다.
 
```python
{
    "question": "세종대왕은 어떤 업적으로 유명한가?",
    "expected_answer": "1443년 한글(훈민정음)을 창제하기 시작하여 1446년 반포했다.",
    "known_retrieval_issue": True,
    "note": "주어 추출 정규식이 '어떤 ~로 유명한가' 패턴을 처리하지 못해 검색 실패..."
}
```
 
재실행해도 중복 등록되지 않도록, Dataset과 example 존재 여부를 먼저
조회한 뒤 없는 것만 추가하는 방식으로 스크립트를 작성했습니다.
 
## 9. LangSmith Evaluation
 
`rag/langchain/run_evaluation.py`로 `/rag-chat`과 동일한 체인을
평가 대상으로 삼아 LangSmith Evaluation을 실행했습니다.
 
### Judge 설계
 
`evaluation_dataset_v1.md`에 이미 정의된 정확성/질문 관련성/자연스러움
3축을 그대로 Gemini 2.5 Flash judge 프롬프트로 사용했습니다. Q10처럼
검색 실패가 알려진 문항에는 "정보 부족을 솔직히 인정하면
정확성 만점" 특례를 프롬프트에 동적으로 주입하도록 설계했습니다.
 
### 특례가 발동하지 않은 이유
 
Q10 평가 결과, accuracy/relevance/fluency 모두 0점이 나왔습니다.
처음엔 특례가 제대로 작동하지 않은 버그로 의심했으나, Input/Output을
직접 확인한 결과 정확한 채점이었습니다. 엘라는 검색 실패 시 "모른다"고
답하는 능력이 없어 무관한 내용을 그대로 생성(hallucination)했고,
judge는 그것을 정확히 "무관한 인물 목록 + 비문법적"이라고 지적했습니다.
특례 조항은 정보 부족을 솔직히 인정하는 모델(Gemma, Gemini)에만
해당하는 것이었고, 엘라 평가에서는 애초에 트리거될 상황이 아니었습니다.
 
### 기술적 이슈
 
`google.generativeai` 패키지가 deprecated라는 FutureWarning이 발생해,
신규 `google.genai` SDK(`genai.Client().models.generate_content(...)`)로
즉시 마이그레이션했습니다. 또한 API 키를 매번 `export`해야 하는 불편을
없애기 위해 `python-dotenv` + `.env`(`.gitignore`에 추가)로 전환했습니다.
 
### 결과
 
10개 문항 모두 채점 완료. 점수 분포는 0~2점대가 대부분으로,
`evaluation_v1_results_ella.md`(5절)의 결론("9개 중 완전 성공 없음,
부분 성공 3건")과 일치했습니다. LLM-as-a-Judge(정성적 판단)로 얻은
결론이 LangSmith Evaluation(운영 도구)에서도 동일하게 재현된 것입니다.
 
## 10. RAGAS 평가
 
Retriever와 생성 모델의 기여도를 분리해서 정량화하기 위해 RAGAS
(Faithfulness / Context Precision / Context Recall / Answer Relevancy)를
도입했습니다(`rag/langchain/evaluate_ragas.py`).
 
### 의존성 버그: langchain-community 버전 호환성
 
```
ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
```
 
`langchain-community==0.4.2`에서 `chat_models.vertexai` 서브모듈이
삭제되었는데, `ragas==0.4.3`(최신 버전) 내부 코드가 여전히 이 경로를
import하고 있어 발생하는, ragas 쪽의 알려진 버그였습니다. `sys.modules`에
가짜 모듈을 주입해 import를 우회하는 방법도 검토했으나, 유지보수성과
코드 명확성을 고려해 채택하지 않았습니다.
 
대신 GitHub Issue/PR을 확인해 원인을 확정한 뒤, 로컬 가상환경에서만
`langchain-community<0.4.2`(0.4.1로 다운그레이드)를 적용했습니다.
프로젝트 전체 의존성(requirements/pyproject)에는 반영하지 않았고,
LangChain RAG·Chroma·LangSmith 등 다른 부분은 최신 버전을 그대로
유지합니다. RAGAS 하나 때문에 핵심 의존성 전체를 낮게 고정하면
"LangChain RAG는 최신, RAGAS는 구버전 대응" 상태가 프로젝트에 영구히
남기 때문입니다.
 
judge LLM도 `google.generativeai.GenerativeModel`을 그대로
넘기면 안 되고, RAGAS가 기대하는 `BaseChatModel` 인터페이스에 맞춰
`ChatGoogleGenerativeAI`로 감싸야 했습니다. 같은 이유로
`context_precision`/`answer_relevancy` 계산에 필요한 임베딩도
명시하지 않으면 RAGAS가 기본값으로 OpenAI 임베딩을 요구해
`OPENAI_API_KEY` 에러가 났습니다. Gemini 임베딩(`GoogleGenerativeAIEmbeddings`,
`models/gemini-embedding-001`)을 명시적으로 지정해 해결했습니다.
 
### 코드 정리
 
- vectorstore/rag_chain을 모듈 레벨에서 한 번만 생성하도록 변경
  (기존에는 10문항마다 매번 재로드하고 있었음)
- `results`(RAGAS의 `EvaluationResult` 객체)는 `json.dump()`로 바로
  직렬화되지 않아 `to_pandas()`로 변환한 뒤 CSV/JSON 둘 다 저장
### 결과
 
| 메트릭 | 점수 |
|---|---|
| Context Precision | 1.000 |
| Context Recall | 1.000 |
| Faithfulness | 0.739 |
| Answer Relevancy | 0.787 |
 
Retriever 성능(precision/recall)은 거의 완벽한 반면, 생성 단계
(faithfulness/answer_relevancy)에서 점수가 떨어졌습니다. 이는
5절(Vanilla RAG), 9절(LangSmith Evaluation)에서 이미 확인한 "검색은
잘 되는데 생성이 약하다"는 결론을, RAGAS가 Retriever와 생성 모델을
분리한 지표로 다시 한번 정량적으로 재확인해준 것입니다.
 
LangSmith Evaluation은 생성된 답변의 품질을 LLM Judge로 평가했고,
RAGAS는 검색과 생성을 각각 분리해 측정했습니다. 서로 다른 방식과
관점으로 접근한 두 평가가 결국 같은 결론(검색보다 생성 모델이 병목)을
가리켰다는 점이 이번 평가 단계에서 가장 확실히 얻은 소득이었습니다.
 
## 11. 향후 실험 (예정)
 
- README 최종 업데이트 (LangSmith Evaluation, RAGAS 결과 반영)
- GitHub Release v1.0
- (가능하다면) 더 큰 사전학습 LLM을 생성 엔진으로 사용했을 때 RAG 결과
  비교 — 검색 vs 생성 능력의 기여도를 더 명확히 분리하기 위함
(이 섹션은 각 단계가 진행되면서 계속 갱신됩니다.)