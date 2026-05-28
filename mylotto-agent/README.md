# 🎱 mylotto-agent

Python 기반 로또 번호 생성 & 학습 기반 번호 생성 실험 플랫폼

> ⚠️ **중요**: 로또는 독립 확률 추첨입니다. 이 프로젝트는 당첨 확률 향상을 보장하지 않으며,
> "예측"이 아닌 **"학습 기반 번호 생성 전략 실험"** 을 목적으로 합니다.

---

## 목차

- [기능](#기능)
- [전체 아키텍처](#전체-아키텍처)
- [디렉터리 구조](#디렉터리-구조)
- [설치](#설치)
- [환경변수](#환경변수)
- [CLI 사용법](#cli-사용법)
- [전략 설명](#전략-설명)
- [ML 파이프라인 실행 흐름](#ml-파이프라인-실행-흐름)
- [테스트](#테스트)
- [로드맵](#로드맵)

---

## 기능

| 기능 | 상태 |
| --- | --- |
| 동행복권 회차별 당첨번호 수집 | ✅ |
| `data/lotto_draw_results.csv` 증분 업데이트 | ✅ |
| 순수 랜덤 전략 (RandomStrategy) | ✅ |
| 균형 전략 (BalancedStrategy — 구간/홀짝/빈도) | ✅ |
| **고도화 균형 전략 (BalancedV2Strategy)** | ✅ |
| **번호별 통계 분석 (analyze)** | ✅ |
| **Leakage-free Feature 생성 (build-features)** | ✅ |
| **LogisticRegression 모델 학습 (train-model)** | ✅ |
| **모델 score 기반 전략 (model_score)** | ✅ |
| **Gap 기반 전략 (GapBasedStrategy)** | ✅ |
| **앙상블 전략 (EnsembleStrategy)** | ✅ |
| **전략별 백테스트 (backtest)** | ✅ |
| **멀티 시드 백테스트 (backtest-multiseed)** | ✅ |
| `data/generated_games.csv` 저장 | ✅ |
| Telegram Bot 알림 | 🔲 skeleton |
| AWS S3 백업 | 🔲 skeleton |
| AWS EC2 원격 실행 | 🔲 skeleton |
| Playwright 자동 구매 | 🔲 skeleton |

---

## 전체 아키텍처

```text
동행복권 API / lotto.co.kr
        │ collect
        ▼
data/lotto_draw_results.csv
        │ analyze          │ build-features
        ▼                  ▼
data/number_stats.csv  data/features.parquet
                           │ train-model
                           ▼
                   data/models/lr_model.pkl
                           │ generate --strategy model_score
                           ▼
                   data/generated_games.csv

backtest: 회차 t에서 history[round_no < t]만 전략에 전달 (leakage 차단)
```

### 모듈 구조

| 모듈 | 역할 |
| --- | --- |
| `collector/` | 동행복권·lotto.co.kr 데이터 수집 (DhCollector / LottoKrCollector) |
| `analysis/` | 번호별 통계 분석 (StatsAnalyzer) |
| `ml/feature_builder.py` | Leakage-free feature 행렬 생성 |
| `ml/model_trainer.py` | LogisticRegression 시간순 학습 |
| `ml/backtest.py` | 전략별 과거 성능 검증 (배치 사전 계산으로 고속화) |
| `strategy/` | BaseStrategy + 6개 구현체 (random / balanced / balanced_v2 / gap_based / model_score / ensemble) |
| `storage/` | 로컬 CSV / parquet I/O |
| `agent.py` | 오케스트레이터 (단일 인터페이스) |
| `cli.py` | Typer CLI 커맨드 정의 |

---

## 디렉터리 구조

```text
mylotto-agent/
├── main.py                          # 엔트리포인트
├── pyproject.toml
├── requirements.txt
├── .env.example
│
├── data/
│   ├── lotto_draw_results.csv       # 당첨번호 (자동 생성)
│   ├── generated_games.csv          # 생성 게임 (자동 생성)
│   ├── number_stats.csv             # 번호별 통계 (analyze 후 생성)
│   ├── features.parquet             # ML feature 행렬 (build-features 후 생성)
│   ├── backtest_results.csv         # 백테스트 결과 (backtest 후 생성)
│   ├── backtest_multiseed_results.csv  # 멀티 시드 상세 결과 (backtest-multiseed 후 생성)
│   ├── backtest_multiseed_summary.csv  # 멀티 시드 전략별 요약 (backtest-multiseed 후 생성)
│   └── models/
│       ├── lr_model.pkl             # 학습된 모델 (train-model 후 생성)
│       └── lr_model.json            # 모델 메타데이터
│
├── src/lotto/
│   ├── agent.py                     # 오케스트레이터
│   ├── cli.py                       # Typer CLI 커맨드
│   │
│   ├── collector/
│   │   ├── base.py                  # 추상 BaseCollector
│   │   ├── dh_collector.py          # 동행복권 공식 API (한국 IP용)
│   │   ├── lotto_kr_collector.py    # lotto.co.kr 스크래핑 (해외 IP 폴백)
│   │   └── validator.py             # 데이터 검증
│   │
│   ├── analysis/
│   │   └── stats_analyzer.py        # 번호별 통계 분석
│   │
│   ├── ml/
│   │   ├── feature_builder.py       # Leakage-free feature 생성
│   │   ├── model_trainer.py         # LogisticRegression 학습
│   │   └── backtest.py              # 전략 백테스트 엔진
│   │
│   ├── strategy/
│   │   ├── base.py                  # 추상 BaseStrategy
│   │   ├── random_strategy.py       # 순수 랜덤
│   │   ├── balanced_strategy.py     # 균형 전략
│   │   ├── balanced_v2_strategy.py  # 고도화 균형 전략 (엄격한 제약 + 다양성)
│   │   ├── gap_based_strategy.py    # Gap 기반 전략 (미출현 번호 가중치)
│   │   ├── model_score_strategy.py  # 모델 score 기반 전략
│   │   └── ensemble_strategy.py     # 앙상블 전략 (5전략 혼합)
│   │
│   ├── storage/
│   │   ├── local_storage.py         # 로컬 CSV I/O
│   │   └── s3_storage.py            # S3 백업 (skeleton)
│   │
│   ├── notify/
│   │   └── telegram_notifier.py     # Telegram Bot (skeleton)
│   │
│   └── automation/
│       ├── playwright_buyer.py      # 자동 구매 (skeleton)
│       └── ec2_runner.py            # EC2 원격 실행 (skeleton)
│
├── scripts/
│   └── fetch_latest.py              # 최근 N회차 실시간 조회 스크립트
│
└── tests/
    ├── conftest.py
    ├── test_collector.py
    ├── test_features.py             # Feature 생성 & leakage 검증
    ├── test_stats.py                # 통계 분석 테스트
    ├── test_backtest.py             # 백테스트 엔진 테스트
    ├── test_model_strategy.py       # ModelScoreStrategy 테스트
    ├── test_gap_strategy.py         # GapBasedStrategy 테스트
    ├── test_ensemble_strategy.py    # EnsembleStrategy + compute_diversity 테스트
    ├── test_balanced_v2_strategy.py # BalancedV2Strategy 테스트 (43개)
    ├── test_backtest_multiseed.py   # 멀티 시드 백테스트 테스트 (25개)
    ├── test_strategy.py
    ├── test_storage.py
    └── test_validator.py
```

---

## 설치

```bash
# 1. 가상환경 활성화
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경변수 설정
copy .env.example .env          # Windows
cp .env.example .env            # macOS/Linux
```

---

## 환경변수

| 변수 | 설명 | 기본값 |
| --- | --- | --- |
| `LOTTO_RESULTS_CSV` | 당첨번호 CSV 경로 | `data/lotto_draw_results.csv` |
| `GENERATED_GAMES_CSV` | 생성게임 CSV 경로 | `data/generated_games.csv` |
| `TELEGRAM_BOT_TOKEN` | Telegram 봇 토큰 | — |
| `TELEGRAM_CHAT_ID` | 메시지 수신 채팅 ID | — |
| `AWS_ACCESS_KEY_ID` | AWS 액세스 키 | — |
| `S3_BUCKET_NAME` | S3 버킷 이름 | — |
| `EC2_INSTANCE_ID` | EC2 인스턴스 ID | — |
| `DH_LOGIN_ID` | 동행복권 로그인 ID | — |
| `DH_LOGIN_PW` | 동행복권 비밀번호 | — |

---

## CLI 사용법

```bash
python main.py --help
python main.py <command> --help
```

### 1. 당첨번호 수집

```bash
# 증분 수집 (로컬 CSV 이후 회차만)
python main.py collect

# 1회차부터 전체 재수집
python main.py collect --full

# 디버그 로그 출력
python main.py collect --verbose
```

### 2. 통계 분석

```bash
# 번호별 출현빈도·gap·홀짝·구간·합계 분석 + CSV 저장
python main.py analyze

# 상위 20개 표시
python main.py analyze --top-n 20

# CSV 저장 없이 출력만
python main.py analyze --no-save
```

#### 출력 내용

- 번호별 전체/최근 10·30·100회 출현 횟수
- 마지막 출현 회차, 미출현 간격(gap)
- 회차별 합계 분포 (평균·표준편차·범위)
- 홀수 개수 분포
- 구간별 평균 개수
- 빈출·미출현 상위 N개 순위

### 3. Feature 생성

```bash
# Leakage-free feature 행렬 생성 → data/features.parquet
python main.py build-features

# 최소 과거 회차 수 지정
python main.py build-features --min-history 50
```

#### Feature 목록

| Feature | 설명 |
| --- | --- |
| `total_frequency_before` | 전체 과거 출현율 |
| `recent_10/30/50/100_frequency` | 최근 N회 출현율 |
| `gap_since_last_seen` | 마지막 출현 이후 경과 회차 수 |
| `rolling_avg_gap` | 출현 간격 평균 |
| `appeared_in_previous_round` | 직전 회차 출현 여부 |
| `number_mod_2` | 홀/짝 (0/1) |
| `number_decade` | 구간 (1=1~9, 2=10~19, ..., 5=40~45) |

> 핵심 보장: 회차 t의 feature는 반드시 t보다 앞선 회차 데이터만 사용 (leakage 방지)

### 4. 모델 학습

```bash
# LogisticRegression 시간순 학습 → data/models/lr_model.pkl
python main.py train-model

# 검증셋 비율 조정 (기본 15%)
python main.py train-model --val-ratio 0.2

# L2 규제 강도 조정 (기본 0.1)
python main.py train-model --C 0.5
```

#### 출력 지표

| 지표 | 설명 |
| --- | --- |
| Val AUC | 검증셋 ROC-AUC (0.5 기준선 = 랜덤과 동일) |
| Train/Val Brier | Brier score (확률 보정 품질) |
| Top-6 Hit 평균 | 검증 회차별 상위 6개 중 실제 당첨 번호 평균 수 |
| Top-10 Hit 평균 | 검증 회차별 상위 10개 중 실제 당첨 번호 평균 수 |

> 로또는 독립 확률이므로 AUC ≈ 0.5는 정상입니다.

### 5. 번호 생성

```bash
# random + balanced 각 5게임 (기본)
python main.py generate

# balanced_v2 전략 5게임
python main.py generate --strategy balanced_v2

# model_score 전략 5게임
python main.py generate --strategy model_score

# 전략 조합 + 게임 수 지정
python main.py generate --strategy random --strategy balanced_v2 --n-games 3

# 재현 가능한 시드
python main.py generate --seed 42
```

### 6. 최신 당첨번호 조회

```bash
python main.py show
python main.py show --n 10   # 최근 10회차
```

### 7. 백테스트

```bash
# 최근 100회차 random 전략 백테스트 (기본)
python main.py backtest --strategy random

# 최근 30회차 model_score 백테스트
python main.py backtest --strategy model_score --recent 30

# ensemble 전략 (gap_based + model_score + balanced + random 혼합)
python main.py backtest --strategy ensemble --recent 50
python main.py backtest --strategy ensemble --recent 300

# 특정 회차 범위
python main.py backtest --strategy balanced --start-round 1100 --end-round 1200

# 3개 전략 동시 비교 (300회차 기준 약 3초)
python main.py backtest -s random -s balanced -s balanced_v2 --recent 300

# 전체 전략 비교 (300회차 기준 약 9초)
python main.py backtest -s random -s balanced -s balanced_v2 -s model_score -s ensemble --recent 300

# 게임 수 지정 + 저장 없이 실행
python main.py backtest --strategy random --n-games 10 --no-save
```

**주의사항**: model_score / ensemble 전략 백테스트는 **매 회차 모델을 재학습하지 않고**,
사전 학습된 모델로 각 회차 t-1까지의 feature를 즉석 계산하는 경량 방식으로 동작합니다.

#### 백테스트 성능

배치 사전 계산(Batch Pre-computation) 최적화 적용:

| 전략 | 최적화 전 (recent 300) | 최적화 후 (recent 300) | 속도 향상 |
| --- | --- | --- | --- |
| random | ~0.1초 | ~0.1초 | — |
| balanced | ~0.8초 | ~0.8초 | — |
| model_score | ~18분 | ~0.5초 | **~2,000×** |
| gap_based | ~5분 | ~0.3초 | **~1,000×** |
| ensemble | ~20분 | ~3초 | **~420×** |
| **4전략 합계** | **~40분** | **~9초** | **~270×** |

**최적화 원리**:

1. **Feature 배치 계산**: 기존에는 회차마다 45번호 × `_compute_number_features()` 호출 (O(45 × n_past) pandas 스캔). 최적화 후 `build_features()`로 전체 feature 행렬을 벡터화 numpy 연산으로 1회 생성.

2. **배치 predict_proba**: `model.predict_proba(300 × 45 = 13,500 samples)` 를 단일 호출로 처리.

3. **Gap 가중치 누적 행렬**: `last_seen[i, n]` 누적 행렬을 1회 빌드 후 O(1) 조회. 기존 회차별 history 전체 스캔 제거.

4. **DataFrame view 사용**: `history[...].copy()` → `history.iloc[:cur_idx]` (view, O(1)).

5. **round_no → index 캐시**: `dict` 기반 O(1) 인덱스 조회로 `history[history["round_no"] == round_no]` 반복 제거.

### 8. 멀티 시드 백테스트

단일 시드 결과의 통계적 노이즈를 제거하고, **seed를 바꿔가며 반복 실행한 평균·표준편차**로 전략의 안정성을 비교합니다.

```bash
# random + balanced 10 seeds × 최근 300회차 (기본)
python main.py backtest-multiseed --seeds 10 --recent 300

# 4개 전략 10 seeds 비교
python main.py backtest-multiseed -s random -s balanced -s model_score -s ensemble \
    --seeds 10 --recent 300

# 30 seeds로 더 안정적인 통계 (약 2분 30초)
python main.py backtest-multiseed -s random -s balanced -s balanced_v2 --seeds 30 --recent 300

# 특정 회차 범위 지정
python main.py backtest-multiseed -s random -s balanced \
    --start-round 1000 --end-round 1200 --seeds 20

# CSV 저장 없이 출력만
python main.py backtest-multiseed -s random -s balanced --seeds 5 --recent 100 --no-save
```

#### 출력 예시 (10 seeds × 300회차)

```text
────── 📊 멀티 시드 백테스트 결과  10 seeds × 300회차 ──────
                 최고일치       3개↑               4개↑
  전략             평균±σ     평균±σ    3개↑%    평균±σ   cover%   다양성
 ─────────────────────────────────────────────────────────────
  random        1.74±0.03   35.0±6.1    11.7%   2.6±1.3    51.1%     71.8
  balanced      1.72±0.04   34.0±5.0    11.3%   2.4±1.4    51.5%     71.6
  model_score   1.72±0.04   34.5±5.1    11.5%   2.1±1.5    50.0%     70.9
  ensemble      1.74±0.04   32.6±6.3    10.9%   2.7±1.4    52.0%     72.5

  vs random (3개↑):  balanced -2.9%  |  model_score -1.4%  |  ensemble -6.9%
  소요 시간: 43.3초  |  전략 4개 × 10 seeds
✓ 멀티 시드 결과 저장: data/backtest_multiseed_results.csv (60,000행)
✓ 멀티 시드 요약 저장: data/backtest_multiseed_summary.csv (4행)
```

#### 지표 설명

| 지표 | 설명 |
| --- | --- |
| `최고일치 평균±σ` | seed별 (회차당 최고 일치 번호 수 평균)의 평균과 표준편차 |
| `3개↑ 평균±σ` | seed별 3개 이상 일치 회차 수의 평균과 표준편차 |
| `3개↑%` | 3개↑ 평균 ÷ 총 회차 × 100 (역사적 3등↑ 확률 기준 ≈ 1.4%) |
| `4개↑ 평균±σ` | seed별 4개 이상 일치 회차 수의 평균과 표준편차 |
| `cover%` | 생성된 게임들이 1~45 번호를 커버하는 비율 평균 |
| `다양성` | 게임 간 Jaccard 기반 다양성 점수 평균 (0~100) |
| `vs random (3개↑)` | random 대비 3개↑ 회차 수의 상대적 개선율 (+ 많음, - 적음) |

#### 저장 CSV 포맷

**`data/backtest_multiseed_results.csv`**: 기존 `backtest_results.csv`와 동일 포맷 + `seed` 컬럼 추가

```text
strategy, seed, round_no, game_no, n1, n2, n3, n4, n5, n6,
winning_1...winning_6, bonus, match_count, best_match_in_round, coverage, diversity_score
```

**`data/backtest_multiseed_summary.csv`**: 전략별 집계 요약

```text
strategy, n_seeds, total_rounds,
avg_best_match_mean, avg_best_match_std,
match_3_plus_mean, match_3_plus_std,
match_4_plus_mean, match_4_plus_std,
coverage_mean, diversity_score_mean, vs_random_3plus_pct
```

#### 성능 (배치 사전계산 최적화 적용)

| 구성 | 소요 시간 |
| --- | --- |
| 4전략 × 10 seeds × 300회차 × 5게임 | ~43초 |
| 2전략 × 30 seeds × 300회차 × 5게임 | ~60초 |
| model_score/ensemble 포함 시 | feature 배치 계산이 전략당 1회만 수행됨 |

> 각 전략의 모델 score / gap 가중치는 seed 루프 **바깥**에서 1회 사전 계산하여 모든 seed가 공유합니다.

### 9. 올인원 실행

```bash
python main.py run

# 오프라인 모드 (수집 건너뜀)
python main.py run --skip-collect
```

---

## collect 실행 예시

### 증분 수집 (`python main.py collect`)

```text
──────────────── 📡 동행복권 당첨번호 수집 ────────────────
  API 최신: 1,225회차  |  로컬 최신: 1,220회차  |  수집 대상: 5회차  |  모드: 증분 수집
⠹ ✓ 회차 1225  ████████████████████████████████████  5/5  100%  0:00:02  0:00:00

╭──────────────── ✓ 수집 완료 ─────────────────╮
│  수집 범위       1,221 ~ 1,225 회차           │
│  수집 성공       5 회차                       │
│  실패 (API 오류) 0 회차                       │
│  성공률          100.0%                       │
│  소요 시간       2.1초                        │
│  누적 저장       1,225 회차                   │
╰───────────────────────────────────────────────╯
```

---

## 전략 설명

### `random` — 순수 랜덤

1~45 중 6개를 완전 무작위로 추출합니다.
과거 데이터와 무관하며 가장 단순한 기준선(baseline)입니다.

### `balanced` — 균형 전략

| 규칙 | 내용 |
| --- | --- |
| 구간 균형 | 5개 구간 중 4개 이상 커버 |
| 홀짝 균형 | 홀수 2~4개 |
| 연속 제한 | 3개 이상 연속 번호 없음 |
| 빈도 가중치 | 과거 출현 빈도를 가중치로 반영 |

### `balanced_v2` — 고도화 균형 전략

기존 `balanced` 대비 더 엄격한 제약과 최근 빈도 가중치, 게임 간 다양성 보장을 추가한 고도화 전략.

| 규칙 | 내용 |
| --- | --- |
| 홀짝 균형 | 홀수 2:4 / 3:3 / 4:2 만 허용 |
| 합계 범위 | history 실제 분포 5th~95th 백분위 (동적) |
| 십단위 구간 | 구간당 최대 3개 & 최소 3구간 커버 |
| 연속 제한 | 3개 이상 연속 번호 없음 |
| 끝자리 제한 | 같은 끝자리 최대 2개 |
| 이전 회차 겹침 | 직전 회차 번호와 최대 2개 겹침 |
| 핫 번호 제한 | 최근 5회차 2회↑ 출현 번호 최대 3개 |
| 게임 간 다양성 | Jaccard 유사도 ≤ 0.5 보장 (후처리) |
| 빈도 가중치 | 최근 100회차 출현 빈도 가중치 |

**백테스트 비교 (최근 300회차 기준)**:

| 전략 | 3개↑ 회차 | 4개↑ 회차 | 평균 합계 | 홀짝 분포 | 평균 십단위 구간 | 연속번호 포함 | 소요 시간 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 39회 (13.0%) | 1회 | 137.4 | 홀0~6 | 3.76 | 52.4% | 0.1초 |
| balanced | 32회 (10.7%) | 4회 | 143.2 | 홀2~4 | 4.39 | 43.8% | 0.7초 |
| balanced_v2 | 32회 (10.7%) | 3회 | 141.5 | 홀2~4 | 4.09 | 47.8% | 1.6초 |

> 로또는 독립 확률이므로 단기 샘플(300회차)에서의 차이는 통계적 노이즈 범위입니다.

### `gap_based` — Gap 기반 전략

최근 장기간 미출현 번호에 가중치를 부여하되, 극단적 집중을 소프트맥스 temperature로 방지합니다.

| 파라미터 | 기본값 | 설명 |
| --- | --- | --- |
| `alpha` | 1.2 | gap 가중치 지수 (높을수록 미출현 번호 선호) |
| `temperature` | 2.0 | 소프트맥스 온도 (높을수록 균등) |
| `max_gap_ratio` | 2.5 | gap 클리핑 임계값 (극단 집중 방지) |

### `ensemble` — 앙상블 전략

5가지 서브전략을 혼합하여 번호 커버리지를 높이는 실험적 접근입니다.

| 서브전략 | 게임 수 |
| --- | --- |
| model_score | 2게임 |
| balanced | 1게임 |
| random | 1게임 |
| gap_based | 1게임 |

게임 간 Jaccard 유사도 ≤ 0.5 다양성 후처리 포함.

### `model_score` — 모델 score 기반 전략

LogisticRegression 모델이 산출한 score를 소프트맥스 가중치로 변환하여
확률적 샘플링을 수행합니다. `train-model` 선행 실행 필요.

| 제약 조건 | 내용 |
| --- | --- |
| 홀짝 균형 | 홀수 2~4개 |
| 합계 범위 | 70~200 |
| 연속 제한 | 3개 이상 연속 번호 없음 |
| 구간 제한 | 한 구간에 3개 이상 금지 |

> 로또는 독립 확률이므로 이 전략은 **당첨 확률 향상을 보장하지 않습니다**.

---

## ML 파이프라인 실행 흐름

```bash
# Step 1. 당첨번호 수집 (최초 1회)
python main.py collect

# Step 2. 통계 분석 (선택)
python main.py analyze

# Step 3. Feature 생성 (약 0.1초 — 1,225회차 기준)
python main.py build-features

# Step 4. 모델 학습 (약 1~2초)
python main.py train-model

# Step 5. 모델 기반 번호 생성
python main.py generate --strategy model_score

# Step 6. 백테스트 검증
python main.py backtest --strategy model_score --recent 100
```

---

## 테스트

```bash
# 전체 테스트 (248개)
pytest

# 커버리지 포함
pytest --cov=src --cov-report=term-missing

# 특정 파일만
pytest tests/test_features.py -v   # Feature & leakage 테스트
pytest tests/test_backtest.py -v   # 백테스트 엔진 테스트
pytest tests/test_model_strategy.py -v

# Leakage 방지 테스트만
pytest tests/test_features.py -k "Leakage" -v
```

### 테스트 커버리지

| 테스트 파일 | 대상 |
| --- | --- |
| `test_collector.py` | DhCollector, LottoKrCollector |
| `test_validator.py` | 데이터 검증 |
| `test_storage.py` | LocalStorage CSV I/O |
| `test_strategy.py` | RandomStrategy, BalancedStrategy |
| `test_stats.py` | StatsAnalyzer |
| `test_features.py` | FeatureBuilder + **leakage 방지 검증** |
| `test_model_strategy.py` | ModelScoreStrategy, 제약 조건 |
| `test_gap_strategy.py` | GapBasedStrategy (16개) |
| `test_ensemble_strategy.py` | EnsembleStrategy, compute_diversity (22개) |
| `test_balanced_v2_strategy.py` | BalancedV2Strategy, game_stats (43개) |
| `test_backtest.py` | 백테스트 엔진, CSV 저장 |
| `test_backtest_multiseed.py` | MultiSeedResult, run_multiseed_backtest, 저장 함수 (25개) |

---

## 로드맵

- [ ] **Telegram 알림** — `notify/telegram_notifier.py` 구현
- [ ] **S3 백업** — `storage/s3_storage.py` 구현
- [ ] **EC2 원격 실행** — `automation/ec2_runner.py` 구현
- [ ] **Playwright 자동 구매** — `automation/playwright_buyer.py` 구현
- [ ] **GitHub Actions 스케줄** — 매주 토요일 자동 수집·생성·알림
- [ ] **백테스트 완전 모드** — 매 회차 t-1까지 데이터로 모델 재학습 (현재는 사전학습 모델 경량 방식)
- [ ] **더 많은 feature** — 홀짝 비율 히스토리, 누적 gap 분포 등
- [ ] **앙상블 전략** — RandomForest / GradientBoosting 추가
- [ ] **당첨 분석 대시보드** — 번호별 출현 빈도, 조합 통계 시각화

---

## 라이선스

MIT
