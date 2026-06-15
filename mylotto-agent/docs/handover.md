# mylotto-agent — Project Handover Document

> **대상 독자:** 이 프로젝트를 이어받는 AI (GPT, Claude 등) 또는 신규 개발자  
> **작성일:** 2026-06-15  
> **현재 상태:** 핵심 기능 구현 완료, 구매 자동화 디버깅 중

---

## Executive Summary (1-page)

`mylotto-agent`는 한국 동행복권 로또 6/45 번호를 **데이터 기반으로 생성하고, 자동으로 구매까지 보조하는 Python 플랫폼**이다.

**핵심 결론 (백테스트 근거):**
- 로또는 독립 확률 추첨이므로 어떤 전략도 순수 랜덤을 통계적으로 유의미하게 이기지 못한다.
- 30-seed × 10회차 multi-seed 백테스트 결과: **random이 사실상 baseline이자 가장 안정적**이다.
- `model_score`는 random 대비 3개 일치 회차가 -1.75% 낮고, `ensemble`은 -6.03% 낮다.
- `balanced_v2`는 통계적 제약으로 번호 분포를 고르게 하나, 당첨 성과는 random과 동등하다.

**현재 운영 흐름 (매주 토요일 09:00 자동):**
```
Windows Task Scheduler → weekly_lotto.ps1
  → analyze (통계 갱신)
  → build-features + train-model (ML 갱신)
  → run --balanced-v2-games 3 --random-games 2 (5게임 생성)
  → buy-lotto (Playwright 자동 구매)
```

**구매 자동화 상태:** Playwright 기반 동행복권 자동 입력 구현 완료.  
번호 선택 셀렉터(`label[for='check645num{N}']`) 동작 확인됨.  
구매 완료 후 `generated_games.csv`에 `purchased=True` 기록, 다음 주 중복 생성 방지.

**남은 핵심 과제:** Playwright 번호 입력 → "확인" 버튼(적용수량 옆) 클릭 플로우 안정화 중.

---

## 1. 프로젝트 개요

### 목적
1. 동행복권 로또 6/45 당첨번호 이력을 수집·분석
2. 여러 전략(통계, ML)으로 번호 생성
3. Playwright로 동행복권 구매 페이지에 자동 입력 및 구매
4. 백테스트로 전략 성능 비교 (기대값 측면의 학문적 실험)

### 기술 스택
| 구분 | 기술 |
|------|------|
| 언어 | Python 3.12+ |
| CLI | Typer + Rich |
| 데이터 | pandas, numpy |
| ML | scikit-learn (LogisticRegression) |
| 브라우저 자동화 | Playwright (async, Chromium) |
| 스케줄링 | Windows Task Scheduler |
| 알림 (미완) | Telegram Bot API |
| 클라우드 (미완) | AWS S3, EC2 |
| 패키지 관리 | pip + .venv |

### 핵심 원칙
- **구매 안전 원칙:** `buy-lotto`는 `구매하기` 버튼까지 자동 클릭하되, 최종 확인 팝업만 처리함. 번호 추가(`확인`) 버튼과 최종 구매(`구매하기`) 버튼을 엄격히 구분한다.
- **Leakage-free 백테스트:** 회차 t 평가 시 history[round_no < t]만 전략에 전달하여 미래 데이터 누수 원천 차단.
- **구매 이력 추적:** 구매 완료 번호는 `purchased=True`로 표시하여 다음 주 동일 번호 재생성 방지.
- **전략 불가지론:** 코드 전체에 "로또는 독립 확률 추첨"임을 명시. 어떤 전략도 당첨을 보장하지 않음.

### 현재 운영 흐름
```
[토요일 09:00]
1. analyze          번호별 출현빈도·gap·홀짝 통계 CSV 갱신
2. build-features   회차별 ML 피처 행렬 생성 (parquet)
3. train-model      LogisticRegression 재학습
4. run              최신 회차 수집 + 5게임 생성 (구매 이력 제외)
5. buy-lotto        Playwright 자동 입력 + 구매하기 클릭
```

---

## 2. 디렉터리 구조

```
mylotto-agent/
├── main.py                    # CLI 진입점 (python main.py <command>)
├── weekly_lotto.ps1           # 주간 자동화 PowerShell 스크립트
├── .env                       # 환경변수 (DH_LOGIN_ID, DH_LOGIN_PW, HEADLESS)
├── .gitignore                 # .env, data/, screenshots/ 등 제외
│
├── data/                      # 런타임 데이터 (gitignore)
│   ├── lotto_draw_results.csv # 당첨번호 이력 (수집 결과)
│   ├── generated_games.csv    # 생성된 게임 + 구매 이력
│   ├── number_stats.csv       # 번호별 통계 (analyze 결과)
│   ├── features.parquet       # ML 피처 행렬 (build-features 결과)
│   ├── backtest_results.csv   # 단일 시드 백테스트 결과
│   ├── backtest_multiseed_results.csv   # multi-seed 상세 결과
│   ├── backtest_multiseed_summary.csv   # multi-seed 전략별 요약
│   ├── browser-profile/       # Playwright 세션 (로그인 쿠키 유지)
│   └── models/
│       ├── lr_model.pkl       # 학습된 scikit-learn Pipeline
│       └── lr_model.json      # 모델 메타데이터 (AUC, Brier, feature coef)
│
├── screenshots/               # Playwright 캡처 (디버깅용)
│   ├── debug_login.png
│   ├── debug_after_login.png
│   ├── debug_buy_page.png
│   └── YYYYMMDD_HHMMSS_purchase_complete.png
│
├── logs/
│   └── weekly_lotto.log       # 주간 실행 로그 (UTF-8)
│
├── src/lotto/
│   ├── agent.py               # LottoAgent — 모든 기능의 오케스트레이터
│   ├── cli.py                 # Typer CLI 커맨드 정의
│   ├── collector/
│   │   ├── dh_collector.py    # 동행복권 당첨번호 수집
│   │   └── lotto_kr_collector.py  # lotto.kr 폴백 수집기
│   ├── strategy/
│   │   ├── base.py            # BaseStrategy 인터페이스
│   │   ├── random_strategy.py
│   │   ├── balanced_strategy.py
│   │   ├── balanced_v2_strategy.py
│   │   ├── gap_based_strategy.py
│   │   ├── model_score_strategy.py
│   │   └── ensemble_strategy.py
│   ├── ml/
│   │   ├── feature_builder.py # leakage-free 피처 생성
│   │   ├── model_trainer.py   # LogisticRegression 학습
│   │   └── backtest.py        # 전략 백테스트 엔진
│   ├── analysis/
│   │   └── stats_analyzer.py  # 번호 통계 분석
│   ├── storage/
│   │   ├── local_storage.py   # CSV 읽기/쓰기 + purchased 관리
│   │   └── s3_storage.py      # S3 백업 (skeleton)
│   ├── automation/
│   │   ├── playwright_buyer.py  # 동행복권 자동 구매 (핵심)
│   │   └── ec2_runner.py        # EC2 원격 실행 (skeleton)
│   └── notify/
│       └── telegram_notifier.py # Telegram 알림 (skeleton)
│
└── docs/
    └── handover.md            # 이 문서
```

---

## 3. 데이터 구조

### `data/lotto_draw_results.csv`
```
round_no, date, num1, num2, num3, num4, num5, num6, bonus
1229,     2026-06-14, 3, 11, 24, 35, 38, 44, 17
```
- `round_no`: 회차 번호 (int)
- `date`: 추첨일 (YYYY-MM-DD)
- `num1`~`num6`: 당첨번호 오름차순 정렬
- `bonus`: 보너스 번호
- 수집 소스: 동행복권 공식 API (`dhlottery.co.kr`) + lotto.kr 폴백

### `data/generated_games.csv`
```
generated_at,          strategy,    game_no, num1,num2,num3,num4,num5,num6, purchased, purchased_at
2026-06-15T09:00:00Z,  balanced_v2, 1,       3,  11,  22,  31,  38,  44,   True,      2026-06-15 09:26:00
2026-06-15T09:00:00Z,  random,      2,       7,  14,  19,  27,  35,  42,   False,     ""
```
- `generated_at`: 생성 배치 타임스탬프 (UTC ISO 8601)
- `purchased`: 구매 완료 여부 (True/False)
- `purchased_at`: 구매 시각 (UTC)
- `buy-lotto`는 `generated_at` 최신 배치에서 `max_games`개만 읽음
- 다음 `run` 시 `purchased=True`인 조합은 자동 제외

### `data/number_stats.csv`
`analyze` 명령이 생성. 번호별(1~45):
- `total_count`, `frequency_pct`: 전체 출현 빈도
- `last_seen_round`, `gap`: 마지막 출현 이후 경과 회차
- `odd_pct`, `even_pct`: 홀짝 비율
- `band`: 십단위 구간 (1~5)
- `streak`: 연속 출현 회차 수

### `data/features.parquet`
`build-features` 명령이 생성하는 ML 피처 행렬.
- 행: 회차 × 번호 조합 (각 회차에서 당첨 여부 레이블)
- 열: 출현빈도, gap, 홀짝, 구간, 이전 회차 출현 여부 등 ~20개 피처
- **leakage-free 설계:** 회차 t의 피처는 t-1까지 데이터만 사용

### `data/models/lr_model.pkl`
`train-model` 명령이 저장하는 scikit-learn Pipeline:
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    LogisticRegression(C=0.1, max_iter=1000))
])
```
- 입력: 번호별 피처 벡터
- 출력: 해당 번호가 다음 회차에 당첨될 확률 (`predict_proba`)
- `lr_model.json`: AUC, Brier score, feature coefficient 메타데이터

---

## 4. 구현된 전략 설명

### 4.1 RandomStrategy (`random`)
**동작 방식:** `random.sample(1~45, 6)` — 순수 무작위 추출. 과거 데이터 미사용.

| 항목 | 내용 |
|------|------|
| 장점 | 가장 단순. 백테스트 baseline. 편향 없음 |
| 단점 | 번호 분포 제약 없어 극단적 조합 가능 |
| 권장 사용 | 항상 1~2게임 포함 권장 (편향 방지) |

---

### 4.2 BalancedStrategy (`balanced`)
**동작 방식:**
1. 5개 구간(1~9, 10~19, …, 40~45) 중 최소 4개 구간에서 1개씩 강제 선택
2. 홀수 2~4개 비율 제한
3. 연속 3개 이상 금지
4. 과거 빈도 가중치 반영 (없으면 균등)

| 항목 | 내용 |
|------|------|
| 장점 | 번호 구간 균형 보장. 직관적 규칙 |
| 단점 | 고정 구간 규칙이 실제 당첨 패턴과 무관할 수 있음 |
| 백테스트 | random 대비 3개+ 일치 -1.94% (열세) |

---

### 4.3 BalancedV2Strategy (`balanced_v2`)
**동작 방식:** balanced 대비 7가지 추가 제약:
1. 합계 범위: history 실제 분포 5th~95th 백분위 (동적)
2. 십단위 구간당 최대 3개 & 최소 3구간 커버
3. 끝자리 같은 숫자 최대 2개
4. 이전 회차 번호 최대 2개 겹침
5. 핫 번호(최근 5회 2회 이상) 최대 3개
6. 게임 간 Jaccard ≤ 0.5 다양성 보장
7. 최근 100회차 빈도 가중치 샘플링

| 항목 | 내용 |
|------|------|
| 장점 | 가장 엄격한 통계 제약. 분포 품질 최고 |
| 단점 | 제약이 많아 생성 속도 느림. 성과는 random과 동등 |
| 현재 주간 사용 | 3게임 생성 (weekly_lotto.ps1) |

---

### 4.4 GapBasedStrategy (`gap_based`)
**동작 방식:**
1. 번호별 마지막 출현 후 경과 회차(gap) 계산
2. `gap^alpha` 비례 가중치 → 소프트맥스 온도 적용
3. gap 평균의 2.5배로 클리핑 (극단 집중 방지)
4. 홀짝·합계·연속·구간 제약 적용

| 항목 | 내용 |
|------|------|
| 장점 | 장기 미출현 번호를 우선 탐색 (직관적) |
| 단점 | gap 기반 선택이 독립 추첨에서 이론적 근거 없음 |
| 백테스트 | 미포함 (ensemble 내 1게임으로만 사용) |

---

### 4.5 ModelScoreStrategy (`model_score`)
**동작 방식:**
1. `lr_model.pkl` 로드 (lazy)
2. 현재 history 기반으로 번호별 피처 계산
3. `predict_proba`로 각 번호의 score 산출
4. 소프트맥스 가중치 + 제약 조건으로 6개 샘플링

| 항목 | 내용 |
|------|------|
| 장점 | 데이터 기반 score 반영 |
| 단점 | 로또 독립 추첨 특성상 ML이 미래 예측 불가. 실제로 random보다 낮음 |
| 백테스트 | random 대비 3개+ -1.75% (열세). 단독 사용 비권장 |
| 의존성 | `build-features` + `train-model` 선행 필수 |

---

### 4.6 EnsembleStrategy (`ensemble`)
**동작 방식:** 고정 구성 5게임:
- model_score 2게임 + balanced 1게임 + random 1게임 + gap_based 1게임

| 항목 | 내용 |
|------|------|
| 장점 | 다양한 전략 혼합으로 번호 커버리지 향상 |
| 단점 | multi-seed 평균 기준 random보다 낮음 (-6.03%). 복잡성 대비 효과 미미 |
| 백테스트 | 가장 낮은 3개+ 회차 (32.23회/300 vs random 34.3회) |
| 의존성 | model_score 포함이므로 lr_model.pkl 필요 |

---

## 5. 백테스트 결과 요약

### 5.1 실험 설계
- **기간:** 최신 300회차 (backtest 실행 기준)
- **게임 수:** 전략당 5게임/회차
- **leakage-free:** 회차 t → history[round_no < t]만 사용
- **단일 시드:** seed 고정 1회 실행
- **multi-seed:** 30개 시드 × 10회차 × 각 전략

### 5.2 Multi-Seed 요약 결과 (30 seeds × 300 rounds)

| 전략 | 평균 최고 일치 | 3개+ 회차 (평균) | 4개+ 회차 (평균) | Coverage | Diversity Score | vs random (3개+) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **random** | **1.728** | **34.3** | 2.23 | 51.2% | 71.7 | — (baseline) |
| balanced | 1.716 | 33.6 | 1.87 | 50.8% | 71.4 | **-1.94%** |
| model_score | 1.719 | 33.7 | **2.23** | 49.9% | 70.8 | -1.75% |
| ensemble | 1.723 | 32.2 | 2.03 | 51.0% | 71.6 | **-6.03%** |

*300회차 중 3개 이상 일치한 회차 수 평균. 5게임/회차 기준.*

### 5.3 해석
- **평균 최고 일치** 기준 모든 전략이 1.72~1.73으로 거의 동일
- **3개+ 회차 수**: random(34.3) > model_score(33.7) > balanced(33.6) > ensemble(32.2)
- **4개+ 회차 수**: random = model_score(2.23) > ensemble(2.03) > balanced(1.87)
- **Coverage**: random이 가장 높고 model_score가 가장 낮음 (score 집중 효과)
- **표준편차**: balanced와 model_score가 낮아 안정적이나, 평균 자체가 random보다 낮음

---

## 6. 지금까지 얻은 결론

1. **model_score는 random을 이기지 못한다.**  
   LogisticRegression은 로또의 독립 추첨 특성을 극복할 수 없다. 3개+ 기준 -1.75%. ML 복잡성 대비 효과 없음.

2. **ensemble도 multi-seed 평균 기준 random보다 낮다.**  
   4개 전략 혼합에도 3개+ 기준 -6.03%. 전략 혼합이 성과를 낮추는 역설적 결과.

3. **현재 가장 안정적인 baseline은 random이다.**  
   단순하고 편향 없으며, 백테스트 기준 모든 지표에서 우위 또는 동등.

4. **balanced는 규칙 기반 전략으로 유지 가치 있다.**  
   성과는 random에 미치지 못하나, 번호 분포 품질(구간 균형, 홀짝)이 높아 심리적 만족도 기여.

5. **balanced_v2는 실험적으로 유의미하다.**  
   다양성 제약이 가장 엄격하고 hot-number 회피 등 직관적 논리를 갖추고 있어, 주간 생성의 주력 전략으로 채택(3게임).

6. **더 복잡한 ML 모델이 도움이 될 가능성은 낮다.**  
   로또 번호는 이론적으로 예측 불가능한 독립 사건이다. 모델 고도화에 시간을 쓰기보다 전략 다양성 유지가 바람직하다.

---

## 7. 남은 작업 (우선순위 순)

### P0 — 즉시 필요 (현재 진행 중)
- [ ] **Playwright 구매 플로우 안정화**
  - `a#num1` (혼합선택 탭) 클릭 → `label[for='check645num{N}']` 번호 선택 → `a:text-is('확인')` (적용수량 옆) 플로우 검증
  - `button#btnBuy`("구매하기") 이후 확인 팝업 처리 안정화
  - 구매 완료 후 `purchased=True` CSV 업데이트 검증

### P1 — 중요
- [ ] **Telegram 알림** (`telegram_notifier.py` skeleton 구현)
  - 생성된 번호 전송 (`send_games`)
  - 구매 완료 후 결과 전송
  - 환경변수: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`

- [ ] **S3 백업** (`s3_storage.py` skeleton 구현)
  - `lotto_draw_results.csv`, `generated_games.csv` 주기적 업로드
  - 환경변수: `AWS_*`, `S3_BUCKET_NAME`

### P2 — 선택적
- [ ] **EC2 자동 실행** (`ec2_runner.py` skeleton 구현)
  - Windows 환경 의존성 제거 후 서버 실행 가능하도록
  - SSM SendCommand 기반

- [ ] **GitHub Actions**
  - 주간 collect + generate 자동화 (구매 제외)
  - secrets: `DH_LOGIN_ID`, `DH_LOGIN_PW`

### P3 — 개선
- [ ] `backtest-multiseed`에 `balanced_v2`, `gap_based` 포함 (현재 미포함)
- [ ] 당첨 결과 자동 조회 및 일치 수 추적
- [ ] `weekly_lotto.ps1` → 크로스플랫폼 Python 스크립트로 전환

---

## 8. 알려진 문제점

### 기술 부채
| 위치 | 문제 | 영향 |
|------|------|------|
| `playwright_buyer.py:_ADD_SELECTORS` | `a:text-is('확인')`이 경고 팝업의 "확인"과 충돌 가능 | 게임 추가 실패 |
| `playwright_buyer.py:_close_popups` | 경고 다이얼로그 선택자(`ui-dialog`)가 실제 팝업 구조와 다를 수 있음 | 팝업 미해제 |
| `weekly_lotto.ps1` | 인코딩: `chcp 65001` 설정에도 일부 환경에서 한글 깨짐 | 로그 가독성 |
| `local_storage.py:save_games` | `encoding="utf-8-sig"` (BOM 포함). 일부 도구에서 BOM 문제 | 미미함 |

### 미완성 기능 (skeleton 상태)
- `TelegramNotifier`: 메서드 정의만 있고 HTTP 호출 없음
- `S3Storage`: 클래스 정의만 있고 업로드 로직 없음
- `Ec2Runner`: 클래스 정의만 있고 실행 로직 없음

### 주의사항
- `.env`에 `DH_LOGIN_ID`, `DH_LOGIN_PW` 포함. **절대 git 커밋 금지** (`.gitignore`에 명시됨)
- `data/browser-profile/`에 로그인 세션 쿠키 저장. 공유 환경 주의.
- 동행복권 구매는 매주 토요일 20:00 마감. `weekly_lotto.ps1` 09:00 실행 기준 충분한 여유.
- Playwright `--disable-blink-features=AutomationControlled` 비활성화 설정 필수. 미설정 시 로봇 감지.

---

## 9. 빠른 실행 방법

```bash
# 환경 설정 (최초 1회)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium

# .env 파일 설정
DH_LOGIN_ID=your_id
DH_LOGIN_PW=your_password
HEADLESS=false
```

### 주요 명령어

```bash
# 전체 주간 파이프라인
.\weekly_lotto.ps1

# 최신 회차 수집
python main.py collect

# 번호 생성 (5게임: balanced_v2 3 + random 2)
python main.py run --balanced-v2-games 3 --random-games 2 --balanced-games 0

# 번호 생성 (5게임: 전략 균등 분배)
python main.py run --total-games 5

# 통계 분석
python main.py analyze

# ML 피처 생성
python main.py build-features

# 모델 학습
python main.py train-model

# 단일 시드 백테스트 (model_score vs random, 최근 100회차)
python main.py backtest --strategy model_score random --rounds 100

# multi-seed 백테스트 (30 seeds × 10회차)
python main.py backtest-multiseed --strategy random balanced model_score ensemble --seeds 30 --rounds 10

# 구매 자동화 (브라우저 열기 + 번호 입력 + 구매)
python main.py buy-lotto --max-games 5

# 드라이런 (번호 확인만, 브라우저 미실행)
python main.py buy-lotto --dry-run
```

---

## 10. 다음 AI에게 전달할 조언

### 발전 방향 (권장)

1. **Playwright 플로우 안정화가 1순위다.**  
   번호 클릭(`check645num`)은 동작 확인됨. `a:text-is('확인')`(적용수량 확인) 버튼 탐지가 불안정하면, JS로 `document.querySelector('.btn_apply').click()` 같은 직접 클릭을 fallback으로 추가하라. 구매 페이지 HTML을 `page.content()`로 덤프해서 실제 버튼 구조를 확인하는 것이 가장 빠르다.

2. **Telegram 알림을 먼저 구현하라.**  
   skeleton이 이미 있고 간단하다. `requests.post`로 메시지 전송만 추가하면 된다. 구매 완료 알림은 실용적 가치가 크다.

3. **전략 실험보다 인프라 안정성이 중요하다.**  
   백테스트 결과로 이미 ML 효과의 한계가 확인됐다. 새 전략보다 S3 백업, 에러 알림, 구매 결과 추적에 투자하는 것이 낫다.

4. **balanced_v2를 주력으로 유지하고, random은 항상 포함하라.**  
   balanced_v2는 논리적으로 가장 잘 설계된 전략이고, random은 baseline 편향 방지 역할을 한다.

### 피해야 할 방향

1. **더 복잡한 ML 모델 (딥러닝, XGBoost 등)에 투자하지 마라.**  
   로또는 독립 확률 추첨이다. LogisticRegression조차 random을 이기지 못했다. 더 강력한 모델도 마찬가지다.

2. **구매 버튼(`button#btnBuy` = "구매하기")을 번호 추가 버튼으로 착각하지 마라.**  
   이전에 이 버튼을 "카트 추가"로 잘못 사용한 이력이 있다. 이 버튼은 **최종 구매**를 완료한다.

3. **`.env`를 git에 커밋하지 마라.**  
   로그인 정보가 포함돼 있다.

4. **backtest 없이 전략을 production에 투입하지 마라.**  
   `backtest-multiseed --seeds 30`으로 충분히 검증한 후 채택하라.

### 이미 검증된 사실

| 사실 | 근거 |
|------|------|
| random > model_score (3개+ 기준) | multi-seed 30×10: 34.3 vs 33.7 |
| random > ensemble (3개+ 기준) | multi-seed 30×10: 34.3 vs 32.2 |
| balanced_v2 제약 로직은 올바름 | 코드 리뷰 + leakage-free 설계 확인 |
| 로그인 플로우 작동 | `#inpUserId` + `type()` + URL 감지 |
| 번호 입력 셀렉터 확인 | `label[for='check645num{N}']` 동작 확인 |
| `button#btnBuy` = 최종 구매 버튼 | 진단 JS + 스크린샷 확인 |
| `a:text-is('확인')` = 적용수량 확인 | 진단 JS + 스크린샷 확인 |

---

*이 문서는 mylotto-agent v1.x 기준으로 작성됨. 주요 변경 시 갱신 필요.*
