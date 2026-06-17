# mylotto-agent — Handover Document (최신)

> 기준일: 2026-06-17  
> 분석 대상: `develop` 브랜치, 커밋 `8dbfc75`

---

## 1. 프로젝트 목적과 현재 완성도

### 목적
동행복권 로또 6/45 번호를 전략적으로 생성하고, Playwright 브라우저 자동화로 자동 구매까지 수행하는 CLI 기반 에이전트.  
구매 후 당첨번호와 대조해 전략별 실적을 누적 추적하는 **피드백 루프**까지 구현됨.

### 완성도 요약

| 기능 영역 | 상태 |
|---|---|
| 당첨번호 수집 (동행복권 API + lotto.co.kr 스크래핑 fallback) | ✅ 완성 |
| 번호 생성 (6개 전략) | ✅ 완성 |
| 통계 분석 | ✅ 완성 |
| ML feature 생성 + 모델 학습 | ✅ 완성 |
| 백테스트 (단일·멀티시드) | ✅ 완성 |
| Playwright 자동 구매 | ✅ 완성 (실제 구매 확인됨) |
| 피드백 루프 (check-results / strategy-report) | ✅ 완성 |
| `target_round_no` 컬럼 + 자동 마이그레이션 | ✅ 완성 |
| 주간 파이프라인 스크립트 (Windows + Linux) | ✅ 완성 |
| Telegram 알림 / S3 저장 / EC2 실행 | ⬜ 구현됨 (미검증) |

---

## 2. 전체 디렉터리 구조

```
mylotto-agent/
├── .env                          ← 실제 credential (gitignored)
├── .env.example                  ← 환경변수 템플릿
├── main.py                       ← 진입점 (python main.py <command>)
├── weekly_lotto.ps1              ← Windows 주간 파이프라인
├── weekly_lotto.sh               ← Linux 주간 파이프라인
├── pyproject.toml
├── requirements.txt
│
├── data/
│   ├── lotto_draw_results.csv    ← 수집된 당첨번호 (1회~최신)
│   ├── generated_games.csv       ← 생성된 게임 (target_round_no 포함)
│   ├── prediction_results.csv    ← 적중 결과 누적
│   ├── backtest_results.csv      ← 단일 백테스트 결과
│   ├── backtest_multiseed_results.csv
│   ├── backtest_multiseed_summary.csv
│   ├── number_stats.csv          ← 번호별 통계
│   ├── features.parquet          ← ML feature 행렬
│   ├── models/
│   │   └── lr_model.pkl          ← 학습된 LogisticRegression 모델
│   └── browser-profile/          ← Playwright persistent 세션 (gitignored)
│
├── docs/
│   ├── handover.md               ← 이전 버전 handover
│   └── handover_latest.md        ← 이 파일
│
├── logs/                         ← 파이프라인 실행 로그
├── screenshots/                  ← 구매 자동화 스크린샷 (gitignored)
│
├── src/lotto/
│   ├── agent.py                  ← 오케스트레이터 (LottoAgent)
│   ├── cli.py                    ← Typer CLI (12개 커맨드)
│   ├── analysis/
│   │   ├── result_checker.py     ← 피드백 루프 (ResultChecker)
│   │   └── stats_analyzer.py
│   ├── automation/
│   │   ├── playwright_buyer.py   ← 자동 구매 (LottoBuyer)
│   │   └── ec2_runner.py
│   ├── collector/
│   │   ├── dh_collector.py       ← 동행복권 공식 API
│   │   └── lotto_kr_collector.py ← lotto.co.kr 스크래핑 (fallback)
│   ├── ml/
│   │   ├── backtest.py
│   │   ├── feature_builder.py
│   │   └── model_trainer.py
│   ├── storage/
│   │   ├── local_storage.py      ← CSV 읽기/쓰기
│   │   └── s3_storage.py
│   └── strategy/
│       ├── random_strategy.py
│       ├── balanced_strategy.py
│       ├── balanced_v2_strategy.py
│       ├── gap_based_strategy.py
│       ├── model_score_strategy.py
│       └── ensemble_strategy.py
│
└── tests/                        ← pytest 테스트 (14개 파일)
```

---

## 3. 주요 CLI 명령어와 실행 예시

```bash
# 진입점
python main.py <command> [options]

# 당첨번호 수집
python main.py collect

# 번호 생성 (balanced_v2 x3 + random x2)
python main.py generate --balanced-v2-games 3 --random-games 2 --balanced-games 0

# 수집 + 생성 올인원 (주간 파이프라인에서 사용)
python main.py run --balanced-v2-games 3 --random-games 2 --balanced-games 0

# 최신 당첨번호 출력
python main.py show

# 통계 분석
python main.py analyze

# ML feature 생성
python main.py build-features

# 모델 학습
python main.py train-model

# 백테스트 (최근 300회차)
python main.py backtest -s random -s balanced -s balanced_v2 --recent 300

# 멀티시드 백테스트
python main.py backtest-multiseed -s random -s balanced --seeds 20 --recent 300

# 자동 구매 (최대 5게임, 1게임씩 5회 반복)
python main.py buy-lotto --max-games 5
python main.py buy-lotto --dry-run          # 번호만 출력, 브라우저 미실행

# 피드백: 적중 결과 확인
python main.py check-results

# 피드백: 전략별 성과 리포트
python main.py strategy-report

# 주간 파이프라인 (Windows)
.\weekly_lotto.ps1

# 주간 파이프라인 (Linux)
bash weekly_lotto.sh
```

---

## 4. 데이터 파일 목록

### `data/lotto_draw_results.csv`
```
round_no, date, num1, num2, num3, num4, num5, num6, bonus
1, 2002-12-07, 10, 23, 29, 33, 37, 40, 16
...
```
- 동행복권 1회차부터 누적
- `collect` 명령으로 증분 업데이트

### `data/generated_games.csv`
```
generated_at, strategy, game_no, target_round_no,
num1, num2, num3, num4, num5, num6, purchased, purchased_at
```
- `target_round_no`: 생성 시점의 `최신 회차 + 1` (번호를 노린 대상 회차)
- `purchased`: 구매 완료 여부
- 기존 데이터 없는 경우 `check-results` 실행 시 날짜 기반으로 자동 마이그레이션

### `data/prediction_results.csv`
```
generated_at, target_round_no, strategy_name, numbers,
winning_numbers, bonus, match_count, bonus_matched,
rank, reward_estimate, checked_at
```
- `check-results` 실행마다 누적 추가 (중복 없음)
- `rank`: 1~5등, 낙첨은 빈 문자열

### 백테스트 관련
| 파일 | 내용 |
|---|---|
| `backtest_results.csv` | 단일 시드 백테스트 |
| `backtest_multiseed_results.csv` | 전략×시드별 전체 결과 |
| `backtest_multiseed_summary.csv` | 전략별 평균±std 요약 |

---

## 5. 구현된 전략 목록과 현재 결론

| 전략 | 설명 | 결론 |
|---|---|---|
| `random` | 완전 랜덤 | **현재 baseline. 가장 안정적.** 멀티시드 백테스트 기준 avg match ≈ 1.0 |
| `balanced` | 홀짝·합계·구간 균형 | random과 유사 수준. 안정적인 보조 전략 |
| `balanced_v2` | balanced + gap 혼합 | balanced보다 소폭 개선. 운영 주력 전략 |
| `gap_based` | 미출현 간격 기반 | 실험적. 안정성 불확실 |
| `model_score` | LogisticRegression 확률 기반 | Experimental. 백테스트에서 random 대비 유의미한 차이 없음 |
| `ensemble` | random + balanced + model_score 앙상블 | Experimental. model_score 없으면 자동 degradation |

**현재 주간 파이프라인 설정**: `balanced_v2 × 3 + random × 2`

---

## 6. check-results / strategy-report 기능 설명

### `check-results`
- **입력**: `data/generated_games.csv` + `data/lotto_draw_results.csv`
- **처리 흐름**:
  1. `generated_games.csv`의 `target_round_no`를 기준으로 당첨번호 대조
  2. `target_round_no == 0`인 레거시 행은 `generated_at` 날짜 기반으로 자동 마이그레이션
  3. 이미 체크된 게임(중복 키: `generated_at + numbers`)은 건너뜀
  4. `target_round_no`에 해당하는 당첨 데이터가 없으면 스킵 (추첨 전)
- **출력**: `data/prediction_results.csv` 누적 저장 + Rich 테이블 (최근 20건)

### `strategy-report`
- **입력**: `data/prediction_results.csv`
- **집계**: 전략별 총게임 수, 평균 적중, 3+/4+/5+ 매치 수, 최대 매치, 등수 분포
- **활용**: 전략 성능을 실제 구매 결과 기반으로 비교

---

## 7. buy-lotto 구매 보조 기능의 실제 동작 흐름

> ⚠️ **중요 — 코드 vs 문서 불일치**
>
> `playwright_buyer.py` 모듈 docstring과 `cli.py`의 커맨드 설명에는
> "구매 최종 확인 버튼은 자동 클릭하지 않는다"고 적혀 있으나,
> **실제 구현은 구매 버튼 클릭 및 확인 팝업까지 전부 자동 처리한다.**
> 이 문서는 실제 코드 기준으로 기술한다.

### 7-1. generated_games.csv 읽기
- `LottoBuyer.load_latest_games()` 호출
- `generated_at` 최댓값(최신 배치) 기준 필터링
- `purchased == False`인 게임만 추출
- 최대 `max_games`(기본 5)개 반환

### 7-2. 브라우저 실행
- `playwright.chromium.launch_persistent_context()` 사용
- `user_data_dir = data/browser-profile` → 쿠키·세션 영구 저장
- `headless`: `.env`의 `HEADLESS` 환경변수로 제어 (기본 `false`)
- 브라우저 자동화 감지 우회: `--disable-blink-features=AutomationControlled`, `ignore_default_args=["--enable-automation"]`
- 첫 이동 URL: `https://www.dhlottery.co.kr`

### 7-3. 로그인 방식 (3단계 폴백)
```
1단계: 기존 세션 유효성 확인
   → _is_logged_in(): 로그아웃 버튼 또는 사용자 정보 영역 존재 여부 확인
   → 세션이 살아있으면 즉시 통과 (재로그인 불필요)

2단계: 자동 로그인 (환경변수 설정 시)
   → .env의 DH_LOGIN_ID / DH_LOGIN_PW 사용
   → #inpUserId → type() (키 입력 시뮬레이션)
   → #inpUserPswdEncn → type() (클라이언트 암호화 트리거)
   → #btnLogin 클릭
   → /login URL에서 벗어나면 성공 판정

3단계: 수동 로그인 대기
   → "브라우저에서 직접 로그인하세요" 출력
   → 최대 180초 대기 (wait_for_url: /login 이탈)
```

### 7-4. 구매 페이지 이동
- `https://ol.dhlottery.co.kr/olotto/game/game645.do` 직접 이동
- 팝업 감지 후 닫기 (`_close_popups()`)
- `span[class*='ball']` 요소가 있는 iframe 프레임 탐색

### 7-5. 번호 입력 방식
```
각 번호 N에 대해 4단계 순서로 클릭 시도:
  1순위: label[for='check645num{N}'] 클릭
  2순위: #check645num{N} 직접 클릭 (force=True)
  3순위: JS evaluate로 click() 직접 호출
  4순위: span.ball 계열 선택자 (폴백)
```
- 클릭 전 a#num1("혼합선택") 탭 활성화, 기존 선택 초기화
- 6개 체크 후 `input[value='확인']`(카트 추가 버튼) 클릭

### 7-6. 최종 구매 버튼 처리 (실제 동작)
```python
# purchase_games() 실제 흐름:
1. button#btnBuy (구매하기 버튼) 클릭
2. window.confirm() 네이티브 dialog → page.on("dialog") 핸들러로 accept()
3. DOM 확인 팝업 → _handle_purchase_popup()
     - 구매하기 클릭 전/후 input[value='확인'] 위치 스냅샷 비교
     - 새로 나타난 버튼을 locator.nth(idx).click(force=True)로 클릭
4. _detect_purchase_success()로 완료 메시지 감지
```
**결론: 구매하기 버튼 및 최종 확인까지 전부 자동 클릭됨. 실제 금전 거래 발생.**

### 7-7. 게임당 브라우저 독립 실행
- `run()` 메서드가 게임 1개마다 `open_browser() → login() → navigate() → fill() → purchase() → close()` 순환
- 세션 오염 방지 목적
- 게임 사이 2초 대기

### 7-8. 스크린샷 저장
| 파일명 패턴 | 시점 |
|---|---|
| `game{N}_before_add.png` | 카트 추가 버튼 클릭 직전 |
| `game{N}_after_add.png` | 카트 추가 완료 직후 |
| `{ts}_before_purchase.png` | 구매하기 클릭 직전 |
| `{ts}_buy_confirm_popup.png` | 확인 팝업 출현 직후 |
| `{ts}_after_purchase.png` | 구매 완료 직후 |
| `game{N}_ok.png` / `game{N}_fail.png` | 최종 결과 |
- 저장 위치: `screenshots/` (gitignored)

### 7-9. browser-profile 세션 저장
- 위치: `data/browser-profile/`
- Chromium Persistent Context — 로그인 쿠키, 세션 토큰 영구 보관
- gitignored 처리됨
- 이 디렉터리가 있으면 두 번째 실행부터 자동 로그인 건너뜀

---

## 8. 현재까지 확인된 정상 실행 명령

모두 Windows(`weekly_lotto.ps1`) 및 실제 테스트에서 동작 확인됨:

```powershell
# 수집 + 생성
python main.py run --balanced-v2-games 3 --random-games 2 --balanced-games 0

# 자동 구매 (5개 게임, 1회 5천원 한도 내 동작 확인)
python main.py buy-lotto --max-games 5

# 피드백 루프
python main.py check-results
python main.py strategy-report

# 전체 파이프라인
.\weekly_lotto.ps1
```

---

## 9. 아직 미구현 또는 불안정한 부분

| 항목 | 상태 | 비고 |
|---|---|---|
| 코드 vs docstring 불일치 | ⚠️ 방치 중 | `playwright_buyer.py` 모듈 docstring, `cli.py` `buy-lotto` 설명이 구현과 다름 |
| 동행복권 1회 5천원 구매한도 | 제약 | 코드 외 서비스 정책. `max_games=1`(1게임×1,000원)으로 우회 가능 |
| `window.confirm()` 실패 케이스 | 가끔 불안정 | 사이트가 dialog 대신 DOM 팝업으로 변경 시 `_handle_purchase_popup()`이 처리하나, 환경에 따라 클릭 실패 가능 |
| 헤드리스 모드 구매 | 미검증 | `HEADLESS=true`로 실제 구매 성공 여부 미확인 |
| Telegram 알림 | 구현됨 | `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` 미설정 상태. 통합 테스트 없음 |
| S3 저장 / EC2 실행 | 구현됨 | 실제 AWS 환경 테스트 없음 |
| 전략 성능 비교 (피드백 기반) | 데이터 부족 | 아직 구매 이력이 적어 `strategy-report` 통계 의미 없음 |
| ML 모델 개선 | 미진행 | 현재 LogisticRegression 기준 성능이 random과 차이 없음 |

---

## 10. 다음 단계 추천 우선순위

1. **docstring 정합성 수정** (간단, 즉시 가능)  
   `playwright_buyer.py` 모듈 docstring, `cli.py` `buy-lotto` 설명을 실제 구현에 맞게 수정

2. **구매 한도 회피 전략 검토**  
   동행복권은 1회 최대 5게임(5,000원). 매주 1회 파이프라인이면 이미 한계. 금액 초과 케이스 처리 필요 여부 확인

3. **피드백 루프 데이터 축적**  
   `weekly_lotto.sh / .ps1`에 `check-results` 단계 추가해 매주 자동 집계

4. **헤드리스 구매 검증 (Linux 환경)**  
   서버 배포 전 `HEADLESS=true` 상태에서 실제 구매 흐름 테스트 필요

5. **전략 피드백 반영 (장기)**  
   `prediction_results.csv` 데이터가 충분히 쌓이면 전략별 실적을 다음 번호 생성 가중치에 반영

---

*분석자: Claude Sonnet 4.6 / 2026-06-17*
