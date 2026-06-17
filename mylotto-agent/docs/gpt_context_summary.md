# mylotto-agent — GPT 전달용 컨텍스트 요약

> 이 파일은 GPT에게 바로 붙여넣을 수 있도록 작성된 요약본입니다.  
> 기준: 2026-06-17, develop 브랜치

---

## 프로젝트 개요

**mylotto-agent**는 동행복권 로또 6/45 번호를 전략적으로 생성하고, Playwright 브라우저 자동화로 자동 구매까지 수행하는 Python CLI 에이전트입니다. 구매 후 당첨번호 대조 → 전략별 실적 추적 피드백 루프까지 구현 완료 상태입니다.

- **언어**: Python 3.11+
- **프레임워크**: Typer CLI, Playwright (async), pandas, scikit-learn
- **진입점**: `python main.py <command>`

---

## 핵심 데이터 흐름

```
[collect] → lotto_draw_results.csv
    ↓
[run/generate] → generated_games.csv (target_round_no 포함)
    ↓
[buy-lotto] → purchased=True 표시
    ↓
[check-results] → prediction_results.csv (적중 결과 누적)
    ↓
[strategy-report] → 전략별 성과 출력
```

---

## 주요 CLI 명령어

| 명령어 | 역할 |
|---|---|
| `collect` | 동행복권 당첨번호 수집 (API + 스크래핑 fallback) |
| `run` | 수집 + 번호 생성 올인원 |
| `generate` | 전략별 번호 생성만 |
| `analyze` | 번호 통계 분석 |
| `build-features` / `train-model` | ML 파이프라인 |
| `backtest` / `backtest-multiseed` | 전략 백테스트 |
| `buy-lotto` | Playwright 자동 구매 |
| `check-results` | 생성번호 vs 당첨번호 대조 |
| `strategy-report` | 전략별 실적 리포트 |

---

## generated_games.csv 컬럼 구조

```
generated_at, strategy, game_no, target_round_no,
num1~num6, purchased, purchased_at
```

- `target_round_no`: 생성 시점의 `최신 회차 + 1`
- 기존 데이터 (`target_round_no=0`)는 `check-results` 실행 시 날짜 기반으로 자동 마이그레이션

---

## buy-lotto 실제 동작 (코드 기준)

1. `generated_games.csv` 최신 배치의 미구매 게임 로드
2. `data/browser-profile` Persistent Context로 Chromium 실행 (세션 재사용)
3. **로그인 3단계**: ① 기존 세션 유효 → 통과 / ② `.env`의 `DH_LOGIN_ID`/`DH_LOGIN_PW`로 자동 로그인 / ③ 수동 로그인 180초 대기
4. `https://ol.dhlottery.co.kr/olotto/game/game645.do` 직접 이동
5. 번호 클릭 (label → input force → JS evaluate 순 4단계 폴백)
6. **구매 버튼 자동 클릭**: `button#btnBuy` 클릭 → `window.confirm()` dialog accept → DOM 팝업 확인 버튼 클릭
7. 게임마다 브라우저를 열고 닫음 (세션 오염 방지)
8. 스크린샷 `screenshots/`에 자동 저장

> ⚠️ **주의**: 모듈 docstring과 CLI 설명에는 "구매 버튼 클릭 안 함"이라고 적혀 있으나, **실제 코드는 구매 확인까지 전부 자동 처리합니다**. 실제 금전 거래가 발생합니다.

---

## 전략 결론

| 전략 | 상태 | 결론 |
|---|---|---|
| `random` | 운영 중 | **Baseline. 가장 안정적** |
| `balanced` | 운영 중 | random과 유사 수준 |
| `balanced_v2` | 운영 주력 | balanced보다 소폭 개선 |
| `gap_based` | Experimental | 안정성 불확실 |
| `model_score` / `ensemble` | Experimental | 백테스트 기준 random 대비 차이 없음 |

현재 주간 파이프라인: `balanced_v2 × 3 + random × 2`

---

## 환경설정 (.env)

```env
DH_LOGIN_ID=<동행복권 아이디>
DH_LOGIN_PW=<동행복권 비밀번호>
HEADLESS=false          # 서버: true
```

- `data/browser-profile/`: Playwright 세션 저장 (gitignored)
- 첫 실행 후 세션이 유지되므로 재로그인 불필요

---

## 현재 미해결 이슈

1. **docstring 불일치**: `playwright_buyer.py` 모듈 설명이 실제 구현과 다름 (수정 필요)
2. **동행복권 5,000원/회 구매한도**: 1회 최대 5게임(1게임=1,000원)
3. **헤드리스 구매 미검증**: `HEADLESS=true` 환경에서 실제 구매 성공 여부 확인 필요
4. **피드백 데이터 부족**: 아직 구매 이력이 적어 `strategy-report` 통계 의미 없음

---

## 다음 단계 추천

1. `weekly_lotto.ps1 / .sh`에 `check-results` 단계 추가 (매주 자동 집계)
2. docstring 실제 구현에 맞게 수정
3. 헤드리스 모드 구매 검증
4. 데이터 축적 후 전략 성능 재평가
