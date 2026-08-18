# ASCP-2026 시스템 설계 문서

> 코드를 직접 읽지 않아도 전체 구조·로직·입출력을 파악할 수 있도록 작성된 참조 문서.
> 기준 커밋: `9467c1e` (2026-05-16)
>
> **파트별 상세 문서는 `docs/` 폴더를 참조.** 각 문서에는 구현 설명과 함께 현재 한계·버그·미구현 사항이 포함되어 있다.
>
> | 파일 | 내용 |
> |------|------|
> | [docs/00_overview.md](docs/00_overview.md) | 전체 파이프라인, 문제 정의, 흐름도 |
> | [docs/01_data.md](docs/01_data.md) | 데이터 로딩, hub_only 필터링 |
> | [docs/02_environment.md](docs/02_environment.md) | RL 환경, 상태, 마스킹, step |
> | [docs/03_embedding.md](docs/03_embedding.md) | 공항 Embedding, flight 표현 |
> | [docs/04_film.md](docs/04_film.md) | FiLM 변조 모듈 |
> | [docs/05_transformer.md](docs/05_transformer.md) | Transformer Encoder |
> | [docs/06_encoder.md](docs/06_encoder.md) | FlightEncoder 전체 파이프라인 |
> | [docs/07_decoder.md](docs/07_decoder.md) | PointerDecoder, Pointer Attention |
> | [docs/08_training.md](docs/08_training.md) | REINFORCE, 커리큘럼 학습 |
> | [docs/09_reward.md](docs/09_reward.md) | 보상 함수 설계 |
> | [docs/10_ip.md](docs/10_ip.md) | Set Partitioning IP |
> | [docs/11_evaluation.md](docs/11_evaluation.md) | 평가 파이프라인, Ablation |
> | [docs/12_data_trace.md](docs/12_data_trace.md) | 예제 데이터 기준 전체 흐름 추적 |
> | [docs/13_design_vision.md](docs/13_design_vision.md) | **연구 설계 구상 (이상적 목표)** |
> | [docs/14_issues.md](docs/14_issues.md) | **현재 코드의 모든 문제점 정리** |

---

## 목차

1. [문제 정의 (CPP)](#1-문제-정의-cpp)
2. [전체 파이프라인 — 큰 그림](#2-전체-파이프라인--큰-그림)
3. [데이터 레이어](#3-데이터-레이어)
4. [제약 조건](#4-제약-조건)
5. [RL 환경](#5-rl-환경)
6. [신경망 구조](#6-신경망-구조)
7. [학습 루프 (REINFORCE)](#7-학습-루프-reinforce)
8. [커리큘럼 학습 (3단계)](#8-커리큘럼-학습-3단계)
9. [보상 함수 설계](#9-보상-함수-설계)
10. [IP 후처리 (Set Partitioning)](#10-ip-후처리-set-partitioning)
11. [평가 파이프라인](#11-평가-파이프라인)
12. [Ablation 실험](#12-ablation-실험)
13. [하이퍼파라미터 일람](#13-하이퍼파라미터-일람)
14. [파일별 역할 요약](#14-파일별-역할-요약)
15. [핵심 설계 결정 이유](#15-핵심-설계-결정-이유)

---

## 1. 문제 정의 (CPP)

### 1.1 무엇을 푸는 문제인가

항공사 운항 스케줄에는 수백 개의 비행편(leg)이 있다. 각 비행편은 출발 공항, 도착 공항, 출발 시각, 도착 시각이 정해져 있다. 항공사는 이 비행편들을 승무원이 실제로 수행할 수 있는 "근무 단위"로 묶어야 한다. 이것이 Crew Pairing Problem(CPP)이다.

**핵심 제약**: 승무원은 연속으로 너무 오래 일할 수 없고, 비행편 사이 대기 시간이 너무 짧거나 길면 안 되며, 근무가 끝나면 반드시 출발했던 base 공항으로 돌아와야 한다.

**목표**: 모든 비행편을 커버하되, 승무원이 낭비하는 시간(dead time)을 최소화하는 pairing 집합을 찾는 것.

### 1.2 용어 정의

| 용어 | 정의 |
|------|------|
| **leg** | 단일 항공편 (출발공항 → 도착공항, 시각 고정) |
| **duty** | 하루 안에 연속으로 수행하는 leg들의 묶음. duty 내에서는 연결 시간(connection)이 존재 |
| **duty period** | duty 하나. overnight rest를 기점으로 구분됨 |
| **pairing** | 하나 이상의 duty로 구성된 완전한 근무 단위. **base 공항에서 출발해 base 공항으로 복귀**해야 함 |
| **dead time** | duty 내에서 leg와 leg 사이의 대기 시간. 비용의 핵심 지표. 예: 10:30 도착 → 13:00 출발이면 dead time = 2.5h |
| **overnight rest** | duty 간 휴식. 최소 9.5시간. 이 시간은 dead time에 포함하지 않음 (의도된 쉬는 시간) |
| **deadhead (DH)** | 하나의 flight를 두 pairing이 공유하는 상황. 실제로는 한 비행편에 승무원이 2팀 배정된다는 의미. IP에서만 발생 가능 |
| **ManDays** | 선택된 pairing의 수. 적을수록 좋음 (Zeren baseline의 평가 지표) |
| **FTC** | Fleet Time Cost = `dead_time / fly_time × 100%`. 비행 시간 대비 낭비 시간 비율 |

### 1.3 왜 어려운가

- leg 수가 늘어나면 가능한 pairing 조합이 기하급수적으로 증가 (NP-hard)
- 시간·공항 연결 제약이 복잡하게 얽혀 있음
- ManDays 최소화와 DH 최소화가 부분적으로 상충함

### 1.4 ASCP-2026의 접근법

RL로 좋은 pairing 후보들을 대량 생성하고, IP로 최적 조합을 선택한다.

```
RL (candidate generation) + IP (optimal selection)
```

RL은 constraint를 만족하는 feasible pairing을 빠르게 만들어낸다. IP는 이 후보들 중에서 비용 최소 조합을 고른다. 두 단계를 분리함으로써 RL의 탐색 능력과 IP의 최적화 능력을 결합한다.

---

## 2. 전체 파이프라인 — 큰 그림

### 2.1 한 문장 요약

> BTS 항공 데이터를 읽어 flight 목록을 만들고 → 신경망이 RL 환경과 상호작용하며 pairing을 구성하는 법을 학습하고 → 학습된 모델로 pairing 후보를 대량 생성한 뒤 → IP가 최적 pairing 집합을 선택한다.

### 2.2 두 가지 실행 모드

시스템은 **학습 모드**와 **추론/평가 모드**로 나뉜다.

- **학습 모드** (`experiments/train.py`): 신경망 파라미터를 업데이트한다. 에피소드마다 RL 환경과 반복 상호작용하며 보상을 최대화하도록 학습.
- **추론/평가 모드** (`evaluate_ip.py`, `eval_vs_baseline.py`): 학습된 체크포인트를 불러와 새로운 flight에 대해 pairing을 생성하고 IP로 최적화한다.

### 2.3 전체 흐름 다이어그램

```
[CSV 파일: T_ONTIME_MARKETING.csv]
    │
    ▼  load_flights() / load_flights_multiday()
    │  - 공항 ID 정수 변환 (허브=0)
    │  - 시간 float 변환 (HHMM → 시간)
    │  - dep_time 기준 정렬
    │
[flight dict 리스트]   {id, origin, dest, dep_time, arr_time}
    │
    ▼  flights_to_tensors()
[4개 Tensor]   origins(N,) / dests(N,) / dep_times(N,) / arr_times(N,)
    │
    ├──────────────────────────────────────────────────────┐
    │  [학습 모드]                                           │  [추론/평가 모드]
    │                                                      │
    ▼                                                      ▼
[FlightEncoder.forward()]                       [체크포인트 로드]
    │  constraint(7,) + 4 tensors → (N, 128)         │  encoder.load_state_dict()
    │  에피소드당 1회 호출                               │  decoder.load_state_dict()
    │                                                  │
    │  ┌── 에피소드 루프 (모든 flight 배정될 때까지) ──┐   │
    │  │                                            │   ▼
    │  │  state_to_vec(state, encoder, constraint)  │  [RL rollout × n_rollouts=100]
    │  │  → state_vec (38,)                         │   stochastic 샘플링
    │  │                                            │   + greedy rollout × 1
    │  │  PointerDecoder.forward(encoded, state_vec, mask)
    │  │  → probs (N+2,)                            │   중복 제거 (legs tuple key)
    │  │                                            │       │
    │  │  action 샘플링 (stochastic)                 │       ▼
    │  │  or argmax (greedy)                        │  [pairing pool]
    │  │                                            │   각 pairing: legs, cost, dead_time
    │  │  if flight: environment.step()             │       │
    │  │  if END_DUTY: step_end_duty()              │       ▼
    │  │  if END_PAIRING: 새 pairing 시작           │  [solve_set_partitioning()]
    │  │                                            │   LP relaxation → column reduction
    │  └────────────────────────────────────────────┘   → IP (x_j ∈ {0,1})
    │                                                       │
    ▼                                                       ▼
[REINFORCE 업데이트]                             [최적 pairing 집합]
    loss = -Σ log_prob × advantage               n_pairings, coverage, dead_time, FTC
         - 0.01 × entropy
    clip_grad_norm(max_norm=1.0)
    optimizer.step()
    │
    ▼
[체크포인트 저장]
    checkpoints/stage{k}_best.pt  (25ep 이동평균 최소 pairings 갱신 시)
    checkpoints/model_latest.pt   (학습 완료 후)
```

### 2.4 에피소드 하나의 흐름 (상세)

에피소드는 "모든 flight가 배정될 때까지" 반복되는 루프다.

```
에피소드 시작
  assigned = {모든 flight_id: False}
  state = init_state(flights)  # 첫 번째 flight 기준 초기화

  while True:
    [1] get_mask() → 현재 state에서 선택 가능한 action 목록 (0=불가, 1=가능)

    [2] 모든 action이 불가능한 경우 (flight 없음 + END_DUTY 불가):
        → 현재 pairing 강제 종료 (deadhead)
        → 미배정 flight 중 가장 이른 것으로 새 pairing 강제 시작
        → continue

    [3] state_to_vec() → decoder 입력 벡터 생성

    [4] decoder(encoded, state_vec, mask) → probs
        stochastic: Categorical(probs).sample() → action + log_prob + entropy 기록
        greedy: probs.argmax() → action

    [5] action 실행:
        flight(0~N-1): environment.step() → next_state, reward
                       LEG_BONUS 적용 (2번째 leg부터)
        END_DUTY(N):   step_end_duty() → rest period 진입
                       OVERNIGHT_PENALTY 적용
        END_PAIRING(N+1): PAIRING_COST + BASE_PENALTY(조건부) 적용
                          미배정 flight로 새 pairing 시작

    [6] 모든 flight 배정 완료 → break

  total_reward += final_reward()  # 미배정 flight당 -10

에피소드 종료
  → log_probs, entropies, total_reward 반환
```

---

## 3. 데이터 레이어

### 3.1 원본 데이터 구조

`T_ONTIME_MARKETING.csv`는 미국 BTS(Bureau of Transportation Statistics)의 실제 항공편 데이터다. 사용하는 컬럼은 다음 5개:

| 컬럼 | 의미 |
|------|------|
| `ORIGIN` | 출발 공항 코드 (문자열, e.g. "ATL") |
| `DEST` | 도착 공항 코드 |
| `CRS_DEP_TIME` | 예정 출발 시각 (HHMM 정수, e.g. 630 = 06:30) |
| `CRS_ARR_TIME` | 예정 도착 시각 |
| `FL_DATE` | 비행 날짜 |

### 3.2 `load_flights()` 처리 과정

**`RL/loader.py`** — 전체 파이프라인에서 데이터를 공급하는 진입점.

#### Step 1: 허브 결정

전체 데이터에서 출발·도착을 합산하여 가장 빈도 높은 공항을 **허브**로 선정한다. Delta Airlines 데이터라면 보통 ATL(애틀랜타)이 허브가 된다. 이 허브가 `base_airport = 0`이 된다. 허브를 index 0으로 고정하는 이유는 `constraint["base_airport"] = 0`과 일치시키기 위함이다.

#### Step 2: hub_only 필터링 (선택적)

`hub_only=True`이면 오직 "허브 ↔ 스포크" 비행편만 남긴다.

**왜 hub_only를 쓰는가?**
hub_only 없이 임의의 flight를 쓰면 base-to-base pairing이 불가능한 flight가 섞여 들어간다. 예를 들어 ATL→BOS→ORD처럼 허브를 거치지 않는 경로는 ATL 복귀가 불가능하다. 이런 flight가 있으면 RL이 항상 강제 종료(deadhead)를 만나고, 신호가 노이즈로 오염된다. hub_only는 **모든 flight가 허브를 출발하거나 허브에 도착하도록 보장**하여 항상 base-to-base pairing이 가능한 깔끔한 문제를 만든다.

**round-trip 보장 로직**:
1. ATL→X 편이 있고 X→ATL 편도 있는 도시 X만 유지 ("round-trip city")
2. 샘플링 후 단방향만 남은 도시를 제거하고 pool에서 보충 (최대 10회 반복)
3. overnight timing 호환성 체크: X→ATL 도착 후 min_rest(8h)를 쉬어도 다음 날 ATL→X 출발이 가능한지 확인. 불가능한 flight는 제거.

#### Step 3: 시간 변환

```python
def convert_time(hhmm):
    h = hhmm // 100
    m = hhmm % 100
    return h + m / 60   # e.g. 630 → 6.5 (= 06:30)
```

이후 날짜별 offset을 더한다. 기준일이 day 0이면 다음 날 flight는 `dep_time += 24`, 이틀 후는 `+48`. 이렇게 하면 "24시간 넘어가는 dep_time"이 자연스럽게 multi-day 연결을 표현한다. 예: day 0 dep_time=22.5, day 1 dep_time=30.0이면 두 flight는 7.5시간 차이.

#### Step 4: 공항 ID 정수 변환

공항 코드(문자열) → 정수 ID. 빈도 내림차순 정렬이므로 허브가 항상 0번이다. Embedding 테이블의 index로 사용된다.

#### Step 5: dep_time 기준 정렬

RL 에피소드에서 unassigned flight 중 "dep_time이 가장 이른 것"부터 새 pairing을 시작하는 로직과 일관성을 맞추기 위해 정렬한다.

#### 최종 출력 포맷

```python
{
    "id":       int,    # 0부터 순서대로
    "origin":   int,    # 정수 공항 ID (0 = 허브 = base)
    "dest":     int,
    "dep_time": float,  # 시간 단위. e.g. 6.5 = 06:30, 30.0 = 다음날 06:00
    "arr_time": float,
}
```

---

### 3.3 `load_flights_multiday()` — 멀티데이 데이터 생성

**`RL/loader.py`** — multi-day pairing 학습을 위한 데이터 복제 함수.

```python
def load_flights_multiday(path, limit=200, n_days=4, seed=42, hub_only=False):
    base = load_flights(path, limit=limit, ...)  # 하루치 flight
    for day in range(n_days):
        for f in base:
            all_flights.append({
                "id":       day * limit + f["id"],
                "dep_time": f["dep_time"] + day * 24.0,
                "arr_time": f["arr_time"] + day * 24.0,
                ...
            })
```

**왜 복제하는가?**
실제 항공 운항은 매일 비슷한 스케줄이 반복된다. 하루치 50개 flight를 4일 복제하면 200개가 되고, day 0의 ATL→BOS와 day 1의 BOS→ATL 사이에 overnight connection이 자연스럽게 생긴다. 이 구조 덕분에 RL이 overnight rest(END_DUTY)를 학습할 기회가 생긴다.

`train.py`는 `limit=50, n_days=4, hub_only=True` → **총 200 flights**로 학습한다.

---

### 3.4 `RL/cppsc_loader.py` — 벤치마크 인스턴스 로더

Tahir 논문의 CPPSC 벤치마크 데이터를 ASCP-2026 포맷으로 변환한다. `eval_vs_baseline.py`에서 사용.

```
CPPSC 시간 단위: 분 (2000-01-01 기준 절대 시각)
ASCP-2026 단위: 시간 (float)
변환: dep_time = dep_abs / 60.0
```

**지원 항공기 타입**: `727, 09, 94, 95, 757, 319, 320`
**tightness 1~5**: availability constraint 수준. 숫자가 클수록 제약이 빡빡함.

CPPSC 전용 제약값 (`get_cppsc_constraints()`):
- `max_duty=12h, min_conn=0.5h, max_conn=4h, max_legs=5`
- Delta 제약(`max_conn=8h`)과 일부 다름에 주의.

---

## 4. 제약 조건

**`RL/constraints.py`** — 모든 제약의 단일 진실 소스(single source of truth).

### 4.1 Delta Airlines 제약값

| 키 | 값 | 설명 |
|----|-----|------|
| `max_duty` | 13.0h | FAA Part 117 최댓값. 실제로는 legs 수에 따라 동적 계산 |
| `min_conn` | 0.5h | 최소 연결 시간 (30분). 이보다 짧으면 환승 불가 |
| `max_conn` | 8.0h | 최대 연결 시간 (8시간). 이보다 길면 새 duty를 시작해야 함 |
| `max_legs` | 4 | duty당 최대 leg 수. 하루에 4개 비행편이 한계 |
| `base_airport` | 0 | base 공항 ID. pairing은 여기서 출발해 여기로 복귀해야 함 |
| `min_rest` | 9.5h | duty 간 최소 휴식 시간 |
| `max_duty_periods` | 4 | pairing당 최대 duty 횟수. 4일짜리 pairing이 최대 |
| `max_pairing_days` | 5 | pairing 최대 기간 (일) |

### 4.2 FiLM에 입력되는 constraint 벡터

`FILM_CONSTRAINT_KEYS`는 `base_airport`를 **제외**한 7개 키다:

```python
FILM_CONSTRAINT_KEYS = [
    "max_duty", "min_conn", "max_conn", "max_legs",
    "min_rest", "max_duty_periods", "max_pairing_days"
]
```

`base_airport`를 제외하는 이유: 공항 ID는 정수 카테고리값이라 연속값 FiLM 입력에 부적합. 공항 정보는 Embedding으로 별도 처리.

`constraint_to_tensor(constraint)` 함수가 이 7개 값을 float tensor로 변환한다:
```python
torch.tensor([constraint[k] for k in FILM_CONSTRAINT_KEYS], dtype=torch.float32)
# e.g. [13.0, 0.5, 8.0, 4.0, 9.5, 4.0, 5.0]
```

### 4.3 FAA Part 117 동적 max_duty

duty에 leg가 추가될수록 최대 duty 시간이 줄어든다. 연속 비행이 많을수록 피로도가 높기 때문이다.

```python
def get_max_duty(legs_after, constraint):
    table = {1: 13.0, 2: 13.0, 3: 12.0, 4: 11.5, 5: 11.0, 6: 10.5}
    return table.get(min(legs_after, 6), 10.0)
```

마스킹에서 실제 적용값:
```python
effective_max_duty = min(get_max_duty(legs_after, constraint), constraint["max_duty"])
```

즉, FAA 테이블값과 constraint의 `max_duty` 중 **작은 값**을 사용한다. constraint를 더 빡빡하게 설정하면 FAA 기준보다 엄격하게 적용된다.

---

## 5. RL 환경

**`RL/environment.py`** — RL 에이전트가 상호작용하는 환경. 상태 전이, 마스킹, 보상을 담당.

### 5.1 액션 공간

```
인덱스 0 ~ N-1  : flight i 선택 (N = 총 flight 수)
인덱스 N        : END_DUTY — 현재 duty 종료, rest period 진입, pairing은 계속
인덱스 N+1      : END_PAIRING — pairing 완전 종료, base 복귀 선언
```

총 `N+2`개의 discrete action. PointerDecoder가 이 `N+2`개에 대한 확률 분포를 출력한다.

### 5.2 상태 구조

`RL/state.py`의 `init_state()`가 에피소드 시작 시 초기 상태를 만든다.

| 필드 | 초기값 | 의미 |
|------|--------|------|
| `current_airport` | `flights[0]["origin"]` | 에이전트의 현재 위치 공항 |
| `current_time` | `flights[0]["dep_time"]` | 현재 시각 (시간 단위 float) |
| `duty_time` | 0.0 | 현재 duty에서 누적된 비행 시간 |
| `duty_start_time` | `flights[0]["dep_time"]` | 현재 duty 시작 시각 |
| `legs` | 0 | 현재 duty의 leg 수 |
| `remaining` | `len(flights)` | 아직 배정되지 않은 flight 수 |
| `pairing_start` | `True` | 새 pairing 첫 번째 leg 여부. True면 공항 연결 제약 없음 |
| `duty_period` | 0 | 지금까지 완료된 duty 횟수 (END_DUTY 누적 수) |
| `pairing_start_time` | `flights[0]["dep_time"]` | 현재 pairing 전체 시작 시각 |
| `is_resting` | `False` | 현재 rest 중인지 여부 |
| `rest_end_time` | `None` | rest가 끝나는 시각. is_resting=True일 때 사용 |

**`pairing_start` 필드의 역할:**
새 pairing의 첫 번째 leg는 어느 공항에서든 출발할 수 있어야 한다 (강제로 새 pairing을 시작하기 때문). `pairing_start=True`이면 `origin == current_airport` 체크를 건너뛴다.

### 5.3 마스킹 — `get_mask(state, flights, assigned, constraint)`

마스킹은 **현재 state에서 규정상 불가능한 action을 0으로 막는** 하드 필터다. PointerDecoder가 `-inf` 처리해서 softmax에서 확률이 0이 된다. 이를 통해 신경망이 아무리 불법 action을 선호해도 절대 선택되지 않는다.

반환값: `[f0, f1, ..., fN-1, END_DUTY, END_PAIRING]` 각각 0 또는 1.

**각 flight에 대한 유효성 체크 로직 (순서대로 AND 조건):**

```
[1] assigned[flight_id] == False        → 이미 배정된 flight 제외

[2] is_resting == True인 경우 (rest 중):
    dep_time >= rest_end_time           → rest가 끝난 뒤 출발하는 flight만
    origin == current_airport           → 공항 연결 (connection 시간 제약 없음)
    * 정상 connection 제약(min_conn, max_conn)은 적용 안 함

[3] pairing_start == True인 경우 (duty 첫 leg):
    duty_time + flight_time <= effective_max_duty
    legs + 1 <= max_legs
    * origin 체크 없음 (새 pairing은 어디서든 시작 가능)

[4] 그 외 (duty 진행 중, not resting):
    origin == current_airport           → 현재 위치 공항에서 출발
    min_conn <= gap <= max_conn         → gap = dep_time - current_time
    duty_time + flight_time <= effective_max_duty
    legs + 1 <= max_legs

[5] 공통:
    (dep_time - pairing_start_time) / 24 <= max_pairing_days
```

**END_DUTY 허용 조건:**
```
legs > 0                              → 현재 duty에 적어도 1개 leg 있음
not is_resting                        → rest 중이 아님
not pairing_start                     → duty 진행 중
duty_period < max_duty_periods - 1    → 아직 duty 횟수 여유 있음
```

`max_duty_periods - 1`인 이유: duty_period는 완료된 duty 수이므로, 현재 duty까지 포함하면 duty_period+1이 된다. `max_duty_periods=4`이면 duty_period가 0,1,2까지만 END_DUTY 가능. duty_period=3이면 더 이상 새 duty를 시작할 수 없으므로 END_DUTY 불가.

**END_PAIRING 허용 조건:**
```
current_airport == base_airport       → base 공항에 있어야 함
legs > 0                              → 적어도 1개 leg 있음
(current_time - pairing_start_time) / 24 <= max_pairing_days
```

### 5.4 `step()` — flight 선택 실행

flight action이 선택되면 이 함수가 상태를 전이시키고 즉시 보상을 반환한다.

```python
def step(state, action, flights, assigned, constraint):
    f = flights[action]
    assigned[f["id"]] = True   # 배정 완료 마킹

    flight_time = f["arr_time"] - f["dep_time"]

    next_state = {
        "current_airport":    f["dest"],
        "current_time":       f["arr_time"],
        "duty_time":          state["duty_time"] + flight_time,
        "duty_start_time":    state["duty_start_time"],
        "legs":               state["legs"] + 1,
        "remaining":          state["remaining"] - 1,
        "pairing_start":      False,   # 첫 leg가 지나면 False
        "duty_period":        state["duty_period"],
        "pairing_start_time": state["pairing_start_time"],
        "is_resting":         False,
        "rest_end_time":      None,
    }

    # 보상: 연결 대기 시간 (음수)
    if not state["pairing_start"] and not state["is_resting"]:
        reward = -(f["dep_time"] - state["current_time"])
    else:
        reward = 0.0   # 첫 leg 또는 rest 직후는 대기가 아님
```

**보상 설명:**
`f["dep_time"] - state["current_time"]`은 이전 leg 도착 후 이번 leg 출발까지의 gap이다. 이 gap이 dead time이다. 음수로 반환하여 dead time이 클수록 총 보상이 낮아지도록 한다.

`pairing_start=True`이거나 `is_resting=True`인 경우 gap을 계산하지 않는다. 새 pairing의 첫 leg는 선택의 여지가 없이 강제로 시작되는 것이고, rest 직후는 overnight rest가 포함된 gap이라 dead time이 아니기 때문이다.

### 5.5 `step_end_duty()` — duty 종료, rest 진입

END_DUTY action이 선택되면 이 함수가 상태를 전이시킨다.

```python
def step_end_duty(state, constraint):
    min_rest = constraint["min_rest"]   # 9.5h
    return {
        **state,
        "duty_time":       0.0,                              # duty 시간 리셋
        "duty_start_time": state["current_time"] + min_rest, # 다음 duty 시작 시각
        "legs":            0,                                # leg 수 리셋
        "is_resting":      True,                             # rest 중 플래그
        "rest_end_time":   state["current_time"] + min_rest, # rest 끝나는 시각
        "duty_period":     state["duty_period"] + 1,         # duty 횟수 증가
        "pairing_start":   False,
    }
```

**이 상태에서 다음 flight 마스킹:**
`is_resting=True`이므로 `dep_time >= rest_end_time`이어야 하고, 공항은 그대로 `current_airport`여야 한다. rest 시간 동안 공항을 이동하지 않는다고 가정한다.

train.py에서 END_DUTY를 선택하면 `OVERNIGHT_PENALTY = -0.5`를 즉시 부과한다. 이는 multi-day pairing을 허용하되 비용으로 기록한다.

### 5.6 `final_reward()` — 에피소드 종료 시 보상

```python
def final_reward(assigned, uncovered_penalty=10.0):
    remaining = sum(1 for v in assigned.values() if not v)
    return -uncovered_penalty * remaining
```

에피소드가 끝났을 때 배정되지 않은 flight가 있으면 flight 1개당 -10을 부과한다. 이 큰 페널티가 RL로 하여금 모든 flight를 빠짐없이 커버하도록 강제한다.

### 5.7 강제 이동 (deadhead handling)

`mask_list`에서 flight도 END_DUTY도 모두 0인 경우, 즉 어떤 합법적인 action도 없을 때 처리:

```python
# 1. 현재 진행 중인 pairing이 있으면 강제 종료 (pairing_start=False인 경우)
if not state["pairing_start"]:
    n_pairings += 1
    n_deadheads += 1
    if state["current_airport"] != base_airport:
        total_reward -= BASE_PENALTY   # -5.0 (base 복귀 못함)
    total_reward -= PAIRING_COST       # -5.0

# 2. 미배정 flight 중 dep_time 최소인 것으로 새 pairing 강제 시작
earliest = min(unassigned, key=lambda x: x["dep_time"])
state = {
    "current_airport":    earliest["origin"],
    "current_time":       earliest["dep_time"],
    "duty_time":          0.0,
    "legs":               0,
    "pairing_start":      True,
    "duty_period":        0,
    ...
}
```

이 강제 이동이 발생하는 경우는 연결 가능한 flight가 없거나 duty 한계에 다다랐지만 END_PAIRING 조건(base 복귀)도 안 되는 상황이다. RL이 이 상황을 최대한 피하도록 학습하는 것이 목표다.

---

## 6. 신경망 구조

### 6.1 전체 구조의 직관적 이해

신경망은 두 부분으로 나뉜다:

- **FlightEncoder**: "어떤 비행편이 있고, 서로 어떤 관계인가"를 한 번만 계산한다. 에피소드 당 1회.
- **PointerDecoder**: "지금 이 상황에서 어디로 가야 하는가"를 매 step 계산한다.

이 분리가 효율적인 이유: flight 목록은 에피소드 내에서 바뀌지 않는다. 같은 flight를 매 step 다시 encode하는 것은 낭비다.

---

### 6.2 FiLM — `model/film.py`

**Feature-wise Linear Modulation**: "항공사 규정에 따라 같은 비행편도 다르게 해석한다"는 개념.

```
constraint 벡터 (7차원)
        │
    MLP (Linear → ReLU → Linear)
        │
   gamma (128차원) + beta (128차원)   ← 파라미터 두 세트를 한번에 생성
        │
flight_vecs (N, 128) × gamma.unsqueeze(0) + beta.unsqueeze(0)
        │
변조된 flight_vecs (N, 128)
```

수식으로 표현하면:
```
output = gamma * flight_vecs + beta
```

이것이 Feature-wise Linear Modulation이다. gamma는 각 차원을 얼마나 강조할지 (scaling), beta는 얼마나 이동시킬지 (shift)를 결정한다. `max_duty`가 6h인 제약이면 flight_time이 긴 비행편에 불리하게 변조되는 식이다.

`use_skip=True`이면 `output += flight_vecs` (residual connection). FiLM이 원본 embedding을 완전히 덮어쓰지 않도록, 원본 정보를 보존한다. Ablation에서 `skip_film=True`를 설정하면 이 mode로 동작.

FiLM의 핵심 가치: **하나의 모델이 다른 제약 조건에서도 동작한다.** max_duty가 6h인 항공사와 14h인 항공사에 같은 모델을 쓸 수 있다. Stage 3에서 `max_duty`를 랜덤 샘플링하며 학습하는 이유가 바로 이것이다.

---

### 6.3 FlightEncoder — `model/encoder.py`

에피소드당 1회 호출. flight들의 정적 표현(static representation)을 만든다.

#### 입력

```
origins   (N,)  int    — 출발 공항 ID
dests     (N,)  int    — 도착 공항 ID
dep_times (N,)  float  — 출발 시각 (시간 단위)
arr_times (N,)  float  — 도착 시각
constraint (7,) float  — FiLM 변조용 제약 벡터
```

#### 처리 순서

**Step 1: 공항 Embedding**
```python
o_emb = airport_emb(origins)   # (N, 32) — 출발 공항 특성 벡터
d_emb = airport_emb(dests)     # (N, 32) — 도착 공항 특성 벡터
times = [dep_time, arr_time]   # (N, 2)
x = cat([o_emb, d_emb, times]) # (N, 66)
x = flight_mlp(x)              # (N, 128) — 2층 MLP로 d_model 차원으로 압축
```

`nn.Embedding(n_airports, 32)`: 공항 ID를 32차원 학습 가능 벡터로 변환. ATL이 hub이면 index 0인 벡터가 ATL의 특성을 담는다. 학습을 통해 지리적·운항 특성이 자동으로 인코딩된다.

`flight_mlp`: `Linear(66→128) → ReLU → Linear(128→128)`. 출발지, 도착지, 시간 정보를 하나의 128차원 flight 표현으로 통합.

**Step 2: FiLM (before Transformer)**
```python
x = film_before(x, constraint)   # (N, 128)
```

Transformer 전에 constraint를 반영한다. Transformer가 constraint를 고려한 상태에서 flight 간 attention을 계산하도록 한다.

**Step 3: Transformer Encoder**
```python
x = transformer(x.unsqueeze(0)).squeeze(0)   # (N, 128)
```

2-layer, 4-head Transformer. flight들이 서로 attention을 주고받으며 관계를 파악한다. "이 비행편 다음에 저 비행편을 연결할 수 있다"는 시퀀스 관계가 attention으로 포착된다.

`skip_transformer=True`이면 Transformer 입력을 출력에 더한다: `output += input_before_transformer`. FiLM이 만든 정보가 Transformer를 통과하면서 희석될 수 있는데, skip이 이를 방지한다.

**Step 4: FiLM (after Transformer)**
```python
x = film_after(x, constraint)    # (N, 128)
```

Transformer 후에 constraint를 한 번 더 반영한다. Transformer가 flight 간 관계를 파악한 후에도 constraint 정보를 강조한다. FiLM이 Transformer 전후 양쪽에 적용되므로 constraint 영향이 더 강하게 작용한다.

#### 출력

```
encoded_flights (N, 128) — 에피소드 내내 재사용되는 정적 flight 표현
```

---

### 6.4 PointerDecoder — `model/decoder.py`

매 step 호출. "지금 state에서 어느 flight (또는 END_DUTY/END_PAIRING)를 선택해야 하는가"를 결정하는 확률 분포를 생성한다.

#### 핵심 아이디어: Pointer Attention

Pointer Network의 핵심은 "입력 시퀀스의 원소를 그대로 가리키는(point)' attention이다. flight를 새로운 공간으로 변환하는 것이 아니라, 기존에 encode된 flight를 직접 참조한다.

#### state_vec 구성

```python
airport_emb = encoder.airport_emb(current_airport)   # (32,)
state_scalars = [
    time_of_day,      # current_time % 24 / 24      → 하루 중 몇 시인지 (0~1)
    day_norm,         # current_time // 24 / max_pairing_days → 며칠째인지 (0~1)
    duty_time_norm,   # duty_time / max_duty          → duty 시간 얼마나 썼는지 (0~1)
    legs_norm,        # legs / max_legs               → leg 몇 개 썼는지 (0~1)
    duty_period_norm, # duty_period / max_duty_periods → duty 몇 번째인지 (0~1)
    is_resting,       # 0.0 or 1.0
]
state_vec = cat([airport_emb, state_scalars])   # (38,)
```

**정규화 이유**: 다양한 스케일의 값들을 0~1 사이로 맞춰 학습 안정화. 예를 들어 `duty_time`은 0~13h이지만 `max_duty=13`으로 나누면 0~1 사이가 된다.

#### 처리 순서

```python
# 1. state → query 벡터
q = state_mlp(state_vec)            # (38,) → (128,)
if skip_state_mlp:
    q = q + skip_proj(state_vec)    # skip_proj: (38,→128) linear
q = W_q(q)                          # (128,)

# 2. key 벡터 구성 (flight + 특수 토큰)
keys = cat([
    encoded_flights,                # (N, 128) — encoder 출력 재사용
    end_duty_token.unsqueeze(0),    # (1, 128) — 학습 가능 파라미터
    end_pairing_token.unsqueeze(0), # (1, 128) — 학습 가능 파라미터
])                                  # (N+2, 128)
k = W_k(keys)                       # (N+2, 128)

# 3. dot-product attention
scores = (k @ q) / sqrt(128)        # (N+2,) — scaled dot product
scores[mask == 0] = -inf            # 불가 action 제거
probs = softmax(scores)             # (N+2,)
```

**Scaled dot product attention**: `Q·K / sqrt(d_model)`. `sqrt(d_model)`로 나누는 이유는 dot product 값이 차원이 커질수록 커지는 것을 막기 위함이다. 크기가 너무 크면 softmax가 거의 one-hot이 되어 탐색이 안 된다.

**`end_duty_token`과 `end_pairing_token`**: `nn.Parameter`로 선언된 학습 가능 벡터. RL이 학습되면서 "END_DUTY를 선택하기 좋은 상황"의 key 표현을 자동으로 학습한다.

#### 출력

```
probs (N+2,)
  [0~N-1]: 각 flight 선택 확률
  [N]:     END_DUTY 확률
  [N+1]:   END_PAIRING 확률
```

---

## 7. 학습 루프 (REINFORCE)

**`experiments/train.py` — `run_curriculum_stage()` 내부**

### 7.1 REINFORCE 알고리즘

Policy Gradient의 가장 기본 형태. 보상이 높은 action의 확률을 높이고, 낮은 action의 확률을 낮춘다.

**핵심 손실 함수:**
```
L = -Σ_t [ log π(a_t | s_t) × A ]  +  entropy 정규화
```

- `log π(a_t | s_t)`: 선택한 action의 log 확률 (Categorical.log_prob)
- `A` (advantage): 이 에피소드가 베이스라인보다 얼마나 좋았는가
- `entropy`: 정책의 불확실성. 높을수록 다양한 action을 탐색

**advantage 계산:**
```python
advantage = (reward_sample - reward_greedy) / (abs(reward_greedy) + 1e-6)
```

- `reward_sample`: 이번 에피소드의 stochastic rollout 보상
- `reward_greedy`: 현재 파라미터로 greedy rollout한 보상 (베이스라인)
- `|reward_greedy| + 1e-6`로 나눠 정규화 (분모 0 방지)

advantage가 **양수**이면: sample이 greedy보다 좋았다 → 이 에피소드에서 선택한 action들의 확률을 높임
advantage가 **음수**이면: sample이 greedy보다 나빴다 → 이 에피소드에서 선택한 action들의 확률을 낮춤

### 7.2 greedy baseline의 역할

매 업데이트마다 **같은 파라미터**로 greedy rollout을 한 번 더 수행한다 (no_grad). 이것이 베이스라인이다.

```python
with torch.no_grad():
    encoded_g = encoder(origins, dests, dep_times, arr_times, c_tensor)
    reward_g, _, _, metrics_g = run_episode(..., greedy=True)
```

greedy rollout은 항상 확률이 가장 높은 action만 선택한다. 즉, "현재 정책이 가장 자신 있는 결과"가 베이스라인이 된다. sample이 이보다 좋아야 강화되고, 나빠야 약화된다.

이 방식의 장점: 별도의 value network를 학습하지 않아도 된다. 단점: 베이스라인 추정이 noisy할 수 있다.

### 7.3 entropy 보너스

```python
loss = torch.stack([
    -lp * advantage - 0.01 * ent
    for lp, ent in zip(log_probs, entropies)
]).sum()
```

`-0.01 * entropy`는 loss를 낮추는 방향으로 entropy를 높게 유지한다. entropy가 높다는 것은 여러 action에 고루 확률이 분포한다는 뜻이다. 초반 학습에서 너무 일찍 하나의 action에만 집중하는 것(premature convergence)을 방지한다.

### 7.4 파라미터 업데이트

```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
optimizer.step()
```

`params = encoder.parameters() + decoder.parameters()` — encoder와 decoder가 동시에 업데이트된다.

`clip_grad_norm_(max_norm=1.0)`: gradient의 L2 norm이 1.0을 초과하면 1.0으로 클리핑. 큰 advantage에 의한 gradient 폭발을 방지한다.

### 7.5 체크포인트 저장 기준

```python
if len(greedy_pairings) >= 25:
    recent_avg = sum(greedy_pairings[-25:]) / 25
    if recent_avg < best_avg_pairings:
        best_avg_pairings = recent_avg
        torch.save({...}, f"stage{stage}_best.pt")
```

greedy rollout의 pairing 수 **25 에피소드 이동평균**이 역대 최솟값을 갱신할 때마다 저장. pairing 수가 적을수록 좋은 해이므로, 이동평균이 낮아진다는 것은 정책이 개선되고 있다는 신호다.

이동평균을 쓰는 이유: 단일 에피소드 결과는 noisy하기 때문. 25ep 평균이 실질적인 개선을 더 잘 반영한다.

---

## 8. 커리큘럼 학습 (3단계)

### 8.1 왜 커리큘럼이 필요한가

multi-day pairing 전체 문제를 처음부터 학습하면 학습 공간이 너무 넓다. RL이 의미 있는 신호를 찾기 전에 수렴에 실패한다. 쉬운 버전부터 점진적으로 난이도를 높이는 커리큘럼 방식으로 학습 효율을 높인다.

### 8.2 Stage 1 — 단일 duty (1000 에피소드)

```python
stage1_c = {**base, "max_duty_periods": 1, "max_pairing_days": 1}
```

**`max_duty_periods=1`의 효과**: END_DUTY를 선택하면 duty_period가 1이 되는데, `can_end_duty`의 조건인 `duty_period < max_duty_periods - 1 = 0`을 절대 만족할 수 없다. 따라서 **END_DUTY는 항상 mask=0**이 된다.

**이 단계에서 학습하는 것**: flight-to-flight connection. "이 비행편 다음에 저 비행편을 연결할 수 있는가", "dead time을 최소화하는 연결은 무엇인가".

overnight과 multi-day가 없으므로 action space가 훨씬 작다. 기본적인 pairing 로직이 빠르게 수렴한다.

### 8.3 Stage 2 — Full multi-day (2000 에피소드)

```python
stage2_c = {**base, "max_duty_periods": 4, "max_pairing_days": 5}
```

Stage 1에서 학습된 파라미터를 그대로 이어받아 (hot-start) multi-day 구조를 추가로 학습한다.

**이 단계에서 학습하는 것**: END_DUTY 선택 타이밍. "언제 현재 duty를 마치고 rest를 취하는 것이 좋은가". overnight rest 후 다음 날 다시 연결하는 패턴.

Stage 1 파라미터에서 시작하므로 기본 connection 로직은 이미 갖춰진 상태. Stage 2는 END_DUTY를 언제 사용할지만 추가로 학습하면 된다.

### 8.4 Stage 3 — Constraint 랜덤 augmentation (2000 에피소드)

```python
def sample_constraint():
    return {**stage3_base, "max_duty": random.uniform(6.0, 14.0)}
```

매 에피소드마다 `max_duty`를 6~14h 사이 균등 분포에서 랜덤 샘플링.

**이 단계에서 학습하는 것**: FiLM이 constraint 변화에 반응하는 법. `max_duty=6h`이면 leg를 적게 연결해야 하고, `max_duty=14h`이면 더 많이 연결할 수 있다. FiLM의 gamma/beta가 이 차이를 flight embedding에 반영하도록 학습된다.

**FiLM 검증 (Stage 3 완료 후)**: 같은 flight, 같은 encoder/decoder로 `max_duty ∈ {6, 8, 10, 12, 14}`를 각각 greedy 평가. constraint에 따라 pairing 수와 coverage가 달라지는지 확인.

---

## 9. 보상 함수 설계

### 9.1 설계 철학

RL의 보상은 다음 두 조건을 만족해야 효과적으로 학습된다:
1. **Dense signal**: 매 step마다 의미 있는 피드백이 있어야 한다.
2. **목표와 일치**: 최종 목표(ManDays 최소화, dead time 최소화)와 보상 방향이 일치해야 한다.

ManDays를 직접 보상으로 쓰면 `END_PAIRING`에서만 신호가 온다 (sparse). Dead time은 매 flight 선택마다 관측 가능하므로 dense signal이다.

### 9.2 보상 구성 요소

#### step 보상 — dead time (매 flight 선택 시)

```python
if not pairing_start and not is_resting:
    reward = -(f["dep_time"] - state["current_time"])
```

`dep_time - current_time`이 연결 대기 시간(dead time)이다. 이를 음수로 주면 dead time이 클수록 total_reward가 낮아진다. RL은 dead time을 줄이는 방향으로 학습한다.

**`pairing_start=True` 또는 `is_resting=True`이면 reward=0.0**: 새 pairing의 첫 leg는 강제로 배정되는 것이고, rest 직후는 overnight 시간이 포함된 gap이라 실질적 dead time이 아니다.

#### LEG_BONUS — 연결 장려 (2번째 leg부터)

```python
if prev_legs >= 1:
    total_reward += LEG_BONUS   # +1.5
```

같은 duty 내에서 flight를 더 연결할수록 보너스. `prev_legs >= 1`이면 현재 leg가 2번째 이상이라는 뜻이다. 이 보너스가 없으면 RL이 매번 1-leg pairing을 만드는 전략을 취할 수 있다 (dead time이 항상 0인 첫 leg만 쓰면 step 보상에서 손해가 없으므로).

#### OVERNIGHT_PENALTY — multi-day 허용하되 비용 부과

```python
total_reward -= OVERNIGHT_PENALTY   # -0.5 per END_DUTY
```

END_DUTY를 선택할 때마다 -0.5. multi-day pairing을 완전히 금지하지 않지만 비용을 부과하여 불필요한 overnight을 억제한다. LEG_BONUS(+1.5)에 비해 작게 설정하여 overnight 후 leg 연결이 충분히 많으면 overnight이 유리하도록 균형을 맞췄다.

#### PAIRING_COST — pairing 수 최소화 압력

```python
total_reward -= PAIRING_COST   # -5.0 per pairing
```

END_PAIRING 또는 강제 종료 시마다 -5.0. pairing이 적을수록 유리하므로 ManDays를 간접적으로 최소화한다.

#### BASE_PENALTY — feasibility 강제

```python
if current_airport != base_airport:
    total_reward -= BASE_PENALTY   # -5.0
```

base 공항이 아닌 곳에서 pairing이 종료되면 -5.0 추가 패널티. pairing은 반드시 base에서 끝나야 하므로 이 조건을 위반하는 것은 규정 위반이다.

#### UNCOVERED_PENALTY — coverage 강제

```python
total_reward += -10.0 × n_uncovered_flights
```

에피소드 종료 시 미배정 flight당 -10. 가장 큰 단위의 패널티. RL이 모든 flight를 커버하도록 강하게 강제한다.

### 9.3 보상 스케일 관계

```
LEG_BONUS(+1.5) vs OVERNIGHT_PENALTY(-0.5)
  → overnight 후 추가 leg 연결이 있으면 net gain. multi-day 적극 허용.

PAIRING_COST(-5.0) vs UNCOVERED_PENALTY(-10.0)
  → 미배정 flight 1개 = pairing 2개 추가와 동일 비용.
     커버를 포기하는 것보다 pairing을 하나 더 만드는 게 낫다.

dead_time penalty vs UNCOVERED_PENALTY
  → dead_time은 보통 수 시간 단위이므로 -수 정도.
     UNCOVERED_PENALTY는 -10. 미배정 1개가 dead_time 10h와 동일.
```

---

## 10. IP 후처리 (Set Partitioning)

### 10.1 왜 IP가 필요한가

RL만으로는 최적 pairing 집합을 보장할 수 없다. RL은 greedy하게 하나의 pairing 시퀀스를 만들어낸다. 하지만 최적 해는 여러 pairing 후보 중 최선의 조합을 선택해야 한다. 이 조합 최적화 문제를 IP가 해결한다.

또한 RL은 deadhead(DH)를 생성할 수 없다. RL에서 각 flight는 `assigned` 딕셔너리에 의해 정확히 1번만 배정된다. IP의 Set Covering(≥1) 제약은 하나의 flight를 여러 pairing이 공유할 수 있게 허용한다. 이것이 deadhead다.

### 10.2 pairing pool 생성 — `evaluate_ip.py`

```python
pool = collect_pool(flights, constraint, encoder, decoder, encoded, n_rollouts=100)
```

**과정:**
1. **stochastic rollout × 100번**: Categorical 분포에서 action 샘플링. 매 rollout마다 다른 pairing 시퀀스가 생성됨.
2. **greedy rollout × 1번**: 확실히 feasible한 후보 1개를 추가.
3. **중복 제거**: `tuple(sorted(legs))` 기준. 같은 flight 집합을 담은 pairing은 한 번만 포함.

**각 pairing의 비용 계산:**

```python
elapsed   = pairing_last_arr - pairing_dep      # 전체 경과 시간
fly       = sum(flight_time for each leg)       # 실제 비행 시간
rest      = sum(min_rest per overnight)          # overnight rest 누적 (dead time 제외)
dead_time = elapsed - fly - rest                # 실제 dead time

rl_bonus  = 1.5 × max(n_legs - 1, 0)           # leg 많을수록 IP 비용 감소
dh_penalty = 5.0 if is_forced else 0.0          # 강제 종료 pairing 억제

cost = dead_time - rl_bonus + dh_penalty
```

**overnight rest를 dead_time에서 제외하는 이유**: overnight rest는 규정상 반드시 취해야 하는 것으로, 낭비 시간이 아니다. 포함하면 multi-day pairing이 불합리하게 불리해진다.

**`rl_bonus`**: RL 학습에서 쓴 `LEG_BONUS`와 동일한 값(1.5)으로, IP 비용에도 반영한다. legs가 많은 pairing일수록 IP가 선호하도록 유도.

### 10.3 Set Partitioning IP — `set_partition.py`

#### 수식 (Klabjan et al. 2001)

```
min  Σ_j  c_j × x_j                              목적함수: 총 비용 최소화
s.t. Σ_{j: i ∈ j} x_j = 1   ∀ flight i          각 flight 정확히 1번 커버
     x_j ∈ {0, 1}                                 pairing 선택 여부
```

실제로는 slack 변수 `s_i ≥ 0`를 더해 infeasible 방지:
```
Σ_{j: i ∈ j} x_j + s_i = 1
```
slack이 양수가 되면 penalty=1e6이 부과되므로 실질적으로 flight를 커버하도록 강제되나, 어떤 pairing도 커버 못하는 flight가 있을 때 solver가 죽지 않게 한다.

#### 3단계 풀이 전략

**Step 1: LP Relaxation**

`x_j ∈ {0,1}` 대신 `x_j ∈ [0,1]`로 완화하여 LP를 푼다. LP는 IP보다 훨씬 빠르게 풀린다.

LP 해에서 **dual variable** π_i를 추출한다. π_i는 flight i에 대한 "shadow price"로, flight i를 커버하는 것의 가치를 나타낸다.

**Reduced cost** 계산:
```
rc_j = c_j - Σ_{i ∈ j} π_i
```

`rc_j < 0`: 이 pairing을 쓰면 dual 기준으로 비용이 줄어든다 → IP 최적해에 포함될 가능성이 높음
`rc_j ≥ 0`: 비용이 줄지 않는다 → IP 최적해에서 제외될 가능성이 높음

**Step 2: Column Reduction**

`rc_j ≤ 1e-6` (≈ 0)인 pairing만 유지. 이것이 column generation의 핵심이다. 수천 개의 후보를 수십~수백 개로 줄인다.

안전장치: 각 flight를 커버하는 pairing이 최소 1개는 남아야 한다. 모든 후보가 제거될 위기인 flight는 rc가 가장 낮은 pairing 1개를 강제 유지한다.

**Step 3: IP 풀기**

줄어든 후보 집합으로 실제 `x_j ∈ {0,1}` IP를 풀어 최적 pairing 조합을 선택한다.

solver: PuLP + CBC (기본), Gurobi 옵션. 시간 제한 300초.

#### 출력

```python
{
    "selected":    [pairing_dict, ...],  # 선택된 pairing 목록
    "n_pairings":  int,                  # = ManDays
    "total_cost":  float,                # 선택된 pairing의 cost 합계
    "coverage":    float,                # 0.0~1.0 (1.0 = 전체 커버)
    "status":      str,                  # "Optimal", "Infeasible", etc.
    "uncoverable": int,                  # 어떤 후보도 없는 flight 수
}
```

---

## 11. 평가 파이프라인

### 11.1 `evaluate_ip.py` — RL+IP 통합 평가

학습된 모델로 pairing을 생성하고 IP로 최적화한 결과를 평가한다.

```bash
python evaluate_ip.py checkpoints/model_latest.pt [max_duty=10.0]
```

**과정:**

1. `load_flights(limit=200, hub_only=True, n_days_max=4)` 로드
2. constraint 설정 (`max_duty=10.0h`)
3. 체크포인트 로드 → `encoder.eval()`, `decoder.eval()` (dropout off, BN 추론 모드)
4. `collect_pool(n_rollouts=100)` → pairing pool 생성
5. `solve_set_partitioning(pool)` → 최적 pairing 선택
6. 결과 출력

**출력 지표:**

| 지표 | 의미 |
|------|------|
| `n_pairings` | 선택된 pairing 수 = ManDays |
| `total_cost` | IP 목적함수 값 (dead_time - rl_bonus + dh_penalty 합계) |
| `coverage` | 커버된 flight 비율 |
| `fly_time` | 선택된 pairing의 실제 비행 시간 합계 |
| `dead_time` | 선택된 pairing의 실제 dead_time 합계 |
| `FTC` | `dead_time / fly_time × 100%` |

### 11.2 `eval_vs_baseline.py` — Tahir I²CGp와 gap 비교

CPPSC 벤치마크 인스턴스에서 RL 결과를 Tahir의 I²CGp baseline과 비교한다.

```bash
python eval_vs_baseline.py --checkpoint checkpoints/model_latest.pt
                           [--at 09]          # 항공기 타입 필터
                           [--tightness 1]    # tightness 1~5
                           [--results path/to/i2cgp_results.json]
```

**평가 방법**: greedy rollout만 사용 (stochastic rollout + IP 없이 직접 비교).

**gap 공식:**
```
gap = (n_RL_pairings - n_baseline_pairings) / n_baseline_pairings × 100%
```

양수 = RL이 pairing 더 많음 (나쁨), 음수 = RL이 pairing 더 적음 (좋음).

**주의사항**: 학습 데이터의 `n_airports`보다 큰 CPPSC 인스턴스는 Embedding 범위를 초과하므로 SKIP된다.

baseline 데이터 소스: `../Tahir/experiments/i2cgp_results.json`. 같은 인스턴스에 i2cgp와 i2cg 둘 다 있으면 i2cgp 우선 사용.

---

## 12. Ablation 실험

### 12.1 목적

skip connection이 학습에 도움이 되는지 검증. 3개 위치(FiLM, Transformer, state_mlp)에서 각각, 그리고 전부 skip을 적용한 구성과 적용 안 한 구성을 비교.

**`experiments/skip_ablation.py`**

### 12.2 5개 구성

| 구성명 | skip_film | skip_transformer | skip_state_mlp | 설명 |
|--------|-----------|-----------------|----------------|------|
| `baseline` | False | False | False | control 그룹. skip 없음 |
| `film_skip` | True | False | False | FiLM에만 residual 추가 |
| `transformer_skip` | False | True | False | Transformer에만 residual 추가 |
| `decoder_skip` | False | False | True | Decoder state_mlp에만 residual 추가 |
| `all_skip` | True | True | True | 3개 모두 skip |

**각 skip의 의미:**
- `skip_film=True`: FiLM 출력 += 원본 flight_vecs. FiLM이 flight 정보를 완전히 덮어쓰지 않도록.
- `skip_transformer=True`: Transformer 출력 += Transformer 입력(= film_before 출력). Transformer가 FiLM 정보를 희석하지 않도록.
- `skip_state_mlp=True`: state_mlp 출력 += skip_proj(state_vec). state_mlp가 원본 state 정보를 보존하도록.

### 12.3 실험 설정

- 600 에피소드, 50 flights, seed=42 고정 (재현성)
- 학습 완료 후 `max_duty ∈ {6, 8, 10, 12, 14}` 5개 값으로 greedy 평가
- 결과: `experiments/results/skip_ablation_results.json`

---

## 13. 하이퍼파라미터 일람

### 신경망

| 파라미터 | 값 | 위치 |
|---------|-----|------|
| `d_model` | 128 | encoder, decoder |
| `airport_emb_dim` | 32 | encoder, decoder |
| `nhead` | 4 | Transformer |
| `num_layers` | 2 | Transformer |
| `dim_feedforward` | 512 (= d_model × 4) | Transformer |
| `constraint_dim` | 7 | FiLM (FILM_CONSTRAINT_KEYS 수) |
| FiLM hidden_dim | d_model=128 | FiLM MLP |

### 학습

| 파라미터 | 값 | 위치 |
|---------|-----|------|
| `lr` | 1e-4 | Adam optimizer |
| `max_norm` | 1.0 | gradient clip |
| `entropy_coef` | 0.01 | REINFORCE loss |
| Stage 1 에피소드 수 | 1000 | train.py |
| Stage 2 에피소드 수 | 2000 | train.py |
| Stage 3 에피소드 수 | 2000 | train.py |
| checkpoint 기준 | 25ep 이동평균 최소 pairings | train.py |
| 무한루프 방지 | `max_steps = len(flights) × 20` | train.py |

### 보상

| 상수 | 값 | 의미 |
|-----|-----|------|
| `PAIRING_COST` | 5.0 | pairing당 페널티 |
| `BASE_PENALTY` | 5.0 | base 미복귀 페널티 |
| `UNCOVERED_PENALTY` | 10.0 | 미배정 flight당 페널티 |
| `OVERNIGHT_PENALTY` | 0.5 | END_DUTY당 페널티 |
| `LEG_BONUS` | 1.5 | 2번째 leg부터 보너스 |

### IP / 평가

| 파라미터 | 값 | 위치 |
|---------|-----|------|
| `n_rollouts` | 100 | evaluate_ip.py |
| `LEG_BONUS_IP` | 1.5 | evaluate_ip.py |
| `DEADHEAD_PENALTY_IP` | 5.0 | evaluate_ip.py |
| LP relaxation threshold | 1e-6 | set_partition.py |
| IP time_limit | 300s | set_partition.py |
| LP slack penalty | 1e6 | set_partition.py |

### 데이터 (train.py 기준)

| 파라미터 | 값 |
|---------|-----|
| limit | 50 |
| n_days | 4 |
| hub_only | True |
| 총 flights | ~200 |

---

## 14. 파일별 역할 요약

```
ASCP-2026/
├── model/
│   ├── film.py
│   │     FiLM 변조 모듈.
│   │     constraint 벡터 → gamma/beta → flight_vecs 변조.
│   │     use_skip=True이면 residual 추가.
│   │
│   ├── encoder.py
│   │     FlightEncoder: Embedding + FiLM_before + Transformer + FiLM_after.
│   │     에피소드당 1회 호출. (N, 128) encoded_flights 반환.
│   │     skip_film, skip_transformer ablation 플래그 지원.
│   │
│   ├── decoder.py
│   │     PointerDecoder: state_vec → W_q → query.
│   │     [encoded_flights | end_duty | end_pairing] → W_k → keys.
│   │     scaled dot-product → mask → softmax → probs (N+2,).
│   │     skip_state_mlp ablation 플래그 지원.
│   │
│   └── __init__.py
│         FlightEncoder, PointerDecoder export.
│
├── RL/
│   ├── constraints.py
│   │     get_delta_constraints(): Delta 제약값 dict 반환.
│   │     FILM_CONSTRAINT_KEYS: FiLM 입력용 7개 키.
│   │
│   ├── state.py
│   │     init_state(flights, constraint): 에피소드 시작 시 초기 상태 dict 생성.
│   │
│   ├── environment.py
│   │     get_max_duty(): FAA Part 117 동적 duty 한도.
│   │     get_mask(): (N+2,) 0/1 mask 반환.
│   │     step(): flight action 실행 → (next_state, reward, done).
│   │     step_end_duty(): END_DUTY 실행 → rest period 진입.
│   │     final_reward(): 에피소드 종료 시 미배정 패널티.
│   │
│   ├── loader.py
│   │     load_flights(): BTS CSV → flight dict 리스트.
│   │     load_flights_multiday(): 하루치 flight를 n_days 복제.
│   │
│   └── cppsc_loader.py
│         load_cppsc_flights(): Tahir CPPSC → ASCP-2026 포맷.
│         get_cppsc_constraints(): CPPSC 전용 제약값.
│
├── experiments/
│   ├── train.py
│   │     메인 학습 스크립트.
│   │     커리큘럼 3단계 (Stage 1→2→3).
│   │     constraint_to_tensor, flights_to_tensors, state_to_vec 헬퍼.
│   │     run_episode(): 에피소드 1회 실행.
│   │     run_curriculum_stage(): REINFORCE 학습 루프.
│   │     train(): 전체 학습 orchestration.
│   │
│   ├── train_step1.py
│   │     구버전 Step 1 (50 flights, max_duty 고정). 단순 버전.
│   │     multi-day, END_DUTY 미지원. ablation 비교용.
│   │
│   ├── train_step2.py
│   │     구버전 Step 2 (300 flights, max_duty 고정). 단순 버전.
│   │
│   ├── train_step1_simple.py
│   │     단순 보상 버전 (dead time만 사용). ablation 비교용.
│   │
│   ├── skip_ablation.py
│   │     Skip connection ablation. 5개 구성 × 600 에피소드 독립 학습.
│   │     결과: experiments/results/skip_ablation_results.json.
│   │
│   ├── test_model.py
│   │     모델 유닛 테스트.
│   │
│   └── analyze_results.py
│         결과 분석 및 시각화.
│
├── set_partition.py
│     solve_lp_relaxation(): LP relaxation → dual variable 추출.
│     column_reduction(): reduced cost 기반 후보 정리.
│     solve_set_partitioning(): Set Partitioning IP 최종 풀기.
│
├── evaluate_ip.py
│     rollout_with_pairings(): RL rollout → pairing 리스트 (legs, cost, dead_time).
│     collect_pool(): n_rollouts × stochastic + 1 × greedy → 중복 제거 pool.
│     evaluate(): 전체 RL+IP 평가 파이프라인.
│
├── eval_vs_baseline.py
│     CPPSC 인스턴스별 RL greedy vs Tahir I²CGp gap 비교.
│     gap = (n_RL - n_baseline) / n_baseline × 100%.
│
├── DESIGN.md  ← 이 파일
│
├── checkpoints/  (git-ignored)
│   ├── stage1_best.pt    Stage 1 최고 체크포인트
│   ├── stage2_best.pt    Stage 2 최고 체크포인트
│   ├── stage3_best.pt    Stage 3 최고 체크포인트
│   └── model_latest.pt   학습 완료 후 최종 저장 (n_airports, constraint_dim 포함)
│
└── RL/data/  (git-ignored)
    └── T_ONTIME_MARKETING.csv   BTS 실제 항공 데이터 (~42MB)
```

---

## 15. 핵심 설계 결정 이유

### 15.1 RL은 Deadhead를 생성할 수 없다

RL 환경에서 `assigned = {flight_id: False}` 딕셔너리는 각 flight가 딱 1번만 선택되도록 보장한다. 한 번 선택된 flight는 mask에서 영구 제외된다. 따라서 RL로 생성된 pairing 집합은 절대로 같은 flight를 두 번 커버하지 않는다.

Deadhead는 **IP Set Covering 제약**에서만 발생한다. `Σ x_j ≥ 1` (≥, covering)이 아닌 `= 1` (partitioning)이지만, slack을 허용하고 여러 pairing이 같은 flight를 포함할 수 있는 구조이므로 IP가 두 pairing을 모두 선택하면 DH가 된다.

### 15.2 ManDays 대신 Dead time을 RL 보상으로 사용

ManDays(pairing 수)는 `END_PAIRING` action에서만 관측된다. 에피소드에 수백 step이 있는데 그 중 단 수십 번만 신호가 온다. 이것이 sparse reward 문제다. RL은 이 sparse 신호에서 어떤 step이 좋은 결과에 기여했는지 역추적하기가 매우 어렵다.

Dead time은 매 flight 선택마다 즉시 관측된다. N개의 flight가 있으면 N번의 dense 신호가 온다. RL gradient가 훨씬 효율적으로 흐른다. Dead time은 ManDays와 강한 상관관계를 가지므로 proxy 목표로 적합하다.

### 15.3 FiLM을 Transformer 전후 양쪽에 적용

Transformer 전에만 FiLM을 적용하면: Transformer가 attention 과정에서 constraint 정보를 희석할 수 있다.
Transformer 후에만 적용하면: attention 계산 자체가 constraint-blind하게 이루어진다.

**Transformer 전**: constraint를 반영한 상태로 attention 계산 → "max_duty가 짧으면 짧은 비행편 위주로 attention"
**Transformer 후**: attention으로 파악된 flight 간 관계를 constraint로 다시 강조 → "이 관계를 이 규정 하에서 어떻게 해석하는가"

두 번 적용하여 constraint가 encoding 전체에 깊이 반영되도록 한다.

### 15.4 greedy rollout을 baseline으로 사용

Value network를 따로 두면 별도의 MLP를 학습해야 하고, value 추정의 bias/variance 트레이드오프를 관리해야 한다. Greedy rollout은 같은 파라미터로 단순히 한 번 더 실행하는 것이므로 구현이 단순하고 현재 policy 수준의 기대값에 근접하다.

단점: greedy baseline도 noisy하고, 때로는 sample보다 항상 좋거나 나쁜 경우가 생긴다. 하지만 실험적으로 충분히 수렴하는 것이 확인되어 채택했다.

### 15.5 hub_only 필터링

hub_only 없이 임의 flight를 사용하면 base-to-base pairing이 불가능한 경우가 많다. 예: ATL→BOS→ORD 경로에서 ATL 복귀가 불가능. RL이 항상 강제 종료(deadhead)에 빠지고, BASE_PENALTY가 끊임없이 발생하여 학습 신호가 오염된다.

hub_only는 모든 flight가 ATL↔X 형태임을 보장한다. ATL→X 후 X에서 쉬고 X→ATL로 복귀하면 완전한 pairing이 된다. RL이 이 구조를 빠르게 학습할 수 있다.

### 15.6 커리큘럼 3단계

**Stage 1 → 2**: 단일 duty에서 multi-day로 점진적 확장.
단일 duty만 학습하면 connection 로직이 먼저 안정적으로 수렴한다. 이 기반 위에 overnight 로직을 추가하면 RL이 훨씬 빠르게 수렴한다. 처음부터 multi-day를 학습하면 connection도 overnight도 동시에 불안정한 상태로 장기간 헤맨다.

**Stage 2 → 3**: constraint 고정에서 랜덤으로 확장.
FiLM이 의미 있는 역할을 하려면 다양한 constraint에 노출되어야 한다. 항상 같은 constraint면 FiLM MLP가 상수에 가까운 gamma/beta를 출력하도록 퇴화할 수 있다. Stage 3에서 `max_duty`를 6~14h로 변동시켜 FiLM이 constraint 변화에 민감하게 반응하도록 강제한다.

### 15.7 무한루프 방지 (`max_steps`)

```python
max_steps = len(flights) * 20
```

이론상 RL이 END_DUTY와 특정 flight를 반복 선택하며 루프에 빠질 수 있다. 실제로 mask가 이를 막아야 하지만, 버그나 엣지케이스에서 무한루프가 발생하면 학습 전체가 중단된다. `len(flights) × 20`을 최대 step 수로 설정하여 이를 방지한다. flight당 20 step은 worst case에도 충분한 여유다.
