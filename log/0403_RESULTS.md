# Curriculum Learning 결과 (2026-04-03)

## 실험 환경

- 데이터: BTS Delta (T_ONTIME_MARKETING.csv)
- 모델: FlightEncoder(Embedding + FiLM + Transformer) + PointerDecoder
- 학습: REINFORCE + greedy rollout baseline
- optimizer: Adam, lr=1e-4
- gradient clipping: max_norm=1.0
- **seed: 42 (고정)** - 재현 가능하도록

---

## Step 1: 50 flights + constraint 고정 (max_duty=10h)

> 목표: 모델 구조가 작동하는지 확인

```
처음 50ep 평균 pairings: 24.7
마지막 50ep 평균 pairings: 23.9
→ ✅ 수렴 확인
```

- greedy pairings가 23~25에서 안정
- 중간에 불안정 구간 있음 (Ep 100, 450에서 spike)
- 1000 에피소드 학습

---

## Step 2: 300 flights + constraint 고정 (max_duty=10h)

> 목표: 스케일업해도 수렴하는지 확인

```
처음 50ep 평균 pairings: 136.2
마지막 50ep 평균 pairings: 133.9
→ ✅ 수렴 확인 (seed=42)
```

- 44개 공항, 300개 flight
- 500 에피소드 학습
- 참고: seed 미고정 시 초기화에 따라 불안정할 수 있음 (294까지 발산한 케이스 있었음)

---

## Step 3: FiLM 검증 — 같은 flights, 다른 constraint

> 목표: FiLM이 constraint를 반영하는지 확인 (박사님 핵심 요청)

```
max_duty=  6h → pairings: 30
max_duty=  8h → pairings: 25
max_duty= 10h → pairings: 25
max_duty= 12h → pairings: 21
max_duty= 14h → pairings: 19
```

→ **✅ FiLM 작동 확인**

- constraint가 strict할수록 pairing 수 증가 (짧은 pairing)
- constraint가 relaxed할수록 pairing 수 감소 (긴 pairing)
- 동일 flights에 대해 constraint만 바꿨을 때 결과가 달라짐

---

## 판단 과정

| 기준              | 결과                    |
| ----------------- | ----------------------- |
| reward 증가 추세  | ✅ Step 1, 2 모두 수렴  |
| constraint별 차이 | ✅ FiLM이 반영하고 있음 |
| → RL 진행 가능    | ✅                      |

---

## Reward 비교 실험 (Step 1 기준, 50 flights, max_duty=10h)

### 복잡 reward

```
reward = 연결보상(+1/-1) + waiting penalty(-0.05*wait)
       + duty penalty(-2*excess) + early/late(-0.01*dep)
       + final(-3*remaining)
```

```
처음 50ep 평균 pairings: 24.7
마지막 50ep 평균 pairings: 23.9
```

### 단순 reward

```
reward = -len(pairings)
```

```
처음 50ep 평균 pairings: 28.9
마지막 50ep 평균 pairings: 24.4
```

### FiLM 검증 비교

| max_duty | 복잡 reward | 단순 reward |
| -------- | ----------- | ----------- |
| 6h       | 30          | 32          |
| 8h       | 25          | 27          |
| 10h      | 25          | 24          |
| 12h      | 21          | 19          |
| 14h      | 19          | 18          |

### 비교 분석

- **수렴**: 둘 다 수렴함
- **FiLM**: 둘 다 constraint별 차이 뚜렷 (FiLM 작동)
- **최종 성능**: 단순 reward가 relaxed constraint(12h, 14h)에서 더 적은 pairings
- **안정성**: 단순 reward가 spike 덜 심함
- **초기 수렴 속도**: 복잡 reward가 더 빠름 (24.7 vs 28.9 시작)

→ 단순 reward로 시작하되, 복잡 reward 요소를 하나씩 추가하며 ablation 가능

---

## 발생한 이슈

1. **학습 불안정**: seed 미고정 시 Step 2에서 발산 (294 pairings). seed=42 고정으로 해결
2. **sample < greedy**: sample rollout이 greedy보다 거의 항상 나쁨 (exploration이 아직 비효율적)
3. **Step 2 수렴 폭 작음**: 136.2 → 133.9 (500ep). 에피소드 수 늘리면 더 줄어들 가능성

---

## 다음 단계

- [ ] 학습 안정화 (spike 원인 분석)
- [ ] reward ablation (단순 → +연결보상 → +waiting → ... 하나씩 추가)
- [ ] constraint 확장 (min_rest, max_days 등)
