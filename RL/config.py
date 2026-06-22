# Action 인덱스 (numpy 음수 인덱싱 활용 — mask[-2], mask[-1])
END_DUTY    = -2
END_PAIRING = -1

# FAA Part 117 — duty 내 leg 수 기준 최대 flight duty period (hours)
# 추후 항공사 확장 시 다른 규정 테이블로 교체 가능
FAA_DUTY_TABLE = {1: 13.0, 2: 13.0, 3: 12.0, 4: 11.5, 5: 11.0, 6: 10.5}

# Delta 기본 제약 — FiLM constraint vector(7,) 및 마스킹 기본값
# base_airport = 0은 fallback값 — train.py에서 에피소드마다 실제 ID 주입하므로 수정 불필요
DEFAULT_CONSTRAINTS = {
    "max_duty":         13.0,   # duty 최대 경과 시간 (h) — Delta CBA §12.D.2 / FAR 117 Table B
    "min_conn":         0.65,   # 최소 연결 시간 (h, ~39분) — BTS p5 추정 (Delta 기준)
    "max_conn":          9.0,   # 최대 연결 시간 (h) — BTS p95 추정 (Delta 기준)
    "max_legs":            8,   # duty당 최대 flight 수 — Delta CBA §12.F (8 landings)
    "base_airport":        0,   # base 공항 ID (fallback — train.py에서 에피소드별 주입)
    "min_rest":         10.0,   # duty 간 최소 휴식 (h) — FAR 117 §117.25
    "max_duty_periods":    2,   # pairing당 최대 overnight rest 수 (overnight 횟수 기준)
    "max_pairing_days":    5,   # pairing 최대 기간 (일) — BTS p95 추정
    "pairing_cost":      5.0,   # END_PAIRING reward 패널티
    "uncovered_penalty": 10.0,  # 미배정 flight 1개당 패널티
    "base_penalty":      5.0,   # END_PAIRING 시 base 미복귀 패널티
}

# 항공사 설정 — 항공사 바꿀 때 AIRLINE만 수정하면 됨
AIRLINE = "delta"
AIRLINE_DATA = {
    "delta":   "RL/data/delta_2019_01.csv",    
    "alaska":  "RL/data/alaska_2019_01.csv",
    "jetblue": "RL/data/jetblue_2019_01.csv",
}
AIRLINE_BASES = {
    "delta":   ["ATL", "DTW", "MSP", "JFK", "LAX", "SEA", "SLC"],  # BTS 2019 기준 주요 허브
    "alaska":  ["SEA", "PDX", "ANC", "LAX", "SFO"],                 # BTS 2019 기준 주요 허브
    "jetblue": ["JFK", "BOS", "FLL", "LAX", "MCO"],                 # BTS 2019 기준 주요 허브
}

# FiLM 입력 정규화 기준값 — constraint 값을 [0, 1]로 정규화하기 위한 분모
# 각 항목별 실제 상한값 (항공사 중 최대 or 여유값)
# evaluate_ip.py도 동일한 값을 써야 checkpoint 호환 — 수정 시 혜린 확인 필요
CONSTRAINT_NORMS = {
    "max_duty":         14.0,   # 항공사 최대(13.0)보다 여유
    "min_conn":          1.0,   # Stage3 범위 상한
    "max_conn":         10.0,   # Delta p95(9.0)보다 여유
    "max_legs":         10.0,   # Delta CBA 최대(8)보다 여유
    "min_rest":         12.0,   # FAR 117 기준(10.0)보다 여유
    "max_duty_periods":  4.0,   # JetBlue 최대(2~3)보다 여유
    "max_pairing_days":  8.0,   # JetBlue 최대(6~7)보다 여유
}

# Stage 3 FiLM augmentation constraint 범위
# (min, max) 튜플. train.py sample_constraint()에서 random.uniform/randint(*range)로 사용
#
# 범위 설계 기준:
#   - 실제 항공사 값(12.5~13.0h)만 쓰면 FiLM 입력 변동폭이 3~7%에 불과 → gradient 미미
#   - CONSTRAINT_NORMS 분모 대비 최소 20% 이상 변동폭 확보해야 FiLM MLP가 의미있는 gamma/beta 학습 가능
#   - 검증 범위(12.0~14.0h)가 훈련 범위를 벗어나면 extrapolation → 항상 identity 출력
STAGE3_CONSTRAINT_RANGES = {
    "max_duty":         (10.5, 14.0),   # NORM=14.0 → 0.75~1.0 (25% 변동) — 검증 12.0~14.0 완전 포함
    "min_rest":         (9.5,  12.0),   # NORM=12.0 → 0.79~1.0 (21% 변동) — 고정값 0%에서 확장
    "min_conn":         (0.5,  1.0),    # NORM=1.0  → 0.5~1.0  (50% 변동)
    "max_conn":         (6.0,  10.0),   # NORM=10.0 → 0.6~1.0  (40% 변동)
    "max_legs":         (4,    10),     # NORM=10.0 → 0.4~1.0  (60% 변동)
    "max_duty_periods": (1,    4),      # NORM=4.0  → 0.25~1.0 (75% 변동)
    "max_pairing_days": (3,    7),      # NORM=8.0  → 0.375~0.875 (50% 변동)
}

# 슬라이딩 윈도우 크기 — max_pairing_days + 1 (마지막 날 시작 pairing도 완성 가능하게)
# Stage 2/3, Phase 2 모두 max_pairing_days = WINDOW_DAYS - 1로 설정
WINDOW_DAYS = 5

# 에피소드 최대 flight 수 — base-first sampling으로 이 수를 초과하지 않도록 제한
# base↔X 편 전부 포함 + 나머지 spoke-spoke 랜덤 샘플링
EPISODE_MAX_FLIGHTS = 600

# Phase 2 CG dual feedback 하이퍼파라미터
# pool rollout 수: pool이 너무 작으면 LP가 의미없고, 너무 많으면 느림 → 30번이 균형점
# LP interval: 매 에피소드 LP를 풀면 CBC solver가 병목 → 10 에피소드마다 재풀기
# (dual_vars는 interval 사이에 캐싱되어 재사용됨)
PHASE2_POOL_ROLLOUTS = 50    # pool 수집 rollout 수 (stochastic × 50 + greedy × 1)
PHASE2_LP_INTERVAL   = 5     # LP re-solve 주기 (에피소드) — 간격 줄여 dual_vars 신선도 향상
PHASE2_N_EPISODES    = 1000  # Phase 2 학습 에피소드 수
PHASE2_DUAL_WEIGHT   = 0.6   # LP dual reward 가중치 — v9: bonus=3.0 스케일에 맞게 0.1→0.6 (비율 0.2 유지)

# Reward shaping
LEG_CONN_BONUS = 1.5          # 연결 flight 추가 시 즉각 보너스 (h 단위, dead_time 패널티와 동일 스케일)
LEG_PER_PAIRING_BONUS = 3.0  # flight 선택 시 즉시 지급 (step reward에 반영) — credit assignment 해결
END_DUTY_BONUS = 2.0          # v8: END_DUTY(overnight rest) 시 즉각 보너스 — multi-day pairing 직접 장려
                               # overnight 10h는 dead_time에서 제외 → legs↑ dead_time↑ 없이 avg_legs 개선 가능
                       # v7: END_PAIRING 지연 지급 → per-step 즉시 지급으로 구조 변경
                       # 임계값 = LEG_CONN_BONUS(1.5) + LEG_PER_PAIRING_BONUS(3.0) = 4.5h
                       # avg gap 4.27h < 4.5h → 대부분 연결이 reward-positive → avg_legs 3+ 목표
                       # dead_time 상승 감수 (avg gap 연결 허용하므로)
                       # pairing 첫 편 및 rest 직후 편에는 적용 안 함 (연결이 아니므로)

# IP/LP cost 함수 상수 — evaluate_ip.py, train.py Phase 2 공용
# cost = dead_time - LEG_BONUS_IP*(n_legs-1) + DEADHEAD_PENALTY_IP*(강제종료) + PAIRING_FIXED_COST
IP_LEG_BONUS        = 1.5   # leg 추가될수록 cost 감소 → 효율적 연결 장려
IP_DEADHEAD_PENALTY = 5.0   # 강제 deadhead 발생 시 가산
IP_PAIRING_FIXED_COST = 4.0 # pairing당 고정 비용 — single-leg cost=4.0 → IP가 multi-leg 강하게 선호

# 커리큘럼 스테이지별 허용 규칙
# Stage 1: 단일 duty (END_DUTY 불가) — 기본 연결 패턴 학습
# Stage 2: multi-day (END_DUTY 가능) — overnight + base 복귀 학습
# Stage 3: Stage 2 + constraint 랜덤화 — FiLM 적응 학습
CURRICULUM_CONFIG = {
    1: {"allow_end_duty": False},
    2: {"allow_end_duty": True},
    3: {"allow_end_duty": True},
}
