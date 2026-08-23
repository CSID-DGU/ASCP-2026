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
    "min_pairing_legs":    2,   # END_PAIRING 허용 최소 leg 수 (항공사별 override)
    "pairing_cost":      5.0,   # END_PAIRING reward 패널티
    "uncovered_penalty": 10.0,  # 미배정 flight 1개당 패널티
    "base_penalty":     500.0,
                                 # 기존 값이 pairing_cost와 같아서 사실상 무시됨 — 실측
                                 # base-to-base 비율 10.63%(yvaa65ph)로 확인 후 상향)
}

# 항공사 설정 — 항공사 바꿀 때 AIRLINE만 수정하면 됨
AIRLINE = "delta"
AIRLINE_DATA = {
    "delta":   "RL/data/delta_2019_01.csv",
    "alaska":  "RL/data/alaska_2019_01.csv",
    "jetblue": "RL/data/jetblue_2019_01.csv",
    "turkish": "RL/data/timetables",           # .legs 파일 디렉토리 (Airbus narrow body, 2014)
}
MULTI_AIRLINES = ("delta", "alaska", "jetblue")
AIRLINE_WINDOW_DAYS = {
    "delta": 6,
    "alaska": 6,
    "jetblue": 8,
    "turkish": 4,
}
AIRLINE_BASES = {
    "delta":   ["ATL", "DTW", "MSP", "JFK", "LAX", "SEA", "SLC"],  # BTS 2019 기준 주요 허브
    "alaska":  ["SEA", "PDX", "ANC", "LAX", "SFO"],                 # BTS 2019 기준 주요 허브
    "jetblue": ["JFK", "BOS", "FLL", "LAX", "MCO"],                 # BTS 2019 기준 주요 허브
    "turkish": ["HB1", "HB2"],                                       # Istanbul 기반 두 homebase
}

# FiLM 입력 정규화 기준값 — constraint 값을 [0, 1]로 정규화하기 위한 분모
# 각 항목별 실제 상한값 (항공사 중 최대 or 여유값)
# evaluation/evaluate_ip.py must use the same values for checkpoint compatibility.
CONSTRAINT_NORMS = {
    "max_duty":         14.0,   # 항공사 최대(13.0)보다 여유
    "min_conn":          1.0,   # Stage3 범위 상한
    "max_conn":         14.0,   # Stage3 범위 상한(14.0)에 맞춰 확장
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
#
STAGE3_CONSTRAINT_RANGES = {
    "max_duty":         (10.5, 14.0),
    "min_rest":         (9.5,  12.0),
    "min_conn":         (0.5,  1.0),
    "max_conn":         (3.5,  13.0),
    "max_legs":         (3,    10),
    "max_duty_periods": (2,    4),
    "max_pairing_days": (2,    8),
}

# 매 에피소드 이 확률로 STAGE3_CONSTRAINT_RANGES 랜덤 샘플링 대신 현재 선택된 항공사의
# 진짜 constraint(airline_base)를 그대로 주입한다. 나머지는 연속 랜덤 샘플링 유지 —
# FiLM이 항공사 몇 개를 그냥 암기하지 않고 연속 함수를 배우게 하면서도, 평가 시점에
# 실제로 받을 값을 학습 중 직접 보게 하기 위한 절충.
STAGE3_REAL_CONSTRAINT_INJECT_PROB = 0.2

# 레거시 기본값. 학습은 AIRLINE_WINDOW_DAYS에서 항공사별 값을 선택함.
WINDOW_DAYS = 5

# 에피소드 최대 flight 수 — sample_connected_subnet으로 spoke-spoke 간선 포함해 샘플링
# star graph 버그 수정으로 base-first → connected subnet 방식으로 변경
EPISODE_MAX_FLIGHTS = 600

# Phase 2 LP-dual refinement hyperparameters (Algorithm 1).
# PHASE2_POOL_ROLLOUTS: rollouts used to build the candidate pool before each
#   LP solve -- too small makes the LP duals meaningless, too large is slow.
# PHASE2_LP_INTERVAL: H_LP in Algorithm 1 / Eq. (10) -- the restricted master
#   LP is re-solved every H_LP episodes rather than every episode (solving
#   every episode would bottleneck on the CBC solver); mu^cov/nu^exc are
#   cached and reused between refreshes.
PHASE2_POOL_ROLLOUTS = 50    # pool-collection rollouts (stochastic x 50 + greedy x 1)
PHASE2_LP_INTERVAL   = 1     # instance별 local ID가 달라 매 episode LP를 다시 풂
PHASE2_ARTIFICIAL_COST = 1000.0  # 미생성 flight의 coverage dual을 만드는 LP 전용 singleton 비용
PHASE2_N_EPISODES    = 1000  # number of Phase 2 training episodes
PHASE2_DUAL_WARMUP   = 100   # w_dual(e) warm-up length in episodes -- ramps 0 -> full over the first 100 episodes

# Dead-end 시 base 재시작 횟수 상한 -- base 출발 flight가 수백 개인 인스턴스에서 초기(랜덤에
# 가까운) 정책은 매번 다시 막혀서 재시작을 사실상 무한정 반복할 수 있음. 이 횟수를 넘으면
# base에서 시작할 미배정 flight가 남아있어도 episode를 끝낸다.
# no-op 재시작 차단(run_episode의 restart_candidate_id) 이후 실측: coverage가 약
# 64회 재시작 근방에서 포화 -- 30은 병목이라 80으로 상향(재시작마다 실제 후보가 하나씩
# blocked_ids에 쌓여 진전이 보장되므로 cap을 넉넉히 줘도 낭비가 없음).
MAX_ZERO_MASK_RESTARTS = 80

# Reward shaping

MIN_LEGS_FOR_PAIRING = 3      # target minimum legs per pairing
MIN_LEGS_PENALTY = -3.0       # extra EndPairing penalty when total_legs < MIN_LEGS_FOR_PAIRING
PHASE2_DUAL_WEIGHT   = 0.6   # w_dual(e) target weight (Eq. 10) for the net-dual reward term

# Reward shaping
LEG_CONN_BONUS = 1.5          # 연결 flight 추가 시 즉각 보너스 (h 단위, dead_time 패널티와 동일 스케일)
LEG_PER_PAIRING_BONUS = 5.0  # v13: 3.0→5.0 — leg 선택 즉시 보상 강화로 avg_legs 3+ 유도
END_DUTY_BONUS = 6.0          # v15: 3.0→6.0 — overnight 사용률 강제 상승, FiLM 학습 촉진
                               # overnight 10h는 dead_time에서 제외 → legs↑ dead_time↑ 없이 avg_legs 개선 가능
MIN_LEGS_FOR_DUTY_BONUS = 2    # v16: END_DUTY_BONUS가 무위험 고정보상이라 duty를 짧게 끝내고
                               # 반복 수령하는 유인이 생김(dp 늘어나도 duty당 leg 수가 오히려 감소).
                               # 현재 duty의 leg 수가 이 값 미만이면 보너스를
                               # legs/MIN_LEGS_FOR_DUTY_BONUS 비율로 깎아서 "짧은 duty로 END_DUTY
                               # 남발" 유인 제거, legs≥이 값이면 기존과 동일
                       # v7: END_PAIRING 지연 지급 → per-step 즉시 지급으로 구조 변경
                       # 임계값 = LEG_CONN_BONUS(1.5) + LEG_PER_PAIRING_BONUS(3.0) = 4.5h
                       # avg gap 4.27h < 4.5h → 대부분 연결이 reward-positive → avg_legs 3+ 목표
                       # dead_time 상승 감수 (avg gap 연결 허용하므로)
                       # pairing 첫 편 및 rest 직후 편에는 적용 안 함 (연결이 아니므로)

# IP/LP cost 함수 상수 — evaluation/evaluate_ip.py, train.py Phase 2 공용
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
