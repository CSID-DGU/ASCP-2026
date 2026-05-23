# Action 인덱스 (numpy 음수 인덱싱 활용 — mask[-2], mask[-1])
END_DUTY    = -2
END_PAIRING = -1

# FAA Part 117 — duty 내 leg 수 기준 최대 flight duty period (hours)
# 추후 항공사 확장 시 다른 규정 테이블로 교체 가능
FAA_DUTY_TABLE = {1: 13.0, 2: 13.0, 3: 12.0, 4: 11.5, 5: 11.0, 6: 10.5}

# Delta 기본 제약 — FiLM constraint vector(7,) 및 마스킹 기본값
# TODO: base_airport = 0 하드코딩 → loader.py Global Airport Index 확정 후 ATL 실제 ID로 교체
DEFAULT_CONSTRAINTS = {
    "max_duty":         13.0,   # duty 최대 경과 시간 (h)
    "min_conn":          0.5,   # 최소 연결 시간 (h, 30분)
    "max_conn":          8.0,   # 최대 연결 시간 (h)
    "max_legs":            4,   # duty당 최대 flight 수
    "base_airport":        0,   # base 공항 ID (Global Index 확정 전 임시값)
    "min_rest":          9.5,   # duty 간 최소 휴식 (h)
    "max_duty_periods":    4,   # pairing당 최대 duty 수
    "max_pairing_days":    5,   # pairing 최대 기간 (일)
    "pairing_cost":      5.0,   # END_PAIRING reward 패널티
    "uncovered_penalty": 10.0,  # 미배정 flight 1개당 패널티
}

# 커리큘럼 스테이지별 허용 규칙
# Stage 1: 단일 duty (END_DUTY 불가) — 기본 연결 패턴 학습
# Stage 2: multi-day (END_DUTY 가능) — overnight + base 복귀 학습
# Stage 3: Stage 2 + constraint 랜덤화 — FiLM 적응 학습
CURRICULUM_CONFIG = {
    1: {"allow_end_duty": False},
    2: {"allow_end_duty": True},
    3: {"allow_end_duty": True},
}
