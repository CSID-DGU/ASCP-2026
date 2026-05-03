def get_delta_constraints():
    return {
        # 현재 적용 (5개)
        "max_duty":         14.0,   # duty 최대 경과 시간 (시간)
        "min_conn":          0.5,   # 최소 연결 시간 (30분)
        "max_conn":          4.0,   # 최대 연결 시간 (4시간)
        "max_legs":          4,     # duty당 최대 flights
        "base_airport":      0,     # base 공항 ID (FiLM 제외)

        # multi-day (FAA Part 117 기반)
        "min_rest":          9.5,   # duty 간 최소 휴식 (시간)
        "max_duty_periods":  4,     # pairing당 최대 duty 횟수
        "max_pairing_days":  5,     # pairing 최대 기간 (일)
    }


# FiLM 입력 constraint key (base_airport는 카테고리이므로 제외)
FILM_CONSTRAINT_KEYS = [
    "max_duty",
    "min_conn",
    "max_conn",
    "max_legs",
    "min_rest",
    "max_duty_periods",
    "max_pairing_days",
]
