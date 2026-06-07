# Delta 항공사 운항 제약값 — 2019 기준
# base_airport는 파일에 포함하지 않음 — train.py에서 에피소드마다 실제 ID 주입

DELTA_CONSTRAINTS = {
    "max_duty":        13.0,  # duty 최대 경과 시간 (h) — Delta CBA §12.D.2 / FAR 117 Table B
    "min_conn":        0.65,  # 최소 연결 시간 (h, ~39분) — BTS p5 추정
    "max_conn":         9.0,  # 최대 연결 시간 (h) — BTS p95 추정
    "max_legs":           8,  # duty당 최대 flight 수 — Delta CBA §12.F (8 landings)
    "min_rest":        10.0,  # duty 간 최소 휴식 (h) — FAR 117 §117.25
    "max_duty_periods":   2,  # pairing당 최대 overnight rest 수 (overnight 횟수 기준)
    "max_pairing_days":   5,  # pairing 최대 기간 (일) — BTS p95 추정
    # base_airport 제외 — 에피소드별 주입
}
