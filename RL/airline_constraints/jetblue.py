# JetBlue Airways 운항 제약값 — 2019 기준 (JetBlue 2018 CBA / FAR 117 적용)
# base_airport는 파일에 포함하지 않음 — train.py에서 에피소드마다 실제 ID 주입

JETBLUE_CONSTRAINTS = {
    "max_duty":        13.0,  # duty 최대 경과 시간 (h) — FAR 117 Table B 기준
    "min_conn":        0.58,  # 최소 연결 시간 (h, ~35분) — BTS p5 추정
    "max_conn":         8.0,  # 최대 연결 시간 (h) — BTS p95 추정
    "max_legs":           7,  # duty당 최대 flight 수 — BTS p95 추정
    "min_rest":        10.0,  # duty 간 최소 휴식 (h) — FAR 117 §117.25
    "max_duty_periods":   3,  # pairing당 최대 overnight rest 수 (overnight 횟수 기준, 상한)
    "max_pairing_days":   7,  # pairing 최대 기간 (일) — BTS p95 추정 (상한)
    "min_pairing_legs":   3,  # END_PAIRING 허용 최소 leg 수 — 2-leg Nash 차단
    # base_airport 제외 — 에피소드별 주입
}
