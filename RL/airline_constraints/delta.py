# Delta 항공사 운항 제약값
# TODO: base_airport = 0 하드코딩 → loader.py Global Airport Index 확정 후 ATL 실제 ID로 교체
# 현재 per-airline 고정 방식 → multi-airline 확장 시 unified index 전환 여부 혜린과 합의 필요

DELTA_CONSTRAINTS = {
    "max_duty":        13.0,  # duty 최대 경과 시간 (h), FAA Part 117 테이블 최댓값
    "min_conn":         0.5,  # 최소 연결 시간 (h, 30분)
    "max_conn":         8.0,  # 최대 연결 시간 (h)
    "max_legs":           4,  # duty당 최대 flight 수
    "min_rest":         9.5,  # duty 간 최소 휴식 (h), FAA Part 117 기준
    "max_duty_periods":   4,  # pairing당 최대 duty 수
    "max_pairing_days":   5,  # pairing 최대 기간 (일)
    # base_airport 제외 — get_delta_constraints(base_airport) 호출 시 에피소드별 주입
}
