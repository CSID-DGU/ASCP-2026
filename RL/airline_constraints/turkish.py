# Turkish Airlines (THY) 운항 제약값 — 2014 timetable 데이터 기준
# EASA FTL Subpart Q / SHGM 규정 기반 추정값
# Airbus narrow body fleet (fleet 3xxx) 전용
# base_airport는 포함하지 않음 — train.py에서 에피소드마다 실제 ID 주입

TURKISH_CONSTRAINTS = {
    "max_duty":        13.0,  # EASA FTL §ORO.FTL.205 최대 FDP (h)
    "min_conn":         0.8, # 최소 연결 시간 (h, ~45분) — Istanbul hub 추정
    "max_conn":         4.0,  # 최대 연결 시간 (h)
    "max_legs":           4,  # EASA FTL 기준 duty당 최대 sectors
    "min_rest":        10.0,  # EASA FTL §ORO.FTL.235 최소 휴식 (h)
    "max_duty_periods":   2,  # pairing당 최대 overnight rest 수
    "max_pairing_days":   3,  # 최대 pairing 기간 (일)
    "min_pairing_legs":   2,  # END_PAIRING 허용 최소 leg 수 — max_conn 좁아 3-leg 달성 어려움
}
