# constraints_turkish.py — Turkish 전용 constraint 정의 (HB1/HB2 비대칭 종료 허용)
#
# RL/constraints.py의 get_turkish_constraints()를 그대로 쓰지 않고 이 파일을 쓰는 이유:
# HB1/HB2는 같은 도시(Istanbul)에 있는 사실상 같은 home이라 pairing이 한쪽에서 출발해
# 다른 쪽에서 끝나도 무패널티인 게 맞다 (RL/data/timetables/ttfields.txt 1번 필드 설명 참고).
# 반면 delta 등은 base가 서로 다른 도시에 있어 이 논리가 적용되면 안 되므로, 원본
# RL/constraints.py는 그대로 두고 turkish에만 base_ids(집합) 기반 로직을 별도로 둔다.
# (log/0703/base.md 참고)

from airline_constraints.turkish import TURKISH_CONSTRAINTS


def get_turkish_constraints(base_airport: int, base_ids=None):
    """Turkish Airlines (THY) constraint dict 반환

    base_airport: 에피소드별 pairing 시작 base 공항 ID (HB1, HB2 중 하나)
    base_ids: HB1/HB2 전체 ID 리스트. 지정하면 pairing 종료/재시작이 base_ids 중
        아무 base나 인정하도록 environment_turkish.py가 사용(HB1↔HB2 비대칭 허용).
        None이면 기존처럼 base_airport 하나만 유효한 base로 취급.
    base_airport/base_ids는 FiLM 입력에서 제외 (카테고리형 → FILM_CONSTRAINT_KEYS에 없음)
    """
    c = {**TURKISH_CONSTRAINTS, "base_airport": base_airport}
    if base_ids is not None:
        c["base_ids"] = list(base_ids)
    return c


# FiLM 입력 constraint 키 순서 — RL/constraints.py의 FILM_CONSTRAINT_KEYS와 동일해야 함
# (checkpoint 호환성 유지 목적 — 여기서 새로 정의하지 않고 원본에서 그대로 가져다 씀)
from constraints import FILM_CONSTRAINT_KEYS  # noqa: E402,F401
