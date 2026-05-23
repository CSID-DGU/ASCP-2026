# constraints.py — 항공사별 운항 제약 정의
#
# 역할:
#   - get_<airline>_constraints(): 항공사별 constraint dict 반환 → environment.py 마스킹 + FiLM 입력에 사용
#   - FILM_CONSTRAINT_KEYS: constraint dict 중 FiLM에 넣을 7개 키 순서 정의
#
# config.py와의 관계:
#   - config.DEFAULT_CONSTRAINTS: constraint가 없을 때 environment.py가 쓰는 fallback (+ reward 상수 포함)
#   - constraints.py: 항공사별 실제 값의 출처 (train.py가 이걸 읽어서 environment에 넘기도록)
#   - reward 상수(pairing_cost, uncovered_penalty)는 config.py에서만 관리 — 여기에 넣지 않음


# 항공사별 constraint 값은 airline_constraints/ 에서 관리
# 새 항공사 추가 시 airline_constraints/<airline>.py 파일 추가 후 아래에 get 함수 작성
from airline_constraints.delta import DELTA_CONSTRAINTS
# from airline_constraints.ua import TURKISH_CONSTRAINTS  # 이런식으로 추후 추가


def get_delta_constraints():
    """Delta 항공사 constraint dict 반환

    FiLM constraint vector(7,) 입력 및 environment.py 마스킹에 사용.
    base_airport는 FiLM 입력에서 제외 (카테고리형 → FILM_CONSTRAINT_KEYS에 없음).
    """
    return DELTA_CONSTRAINTS


# FiLM 입력 constraint 키 순서 — constraint_to_tensor()가 이 순서대로 tensor 변환 → shape (7,)
# base_airport는 카테고리형이라 제외
FILM_CONSTRAINT_KEYS = [
    "max_duty",         # 0
    "min_conn",         # 1
    "max_conn",         # 2
    "max_legs",         # 3
    "min_rest",         # 4
    "max_duty_periods", # 5
    "max_pairing_days", # 6
]
