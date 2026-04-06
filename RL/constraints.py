def get_delta_constraints():
    return {
        "max_duty": 14.0,
        "min_conn": 0.5,      # 30분
        "max_conn": 4.0,      # 4시간
        "max_legs": 4,
        "base_airport": 0,
    }


# FiLM에 들어가는 constraint key
# base_airport는 카테고리 값이라 제외
FILM_CONSTRAINT_KEYS = ["max_duty", "min_conn", "max_conn", "max_legs"]