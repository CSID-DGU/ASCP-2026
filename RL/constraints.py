def get_delta_constraints():
    return {
        "max_duty": 14.0,
        "min_conn": 0.5,      # 30분
        "max_conn": 4.0,      # 4시간
        "max_legs": 4,
        "base_airport": 0,
    }