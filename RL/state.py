def init_state(flights, constraint):
    first = flights[0]

    return {
        "current_airport":    first["origin"],
        "current_time":       first["dep_time"],
        "duty_time":          0.0,
        "duty_start_time":    first["dep_time"],
        "legs":               0,
        "remaining":          len(flights),
        "pairing_start":      True,
        # multi-day 필드
        "duty_period":        0,
        "pairing_start_time": first["dep_time"],
        "is_resting":         False,
        "rest_end_time":      None,
    }