def init_state(flights, constraint):
    first = flights[0]

    return {
        "current_airport": first["origin"],
        "current_time": 0.0,
        "duty_time": 0.0,
        "duty_start_time": 0.0,
        "legs": 0,
        "remaining": len(flights),
    }