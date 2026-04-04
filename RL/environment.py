END = -1


def init_state(flights):
    first = flights[0]

    return {
        "current_airport": first["origin"],
        "current_time": 0,
        "duty_time": 0,
        "remaining": len(flights)
    }


def get_mask(state, flights, assigned, constraint):
    mask = []

    for f in flights:
        if assigned[f["id"]]:
            mask.append(0)
            continue

        flight_time = f["arr_time"] - f["dep_time"]
        gap = f["dep_time"] - state["current_time"]

        valid = True

        # 1. 공항 연결
        if f["origin"] != state["current_airport"]:
            valid = False

        # 2. connection time
        if gap < constraint["min_conn"] or gap > constraint["max_conn"]:
            valid = False

        # 3. duty 제한
        if state["duty_time"] + flight_time > constraint["max_duty"]:
            valid = False

        # 4. leg 제한
        if state["legs"] + 1 > constraint["max_legs"]:
            valid = False

        mask.append(1 if valid else 0)

    # END action
    mask.append(1)

    return mask

def step(state, action, flights, assigned, constraint):
    f = flights[action]
    assigned[f["id"]] = True

    flight_time = f["arr_time"] - f["dep_time"]

    next_state = {
        "current_airport": f["dest"],
        "current_time": f["arr_time"],
        "duty_time": state["duty_time"] + flight_time,
        "duty_start_time": state["duty_start_time"],
        "legs": state["legs"] + 1,
        "remaining": state["remaining"] - 1,
    }

    # reward = cost 기반으로 단순화
    reward = -1.0

    return next_state, reward, False


def final_reward(assigned):
    remaining = sum(1 for v in assigned.values() if not v)

    # 덜 사용할수록 좋음
    return -5 * remaining