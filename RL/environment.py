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

        valid = (
            f["origin"] == state["current_airport"] and
            f["dep_time"] >= state["current_time"] and
            state["duty_time"] + flight_time <= constraint["max_duty"]
        )

        mask.append(1 if valid else 0)

    mask.append(1)  # END 항상 허용
    return mask


def step(state, action, flights, assigned, constraint):
    f = flights[action]
    assigned[f["id"]] = True

    flight_time = f["arr_time"] - f["dep_time"]

    next_state = {
        "current_airport": f["dest"],
        "current_time": f["arr_time"],
        "duty_time": state["duty_time"] + flight_time,
        "remaining": state["remaining"] - 1
    }

    reward = 0

    # ---------------------------
    # ✔ 1. 연결 보상 / penalty
    # ---------------------------
    if state["current_airport"] == f["origin"]:
        reward += 1.0
    else:
        reward -= 1.0  # deadhead 개념

    # ---------------------------
    # ✔ 2. waiting time penalty
    # ---------------------------
    if state["current_time"] > 0:
        wait = f["dep_time"] - state["current_time"]
        reward -= 0.05 * abs(wait)

    # ---------------------------
    # ✔ 3. duty time penalty
    # ---------------------------
    if next_state["duty_time"] > constraint["max_duty"]:
        excess = next_state["duty_time"] - constraint["max_duty"]
        reward -= 2.0 * excess

    # ---------------------------
    # ✔ 4. early / late imbalance
    # ---------------------------
    reward -= 0.01 * f["dep_time"]

    return next_state, reward, False


def final_reward(assigned):
    remaining = sum(1 for v in assigned.values() if not v)

    # 덜 사용할수록 좋음
    return -3 * remaining