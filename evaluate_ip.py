"""
RL rollout → pairing 후보 pool 생성 → Set Partitioning IP로 최적 조합 선택

비용함수: cost = elapsed - fly  (dead time = 낭비 시간)
"""

import sys
import torch
from torch.distributions import Categorical

sys.path.insert(0, "RL")
from loader import load_flights
from environment import get_mask, step
from constraints import get_delta_constraints, FILM_CONSTRAINT_KEYS
from state import init_state

from model import FlightEncoder, PointerDecoder
from set_partition import solve_set_partitioning


def constraint_to_tensor(constraint):
    return torch.tensor([constraint[k] for k in FILM_CONSTRAINT_KEYS], dtype=torch.float32)


def flights_to_tensors(flights):
    origins   = torch.tensor([f["origin"]   for f in flights])
    dests     = torch.tensor([f["dest"]     for f in flights])
    dep_times = torch.tensor([f["dep_time"] for f in flights], dtype=torch.float32)
    arr_times = torch.tensor([f["arr_time"] for f in flights], dtype=torch.float32)
    return origins, dests, dep_times, arr_times


def state_to_vec(state, encoder, constraint):
    airport_emb = encoder.airport_emb(torch.tensor(state["current_airport"]))
    return torch.cat([
        airport_emb,
        torch.tensor([
            state["current_time"] / 24.0,
            state["duty_time"]    / constraint["max_duty"],
            float(state["legs"])  / constraint["max_legs"],
        ], dtype=torch.float32)
    ])


def rollout_with_pairings(flights, constraint, encoder, decoder, encoded, greedy=False):
    """
    RL rollout 1번 실행.
    각 pairing의 legs(flight id 리스트), fly, elapsed, cost를 반환.
    """
    assigned = {f["id"]: False for f in flights}
    state    = init_state(flights, constraint)

    pairings = []

    # 현재 pairing 추적용
    current_legs     = []
    pairing_dep      = None
    pairing_fly      = 0.0
    pairing_last_arr = 0.0

    def flush_pairing():
        """현재 pairing 완성 → pairings에 추가"""
        if len(current_legs) < 1 or pairing_dep is None:
            return
        elapsed  = pairing_last_arr - pairing_dep
        fly      = pairing_fly
        cost     = elapsed - fly   # dead time
        pairings.append({
            "legs":    list(current_legs),
            "fly":     fly,
            "elapsed": elapsed,
            "cost":    max(cost, 0.0),
        })

    def start_new_pairing(f):
        """새 pairing 시작"""
        nonlocal pairing_dep, pairing_fly, pairing_last_arr
        current_legs.clear()
        current_legs.append(f["id"])
        pairing_dep      = f["dep_time"]
        pairing_fly      = f["arr_time"] - f["dep_time"]
        pairing_last_arr = f["arr_time"]

    # 첫 flight 강제 시작
    unassigned = [f for f in flights if not assigned[f["id"]]]
    if not unassigned:
        return pairings

    first = sorted(unassigned, key=lambda x: x["dep_time"])[0]
    assigned[first["id"]] = True
    start_new_pairing(first)
    state = {
        "current_airport":  first["dest"],
        "current_time":     first["arr_time"],
        "duty_time":        first["arr_time"] - first["dep_time"],
        "duty_start_time":  first["dep_time"],
        "legs":             1,
        "remaining":        sum(1 for v in assigned.values() if not v),
    }

    while True:
        mask_list = get_mask(state, flights, assigned, constraint)
        mask      = torch.tensor(mask_list, dtype=torch.float32)

        # 갈 수 있는 flight 없음 → pairing 강제 종료
        if sum(mask_list[:-1]) == 0:
            flush_pairing()
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if not unassigned:
                break
            nxt = sorted(unassigned, key=lambda x: x["dep_time"])[0]
            assigned[nxt["id"]] = True
            start_new_pairing(nxt)
            state = {
                "current_airport":  nxt["dest"],
                "current_time":     nxt["arr_time"],
                "duty_time":        nxt["arr_time"] - nxt["dep_time"],
                "duty_start_time":  nxt["dep_time"],
                "legs":             1,
                "remaining":        sum(1 for v in assigned.values() if not v),
            }
            continue

        state_vec = state_to_vec(state, encoder, constraint)
        probs     = decoder(encoded, state_vec, mask)

        if greedy:
            action = probs.argmax().item()
        else:
            action = Categorical(probs).sample().item()

        # END 선택 → pairing 종료
        if action == len(flights):
            flush_pairing()
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if not unassigned:
                break
            nxt = sorted(unassigned, key=lambda x: x["dep_time"])[0]
            assigned[nxt["id"]] = True
            start_new_pairing(nxt)
            state = {
                "current_airport":  nxt["dest"],
                "current_time":     nxt["arr_time"],
                "duty_time":        nxt["arr_time"] - nxt["dep_time"],
                "duty_start_time":  nxt["dep_time"],
                "legs":             1,
                "remaining":        sum(1 for v in assigned.values() if not v),
            }
            continue

        # flight 선택
        f = flights[action]
        current_legs.append(f["id"])
        pairing_fly      += f["arr_time"] - f["dep_time"]
        pairing_last_arr  = f["arr_time"]

        state, _, done = step(state, action, flights, assigned, constraint)
        if done:
            flush_pairing()
            break

    return pairings


def collect_pool(flights, constraint, encoder, decoder, encoded, n_rollouts=100):
    """
    n_rollouts번 rollout → 중복 제거한 pairing pool 반환.
    legs tuple을 key로 중복 제거.
    """
    pool = {}
    for _ in range(n_rollouts):
        pairings = rollout_with_pairings(
            flights, constraint, encoder, decoder, encoded, greedy=False
        )
        for p in pairings:
            key = tuple(sorted(p["legs"]))
            if key not in pool:
                pool[key] = p

    # greedy rollout도 1번 추가 (확실히 feasible한 후보 포함)
    greedy_pairings = rollout_with_pairings(
        flights, constraint, encoder, decoder, encoded, greedy=True
    )
    for p in greedy_pairings:
        key = tuple(sorted(p["legs"]))
        if key not in pool:
            pool[key] = p

    return list(pool.values())


def evaluate(checkpoint_path, data_path="RL/data/T_ONTIME_MARKETING.csv",
             n_rollouts=100, max_duty=10.0):

    flights    = load_flights(data_path, limit=200, hub_only=True)
    n_flights  = len(flights)
    n_airports = max(max(f["origin"], f["dest"]) for f in flights) + 1

    constraint = get_delta_constraints()
    constraint["max_duty"] = max_duty
    c_tensor   = constraint_to_tensor(constraint)

    encoder = FlightEncoder(n_airports=n_airports, constraint_dim=len(FILM_CONSTRAINT_KEYS))
    decoder = PointerDecoder()

    ckpt = torch.load(checkpoint_path, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    origins, dests, dep_times, arr_times = flights_to_tensors(flights)

    with torch.no_grad():
        encoded = encoder(origins, dests, dep_times, arr_times, c_tensor)

        print(f"rollout {n_rollouts}번으로 pairing pool 생성 중")
        pool = collect_pool(flights, constraint, encoder, decoder, encoded, n_rollouts)
        print(f"pool 크기: {len(pool)}개 후보")

        print("IP 풀기...")
        result = solve_set_partitioning(pool, n_flights=n_flights)

    print()
    print("=" * 50)
    print("결과")
    print("=" * 50)
    print(f"  pairing 수:   {result['n_pairings']}")
    print(f"  total cost:   {result['total_cost']:.2f}h  (dead time 합계)")
    print(f"  coverage:     {result['coverage']*100:.1f}%")
    print(f"  uncoverable:  {result['uncoverable']}개 flight")
    print(f"  status:       {result['status']}")

    if result["selected"]:
        fly_total  = sum(p["fly"]  for p in result["selected"])
        dead_total = sum(p["cost"] for p in result["selected"])
        print(f"  fly time:     {fly_total:.2f}h")
        print(f"  dead time:    {dead_total:.2f}h")
        print(f"  FTC:          {dead_total / fly_total * 100:.2f}%")

    return result


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/step2_best.pt"
    evaluate(ckpt)
