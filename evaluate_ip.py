"""
RL rollout → pairing 후보 pool 생성 → Set Covering IP로 최적 조합 선택

비용함수: cost = dead_time - LEG_BONUS*(n_legs-1) + DEADHEAD_PENALTY*(강제종료여부)
"""

import sys
import os
import torch
from torch.distributions import Categorical

sys.path.insert(0, "RL")

DEVICE = torch.device("cpu")
from loader import load_flights_rolling, build_airport_map, bases_to_ids
from environment import get_mask, step
from constraints import get_delta_constraints, FILM_CONSTRAINT_KEYS
from model import FlightEncoder, PointerDecoder
from set_partition import solve_set_covering
import config
import wandb


def constraint_to_tensor(constraint):
    return torch.tensor(
        [constraint[k] / config.CONSTRAINT_NORMS[k] for k in FILM_CONSTRAINT_KEYS],
        dtype=torch.float32,
    ).to(DEVICE)


def flights_to_tensors(flights, max_time=120.0):
    origins  = torch.tensor([f["origin"]   for f in flights])
    dests    = torch.tensor([f["dest"]     for f in flights])
    dep_raw  = torch.tensor([f["dep_time"] for f in flights], dtype=torch.float32)
    arr_raw  = torch.tensor([f["arr_time"] for f in flights], dtype=torch.float32)
    dep_norm = dep_raw / max_time
    arr_norm = arr_raw / max_time
    fly_norm = (arr_raw - dep_raw) / max_time
    return origins, dests, dep_norm, arr_norm, fly_norm


def state_to_vec(state, encoder, constraint):
    """state dict → decoder 입력 tensor (71,) = current_emb(32) + base_emb(32) + scalars(7)"""
    current_emb = encoder.airport_emb(torch.tensor(state["current_airport"]).to(DEVICE))
    base_emb    = encoder.airport_emb(torch.tensor(constraint["base_airport"]).to(DEVICE))

    max_pairing_days = constraint.get("max_pairing_days", 5)
    time_of_day      = (state["current_time"] % 24.0) / 24.0
    day_norm         = (state["current_time"] // 24.0) / max(max_pairing_days, 1)
    duty_period_norm = state.get("duty_period", 0) / max(constraint.get("max_duty_periods", 4), 1)

    if state.get("is_resting", False) or state.get("pairing_start", False):
        duty_elapsed = 0.0
    else:
        duty_elapsed = max(0.0, state["current_time"] - state.get("duty_start_time", state["current_time"]))

    min_rest = constraint.get("min_rest", config.DEFAULT_CONSTRAINTS["min_rest"])
    if state.get("is_resting", False) and state.get("rest_end_time") is not None:
        rest_remaining = max(0.0, state["rest_end_time"] - state["current_time"]) / min_rest
    else:
        rest_remaining = 0.0

    return torch.cat([
        current_emb,
        base_emb,
        torch.tensor([
            time_of_day,
            day_norm,
            duty_elapsed                     / constraint["max_duty"],
            state.get("legs", 0)             / constraint["max_legs"],
            duty_period_norm,
            1.0 if state.get("is_resting", False) else 0.0,
            rest_remaining,
        ], dtype=torch.float32).to(DEVICE)
    ])


LEG_BONUS_IP        = 1.5
DEADHEAD_PENALTY_IP = 5.0


def rollout_with_pairings(flights, constraint, encoder, decoder, encoded, greedy=False):
    """
    RL rollout 1번 실행.
    각 pairing의 legs, fly, elapsed, cost를 반환.
    """
    assigned = {f["id"]: False for f in flights}

    pairings = []

    current_legs     = []
    pairing_dep      = None
    pairing_fly      = 0.0
    pairing_last_arr = 0.0
    pairing_rest     = 0.0

    def flush_pairing(is_forced=False):
        if len(current_legs) < 1 or pairing_dep is None:
            return
        elapsed   = pairing_last_arr - pairing_dep
        fly       = pairing_fly
        n_legs    = len(current_legs)
        dead_time = max(elapsed - fly - pairing_rest, 0.0)
        rl_bonus  = LEG_BONUS_IP * max(n_legs - 1, 0)
        dh_penalty = DEADHEAD_PENALTY_IP if is_forced else 0.0
        cost = dead_time - rl_bonus + dh_penalty
        pairings.append({
            "legs":        list(current_legs),
            "fly":         fly,
            "elapsed":     elapsed,
            "dead_time":   dead_time,
            "cost":        cost,
            "is_deadhead": is_forced,
            "n_legs":      n_legs,
        })

    def start_new_pairing(f):
        nonlocal pairing_dep, pairing_fly, pairing_last_arr, pairing_rest
        current_legs.clear()
        current_legs.append(f["id"])
        pairing_dep      = f["dep_time"]
        pairing_fly      = f["arr_time"] - f["dep_time"]
        pairing_last_arr = f["arr_time"]
        pairing_rest     = 0.0

    unassigned = [f for f in flights if not assigned[f["id"]]]
    if not unassigned:
        return pairings

    episode_base = constraint.get("base_airport", 0)
    base_flights = [f for f in unassigned if f["origin"] == episode_base]
    first = sorted(base_flights or unassigned, key=lambda f: f["dep_time"])[0]
    assigned[first["id"]] = True
    start_new_pairing(first)
    state = {
        "current_airport":    first["dest"],
        "current_time":       first["arr_time"],
        "duty_time":          first["arr_time"] - first["dep_time"],
        "duty_start_time":    first["dep_time"],
        "legs":               1,
        "remaining":          sum(1 for v in assigned.values() if not v),
        "pairing_start":      False,
        "duty_period":        0,
        "pairing_start_time": first["dep_time"],
        "is_resting":         False,
        "rest_end_time":      None,
        "base_airport":       episode_base,
    }

    while True:
        mask_list = get_mask(state, flights, assigned, constraint)
        mask      = torch.tensor(mask_list, dtype=torch.float32).to(DEVICE)

        if sum(mask_list[:-2]) == 0 and mask_list[-2] == 0 and mask_list[-1] == 0:
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if not unassigned:
                flush_pairing(is_forced=False)
                break
            flush_pairing(is_forced=True)
            base_flights = [f for f in unassigned if f["origin"] == episode_base]
            nxt = sorted(base_flights or unassigned, key=lambda f: f["dep_time"])[0]
            assigned[nxt["id"]] = True
            start_new_pairing(nxt)
            state = {
                "current_airport":    nxt["dest"],
                "current_time":       nxt["arr_time"],
                "duty_time":          nxt["arr_time"] - nxt["dep_time"],
                "duty_start_time":    nxt["dep_time"],
                "legs":               1,
                "remaining":          sum(1 for v in assigned.values() if not v),
                "pairing_start":      False,
                "duty_period":        0,
                "pairing_start_time": nxt["dep_time"],
                "is_resting":         False,
                "rest_end_time":      None,
                "base_airport":       episode_base,
            }
            continue

        state_vec = state_to_vec(state, encoder, constraint)
        probs     = decoder(encoded, state_vec, mask)

        if greedy:
            action = probs.argmax().item()
        else:
            action = Categorical(probs).sample().item()

        if action == len(flights):
            pairing_rest += constraint.get("min_rest", 9.5)
            state, _, _ = step(state, action, flights, assigned, constraint)
            continue

        if action == len(flights) + 1:
            flush_pairing(is_forced=False)
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if not unassigned:
                break
            base_flights = [f for f in unassigned if f["origin"] == episode_base]
            nxt = sorted(base_flights or unassigned, key=lambda f: f["dep_time"])[0]
            assigned[nxt["id"]] = True
            start_new_pairing(nxt)
            state = {
                "current_airport":    nxt["dest"],
                "current_time":       nxt["arr_time"],
                "duty_time":          nxt["arr_time"] - nxt["dep_time"],
                "duty_start_time":    nxt["dep_time"],
                "legs":               1,
                "remaining":          sum(1 for v in assigned.values() if not v),
                "pairing_start":      False,
                "duty_period":        0,
                "pairing_start_time": nxt["dep_time"],
                "is_resting":         False,
                "rest_end_time":      None,
                "base_airport":       episode_base,
            }
            continue

        f = flights[action]
        current_legs.append(f["id"])
        pairing_fly      += f["arr_time"] - f["dep_time"]
        pairing_last_arr  = f["arr_time"]

        state, _, done = step(state, action, flights, assigned, constraint)
        if done:
            flush_pairing(is_forced=False)
            break

    return pairings


def rollout_batch(flights, constraint, encoder, decoder, encoded, B=50, greedy=False):
    """B개 rollout을 매 step 배치 decoder call로 동시 실행."""
    n_flights    = len(flights)
    episode_base = constraint.get("base_airport", 0)

    assigned  = [{f["id"]: False for f in flights} for _ in range(B)]
    states    = [None] * B
    cur_legs  = [[] for _ in range(B)]
    pair_dep  = [None] * B
    pair_fly  = [0.0] * B
    pair_arr  = [0.0] * B
    pair_rest = [0.0] * B
    pairings  = [[] for _ in range(B)]
    done      = [False] * B

    def flush_env(i, forced=False):
        if not cur_legs[i] or pair_dep[i] is None:
            return
        elapsed = pair_arr[i] - pair_dep[i]
        fly     = pair_fly[i]
        n_legs  = len(cur_legs[i])
        dead    = max(elapsed - fly - pair_rest[i], 0.0)
        cost    = dead - LEG_BONUS_IP * max(n_legs - 1, 0) + (DEADHEAD_PENALTY_IP if forced else 0.0)
        pairings[i].append({"legs": list(cur_legs[i]), "fly": fly, "elapsed": elapsed,
                             "dead_time": dead, "cost": cost, "is_deadhead": forced, "n_legs": n_legs})

    def start_env(i, f):
        assigned[i][f["id"]] = True
        cur_legs[i]  = [f["id"]]
        pair_dep[i]  = f["dep_time"]
        pair_fly[i]  = f["arr_time"] - f["dep_time"]
        pair_arr[i]  = f["arr_time"]
        pair_rest[i] = 0.0
        states[i] = {
            "current_airport":    f["dest"],
            "current_time":       f["arr_time"],
            "duty_time":          f["arr_time"] - f["dep_time"],
            "duty_start_time":    f["dep_time"],
            "legs":               1,
            "remaining":          sum(1 for v in assigned[i].values() if not v),
            "pairing_start":      False,
            "duty_period":        0,
            "pairing_start_time": f["dep_time"],
            "is_resting":         False,
            "rest_end_time":      None,
            "base_airport":       episode_base,
        }

    base_fs = [f for f in flights if f["origin"] == episode_base]
    first   = sorted(base_fs or flights, key=lambda f: f["dep_time"])[0]
    for i in range(B):
        start_env(i, first)

    for _ in range(n_flights * 6):
        active = [i for i in range(B) if not done[i]]
        if not active:
            break

        normal, zero_mask = [], []
        for i in active:
            ml = get_mask(states[i], flights, assigned[i], constraint)
            if sum(ml[:-2]) == 0 and ml[-2] == 0 and ml[-1] == 0:
                zero_mask.append(i)
            else:
                normal.append((i, ml))

        for i in zero_mask:
            unassigned = [f for f in flights if not assigned[i][f["id"]]]
            if not unassigned:
                flush_env(i)
                done[i] = True
                continue
            flush_env(i, forced=True)
            bf = [f for f in unassigned if f["origin"] == episode_base]
            start_env(i, sorted(bf or unassigned, key=lambda f: f["dep_time"])[0])

        if not normal:
            continue

        idxs   = [i for i, _ in normal]
        masks_t = torch.stack([
            torch.tensor(ml, dtype=torch.float32) for _, ml in normal
        ]).to(DEVICE)                                              # (|normal|, N+2)
        svecs_t = torch.stack([
            state_to_vec(states[i], encoder, constraint) for i in idxs
        ]).to(DEVICE)                                              # (|normal|, state_dim)

        probs = decoder(encoded, svecs_t, masks_t)                 # (|normal|, N+2)
        if greedy:
            actions = probs.argmax(dim=-1).cpu().tolist()
        else:
            actions = Categorical(probs).sample().cpu().tolist()

        for action, i in zip(actions, idxs):
            if action == n_flights:            # END_DUTY
                pair_rest[i] += constraint.get("min_rest", 9.5)
                states[i], _, _ = step(states[i], action, flights, assigned[i], constraint)

            elif action == n_flights + 1:      # END_PAIRING
                flush_env(i)
                unassigned = [f for f in flights if not assigned[i][f["id"]]]
                if not unassigned:
                    done[i] = True
                    continue
                bf = [f for f in unassigned if f["origin"] == episode_base]
                start_env(i, sorted(bf or unassigned, key=lambda f: f["dep_time"])[0])

            else:                              # flight 선택
                f = flights[action]
                cur_legs[i].append(f["id"])
                pair_fly[i] += f["arr_time"] - f["dep_time"]
                pair_arr[i]  = f["arr_time"]
                states[i], _, done_flag = step(states[i], action, flights, assigned[i], constraint)
                if done_flag:
                    flush_env(i)
                    done[i] = True

    return pairings


def collect_pool(flights, constraint, encoder, decoder, encoded, n_rollouts=100):
    """단일 base n_rollouts번 rollout → 중복 제거한 pairing pool 반환."""
    pool = {}
    for p in [p for ps in rollout_batch(flights, constraint, encoder, decoder, encoded, B=n_rollouts)
              for p in ps]:
        key = tuple(sorted(p["legs"]))
        if key not in pool or p["cost"] < pool[key]["cost"]:
            pool[key] = p
    for p in rollout_batch(flights, constraint, encoder, decoder, encoded, B=1, greedy=True)[0]:
        key = tuple(sorted(p["legs"]))
        if key not in pool or p["cost"] < pool[key]["cost"]:
            pool[key] = p
    return list(pool.values())


def collect_pool_multibase(flights, constraint, encoder, decoder, encoded,
                           bases, n_rollouts_per_base=50):
    """각 base에서 n_rollouts_per_base번 배치 rollout → 통합 pool 반환."""
    pool = {}
    for b_idx, base in enumerate(bases):
        c_b = {**constraint, "base_airport": base}
        print(f"  [{b_idx+1}/{len(bases)}] base={base}: stochastic {n_rollouts_per_base}개...", flush=True)
        for p in [p for ps in rollout_batch(flights, c_b, encoder, decoder, encoded, B=n_rollouts_per_base)
                  for p in ps]:
            key = tuple(sorted(p["legs"]))
            if key not in pool or p["cost"] < pool[key]["cost"]:
                pool[key] = p
        print(f"  [{b_idx+1}/{len(bases)}] base={base}: greedy 1개...", flush=True)
        for p in rollout_batch(flights, c_b, encoder, decoder, encoded, B=1, greedy=True)[0]:
            key = tuple(sorted(p["legs"]))
            if key not in pool or p["cost"] < pool[key]["cost"]:
                pool[key] = p
        print(f"  → pool 누계: {len(pool)}개", flush=True)
    return list(pool.values())


def evaluate(checkpoint_path, data_path="RL/data/T_ONTIME_MARKETING.csv",
             n_rollouts=100, window_days=5, offset_days=0,
             bases=("ATL", "DTW", "MSP"),
             lambda_dh=1.0,
             device="cpu",
             n_max=None):
    """
    Args:
        bases: crew base 공항 코드 리스트 (예: ["ATL", "DTW", "MSP"]).
               airport_map을 통해 내부적으로 정수 ID로 변환된다.
        n_max: 에피소드 최대 flight 수 (None이면 config.EPISODE_MAX_FLIGHTS 사용).
               미지정 시 window 전체(~10만 편)가 로드되어 사실상 무한 실행됨.
    """
    global DEVICE
    DEVICE = torch.device(device)

    if n_max is None:
        n_max = config.EPISODE_MAX_FLIGHTS

    airport_map = build_airport_map(data_path)
    base_ids    = bases_to_ids(bases, airport_map)

    flights   = load_flights_rolling(
        data_path, window_days=window_days,
        offset_days=offset_days, airport_map=airport_map,
        base_airport=base_ids[0], n_max=n_max,
    )
    n_flights = len(flights)

    # checkpoint에서 n_airports, max_time 로드
    ckpt       = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    # stage 체크포인트는 n_airports 없음 → encoder embedding에서 유추
    n_airports = ckpt.get("n_airports",
                          ckpt["encoder"]["airport_emb.weight"].shape[0])
    max_time   = ckpt.get("max_time", window_days * 24)

    constraint = get_delta_constraints(base_ids[0])
    c_tensor   = constraint_to_tensor(constraint)

    encoder = FlightEncoder(n_airports=n_airports, constraint_dim=len(FILM_CONSTRAINT_KEYS)).to(DEVICE)
    decoder = PointerDecoder().to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    origins, dests, dep_norm, arr_norm, fly_norm = [
        t.to(DEVICE) for t in flights_to_tensors(flights, max_time)
    ]

    ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    wandb.init(
        project="ASCP-2026",
        name=f"eval_{ckpt_name}",
        config={
            "checkpoint":   checkpoint_path,
            "n_rollouts":   n_rollouts,
            "window_days":  window_days,
            "offset_days":  offset_days,
            "bases":        list(bases),
            "lambda_dh":    lambda_dh,
            "n_flights":    n_flights,
            "device":       device,
        },
    )

    with torch.no_grad():
        encoded = encoder(origins, dests, dep_norm, arr_norm, fly_norm, c_tensor)

        print(f"bases: {list(bases)} → IDs: {base_ids}, rollout {n_rollouts}개/base (배치)", flush=True)
        pool = collect_pool_multibase(
            flights, constraint, encoder, decoder, encoded,
            base_ids, n_rollouts_per_base=n_rollouts,
        )
        print(f"pool 크기: {len(pool)}개 후보", flush=True)

        print("IP 풀기 (Set Covering)...", flush=True)
        result = solve_set_covering(pool, n_flights=n_flights, lambda_dh=lambda_dh)

    fly_total  = sum(p["fly"]                       for p in result["selected"]) if result["selected"] else 0.0
    dead_total = sum(p.get("dead_time", p["cost"])  for p in result["selected"]) if result["selected"] else 0.0
    legs_total = sum(p.get("n_legs", len(p["legs"])) for p in result["selected"]) if result["selected"] else 0
    avg_legs   = legs_total / len(result["selected"]) if result["selected"] else 0.0
    ftc        = dead_total / fly_total * 100 if fly_total > 0 else 0.0

    print()
    print("=" * 50)
    print("결과")
    print("=" * 50)
    print(f"  pairing 수:   {result['n_pairings']}")
    print(f"  total cost:   {result['total_cost']:.2f}h")
    print(f"  coverage:     {result['coverage']*100:.1f}%")
    print(f"  uncoverable:  {result['uncoverable']}개 flight")
    print(f"  deadhead:     {result['deadhead_count']}개 flight")
    print(f"  status:       {result['status']}")
    if result["selected"]:
        print(f"  fly time:     {fly_total:.2f}h")
        print(f"  dead time:    {dead_total:.2f}h")
        print(f"  FTC:          {ftc:.2f}%")
        print(f"  avg legs/pairing: {avg_legs:.2f}")

    wandb.log({
        "n_pairings":    result["n_pairings"],
        "total_cost":    result["total_cost"],
        "coverage_pct":  result["coverage"] * 100,
        "n_deadheads":   result["deadhead_count"],
        "n_uncoverable": result["uncoverable"],
        "fly_time":      fly_total,
        "dead_time":     dead_total,
        "FTC":           ftc,
        "avg_legs":      avg_legs,
        "pool_size":     len(pool),
    })
    wandb.finish()

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="checkpoints/step2_best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-rollouts", type=int, default=100)
    args = parser.parse_args()
    evaluate(args.checkpoint, device=args.device, n_rollouts=args.n_rollouts)
