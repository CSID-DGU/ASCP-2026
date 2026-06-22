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
from constraints import (
    get_delta_constraints, get_alaska_constraints, get_jetblue_constraints,
    FILM_CONSTRAINT_KEYS,
)

_GET_CONSTRAINT = {
    "delta":   get_delta_constraints,
    "alaska":  get_alaska_constraints,
    "jetblue": get_jetblue_constraints,
}
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
    """state dict → decoder 입력 tensor (78,) = current_emb(32) + base_emb(32) + scalars(7) + constraint_vec(7)
    train.py와 동일한 구현 유지 필수 — 불일치 시 로드된 모델이 다른 입력 분포를 받게 됨
    """
    current_emb = encoder.airport_emb(torch.tensor(state["current_airport"]).to(DEVICE))
    base_emb    = encoder.airport_emb(torch.tensor(constraint["base_airport"]).to(DEVICE))

    time_of_day      = (state["current_time"] % 24.0) / 24.0
    day_norm         = (state["current_time"] // 24.0) / config.CONSTRAINT_NORMS["max_pairing_days"]
    duty_period_norm = state.get("duty_period", 0) / config.CONSTRAINT_NORMS["max_duty_periods"]

    if state.get("is_resting", False) or state.get("pairing_start", False):
        duty_elapsed = 0.0
    else:
        duty_elapsed = max(0.0, state["current_time"] - state.get("duty_start_time", state["current_time"]))

    if state.get("is_resting", False) and state.get("rest_end_time") is not None:
        rest_remaining = max(0.0, state["rest_end_time"] - state["current_time"]) / config.CONSTRAINT_NORMS["min_rest"]
    else:
        rest_remaining = 0.0

    return torch.cat([
        current_emb,
        base_emb,
        torch.tensor([
            time_of_day,
            day_norm,
            duty_elapsed / config.CONSTRAINT_NORMS["max_duty"],
            state.get("legs", 0) / config.CONSTRAINT_NORMS["max_legs"],
            duty_period_norm,
            1.0 if state.get("is_resting", False) else 0.0,
            rest_remaining,
        ], dtype=torch.float32).to(DEVICE),
        constraint_to_tensor(constraint),
    ])


LEG_BONUS_IP        = 1.5 
DEADHEAD_PENALTY_IP = 5.0
PAIRING_FIXED_COST  = 1.5  # pairing당 고정 비용 — single-leg pairing을 IP에서 무상으로 두지 않기 위함
                           # 없으면 single-leg cost=0 → IP가 묶을 이유 없음. 있으면 multi-leg 선호 유도


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
        cost = dead_time - rl_bonus + dh_penalty + PAIRING_FIXED_COST
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
        cost    = dead - LEG_BONUS_IP * max(n_legs - 1, 0) + (DEADHEAD_PENALTY_IP if forced else 0.0) + PAIRING_FIXED_COST
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


def evaluate(checkpoint_path, data_path=None,
             n_rollouts=100, window_days=5, offset_days=0,
             bases=("ATL", "DTW", "MSP"),
             lambda_dh=1.0,
             device="cpu",
             airline="delta"):
    """
    Args:
        bases: crew base 공항 코드 리스트 (예: ["ATL", "DTW", "MSP"]).
               airport_map을 통해 내부적으로 정수 ID로 변환된다.
    """
    global DEVICE
    DEVICE = torch.device(device)

    if data_path is None:
        data_path = config.AIRLINE_DATA[airline]

    # checkpoint를 먼저 로드해 vocab 크기 확인 — multi-airline 모델(n_airports=168)은
    # 통합 공항 맵이 필요. 단일 항공사 맵으로 빌드하면 ID 불일치로 임베딩 오류 발생.
    ckpt       = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    n_airports = ckpt.get("n_airports",
                          ckpt["encoder"]["airport_emb.weight"].shape[0])
    max_time   = ckpt.get("max_time", window_days * 24)

    # multi-airline 모델(vocab>145)은 통합 맵; 단일 항공사 모델은 해당 CSV만
    if n_airports > 145:
        map_paths = list(config.AIRLINE_DATA.values())
    else:
        map_paths = data_path
    airport_map = build_airport_map(map_paths)
    base_ids    = bases_to_ids(bases, airport_map)

    flights   = load_flights_rolling(
        data_path, window_days=window_days,
        offset_days=offset_days, airport_map=airport_map,
        base_airport=base_ids[0],
    )
    n_flights = len(flights)

    constraint = _GET_CONSTRAINT[airline](base_ids[0])
    c_tensor   = constraint_to_tensor(constraint)

    encoder = FlightEncoder(n_airports=n_airports, constraint_dim=len(FILM_CONSTRAINT_KEYS)).to(DEVICE)
    decoder = PointerDecoder(constraint_dim=len(FILM_CONSTRAINT_KEYS)).to(DEVICE)
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
    parser.add_argument("checkpoint", nargs="?", default="checkpoints/step2_best.pt",
                        help="체크포인트 파일 또는 디렉토리 경로. "
                             "디렉토리를 넘기면 stage1~phase2_best.pt 4개를 순서대로 평가 후 요약 출력. "
                             "예) checkpoints/di83hxpy")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-rollouts", type=int, default=500)
    parser.add_argument("--n-runs", type=int, default=1,
                        help="evaluate 반복 횟수 (기본: 1). 1보다 크면 평균/std 출력")
    parser.add_argument("--airline", default="delta", choices=["delta", "alaska", "jetblue"],
                        help="평가 항공사 (데이터 + constraint 선택). 기본: delta")
    args = parser.parse_args()

    # checkpoints/ 생략 허용 — "di83hxpy" → "checkpoints/di83hxpy"
    ckpt_arg = args.checkpoint
    if not os.path.exists(ckpt_arg):
        candidate = os.path.join("checkpoints", ckpt_arg)
        if os.path.exists(candidate):
            ckpt_arg = candidate

    import numpy as np

    def _collect_metrics(result):
        sel  = result["selected"]
        fly  = sum(p["fly"]                        for p in sel) if sel else 0.0
        dead = sum(p.get("dead_time", p["cost"])   for p in sel) if sel else 0.0
        legs = sum(p.get("n_legs", len(p["legs"])) for p in sel) if sel else 0
        return {
            "ftc":        dead / fly * 100 if fly > 0 else 0.0,
            "dead":       dead,
            "fly":        fly,
            "avg_legs":   legs / len(sel) if sel else 0.0,
            "n_pairings": result["n_pairings"],
            "dh":         result["deadhead_count"],
            "coverage":   result["coverage"] * 100,
        }

    def _print_stats(label, runs):
        keys = ["ftc", "dead", "fly", "avg_legs", "n_pairings"]
        vals = {k: [r[k] for r in runs] for k in keys}
        n = len(runs)
        print(f"\n{'='*60}")
        print(f"{label}  ({n}회 평균)")
        print(f"{'='*60}")
        print(f"  FTC:        {np.mean(vals['ftc']):.2f}% ± {np.std(vals['ftc']):.2f}%")
        print(f"  dead_time:  {np.mean(vals['dead']):.2f}h ± {np.std(vals['dead']):.2f}h")
        print(f"  n_pairings: {np.mean(vals['n_pairings']):.1f} ± {np.std(vals['n_pairings']):.1f}")
        print(f"  avg_legs:   {np.mean(vals['avg_legs']):.2f} ± {np.std(vals['avg_legs']):.2f}")
        print(f"  개별 FTC:   {[f'{v:.2f}%' for v in vals['ftc']]}")

    if os.path.isdir(ckpt_arg):
        stages     = ["stage1_best", "stage2_best", "stage3_best", "phase2_best"]
        ckpt_dir   = ckpt_arg
        eval_bases = config.AIRLINE_BASES[args.airline]
        summary    = []

        for stage in stages:
            ckpt = os.path.join(ckpt_dir, f"{stage}.pt")
            if not os.path.exists(ckpt):
                print(f"[SKIP] {ckpt} not found")
                continue
            stage_runs = []
            for run_i in range(args.n_runs):
                print(f"\n{'='*60}")
                print(f"Evaluating: {stage}  [run {run_i+1}/{args.n_runs}]  [airline={args.airline}]")
                print(f"{'='*60}")
                result = evaluate(ckpt, device=args.device, n_rollouts=args.n_rollouts,
                                  bases=eval_bases, airline=args.airline)
                stage_runs.append(_collect_metrics(result))

            m = {k: np.mean([r[k] for r in stage_runs]) for k in stage_runs[0]}
            s = {k: np.std ([r[k] for r in stage_runs]) for k in stage_runs[0]}
            summary.append({"stage": stage, "mean": m, "std": s, "runs": stage_runs})

        print(f"\n{'='*60}")
        if args.n_runs > 1:
            print(f"Summary — {ckpt_dir}  ({args.n_runs}회 평균)")
        else:
            print(f"Summary — {ckpt_dir}")
        print(f"{'='*60}")
        if args.n_runs > 1:
            hdr = f"  {'Stage':<15} {'Pairs':>12} {'DH':>4} {'Cover':>7} {'FTC%':>14} {'Dead(h)':>14} {'AvgLegs':>12}"
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for r in summary:
                m, s = r["mean"], r["std"]
                pairs_str = f"{m['n_pairings']:.1f}±{s['n_pairings']:.1f}"
                ftc_str   = f"{m['ftc']:.2f}±{s['ftc']:.2f}%"
                dead_str  = f"{m['dead']:.2f}±{s['dead']:.2f}h"
                legs_str  = f"{m['avg_legs']:.2f}±{s['avg_legs']:.2f}"
                print(f"  {r['stage']:<15} {pairs_str:>12} {m['dh']:>4.0f} "
                      f"{m['coverage']:>6.1f}% {ftc_str:>14} {dead_str:>14} {legs_str:>12}")
        else:
            hdr = f"  {'Stage':<15} {'Pairs':>6} {'DH':>4} {'Cover':>7} {'FTC%':>6} {'Dead(h)':>8} {'Fly(h)':>8} {'AvgLegs':>8}"
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for r in summary:
                m = r["mean"]
                print(f"  {r['stage']:<15} {m['n_pairings']:>6.0f} {m['dh']:>4.0f} "
                      f"{m['coverage']:>6.1f}% {m['ftc']:>5.2f}% "
                      f"{m['dead']:>7.2f}h {m['fly']:>7.2f}h {m['avg_legs']:>8.2f}")
    else:
        eval_bases = config.AIRLINE_BASES[args.airline]
        run_results = []
        for run_i in range(args.n_runs):
            if args.n_runs > 1:
                print(f"\n[Run {run_i+1}/{args.n_runs}]", flush=True)
            result = evaluate(ckpt_arg, device=args.device, n_rollouts=args.n_rollouts,
                              bases=eval_bases, airline=args.airline)
            run_results.append(_collect_metrics(result))
        if args.n_runs > 1:
            _print_stats(ckpt_arg, run_results)
