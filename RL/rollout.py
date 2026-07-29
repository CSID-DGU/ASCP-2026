# rollout.py — RL rollout → pairing 구조체 수집 (evaluate_ip.py, train.py Phase2 공용)
#
# rollout_with_pairings: 단건 rollout → pairing 리스트 반환
# rollout_batch: B개 rollout 배치 실행
# collect_pool: 단일 base rollout pool 수집
# collect_pool_multibase: 여러 base rollout pool 수집

import random

import torch
from torch.distributions import Categorical

import config
import environment as _env_default
from base_reach import build_base_reach
from turkish.environment_turkish import get_mask as _get_mask_turkish, step as _step_turkish
from utils import state_to_vec, flight_gap_bias, flight_gap_bias_batch

get_mask, step = _env_default.get_mask, _env_default.step


def set_environment(airline):
    """airline에 맞는 get_mask/step 구현으로 전환 (turkish는 HB1/HB2 비대칭 종료 허용).
    collect_pool_full/rollout_subset_global 등 이 모듈의 get_mask/step을
    참조하는 모든 호출부에 즉시 반영됨 (모듈 전역 rebind)."""
    global get_mask, step
    if airline == "turkish":
        get_mask, step = _get_mask_turkish, _step_turkish
    else:
        get_mask, step = _env_default.get_mask, _env_default.step


def rollout_with_pairings(flights, constraint, encoder, decoder, encoded,
                          greedy=False, device=None):
    """RL rollout 1번 실행 → pairing 구조체 리스트 반환.

    각 pairing: {legs, fly, elapsed, dead_time, cost, is_deadhead, n_legs}
    cost = dead_time - IP_LEG_BONUS*(n_legs-1) + IP_DEADHEAD_PENALTY*(강제종료) + IP_PAIRING_FIXED_COST
    """
    dev = device or torch.device("cpu")
    assigned = {f["id"]: False for f in flights}
    flight_by_id = {f["id"]: f for f in flights}

    pairings = []

    # ── base 처리 옵션 ────────────────────────────────────────────────────────
    # base_ids            : base 목록. 현재 base의 미배정 출발편이 소진되면 다른 base로 회전.
    # strict_base_start   : base 출발편이 전부 소진되면 임의 공항에서 새 pairing을 시작하지
    #                       않고 rollout을 종료 (base 없는 pairing 생성 원천 차단).
    # require_base_return : base 복귀를 hard mask로 강제 (environment.get_mask에서 처리).
    all_bases      = list(constraint.get("base_ids") or [constraint.get("base_airport", 0)])
    strict_start   = bool(constraint.get("strict_base_start", False))
    require_return = bool(constraint.get("require_base_return", False))
    min_rest         = constraint.get("min_rest", 10.0)
    min_pairing_legs = constraint.get("min_pairing_legs", 2)

    _reach_cache = {}

    def constraint_for(base):
        c = {**constraint, "base_airport": base}
        if require_return:
            if base not in _reach_cache:
                _reach_cache[base] = build_base_reach(flights, base, constraint)
            c["_base_reach"] = _reach_cache[base]
        return c

    bad_starters = set()

    current_legs     = []
    leg_recs         = []
    pairing_dep      = None
    pairing_start_ap = None
    pairing_fly      = 0.0
    pairing_last_arr = 0.0
    pairing_rest     = 0.0
    pairing_n_duties = 1   # 현재 pairing의 duty 수 (overnight 발생 시 증가)
    # [진단용] dead_time을 duty 내부 연결 gap과 duty 간 초과 대기(실제 휴식 - min_rest)로
    # 분리 계측. dead_time 계산식/cost 자체는 손대지 않음 — 순수 부가 필드.
    pairing_intra_gap    = 0.0
    pairing_inter_excess = 0.0

    def flush_pairing(is_forced=False):
        if len(current_legs) < 1 or pairing_dep is None:
            return
        elapsed   = pairing_last_arr - pairing_dep
        fly       = pairing_fly
        n_legs    = len(current_legs)
        dead_time = max(elapsed - fly - pairing_rest, 0.0)
        cost = (dead_time
                - config.IP_LEG_BONUS * max(n_legs - 1, 0)
                + (config.IP_DEADHEAD_PENALTY if is_forced else 0.0)
                + config.IP_PAIRING_FIXED_COST)
        # ends_at_base: 이 pairing이 실제로 출발한 base(pairing_start_ap, base 회전 시
        # episode_base와 다를 수 있음)로 정확히 복귀했는지. episode_base 고정값과 비교하면
        # 회전 후 pairing을 전부 오판정하므로 반드시 pairing_start_ap을 써야 한다.
        ends_at_base = (pairing_start_ap == flight_by_id[current_legs[-1]]["dest"])
        pairings.append({
            "legs":        list(current_legs),
            "fly":         fly,
            "elapsed":     elapsed,
            "dead_time":   dead_time,
            "cost":        cost,
            "is_deadhead": is_forced,
            "n_legs":      n_legs,
            "n_duties":    pairing_n_duties,
            "intra_duty_gap":    pairing_intra_gap,
            "inter_duty_excess": pairing_inter_excess,
            "ends_at_base":      ends_at_base,
            "true_start_airport": pairing_start_ap,
        })

    def emit_prefix(recs, end_ap, start_ap):
        """salvage_doomed 전용: recs(leg 기록 prefix)로 base에서 끝나는 pairing을 만든다."""
        if len(recs) < 1:
            return
        fly     = sum(r["arr"] - r["dep"] for r in recs)
        elapsed = recs[-1]["arr"] - recs[0]["dep"]
        n_rest  = sum(1 for r in recs[1:] if r["rested"])
        rest    = min_rest * n_rest
        intra = inter = 0.0
        for prev, r in zip(recs, recs[1:]):
            if r["rested"]:
                inter += max(r["dep"] - (prev["arr"] + min_rest), 0.0)
            else:
                intra += r["dep"] - prev["arr"]
        n_legs    = len(recs)
        dead_time = max(elapsed - fly - rest, 0.0)
        pairings.append({
            "legs":        [r["id"] for r in recs],
            "fly":         fly,
            "elapsed":     elapsed,
            "dead_time":   dead_time,
            "cost":        (dead_time
                            - config.IP_LEG_BONUS * max(n_legs - 1, 0)
                            + config.IP_PAIRING_FIXED_COST),
            "is_deadhead": False,
            "n_legs":      n_legs,
            "n_duties":    n_rest + 1,
            "intra_duty_gap":    intra,
            "inter_duty_excess": inter,
            "ends_at_base":      True,
            "true_start_airport": start_ap,
            "is_truncated":      True,
        })

    def salvage_doomed():
        """base 복귀가 불가능해진 pairing 처리 — base로 끝나는 최장 prefix만 유효 pairing으로
        확정하고, 남은 tail leg는 미배정으로 되돌려 다른 pairing이 재사용하게 한다."""
        k = 0
        for i, r in enumerate(leg_recs):
            if r["dest"] == episode_base:
                k = i + 1
        if k >= min_pairing_legs:
            emit_prefix(leg_recs[:k], episode_base, pairing_start_ap)
            tail = leg_recs[k:]
        else:
            tail = list(leg_recs)
            if leg_recs:
                bad_starters.add(leg_recs[0]["id"])
        for r in tail:
            assigned[r["id"]] = False

    def start_new_pairing(f):
        nonlocal pairing_dep, pairing_fly, pairing_last_arr, pairing_rest, pairing_n_duties
        nonlocal pairing_intra_gap, pairing_inter_excess, pairing_start_ap
        current_legs.clear()
        current_legs.append(f["id"])
        leg_recs.clear()
        leg_recs.append({"id": f["id"], "dest": f["dest"],
                         "dep": f["dep_time"], "arr": f["arr_time"], "rested": False})
        pairing_start_ap = f["origin"]
        pairing_dep      = f["dep_time"]
        pairing_fly      = f["arr_time"] - f["dep_time"]
        pairing_last_arr = f["arr_time"]
        pairing_rest     = 0.0
        pairing_n_duties = 1
        pairing_intra_gap    = 0.0
        pairing_inter_excess = 0.0

    def pick_start():
        """다음 pairing의 (base, 첫 flight) 선택. 현재 base에 미배정 출발편이 남아있으면
        그대로, 없으면 다른 base로 회전. 어디에도 없으면 strict_start 여부에 따라 종료
        또는(기존 동작) 임의 공항에서 시작."""
        unassigned = [f for f in flights if not assigned[f["id"]]]
        if not unassigned:
            return None, None
        startable = [f for f in unassigned if f["id"] not in bad_starters]
        best = None
        for b in [episode_base] + [x for x in all_bases if x != episode_base]:
            cands = [f for f in startable if f["origin"] == b]
            if not cands:
                continue
            f = min(cands, key=lambda f: f["dep_time"])
            if b == episode_base:
                return b, f
            if best is None or f["dep_time"] < best[1]["dep_time"]:
                best = (b, f)
        if best is not None:
            return best
        if strict_start:
            return None, None
        if not startable:
            return None, None
        return episode_base, min(startable, key=lambda f: f["dep_time"])

    def begin_pairing():
        nonlocal state, episode_base, cur_c
        base, f = pick_start()
        if f is None:
            return False
        if base != episode_base:
            episode_base = base
            cur_c = constraint_for(base)
        assigned[f["id"]] = True
        start_new_pairing(f)
        state = {
            "current_airport":    f["dest"],
            "current_time":       f["arr_time"],
            "duty_time":          f["arr_time"] - f["dep_time"],
            "duty_start_time":    f["dep_time"],
            "legs":               1,
            "total_legs":         1,
            "remaining":          sum(1 for v in assigned.values() if not v),
            "pairing_start":      False,
            "duty_period":        0,
            "pairing_start_time": f["dep_time"],
            "is_resting":         False,
            "rest_end_time":      None,
            "base_airport":       episode_base,
        }
        return True

    if not any(not v for v in assigned.values()):
        return pairings

    episode_base = constraint.get("base_airport", 0)
    cur_c        = constraint_for(episode_base)
    state        = None
    if not begin_pairing():
        return pairings

    while True:
        mask_list = get_mask(state, flights, assigned, cur_c)
        mask      = torch.tensor(mask_list, dtype=torch.float32).to(dev)

        if sum(mask_list[:-2]) == 0 and mask_list[-2] == 0 and mask_list[-1] == 0:
            if require_return and state["current_airport"] != episode_base:
                salvage_doomed()
            else:
                flush_pairing(is_forced=any(not v for v in assigned.values()))
            if not begin_pairing():
                break
            continue

        _incl_total = decoder.state_mlp[0].weight.shape[1] > 78
        state_vec = state_to_vec(state, encoder, cur_c, device=dev, include_total_legs=_incl_total)
        gap_bias  = flight_gap_bias(state, flights, cur_c, device=dev)
        probs     = decoder(encoded, state_vec, mask, gap_bias=gap_bias)

        if greedy:
            action = probs.argmax().item()
        else:
            action = Categorical(probs).sample().item()

        if action == len(flights):             # END_DUTY
            pairing_rest     += min_rest
            pairing_n_duties += 1
            state, _, _ = step(state, action, flights, assigned, cur_c)
            continue

        if action == len(flights) + 1:         # END_PAIRING
            flush_pairing(is_forced=False)
            if not begin_pairing():
                break
            continue

        f = flights[action]
        current_legs.append(f["id"])
        leg_recs.append({"id": f["id"], "dest": f["dest"],
                         "dep": f["dep_time"], "arr": f["arr_time"],
                         "rested": bool(state.get("is_resting", False))})
        pairing_fly      += f["arr_time"] - f["dep_time"]
        pairing_last_arr  = f["arr_time"]

        # [진단용, 실험 A] 선택 직전 state 기준으로 gap 종류 분리 집계
        if not state.get("pairing_start", False) and not state.get("is_resting", False):
            pairing_intra_gap += f["dep_time"] - state["current_time"]
        elif state.get("is_resting", False):
            rest_end = state.get("rest_end_time", f["dep_time"])
            pairing_inter_excess += max(f["dep_time"] - rest_end, 0.0)

        state, _, done = step(state, action, flights, assigned, cur_c)
        if done:
            if not state.get("pairing_start", False):
                flush_pairing(is_forced=False)
            break

    return pairings


def rollout_batch(flights, constraint, encoder, decoder, encoded, B=50,
                  greedy=False, device=None):
    """B개 rollout을 매 step 배치 decoder call로 동시 실행."""
    dev = device or torch.device("cpu")
    n_flights    = len(flights)
    episode_base = constraint.get("base_airport", 0)
    flight_by_id = {f["id"]: f for f in flights}

    assigned    = [{f["id"]: False for f in flights} for _ in range(B)]
    states      = [None] * B
    cur_legs    = [[] for _ in range(B)]
    pair_dep    = [None] * B
    pair_fly    = [0.0] * B
    pair_arr    = [0.0] * B
    pair_rest   = [0.0] * B
    pair_duties = [1] * B   # 현재 pairing의 duty 수
    pairings    = [[] for _ in range(B)]
    done        = [False] * B

    def flush_env(i, forced=False):
        if not cur_legs[i] or pair_dep[i] is None:
            return
        elapsed = pair_arr[i] - pair_dep[i]
        fly     = pair_fly[i]
        n_legs  = len(cur_legs[i])
        dead    = max(elapsed - fly - pair_rest[i], 0.0)
        cost    = (dead
                   - config.IP_LEG_BONUS * max(n_legs - 1, 0)
                   + (config.IP_DEADHEAD_PENALTY if forced else 0.0)
                   + config.IP_PAIRING_FIXED_COST)
        ends_at_base = (flight_by_id[cur_legs[i][0]]["origin"] == episode_base
                        and flight_by_id[cur_legs[i][-1]]["dest"] == episode_base)
        pairings[i].append({"legs": list(cur_legs[i]), "fly": fly, "elapsed": elapsed,
                             "dead_time": dead, "cost": cost, "is_deadhead": forced,
                             "n_legs": n_legs, "n_duties": pair_duties[i],
                             "ends_at_base": ends_at_base})

    def start_env(i, f):
        assigned[i][f["id"]] = True
        cur_legs[i]    = [f["id"]]
        pair_dep[i]    = f["dep_time"]
        pair_fly[i]    = f["arr_time"] - f["dep_time"]
        pair_arr[i]    = f["arr_time"]
        pair_rest[i]   = 0.0
        pair_duties[i] = 1
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
        ]).to(dev)
        _incl_total = decoder.state_mlp[0].weight.shape[1] > 78
        svecs_t = torch.stack([
            state_to_vec(states[i], encoder, constraint, device=dev, include_total_legs=_incl_total) for i in idxs
        ]).to(dev)
        gap_bias_t = flight_gap_bias_batch([states[i] for i in idxs], flights, constraint, device=dev)

        probs = decoder(encoded, svecs_t, masks_t, gap_bias=gap_bias_t)
        if greedy:
            actions = probs.argmax(dim=-1).cpu().tolist()
        else:
            actions = Categorical(probs).sample().cpu().tolist()

        for action, i in zip(actions, idxs):
            if action == n_flights:            # END_DUTY
                pair_rest[i]   += constraint.get("min_rest", 10.0)
                pair_duties[i] += 1
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


def collect_pool(flights, constraint, encoder, decoder, encoded,
                 n_rollouts=100, device=None):
    """단일 base n_rollouts번 배치 rollout → 중복 제거한 pairing pool 반환.

    base로 복귀 안 한 pairing(ends_at_base=False)은 pool에서 제외(2026-07-28) —
    IP가 base 미복귀 pairing을 후보로 골라버리는 문제 수정.
    """
    pool = {}
    for p in [p for ps in rollout_batch(flights, constraint, encoder, decoder, encoded,
                                         B=n_rollouts, device=device)
              for p in ps]:
        if not p["ends_at_base"]:
            continue
        key = tuple(sorted(p["legs"]))
        if key not in pool or p["cost"] < pool[key]["cost"]:
            pool[key] = p
    for p in rollout_batch(flights, constraint, encoder, decoder, encoded,
                            B=1, greedy=True, device=device)[0]:
        if not p["ends_at_base"]:
            continue
        key = tuple(sorted(p["legs"]))
        if key not in pool or p["cost"] < pool[key]["cost"]:
            pool[key] = p
    return list(pool.values())


def collect_pool_multibase(flights, constraint, encoder, decoder, encoded,
                           bases, n_rollouts_per_base=50, device=None):
    """각 base에서 n_rollouts_per_base번 배치 rollout → 통합 pool 반환.

    base로 복귀 안 한 pairing은 제외(2026-07-28, collect_pool과 동일).
    """
    pool = {}
    for b_idx, base in enumerate(bases):
        c_b = {**constraint, "base_airport": base}
        print(f"  [{b_idx+1}/{len(bases)}] base={base}: stochastic {n_rollouts_per_base}개...", flush=True)
        for p in [p for ps in rollout_batch(flights, c_b, encoder, decoder, encoded,
                                             B=n_rollouts_per_base, device=device)
                  for p in ps]:
            if not p["ends_at_base"]:
                continue
            key = tuple(sorted(p["legs"]))
            if key not in pool or p["cost"] < pool[key]["cost"]:
                pool[key] = p
        print(f"  [{b_idx+1}/{len(bases)}] base={base}: greedy 1개...", flush=True)
        for p in rollout_batch(flights, c_b, encoder, decoder, encoded,
                                B=1, greedy=True, device=device)[0]:
            if not p["ends_at_base"]:
                continue
            key = tuple(sorted(p["legs"]))
            if key not in pool or p["cost"] < pool[key]["cost"]:
                pool[key] = p
        print(f"  → pool 누계: {len(pool)}개", flush=True)
    return list(pool.values())
