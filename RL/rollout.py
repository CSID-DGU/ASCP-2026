# rollout.py -- run RL rollouts and collect pairing structs (shared by
# evaluate_ip.py and train.py's Phase 2)
#
# rollout_with_pairings: run one rollout -> return a list of pairings
# rollout_batch: run B rollouts in a batch
# collect_pool: collect a rollout pool from a single base
# collect_pool_multibase: collect a rollout pool from multiple bases

import random

import torch
from torch.distributions import Categorical

import config
import environment as _env_default
from base_reach import build_base_reach, can_reach_base
from turkish.environment_turkish import get_mask as _get_mask_turkish, step as _step_turkish
from utils import state_to_vec, flight_gap_bias

get_mask, step = _env_default.get_mask, _env_default.step


def set_environment(airline):
    """Switch to the get_mask/step implementation for the given airline
    (turkish allows asymmetric HB1/HB2 termination). Rebinds this module's
    get_mask/step globals, so all callers that reference them (e.g.
    collect_pool_full, rollout_subset_global) pick up the change immediately."""
    global get_mask, step
    if airline == "turkish":
        get_mask, step = _get_mask_turkish, _step_turkish
    else:
        get_mask, step = _env_default.get_mask, _env_default.step


def rollout_with_pairings(flights, constraint, encoder, decoder, encoded,
                          greedy=False, device=None):
    """Run one RL rollout and return a list of pairing structs.

    Each pairing: {legs, fly, elapsed, dead_time, cost, is_deadhead, n_legs}
    cost = dead_time - IP_LEG_BONUS*(n_legs-1) + IP_DEADHEAD_PENALTY*(if forced) + IP_PAIRING_FIXED_COST
    """
    dev = device or torch.device("cpu")
    assigned = {f["id"]: False for f in flights}
    flight_by_id = {f["id"]: f for f in flights}

    pairings = []

    # 모든 pairing은 허용 base에서 시작하고 해당 pairing의 base로 복귀해야 함.
    all_bases      = list(constraint.get("base_ids") or [constraint["base_airport"]])
    min_rest         = constraint.get("min_rest", 10.0)
    min_pairing_legs = constraint.get("min_pairing_legs", 2)

    _reach_cache = {}
    if constraint.get("_base_reach") is not None:
        # 호출부가 계산한 현재 base의 reachability를 재사용해 rollout별 중복 계산을 막음.
        _reach_cache[constraint["base_airport"]] = constraint["_base_reach"]

    def constraint_for(base):
        c = {**constraint, "base_airport": base}
        if base not in _reach_cache:
            _reach_cache[base] = build_base_reach(flights, base, c)
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
    pairing_n_duties = 1   # number of duties in the current pairing (increments on each overnight)
    # [diagnostic] Split dead_time into within-duty connection gaps and
    # inter-duty excess wait (actual rest - min_rest) for measurement only.
    # Does not affect the dead_time/cost calculation itself -- purely additive fields.
    pairing_intra_gap    = 0.0
    pairing_inter_excess = 0.0

    def flush_pairing(is_forced=False):
        if len(current_legs) < 1 or pairing_dep is None:
            return
        elapsed   = pairing_last_arr - pairing_dep
        fly       = pairing_fly
        n_legs    = len(current_legs)
        # CPP column은 동일 base 복귀·최소 leg·최대 기간을 모두 만족할 때만 저장함.
        if pairing_start_ap != flight_by_id[current_legs[-1]]["dest"]:
            raise ValueError("base로 복귀하지 않은 pairing은 저장할 수 없습니다.")
        if n_legs < min_pairing_legs:
            raise ValueError("최소 leg 수를 충족하지 않은 pairing은 저장할 수 없습니다.")
        if elapsed / 24.0 > cur_c["max_pairing_days"]:
            raise ValueError("최대 pairing 기간을 초과한 pairing은 저장할 수 없습니다.")
        dead_time = max(elapsed - fly - pairing_rest, 0.0)
        cost = (dead_time
                - config.IP_LEG_BONUS * max(n_legs - 1, 0)
                + (config.IP_DEADHEAD_PENALTY if is_forced else 0.0)
                + config.IP_PAIRING_FIXED_COST)
        # ends_at_base: whether this pairing actually returned to the base it
        # departed from (pairing_start_ap, which can differ from episode_base
        # after base rotation). Comparing against the fixed episode_base
        # would misjudge every pairing after a rotation, so pairing_start_ap
        # must be used instead.
        ends_at_base = True
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
        """salvage_doomed only: build a pairing ending at base from a prefix of leg records (recs)."""
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
                            + config.IP_DEADHEAD_PENALTY
                            + config.IP_PAIRING_FIXED_COST),
            # A prefix cut off by salvage was not intentionally ended by the
            # policy here (it's a fragment truncated after hitting a dead
            # end), so it is treated the same as flush_pairing's
            # is_forced=True -- otherwise the IP could mistake this fragment
            # for a "clean" pairing with no deadhead penalty and unfairly prefer it.
            "is_deadhead": True,
            "n_legs":      n_legs,
            "n_duties":    n_rest + 1,
            "intra_duty_gap":    intra,
            "inter_duty_excess": inter,
            # salvage 결과도 실제 마지막 도착지가 목표 base인지 다시 확인함.
            "ends_at_base":      recs[-1]["dest"] == end_ap,
            "true_start_airport": start_ap,
            "is_truncated":      True,
        })

    def salvage_doomed():
        """Handle a pairing that can no longer return to base -- finalize
        only the longest prefix ending at base as a valid pairing, and
        return the remaining tail legs to unassigned so other pairings can reuse them."""
        k = 0
        for i, r in enumerate(leg_recs):
            elapsed_days = (r["arr"] - leg_recs[0]["dep"]) / 24.0
            if (r["dest"] == episode_base
                    and i + 1 >= min_pairing_legs
                    and elapsed_days <= cur_c["max_pairing_days"]):
                k = i + 1
        if k > 0:
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
        """허용 base 중 복귀 가능한 첫 flight를 선택하며 없으면 rollout을 종료함."""
        unassigned = [f for f in flights if not assigned[f["id"]]]
        if not unassigned:
            return None, None
        startable = [f for f in unassigned if f["id"] not in bad_starters]
        best = None
        for b in [episode_base] + [x for x in all_bases if x != episode_base]:
            c_b = constraint_for(b)
            cands = [f for f in startable if f["origin"] == b and can_reach_base(
                c_b["_base_reach"], f, f["dep_time"], c_b["max_pairing_days"],
                duty_period=0, max_duty_periods=c_b["max_duty_periods"],
            )]
            if not cands:
                continue
            f = min(cands, key=lambda f: f["dep_time"])
            if b == episode_base:
                return b, f
            if best is None or f["dep_time"] < best[1]["dep_time"]:
                best = (b, f)
        if best is not None:
            return best
        return None, None

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

    episode_base = constraint["base_airport"]
    cur_c        = constraint_for(episode_base)
    state        = None
    if not begin_pairing():
        return pairings

    while True:
        mask_list = get_mask(state, flights, assigned, cur_c)
        mask      = torch.tensor(mask_list, dtype=torch.float32).to(dev)

        if sum(mask_list[:-2]) == 0 and mask_list[-2] == 0 and mask_list[-1] == 0:
            # 위치와 무관하게 마지막 합법 base 복귀 prefix만 보존함.
            salvage_doomed()
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

        if action == len(flights):             # EndDuty
            pairing_rest     += min_rest
            pairing_n_duties += 1
            state, _, _ = step(state, action, flights, assigned, cur_c)
            continue

        if action == len(flights) + 1:         # EndPairing
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

        # [diagnostic, experiment A] classify the gap type based on state just before this selection
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
    """CPP legality가 검증된 single rollout을 B회 실행해 동일한 반환 형식을 제공함."""
    # 벡터화보다 correctness를 우선하며, 이후 동일 lifecycle을 보존한 최적화로 교체 가능함.
    return [
        rollout_with_pairings(
            flights, constraint, encoder, decoder, encoded,
            greedy=greedy, device=device,
        )
        for _ in range(B)
    ]


def collect_pool(flights, constraint, encoder, decoder, encoded,
                 n_rollouts=100, device=None):
    """Run n_rollouts batched rollouts from a single base and return a deduplicated pairing pool.

    Pairings that don't return to base (ends_at_base=False) are excluded from
    the pool, since the IP should never be offered a candidate that fails
    the base-return requirement of Omega(c).
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
    """Run n_rollouts_per_base batched rollouts from each base and return the merged pool.

    Pairings that don't return to base are excluded, same as collect_pool.
    """
    pool = {}
    for b_idx, base in enumerate(bases):
        c_b = {**constraint, "base_airport": base}
        print(f"  [{b_idx+1}/{len(bases)}] base={base}: {n_rollouts_per_base} stochastic rollouts...", flush=True)
        for p in [p for ps in rollout_batch(flights, c_b, encoder, decoder, encoded,
                                             B=n_rollouts_per_base, device=device)
                  for p in ps]:
            if not p["ends_at_base"]:
                continue
            key = tuple(sorted(p["legs"]))
            if key not in pool or p["cost"] < pool[key]["cost"]:
                pool[key] = p
        print(f"  [{b_idx+1}/{len(bases)}] base={base}: 1 greedy rollout...", flush=True)
        for p in rollout_batch(flights, c_b, encoder, decoder, encoded,
                                B=1, greedy=True, device=device)[0]:
            if not p["ends_at_base"]:
                continue
            key = tuple(sorted(p["legs"]))
            if key not in pool or p["cost"] < pool[key]["cost"]:
                pool[key] = p
        print(f"  -> cumulative pool: {len(pool)}", flush=True)
    return list(pool.values())
