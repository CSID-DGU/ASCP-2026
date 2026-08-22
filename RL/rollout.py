# rollout.py -- run RL rollouts and collect pairing structs (shared by
# evaluation/evaluate_ip.py and experiments/train.py's Phase 2)
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
from base_reach import build_base_reach, can_reach_any_base
import turkish.environment_turkish as _env_turkish
from utils import (
    state_to_vec, flight_gap_bias, state_to_vec_batch, flight_gap_bias_batch,
)

get_mask, step = _env_default.get_mask, _env_default.step
get_mask_batch, step_batch = _env_default.get_mask_batch, _env_default.step_batch


def set_environment(airline):
    """Switch to the get_mask/step implementation for the given airline
    (Turkish는 HB1/HB2 교차 복귀 허용). Rebinds this module's
    get_mask/step globals, so all callers that reference them (e.g.
    collect_pool_full, rollout_subset_global) pick up the change immediately."""
    global get_mask, step, get_mask_batch, step_batch
    if airline == "turkish":
        get_mask, step = _env_turkish.get_mask, _env_turkish.step
        get_mask_batch, step_batch = _env_turkish.get_mask_batch, _env_turkish.step_batch
    else:
        get_mask, step = _env_default.get_mask, _env_default.step
        get_mask_batch, step_batch = _env_default.get_mask_batch, _env_default.step_batch


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
        return_bases = all_bases if c.get("allow_cross_base_return") else [base]
        for target in return_bases:
            if target not in _reach_cache:
                _reach_cache[target] = build_base_reach(flights, target, c)
        c["_base_reach"] = _reach_cache[base]
        c["_base_reaches"] = {target: _reach_cache[target] for target in return_bases}
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
        # 일반 항공사는 동일 base, Turkish는 HB1/HB2 home-base 집합 복귀를 요구함.
        allowed_returns = set(cur_c.get("base_ids") or [pairing_start_ap]) \
            if cur_c.get("allow_cross_base_return") else {pairing_start_ap}
        if flight_by_id[current_legs[-1]]["dest"] not in allowed_returns:
            raise ValueError("허용 home base로 복귀하지 않은 pairing은 저장할 수 없습니다.")
        if n_legs < min_pairing_legs:
            raise ValueError("최소 leg 수를 충족하지 않은 pairing은 저장할 수 없습니다.")
        if elapsed / 24.0 > cur_c["max_pairing_days"]:
            raise ValueError("최대 pairing 기간을 초과한 pairing은 저장할 수 없습니다.")
        dead_time = max(elapsed - fly - pairing_rest, 0.0)
        # leg 수가 많고 연결이 타이트하면(dead_time이 작은데 IP_LEG_BONUS*(n_legs-1)이
        # 큰 경우) 이 공식이 음수가 될 수 있음 -- evaluation/full_flight_master.py::
        # validate_master_inputs()는 cost>=0을 요구하므로(policy pairing도 결국 그
        # pool에 합쳐짐) 방어적으로 0에서 clamp.
        cost = max((dead_time
                    - config.IP_LEG_BONUS * max(n_legs - 1, 0)
                    + (config.IP_DEADHEAD_PENALTY if is_forced else 0.0)
                    + config.IP_PAIRING_FIXED_COST), 0.0)
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
            "true_end_airport":   flight_by_id[current_legs[-1]]["dest"],
            "source_type":         "policy",
            "duty_break_indices": [
                i for i, rec in enumerate(leg_recs) if i > 0 and rec["rested"]
            ],
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
            # flush_pairing()과 동일한 이유로 clamp (leg 수 많고 연결 타이트하면 음수 가능)
            "cost":        max((dead_time
                                - config.IP_LEG_BONUS * max(n_legs - 1, 0)
                                + config.IP_DEADHEAD_PENALTY
                                + config.IP_PAIRING_FIXED_COST), 0.0),
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
            "source_type":       "salvage",
            "duty_break_indices": [
                i for i, rec in enumerate(recs) if i > 0 and rec["rested"]
            ],
        })

    def salvage_doomed():
        """Handle a pairing that can no longer return to base -- finalize
        only the longest prefix ending at base as a valid pairing, and
        return the remaining tail legs to unassigned so other pairings can reuse them."""
        k = 0
        prefix_end_ap = None
        allowed_returns = set(cur_c.get("base_ids") or [episode_base]) \
            if cur_c.get("allow_cross_base_return") else {episode_base}
        for i, r in enumerate(leg_recs):
            elapsed_days = (r["arr"] - leg_recs[0]["dep"]) / 24.0
            if (r["dest"] in allowed_returns
                    and i + 1 >= min_pairing_legs
                    and elapsed_days <= cur_c["max_pairing_days"]):
                k = i + 1
                prefix_end_ap = r["dest"]
        if k > 0:
            emit_prefix(leg_recs[:k], prefix_end_ap, pairing_start_ap)
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
            cands = [f for f in startable if f["origin"] == b and can_reach_any_base(
                c_b["_base_reaches"], f, f["dep_time"], c_b["max_pairing_days"],
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

    # 무한루프 방지(experiments/train.py::run_episode()와 동일한 안전장치) --
    # 정상 종료하는 episode에는 영향 없고, 혹시 모를 예외적 무한루프만 방어함.
    max_steps  = len(flights) * 20
    step_count = 0

    while True:
        step_count += 1
        if step_count > max_steps:
            break

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


class _RolloutCtx:
    """rollout_with_pairings()의 클로저 상태(pairing 조립 진행 상황)를 episode
    하나만큼 담는 컨테이너 -- Phase 4(experiment/rollout-batch-vectorization)의
    실제 배치 rollout_batch()가 B개를 독립적으로 들고 다닌다."""

    __slots__ = (
        "assigned", "bad_starters", "current_legs", "leg_recs",
        "pairing_dep", "pairing_start_ap", "pairing_fly", "pairing_last_arr",
        "pairing_rest", "pairing_n_duties", "pairing_intra_gap", "pairing_inter_excess",
        "episode_base", "cur_c", "state", "pairings", "finished", "step_count",
    )

    def __init__(self, flights):
        self.assigned = {f["id"]: False for f in flights}
        self.bad_starters = set()
        self.step_count = 0
        self.current_legs = []
        self.leg_recs = []
        self.pairing_dep = None
        self.pairing_start_ap = None
        self.pairing_fly = 0.0
        self.pairing_last_arr = 0.0
        self.pairing_rest = 0.0
        self.pairing_n_duties = 1
        self.pairing_intra_gap = 0.0
        self.pairing_inter_excess = 0.0
        self.episode_base = None
        self.cur_c = None
        self.state = None
        self.pairings = []
        self.finished = False


def _flush_pairing(ctx, flight_by_id, min_pairing_legs, is_forced=False):
    if len(ctx.current_legs) < 1 or ctx.pairing_dep is None:
        return
    elapsed = ctx.pairing_last_arr - ctx.pairing_dep
    fly = ctx.pairing_fly
    n_legs = len(ctx.current_legs)
    allowed_returns = set(ctx.cur_c.get("base_ids") or [ctx.pairing_start_ap]) \
        if ctx.cur_c.get("allow_cross_base_return") else {ctx.pairing_start_ap}
    if flight_by_id[ctx.current_legs[-1]]["dest"] not in allowed_returns:
        raise ValueError("허용 home base로 복귀하지 않은 pairing은 저장할 수 없습니다.")
    if n_legs < min_pairing_legs:
        raise ValueError("최소 leg 수를 충족하지 않은 pairing은 저장할 수 없습니다.")
    if elapsed / 24.0 > ctx.cur_c["max_pairing_days"]:
        raise ValueError("최대 pairing 기간을 초과한 pairing은 저장할 수 없습니다.")
    dead_time = max(elapsed - fly - ctx.pairing_rest, 0.0)
    cost = max((dead_time
                - config.IP_LEG_BONUS * max(n_legs - 1, 0)
                + (config.IP_DEADHEAD_PENALTY if is_forced else 0.0)
                + config.IP_PAIRING_FIXED_COST), 0.0)
    ctx.pairings.append({
        "legs":        list(ctx.current_legs),
        "fly":         fly,
        "elapsed":     elapsed,
        "dead_time":   dead_time,
        "cost":        cost,
        "is_deadhead": is_forced,
        "n_legs":      n_legs,
        "n_duties":    ctx.pairing_n_duties,
        "intra_duty_gap":    ctx.pairing_intra_gap,
        "inter_duty_excess": ctx.pairing_inter_excess,
        "ends_at_base":      True,
        "true_start_airport": ctx.pairing_start_ap,
        "true_end_airport":   flight_by_id[ctx.current_legs[-1]]["dest"],
        "source_type":         "policy",
        "duty_break_indices": [
            i for i, rec in enumerate(ctx.leg_recs) if i > 0 and rec["rested"]
        ],
    })


def _emit_prefix(ctx, recs, end_ap, start_ap, min_rest):
    if len(recs) < 1:
        return
    fly = sum(r["arr"] - r["dep"] for r in recs)
    elapsed = recs[-1]["arr"] - recs[0]["dep"]
    n_rest = sum(1 for r in recs[1:] if r["rested"])
    rest = min_rest * n_rest
    intra = inter = 0.0
    for prev, r in zip(recs, recs[1:]):
        if r["rested"]:
            inter += max(r["dep"] - (prev["arr"] + min_rest), 0.0)
        else:
            intra += r["dep"] - prev["arr"]
    n_legs = len(recs)
    dead_time = max(elapsed - fly - rest, 0.0)
    ctx.pairings.append({
        "legs":        [r["id"] for r in recs],
        "fly":         fly,
        "elapsed":     elapsed,
        "dead_time":   dead_time,
        "cost":        max((dead_time
                            - config.IP_LEG_BONUS * max(n_legs - 1, 0)
                            + config.IP_DEADHEAD_PENALTY
                            + config.IP_PAIRING_FIXED_COST), 0.0),
        "is_deadhead": True,
        "n_legs":      n_legs,
        "n_duties":    n_rest + 1,
        "intra_duty_gap":    intra,
        "inter_duty_excess": inter,
        "ends_at_base":      recs[-1]["dest"] == end_ap,
        "true_start_airport": start_ap,
        "is_truncated":      True,
        "source_type":       "salvage",
        "duty_break_indices": [
            i for i, rec in enumerate(recs) if i > 0 and rec["rested"]
        ],
    })


def _salvage_doomed(ctx, min_pairing_legs, min_rest):
    k = 0
    prefix_end_ap = None
    allowed_returns = set(ctx.cur_c.get("base_ids") or [ctx.episode_base]) \
        if ctx.cur_c.get("allow_cross_base_return") else {ctx.episode_base}
    for i, r in enumerate(ctx.leg_recs):
        elapsed_days = (r["arr"] - ctx.leg_recs[0]["dep"]) / 24.0
        if (r["dest"] in allowed_returns
                and i + 1 >= min_pairing_legs
                and elapsed_days <= ctx.cur_c["max_pairing_days"]):
            k = i + 1
            prefix_end_ap = r["dest"]
    if k > 0:
        _emit_prefix(ctx, ctx.leg_recs[:k], prefix_end_ap, ctx.pairing_start_ap, min_rest)
        tail = ctx.leg_recs[k:]
    else:
        tail = list(ctx.leg_recs)
        if ctx.leg_recs:
            ctx.bad_starters.add(ctx.leg_recs[0]["id"])
    for r in tail:
        ctx.assigned[r["id"]] = False


def _start_new_pairing(ctx, f):
    ctx.current_legs.clear()
    ctx.current_legs.append(f["id"])
    ctx.leg_recs.clear()
    ctx.leg_recs.append({"id": f["id"], "dest": f["dest"],
                         "dep": f["dep_time"], "arr": f["arr_time"], "rested": False})
    ctx.pairing_start_ap = f["origin"]
    ctx.pairing_dep      = f["dep_time"]
    ctx.pairing_fly      = f["arr_time"] - f["dep_time"]
    ctx.pairing_last_arr = f["arr_time"]
    ctx.pairing_rest     = 0.0
    ctx.pairing_n_duties = 1
    ctx.pairing_intra_gap    = 0.0
    ctx.pairing_inter_excess = 0.0


def _pick_start(ctx, flights, all_bases, constraint_for):
    unassigned = [f for f in flights if not ctx.assigned[f["id"]]]
    if not unassigned:
        return None, None
    startable = [f for f in unassigned if f["id"] not in ctx.bad_starters]
    best = None
    for b in [ctx.episode_base] + [x for x in all_bases if x != ctx.episode_base]:
        c_b = constraint_for(b)
        cands = [f for f in startable if f["origin"] == b and can_reach_any_base(
            c_b["_base_reaches"], f, f["dep_time"], c_b["max_pairing_days"],
            duty_period=0, max_duty_periods=c_b["max_duty_periods"],
        )]
        if not cands:
            continue
        f = min(cands, key=lambda f: f["dep_time"])
        if b == ctx.episode_base:
            return b, f
        if best is None or f["dep_time"] < best[1]["dep_time"]:
            best = (b, f)
    if best is not None:
        return best
    return None, None


def _begin_pairing(ctx, flights, all_bases, constraint_for):
    base, f = _pick_start(ctx, flights, all_bases, constraint_for)
    if f is None:
        return False
    if base != ctx.episode_base:
        ctx.episode_base = base
        ctx.cur_c = constraint_for(base)
    ctx.assigned[f["id"]] = True
    _start_new_pairing(ctx, f)
    ctx.state = {
        "current_airport":    f["dest"],
        "current_time":       f["arr_time"],
        "duty_time":          f["arr_time"] - f["dep_time"],
        "duty_start_time":    f["dep_time"],
        "legs":               1,
        "total_legs":         1,
        "remaining":          sum(1 for v in ctx.assigned.values() if not v),
        "pairing_start":      False,
        "duty_period":        0,
        "pairing_start_time": f["dep_time"],
        "is_resting":         False,
        "rest_end_time":      None,
        "base_airport":       ctx.episode_base,
    }
    return True


def _apply_action(ctx, action, flights, flight_by_id, all_bases, constraint_for,
                  min_rest, min_pairing_legs):
    if action == len(flights):             # EndDuty
        ctx.pairing_rest     += min_rest
        ctx.pairing_n_duties += 1
        ctx.state, _, _ = step(ctx.state, action, flights, ctx.assigned, ctx.cur_c)
        return

    if action == len(flights) + 1:         # EndPairing
        _flush_pairing(ctx, flight_by_id, min_pairing_legs, is_forced=False)
        if not _begin_pairing(ctx, flights, all_bases, constraint_for):
            ctx.finished = True
        return

    f = flights[action]
    ctx.current_legs.append(f["id"])
    ctx.leg_recs.append({"id": f["id"], "dest": f["dest"],
                         "dep": f["dep_time"], "arr": f["arr_time"],
                         "rested": bool(ctx.state.get("is_resting", False))})
    ctx.pairing_fly      += f["arr_time"] - f["dep_time"]
    ctx.pairing_last_arr  = f["arr_time"]

    if not ctx.state.get("pairing_start", False) and not ctx.state.get("is_resting", False):
        ctx.pairing_intra_gap += f["dep_time"] - ctx.state["current_time"]
    elif ctx.state.get("is_resting", False):
        rest_end = ctx.state.get("rest_end_time", f["dep_time"])
        ctx.pairing_inter_excess += max(f["dep_time"] - rest_end, 0.0)

    ctx.state, _, done = step(ctx.state, action, flights, ctx.assigned, ctx.cur_c)
    if done:
        if not ctx.state.get("pairing_start", False):
            _flush_pairing(ctx, flight_by_id, min_pairing_legs, is_forced=False)
        ctx.finished = True


def rollout_batch(flights, constraint, encoder, decoder, encoded, B=50,
                  greedy=False, device=None):
    """B개 episode를 실제로 배치 처리하는 rollout (Phase 4,
    experiment/rollout-batch-vectorization) -- Phase 1-3에서 만든
    get_mask_batch()/state_to_vec_batch()/flight_gap_bias_batch()를 엮어서,
    매 timestep마다 "아직 안 끝난 episode들"의 decoder 호출을 하나로 묶는다
    (진짜 병목인 신경망 forward pass를 B번에서 그룹 수만큼으로 줄임).
    반환 형식은 rollout_with_pairings()를 B번 호출한 것과 동일
    (List[List[pairing dict]]).

    Turkish HB1/HB2 교차 base 복귀 때문에 episode마다 cur_c(_base_reach 등)가
    달라질 수 있어서, 매 timestep마다 활성 episode를 episode_base별로 그룹
    지어 그룹마다 따로 배치 호출한다(delta는 항상 그룹 1개, Turkish도 보통
    1~2개 -- get_mask_batch()/state_to_vec_batch()가 "배치 전체가 constraint를
    공유한다"고 가정하고 만들어졌으므로 이 전제를 유지하기 위함).
    """
    dev = device or torch.device("cpu")
    flight_by_id = {f["id"]: f for f in flights}
    all_bases = list(constraint.get("base_ids") or [constraint["base_airport"]])
    min_rest = constraint.get("min_rest", 10.0)
    min_pairing_legs = constraint.get("min_pairing_legs", 2)
    # 무한루프 방지(rollout_with_pairings()와 동일한 안전장치, episode별로 독립 집계).
    max_steps = len(flights) * 20

    _reach_cache = {}
    if constraint.get("_base_reach") is not None:
        _reach_cache[constraint["base_airport"]] = constraint["_base_reach"]
    _constraint_cache = {}

    def constraint_for(base):
        if base in _constraint_cache:
            return _constraint_cache[base]
        c = {**constraint, "base_airport": base}
        return_bases = all_bases if c.get("allow_cross_base_return") else [base]
        for target in return_bases:
            if target not in _reach_cache:
                _reach_cache[target] = build_base_reach(flights, target, c)
        c["_base_reach"] = _reach_cache[base]
        c["_base_reaches"] = {target: _reach_cache[target] for target in return_bases}
        _constraint_cache[base] = c
        return c

    ctxs = [_RolloutCtx(flights) for _ in range(B)]
    for ctx in ctxs:
        ctx.episode_base = constraint["base_airport"]
        ctx.cur_c = constraint_for(ctx.episode_base)
        if not any(not v for v in ctx.assigned.values()):
            ctx.finished = True
            continue
        if not _begin_pairing(ctx, flights, all_bases, constraint_for):
            ctx.finished = True

    _incl_total = decoder.state_mlp[0].weight.shape[1] > 78

    while any(not ctx.finished for ctx in ctxs):
        active = [ctx for ctx in ctxs if not ctx.finished]

        groups = {}
        for ctx in active:
            groups.setdefault(ctx.episode_base, []).append(ctx)

        for base, group in groups.items():
            c_b = constraint_for(base)
            states = [ctx.state for ctx in group]
            assigneds = [ctx.assigned for ctx in group]
            masks = get_mask_batch(states, flights, assigneds, c_b)

            decide = []
            for ctx, mask_list in zip(group, masks):
                ctx.step_count += 1
                if ctx.step_count > max_steps:
                    # 무한루프 방지(rollout_with_pairings()와 동일한 안전장치) --
                    # 이 episode만 강제 종료되고 같은 그룹의 다른 episode는 계속 진행됨.
                    ctx.finished = True
                    continue
                if sum(mask_list[:-2]) == 0 and mask_list[-2] == 0 and mask_list[-1] == 0:
                    # 위치와 무관하게 마지막 합법 base 복귀 prefix만 보존함.
                    _salvage_doomed(ctx, min_pairing_legs, min_rest)
                    if not _begin_pairing(ctx, flights, all_bases, constraint_for):
                        ctx.finished = True
                    # base/state가 바뀌었을 수 있어 이번 timestep엔 액션을 뽑지 않고
                    # 다음 while 순회에서 새 state로 다시 처리함.
                else:
                    decide.append((ctx, mask_list))

            if not decide:
                continue

            d_ctxs   = [ctx for ctx, _ in decide]
            d_states = [ctx.state for ctx in d_ctxs]
            d_masks  = [m for _, m in decide]

            state_vecs = state_to_vec_batch(
                d_states, encoder, c_b, device=dev, include_total_legs=_incl_total
            )
            gap_biases = flight_gap_bias_batch(d_states, flights, c_b, device=dev)
            mask_tensor = torch.tensor(d_masks, dtype=torch.float32, device=dev)
            probs = decoder(encoded, state_vecs, mask_tensor, gap_bias=gap_biases)

            if greedy:
                actions = probs.argmax(dim=-1).tolist()
            else:
                actions = Categorical(probs).sample().tolist()

            for ctx, action in zip(d_ctxs, actions):
                # 방어적 invariant 위반(예: flush_pairing()의 base 복귀 실패 체크)이
                # 이 episode 하나에서 터져도 배치 안의 다른 episode는 계속 진행되게
                # 격리함 -- 예전에는 rollout_with_pairings()를 episode마다 개별
                # 호출해서 호출부(evaluation/evaluate_ip.py)가 try/except로 각각
                # 감쌌는데, 배치로 묶은 이제는 이 함수 안에서 직접 격리해야
                # 그 "한 episode 실패가 나머지를 안 죽인다"는 보장이 유지됨.
                try:
                    _apply_action(ctx, action, flights, flight_by_id, all_bases,
                                  constraint_for, min_rest, min_pairing_legs)
                except Exception:
                    ctx.finished = True

    return [ctx.pairings for ctx in ctxs]


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
