import os
import sys
import random
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RL"))
import torch
import torch.optim as optim
from collections import defaultdict
from torch.distributions import Categorical
import wandb

from model import FlightEncoder, PointerDecoder
from loader import build_airport_map, bases_to_ids, load_flights_rolling, airport_map_hash
import environment as _env_default
from turkish.environment_turkish import (
    get_mask as _get_mask_turkish, step as _step_turkish, final_reward as _final_reward_turkish,
    get_mask_batch as _get_mask_batch_turkish,
)
from turkish.constraints_turkish import get_turkish_constraints as get_turkish_constraints_hb
from constraints import (
    get_delta_constraints, get_alaska_constraints,
    get_jetblue_constraints,
    FILM_CONSTRAINT_KEYS,
)

get_mask, step, final_reward = _env_default.get_mask, _env_default.step, _env_default.final_reward
get_mask_batch = _env_default.get_mask_batch


def _select_environment(airline):
    """airline에 맞는 get_mask/step/final_reward 구현으로 전환. run_episode 등 이 모듈의 get_mask/step/final_reward를
    참조하는 모든 호출부에 즉시 반영됨 (모듈 전역 rebind)."""
    global get_mask, step, final_reward, get_mask_batch
    if airline == "turkish":
        get_mask, step, final_reward = _get_mask_turkish, _step_turkish, _final_reward_turkish
        get_mask_batch = _get_mask_batch_turkish
    else:
        get_mask, step, final_reward = _env_default.get_mask, _env_default.step, _env_default.final_reward
        get_mask_batch = _env_default.get_mask_batch


_CONSTRAINT_FN = {
    "delta":   get_delta_constraints,
    "alaska":  get_alaska_constraints,
    "jetblue": get_jetblue_constraints,
    "turkish": get_turkish_constraints_hb,  # Turkish 규정값 및 HB1/HB2 교차 복귀 유지
}


def _constraint_for_episode(airline, base_airport, **overrides):
    """현재 episode의 항공사 규정을 만든 뒤 curriculum 값만 덮어씀."""
    if airline not in _CONSTRAINT_FN:
        raise ValueError(f"지원하지 않는 episode 항공사: {airline}")
    return {**_CONSTRAINT_FN[airline](base_airport), **overrides}


def _unpack_flight_sample(sample, *, require_airline, default_airline):
    """flight와 함께 선택된 항공사를 전달하여 규정 오적용을 차단함."""
    if len(sample) == 8:
        return sample
    if len(sample) == 7 and not require_airline:
        return (*sample, default_airline)
    raise ValueError("멀티에어라인 flight_sampler는 episode 항공사를 함께 반환해야 함")

from state import init_state
from base_reach import build_base_reaches, can_reach_any_base
from utils import (
    flights_to_tensors, constraint_to_tensor, state_to_vec, flight_gap_bias,
    state_to_vec_batch, flight_gap_bias_batch, set_skip_decoder_constraint,
)
import config

DEVICE = torch.device("cpu")  # train() 호출 전 _set_device()로 설정
def _set_device(device_str: str):
    global DEVICE
    DEVICE = torch.device(device_str)


def _is_better_checkpoint(coverage_pct, avg_pairings, best_coverage_pct, best_avg_pairings):
    """CPP 목적 순서대로 coverage를 먼저, pairing 수를 그다음 비교함.

    25-episode 평균 coverage가 부동소수점 단위(1e-9)로 완전히 같을 일은 거의 없어서
    그 기준이면 avg_pairings tiebreak이 사실상 죽은 코드가 됨 -- CHECKPOINT_COVERAGE_TOL_PCT
    (기본 0.5%p) 이내 차이는 "사실상 동률"로 보고 pairing 수 최소화로 비교한다.
    """
    tol = config.CHECKPOINT_COVERAGE_TOL_PCT
    return (
        coverage_pct > best_coverage_pct + tol
        or (
            abs(coverage_pct - best_coverage_pct) <= tol
            and avg_pairings < best_avg_pairings
        )
    )


def _airline_selection_score(histories, expected_airlines, window=25):
    """모든 항공사의 최근 성능에서 worst coverage와 평균 pairing을 계산함."""
    if any(len(histories.get(a, [])) < window for a in expected_airlines):
        return None
    per_airline = {
        airline: {
            "coverage_pct": sum(v["coverage_pct"] for v in histories[airline][-window:]) / window,
            "avg_pairings": sum(v["n_pairings"] for v in histories[airline][-window:]) / window,
        }
        for airline in expected_airlines
    }
    return {
        "coverage_pct": min(v["coverage_pct"] for v in per_airline.values()),
        "avg_pairings": sum(v["avg_pairings"] for v in per_airline.values()) / len(per_airline),
        "per_airline": per_airline,
    }


def _prepare_cpp_constraint(flights, constraint):
    """일반 base 또는 Turkish HB1/HB2 집합에 대한 reachability를 구성함."""
    c = dict(constraint)
    base = c["base_airport"]
    return_bases = list(c.get("base_ids") or [base]) \
        if c.get("allow_cross_base_return") else [base]
    cache_key = tuple(return_bases)
    if c.get("_base_reaches") is not None and c.get("_base_reach_bases") == cache_key:
        return c
    c["_base_reaches"] = build_base_reaches(flights, return_bases, c)
    c["_base_reach"] = c["_base_reaches"][base]
    c["_base_reach_base"] = base
    c["_base_reach_bases"] = cache_key
    return c


def run_episode(flights, constraint, encoder, decoder, encoded, greedy=False, stage=3):
    """
    Returns:
        total_reward, log_probs, entropies, metrics dict
        metrics: {n_pairings, n_deadheads, n_uncovered, coverage_pct}

    stage: config.CURRICULUM_CONFIG의 stage별 규칙(예: Stage1 allow_end_duty=False)을
        get_mask()에 반영하기 위함. 미지정 시 stage=3(제일 느슨한 규칙) 유지 -- 이전엔
        get_mask() 호출부가 stage를 아예 안 넘겨서 Stage1도 항상 stage=3 규칙으로
        학습됐음(Stage1의 allow_end_duty=False가 한 번도 적용된 적 없었음).
    """
    constraint = _prepare_cpp_constraint(flights, constraint)
    assigned = {f["id"]: False for f in flights}
    state = init_state(flights, constraint)

    log_probs = []
    entropies = []
    total_reward = 0
    n_pairings = 0
    n_deadheads = 0  # 강제 시작된 pairing 수 (connection 못 찾아서)
    n_end_duties = 0
    total_legs_sum = 0
    n_zero_mask = 0

    max_steps = len(flights) * 100  # dead-end 재시작 반영해 여유있게
    step_count = 0
    current_pairing_ids = []  # 현재(아직 EndPairing 안 된) pairing에 쓰인 flight id들
    blocked_ids = set()  # 막혀서 버려진 flight id -- 이 episode 안에서는 다시 못 고름
    # (reward pump 방지: assigned만 풀어주면 같은 flight를 다시 골라 LEG_PER_PAIRING_BONUS를
    # 또 받을 수 있음. get_mask에는 assigned|blocked를 합쳐서 넘기고, coverage/reward
    # 계산에 쓰는 진짜 assigned는 완성된 pairing만 반영하도록 분리함.)
    restart_candidate_id = None  # 직전 재시작이 겨냥한 base flight id
    tried_restart_ids = set()  # pairing 시작점으로 시도해서 실패한 flight id -- RL/rollout.py의
    # bad_starters와 동일한 역할: "새 pairing 시작점으로는 다시 안 씀"만 강제하고 blocked_ids
    # (reward 받은/pump 방지용)와는 분리해서, 다른 pairing의 연결편으로는 여전히 고를 수 있게 함.
    # 안 넣으면 재시작 후보(next_first)가 매번 동일하게 재계산돼 동일 dead-end가 재생성 ->
    # cap을 다 소진할 때까지 no-op만 반복.
    while True:
        step_count += 1
        if step_count > max_steps:
            raise RuntimeError(
                f"training rollout max_steps 초과: steps={step_count}, flights={len(flights)}"
            )
        mask_assigned = assigned if not blocked_ids else {
            **assigned, **{fid: True for fid in blocked_ids}
        }
        mask_list = get_mask(state, flights, mask_assigned, constraint, stage=stage)
        mask = torch.tensor(mask_list, dtype=torch.float32).to(DEVICE)

        # 합법 action이 없으면 -- base에서 다시 시작할 수 있는 미배정 flight가 남아있는 한
        # episode 전체를 끝내지 않고, 막힌 (미완성) pairing만 버리고 base에서 재시작한다.
        no_flight     = sum(mask_list[:-2]) == 0
        no_end_duty   = mask_list[-2] == 0
        no_end_pairing = mask_list[-1] == 0
        if no_flight and no_end_duty and no_end_pairing:
            if not current_pairing_ids and restart_candidate_id is not None:
                # 직전 재시작 이후 flight를 하나도 못 고르고 또 막힘 -- 이 flight는 새
                # pairing 시작점으로는 실패가 확정됐으므로 재시작 후보 풀에서만 뺀다
                # (reward를 받은 적 없는 flight라 blocked_ids/reward pump와는 무관).
                tried_restart_ids.add(restart_candidate_id)
            # 막힌 pairing에 쓰인 flight는 완성된 게 아니므로 coverage/reward 계산에서
            # 빠져야 하지만(다시 미배정), reward pump를 막기 위해 이 episode 안에서
            # 다시 고르지는 못 하게 영구 차단한다.
            blocked_ids.update(current_pairing_ids)
            for fid in current_pairing_ids:
                assigned[fid] = False
            current_pairing_ids = []
            unassigned = [f for f in flights if not assigned[f["id"]] and f["id"] not in blocked_ids]
            if not unassigned:
                break
            n_zero_mask += 1
            base = constraint["base_airport"]
            base_unassigned = [f for f in unassigned
                               if f["origin"] == base and f["id"] not in tried_restart_ids]
            if not base_unassigned or n_zero_mask > config.MAX_ZERO_MASK_RESTARTS:
                break
            next_first = min(base_unassigned, key=lambda f: f["dep_time"])
            restart_candidate_id = next_first["id"]
            next_time = next_first["dep_time"]
            state = {
                **state,
                "current_airport":    base,
                "current_time":       next_time,
                "duty_time":          0.0,
                "duty_start_time":    next_time,
                "legs":               0,
                "total_legs":         0,
                "duty_period":        0,
                "is_resting":         False,
                "rest_end_time":      None,
                "pairing_start":      True,
                "pairing_start_time": next_time,
            }
            continue

        # decoder
        state_vec = state_to_vec(state, encoder, constraint, device=DEVICE)
        gap_bias  = flight_gap_bias(state, flights, constraint, device=DEVICE)
        probs = decoder(encoded, state_vec, mask, gap_bias=gap_bias)

        if greedy:
            action = probs.argmax().item()
        else:
            dist = Categorical(probs)
            a = dist.sample()
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())
            action = a.item()

        n_flights = len(flights)

        # END_DUTY (index N): step()이 rest 진입 처리
        if action == n_flights:
            n_end_duties += 1
            state, r, done = step(state, action, flights, assigned, constraint)
            total_reward += r
            continue

        # END_PAIRING (index N+1): step()이 BASE_PENALTY + PAIRING_COST 처리
        if action == n_flights + 1:
            n_pairings += 1
            total_legs_sum += state.get("total_legs", 0)
            current_pairing_ids = []  # pairing 완성됨 -- 다음 pairing부터 새로 추적
            state, r, done = step(state, action, flights, assigned, constraint)
            total_reward += r
            if done:
                break
            continue

        # flight action
        current_pairing_ids.append(flights[action]["id"])
        state, r, done = step(state, action, flights, assigned, constraint)
        total_reward += r

        if done:
            break

    total_reward += final_reward(assigned)

    n_uncovered = sum(1 for v in assigned.values() if not v)
    coverage_pct = (len(flights) - n_uncovered) / len(flights) * 100

    metrics = {
        "n_pairings":   n_pairings,
        "n_deadheads":  n_deadheads,
        "n_uncovered":  n_uncovered,
        "coverage_pct": coverage_pct,
        "avg_legs":     total_legs_sum / n_pairings if n_pairings > 0 else 0.0,
        "avg_overnight": n_end_duties / n_pairings if n_pairings > 0 else 0.0,
        "n_zero_mask": n_zero_mask,
    }
    return total_reward, log_probs, entropies, metrics


# ── Phase 2 helpers ──────────────────────────────────────────────────────────
_LEG_BONUS_IP        = config.IP_LEG_BONUS           
_DEADHEAD_PENALTY_IP = config.IP_DEADHEAD_PENALTY     
_PAIRING_FIXED_COST  = config.IP_PAIRING_FIXED_COST  


def _rollout_with_pairings(flights, constraint, encoder, decoder, encoded, greedy=False):
    constraint = _prepare_cpp_constraint(flights, constraint)
    assigned = {f["id"]: False for f in flights}
    flight_by_id = {f["id"]: f for f in flights}
    pairings = []

    current_legs     = []
    pairing_dep      = None
    pairing_fly      = 0.0
    pairing_last_arr = 0.0
    pairing_rest     = 0.0
    bad_starters     = set()  # RL/rollout.py의 bad_starters와 동일: dead-end로 확인된
    # pairing 시작점(leg[0])을 rollout 전체에서 영구 제외. 매 dead-end마다 새로 계산되는
    # abandoned_ids만으로는 실패한 후보 몇 개를 계속 순환 재시도하게 됨 -- 실측: cap=30/100
    # 둘 다 결과(n_pairings=3) 동일한데 재시작 시도 중 각각 26/96회가 이미 실패했던
    # 후보의 중복 재시도였음(고유 후보는 4개뿐).

    def flush_pairing(is_forced=False):
        if len(current_legs) < 1 or pairing_dep is None:
            return
        elapsed   = pairing_last_arr - pairing_dep
        # 일반 항공사는 동일 base, Turkish는 HB1/HB2 home-base 집합 복귀를 허용함.
        allowed_returns = set(constraint.get("base_ids") or [episode_base]) \
            if constraint.get("allow_cross_base_return") else {episode_base}
        if flight_by_id[current_legs[0]]["origin"] != episode_base \
                or flight_by_id[current_legs[-1]]["dest"] not in allowed_returns:
            raise ValueError("허용 home base로 복귀하지 않은 pairing은 dual pool에 저장할 수 없습니다.")
        if elapsed / 24.0 > constraint["max_pairing_days"]:
            raise ValueError("최대 pairing 기간을 초과한 pairing은 dual pool에 저장할 수 없습니다.")
        dead_time = max(elapsed - pairing_fly - pairing_rest, 0.0)
        # leg 수가 많고 연결이 타이트하면 이 공식이 음수가 될 수 있음(RL/rollout.py/
        # RL/turkish/environment_turkish.py에서 고친 것과 동일한 문제) -- solve_lp_relaxation()이
        # 이 pool의 cost를 그대로 쓰므로 방어적으로 0에서 clamp.
        cost = max((dead_time
                    - _LEG_BONUS_IP * max(len(current_legs) - 1, 0)
                    + (_DEADHEAD_PENALTY_IP if is_forced else 0.0)
                    + _PAIRING_FIXED_COST), 0.0)
        # A pairing must start and end at the base to be a valid column for
        # the LP-dual pool (Eq. 2 requires x_p in Omega(c), which excludes
        # pairings that never return to base); same check as RL/rollout.py.
        ends_at_base = True
        pairings.append({"legs": list(current_legs), "fly": pairing_fly,
                         "elapsed": elapsed, "cost": cost,
                         "ends_at_base": ends_at_base})

    def start_new(f):
        nonlocal pairing_dep, pairing_fly, pairing_last_arr, pairing_rest
        current_legs.clear()
        current_legs.append(f["id"])
        pairing_dep      = f["dep_time"]
        pairing_fly      = f["arr_time"] - f["dep_time"]
        pairing_last_arr = f["arr_time"]
        pairing_rest     = 0.0

    episode_base = constraint["base_airport"]

    def base_start_candidates(candidates):
        base_flights = [f for f in candidates
                        if f["origin"] == episode_base and f["id"] not in bad_starters]
        # 수동 시작 flight도 decoder와 같은 복귀 가능성 검사를 통과해야 함.
        return [f for f in base_flights if can_reach_any_base(
            constraint["_base_reaches"], f, f["dep_time"],
            constraint["max_pairing_days"], duty_period=0,
            max_duty_periods=constraint["max_duty_periods"],
        )]

    # Manually start the first flight -- prefer a base-departing leg
    unassigned   = [f for f in flights if not assigned[f["id"]]]
    base_flights = base_start_candidates(unassigned)
    if not base_flights:
        return pairings
    first        = sorted(base_flights or unassigned, key=lambda f: f["dep_time"])[0]
    assigned[first["id"]] = True
    start_new(first)
    state = {
        "current_airport":    first["dest"],
        "current_time":       first["arr_time"],
        "duty_time":          first["arr_time"] - first["dep_time"],
        "duty_start_time":    first["dep_time"],
        "legs":               1,
        "total_legs":         1,
        "remaining":          sum(1 for v in assigned.values() if not v),
        "pairing_start":      False,
        "duty_period":        0,
        "pairing_start_time": first["dep_time"],
        "is_resting":         False,
        "rest_end_time":      None,
    }

    max_steps  = len(flights) * 100  # dead-end 재시작 반영해 여유있게
    step_count = 0
    zero_mask_restarts = 0

    while True:
        step_count += 1
        if step_count > max_steps:
            raise RuntimeError(
                f"training rollout max_steps 초과: steps={step_count}, flights={len(flights)}"
            )

        mask_list = get_mask(state, flights, assigned, constraint)
        mask      = torch.tensor(mask_list, dtype=torch.float32).to(DEVICE)

        if sum(mask_list[:-2]) == 0 and mask_list[-2] == 0 and mask_list[-1] == 0:
            # 미복귀 partial pairing은 CPP column으로 저장하지 않음(버림). 다만 base에서
            # 다시 시작할 미배정 flight가 남아있으면 rollout 전체를 끝내지 않고 이어감 --
            # 그래야 pool이 첫 dead-end에서 끊기지 않고 여러 pairing을 계속 모을 수 있음.
            zero_mask_restarts += 1
            if current_legs:
                # 이번 pairing의 시작 leg는 dead-end로 확인됐으므로 향후 시작점
                # 후보에서 영구 제외(RL/rollout.py의 bad_starters와 동일). 중간/끝
                # leg는 다른 pairing의 연결편으로 여전히 쓰일 수 있어야 하므로 안 막음.
                bad_starters.add(current_legs[0])
            abandoned_ids = set(current_legs)
            for fid in abandoned_ids:
                assigned[fid] = False
            unassigned = [f for f in flights if not assigned[f["id"]]]
            base_unassigned = base_start_candidates(unassigned)
            if not base_unassigned or zero_mask_restarts > config.MAX_ZERO_MASK_RESTARTS:
                break
            next_first = sorted(base_unassigned, key=lambda f: f["dep_time"])[0]
            assigned[next_first["id"]] = True
            start_new(next_first)
            state = {
                "current_airport":    next_first["dest"],
                "current_time":       next_first["arr_time"],
                "duty_time":          next_first["arr_time"] - next_first["dep_time"],
                "duty_start_time":    next_first["dep_time"],
                "legs":               1,
                "total_legs":         1,
                "remaining":          sum(1 for v in assigned.values() if not v),
                "pairing_start":      False,
                "duty_period":        0,
                "pairing_start_time": next_first["dep_time"],
                "is_resting":         False,
                "rest_end_time":      None,
            }
            continue

        state_vec = state_to_vec(state, encoder, constraint, device=DEVICE)
        gap_bias  = flight_gap_bias(state, flights, constraint, device=DEVICE)
        probs     = decoder(encoded, state_vec, mask, gap_bias=gap_bias)
        action    = probs.argmax().item() if greedy else Categorical(probs).sample().item()

        if action == len(flights):          # END_DUTY
            pairing_rest += constraint.get("min_rest", 10.0)
            state, _, _ = step(state, action, flights, assigned, constraint)
            continue

        if action == len(flights) + 1:      # EndPairing -> start a new pairing
            flush_pairing(is_forced=False)
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if not unassigned:
                break
            base_flights = base_start_candidates(unassigned)
            if not base_flights:
                break
            nxt = sorted(base_flights or unassigned, key=lambda f: f["dep_time"])[0]
            assigned[nxt["id"]] = True
            start_new(nxt)
            state = {
                "current_airport":    nxt["dest"],
                "current_time":       nxt["arr_time"],
                "duty_time":          nxt["arr_time"] - nxt["dep_time"],
                "duty_start_time":    nxt["dep_time"],
                "legs":               1,
                "total_legs":         1,
                "remaining":          sum(1 for v in assigned.values() if not v),
                "pairing_start":      False,
                "duty_period":        0,
                "pairing_start_time": nxt["dep_time"],
                "is_resting":         False,
                "rest_end_time":      None,
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


class _DualPoolCtx:
    """_rollout_with_pairings()의 클로저 상태를 episode 하나만큼 담는 컨테이너 --
    _rollout_batch_dual_pool()(Phase 5, experiment/rollout-batch-vectorization)이
    B개를 독립적으로 유지한다. RL/rollout.py::_RolloutCtx와 같은 목적이지만, 이
    pool은 salvage_doomed/base 회전이 없는 더 단순한 원본 로직을 그대로 따른다."""

    __slots__ = ("assigned", "current_legs", "pairing_dep", "pairing_fly",
                "pairing_last_arr", "pairing_rest", "state", "pairings", "finished",
                "zero_mask_restarts", "bad_starters")

    def __init__(self, flights):
        self.assigned = {f["id"]: False for f in flights}
        self.current_legs = []
        self.zero_mask_restarts = 0
        self.pairing_dep = None
        self.pairing_fly = 0.0
        self.pairing_last_arr = 0.0
        self.pairing_rest = 0.0
        self.state = None
        self.pairings = []
        self.finished = False
        self.bad_starters = set()  # _rollout_with_pairings()와 동일: dead-end로
        # 확인된 pairing 시작점을 이 episode(ctx) 전체에서 영구 제외


def _dual_pool_flush_pairing(ctx, flight_by_id, constraint, episode_base, is_forced=False):
    if len(ctx.current_legs) < 1 or ctx.pairing_dep is None:
        return
    elapsed = ctx.pairing_last_arr - ctx.pairing_dep
    n_legs = len(ctx.current_legs)
    allowed_returns = set(constraint.get("base_ids") or [episode_base]) \
        if constraint.get("allow_cross_base_return") else {episode_base}
    if flight_by_id[ctx.current_legs[0]]["origin"] != episode_base \
            or flight_by_id[ctx.current_legs[-1]]["dest"] not in allowed_returns:
        raise ValueError("허용 home base로 복귀하지 않은 pairing은 dual pool에 저장할 수 없습니다.")
    if elapsed / 24.0 > constraint["max_pairing_days"]:
        raise ValueError("최대 pairing 기간을 초과한 pairing은 dual pool에 저장할 수 없습니다.")
    dead_time = max(elapsed - ctx.pairing_fly - ctx.pairing_rest, 0.0)
    cost = max((dead_time
                - _LEG_BONUS_IP * max(n_legs - 1, 0)
                + (_DEADHEAD_PENALTY_IP if is_forced else 0.0)
                + _PAIRING_FIXED_COST), 0.0)
    ctx.pairings.append({"legs": list(ctx.current_legs), "fly": ctx.pairing_fly,
                         "elapsed": elapsed, "cost": cost, "ends_at_base": True})


def _dual_pool_start_new(ctx, f):
    ctx.current_legs = [f["id"]]
    ctx.pairing_dep = f["dep_time"]
    ctx.pairing_fly = f["arr_time"] - f["dep_time"]
    ctx.pairing_last_arr = f["arr_time"]
    ctx.pairing_rest = 0.0


def _dual_pool_base_start_candidates(flights, ctx, episode_base, constraint):
    unassigned = [f for f in flights if not ctx.assigned[f["id"]]]
    base_flights = [f for f in unassigned if f["origin"] == episode_base
                    and f["id"] not in ctx.bad_starters]
    return [f for f in base_flights if can_reach_any_base(
        constraint["_base_reaches"], f, f["dep_time"],
        constraint["max_pairing_days"], duty_period=0,
        max_duty_periods=constraint["max_duty_periods"],
    )]


def _dual_pool_begin(flights, ctx, episode_base, constraint):
    base_flights = _dual_pool_base_start_candidates(flights, ctx, episode_base, constraint)
    if not base_flights:
        return False
    first = sorted(base_flights, key=lambda f: f["dep_time"])[0]
    ctx.assigned[first["id"]] = True
    _dual_pool_start_new(ctx, first)
    ctx.state = {
        "current_airport":    first["dest"],
        "current_time":       first["arr_time"],
        "duty_time":          first["arr_time"] - first["dep_time"],
        "duty_start_time":    first["dep_time"],
        "legs":               1,
        "total_legs":         1,
        "remaining":          sum(1 for v in ctx.assigned.values() if not v),
        "pairing_start":      False,
        "duty_period":        0,
        "pairing_start_time": first["dep_time"],
        "is_resting":         False,
        "rest_end_time":      None,
    }
    return True


def _rollout_batch_dual_pool(flights, constraint, encoder, decoder, encoded, B, greedy=False):
    """_rollout_with_pairings()의 실제 배치 버전 -- 동작(salvage_doomed 없음, base
    고정, cross-base 회전 없음)은 완전히 동일하게 유지하고 decoder 호출만 배치로
    묶는다. Phase 5, experiment/rollout-batch-vectorization -- _collect_pool()에서
    사용. RL/rollout.py::rollout_batch()와 달리 episode_base가 절대 안 바뀌므로
    (원본 _rollout_with_pairings()도 회전이 없음) 매 timestep 그룹핑이 필요 없음
    -- 활성 episode 전체가 항상 하나의 배치."""
    flight_by_id = {f["id"]: f for f in flights}
    episode_base = constraint["base_airport"]

    ctxs = [_DualPoolCtx(flights) for _ in range(B)]
    for ctx in ctxs:
        if not _dual_pool_begin(flights, ctx, episode_base, constraint):
            ctx.finished = True

    _incl_total = decoder.state_mlp[0].weight.shape[1] > 78
    max_steps = len(flights) * 100  # dead-end 재시작 반영해 여유있게
    step_counts = [0] * B
    n_flights = len(flights)

    while any(not ctx.finished for ctx in ctxs):
        active_idx = [i for i, ctx in enumerate(ctxs) if not ctx.finished]

        states    = [ctxs[i].state for i in active_idx]
        assigneds = [ctxs[i].assigned for i in active_idx]
        masks     = get_mask_batch(states, flights, assigneds, constraint)

        decide_idx   = []
        decide_masks = []
        for local_i, i in enumerate(active_idx):
            step_counts[i] += 1
            mask_list = masks[local_i]
            if step_counts[i] > max_steps:
                raise RuntimeError(
                    "dual-pool batch rollout max_steps 초과: "
                    f"episode={i}, steps={step_counts[i]}, flights={len(flights)}"
                )
            # 합법 action이 없으면 -- run_episode()/_rollout_with_pairings()와 동일하게,
            # 막힌 pairing만 버리고(미배정으로 되돌리되 이번 재시작 후보에서는 제외)
            # base에서 재시작함. 재시작 횟수는 MAX_ZERO_MASK_RESTARTS로 제한.
            if sum(mask_list[:-2]) == 0 and mask_list[-2] == 0 and mask_list[-1] == 0:
                ctx = ctxs[i]
                ctx.zero_mask_restarts += 1
                if ctx.current_legs:
                    # _rollout_with_pairings()와 동일: 시작점만 영구 제외, 중간/끝
                    # leg는 다른 pairing의 연결편으로 여전히 쓰일 수 있어야 함.
                    ctx.bad_starters.add(ctx.current_legs[0])
                for fid in ctx.current_legs:
                    ctx.assigned[fid] = False
                ctx.current_legs = []
                if (ctx.zero_mask_restarts > config.MAX_ZERO_MASK_RESTARTS
                        or not _dual_pool_begin(flights, ctx, episode_base, constraint)):
                    ctx.finished = True
                continue
            decide_idx.append(i)
            decide_masks.append(mask_list)

        if not decide_idx:
            continue

        d_states = [ctxs[i].state for i in decide_idx]
        state_vecs = state_to_vec_batch(
            d_states, encoder, constraint, device=DEVICE, include_total_legs=_incl_total
        )
        gap_biases = flight_gap_bias_batch(d_states, flights, constraint, device=DEVICE)
        mask_tensor = torch.tensor(decide_masks, dtype=torch.float32, device=DEVICE)
        probs = decoder(encoded, state_vecs, mask_tensor, gap_bias=gap_biases)

        if greedy:
            actions = probs.argmax(dim=-1).tolist()
        else:
            actions = Categorical(probs).sample().tolist()

        for i, action in zip(decide_idx, actions):
            ctx = ctxs[i]
            if action == n_flights:               # EndDuty
                ctx.pairing_rest += constraint.get("min_rest", 10.0)
                ctx.state, _, _ = step(ctx.state, action, flights, ctx.assigned, constraint)
                continue

            if action == n_flights + 1:           # EndPairing -> 새 pairing 시작
                _dual_pool_flush_pairing(ctx, flight_by_id, constraint, episode_base, is_forced=False)
                if not _dual_pool_begin(flights, ctx, episode_base, constraint):
                    ctx.finished = True
                continue

            f = flights[action]
            ctx.current_legs.append(f["id"])
            ctx.pairing_fly      += f["arr_time"] - f["dep_time"]
            ctx.pairing_last_arr  = f["arr_time"]
            ctx.state, _, done = step(ctx.state, action, flights, ctx.assigned, constraint)
            if done:
                _dual_pool_flush_pairing(ctx, flight_by_id, constraint, episode_base, is_forced=False)
                ctx.finished = True

    return [ctx.pairings for ctx in ctxs]


def _collect_pool(flights, constraint, encoder, decoder, encoded, n_rollouts):
    constraint = _prepare_cpp_constraint(flights, constraint)
    # Exclude pairings that do not return to base -- the restricted LP of
    # Eq. (2) is defined over Omega(c), and its duals mu^cov/nu^exc (Eq. 9)
    # should not be computed from infeasible columns.
    pool = {}
    for episode_pairings in _rollout_batch_dual_pool(
        flights, constraint, encoder, decoder, encoded, B=n_rollouts, greedy=False
    ):
        for p in episode_pairings:
            if not p["ends_at_base"]:
                continue
            key = tuple(sorted(p["legs"]))
            if key not in pool or p["cost"] < pool[key]["cost"]:
                pool[key] = p
    for p in _rollout_batch_dual_pool(
        flights, constraint, encoder, decoder, encoded, B=1, greedy=True
    )[0]:
        if not p["ends_at_base"]:
            continue
        key = tuple(sorted(p["legs"]))
        if key not in pool or p["cost"] < pool[key]["cost"]:
            pool[key] = p
    return list(pool.values())


def run_episode_with_dual(flights, constraint, encoder, decoder, encoded, dual_vars, greedy=False, dual_weight=None, dh_dual_vars=None):
    """Phase II rollout: same environment as Phase I, but the per-step reward
    is augmented with the net-dual signal of Eq. (9)-(10):

        delta_i = mu_i^cov - nu_i^exc,   r~_t = r^loc_t + w_dual(e) * delta_i

    dual_vars/dh_dual_vars are the cached mu^cov/nu^exc from the most recent
    restricted-master LP solve (Algorithm 1, line 6); dual_weight is
    w_dual(e), ramped up externally by run_phase2() (Algorithm 1, line 8-9).
    """
    constraint = _prepare_cpp_constraint(flights, constraint)
    assigned = {f["id"]: False for f in flights}
    state    = init_state(flights, constraint)

    log_probs    = []
    entropies    = []
    total_reward = 0
    n_pairings    = 0
    n_deadheads   = 0
    n_end_duties  = 0
    total_legs_sum = 0
    n_zero_mask    = 0
    base          = constraint["base_airport"]

    max_steps  = len(flights) * 100  # dead-end 재시작 반영해 여유있게
    step_count = 0
    current_pairing_ids = []
    blocked_ids = set()  # reward pump 방지 -- run_episode()와 동일한 이유
    restart_candidate_id = None  # run_episode()와 동일: no-op 재시작 방지용
    tried_restart_ids = set()  # run_episode()와 동일: 시작점 실패만 기록, 연결편 재사용은 허용

    while True:
        step_count += 1
        if step_count > max_steps:
            raise RuntimeError(
                f"training rollout max_steps 초과: steps={step_count}, flights={len(flights)}"
            )

        mask_assigned = assigned if not blocked_ids else {
            **assigned, **{fid: True for fid in blocked_ids}
        }
        mask_list  = get_mask(state, flights, mask_assigned, constraint)
        mask       = torch.tensor(mask_list, dtype=torch.float32).to(DEVICE)

        no_flight      = sum(mask_list[:-2]) == 0
        no_end_duty    = mask_list[-2] == 0
        no_end_pairing = mask_list[-1] == 0
        if no_flight and no_end_duty and no_end_pairing:
            if not current_pairing_ids and restart_candidate_id is not None:
                # run_episode()와 동일: 재시작 직후 진전 없이 또 막히면 그 후보는 시작점
                # 후보에서만 뺀다(연결편으로는 여전히 재사용 가능, reward 받은 적 없음).
                tried_restart_ids.add(restart_candidate_id)
            # dual 학습도 run_episode()와 동일: 막힌 pairing은 미배정으로 되돌리되
            # reward pump 방지를 위해 이 episode 안에서는 영구 차단.
            blocked_ids.update(current_pairing_ids)
            for fid in current_pairing_ids:
                assigned[fid] = False
            current_pairing_ids = []
            unassigned = [f for f in flights if not assigned[f["id"]] and f["id"] not in blocked_ids]
            if not unassigned:
                break
            n_zero_mask += 1
            base_unassigned = [f for f in unassigned
                               if f["origin"] == base and f["id"] not in tried_restart_ids]
            if not base_unassigned or n_zero_mask > config.MAX_ZERO_MASK_RESTARTS:
                break
            next_first = min(base_unassigned, key=lambda f: f["dep_time"])
            restart_candidate_id = next_first["id"]
            next_time = next_first["dep_time"]
            state = {
                **state,
                "current_airport":    base,
                "current_time":       next_time,
                "duty_time":          0.0,
                "duty_start_time":    next_time,
                "legs":               0,
                "total_legs":         0,
                "duty_period":        0,
                "is_resting":         False,
                "rest_end_time":      None,
                "pairing_start":      True,
                "pairing_start_time": next_time,
            }
            continue

        state_vec = state_to_vec(state, encoder, constraint, device=DEVICE)
        gap_bias  = flight_gap_bias(state, flights, constraint, device=DEVICE)
        probs     = decoder(encoded, state_vec, mask, gap_bias=gap_bias)
        if greedy:
            action = probs.argmax().item()
        else:
            dist = Categorical(probs)
            a    = dist.sample()
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())
            action = a.item()

        n_flights = len(flights)

        if action == n_flights:         # END_DUTY
            n_end_duties += 1
            state, r, done = step(state, action, flights, assigned, constraint)
            total_reward += r
            continue

        if action == n_flights + 1:
            n_pairings += 1
            total_legs_sum += state.get("total_legs", 0)
            current_pairing_ids = []
            state, r, done = step(state, action, flights, assigned, constraint)
            total_reward += r
            if done:
                break
            continue

        flight_id = flights[action]["id"]
        current_pairing_ids.append(flight_id)
        _dw = dual_weight if dual_weight is not None else config.PHASE2_DUAL_WEIGHT  # w_dual(e)
        state, r, done = step(state, action, flights, assigned, constraint)
        _nu_exc = dh_dual_vars.get(flight_id, 0.0) if dh_dual_vars else 0.0
        # r~_t = r^loc_t + w_dual(e) * (mu_i^cov - nu_i^exc), Eq. (9)-(10)
        total_reward += r + (dual_vars.get(flight_id, 0.0) - _nu_exc) * _dw
        if done:
            break

    total_reward += final_reward(assigned)
    n_uncovered  = sum(1 for v in assigned.values() if not v)
    coverage_pct = (len(flights) - n_uncovered) / len(flights) * 100
    return total_reward, log_probs, entropies, {
        "n_pairings":    n_pairings,
        "n_deadheads":   n_deadheads,
        "n_uncovered":   n_uncovered,
        "coverage_pct":  coverage_pct,
        "avg_legs":      total_legs_sum / n_pairings if n_pairings > 0 else 0.0,
        "avg_overnight": n_end_duties / n_pairings if n_pairings > 0 else 0.0,
        "n_zero_mask":   n_zero_mask,
    }


def normalize_phase2_dual_signal(
    coverage_duals, excess_duals=None, mode="real",
    uncovered_flight_ids=None, shuffle_seed=0,
):
    """Phase2 ablation별 flight 신호를 만들고 [-1, 1] 범위로 맞춤."""
    excess_duals = excess_duals or {}
    mode = {"net": "real", "coverage_only": "coverage-only"}.get(mode, mode)
    if mode == "zero":
        return {flight_id: 0.0 for flight_id in coverage_duals}
    if mode == "uncovered-only":
        uncovered = set(uncovered_flight_ids or ())
        return {
            flight_id: 1.0 if flight_id in uncovered else 0.0
            for flight_id in coverage_duals
        }
    if mode == "coverage-only":
        raw = {flight_id: float(value) for flight_id, value in coverage_duals.items()}
    elif mode in ("real", "shuffled", "robust-real", "robust-shuffled"):
        raw = {
            flight_id: float(value) - float(excess_duals.get(flight_id, 0.0))
            for flight_id, value in coverage_duals.items()
        }
    else:
        raise ValueError(f"지원하지 않는 Phase 2 dual mode: {mode}")
    if mode in ("robust-real", "robust-shuffled"):
        # Artificial singleton의 큰 dual이 정상 후보들의 신호를 0에 가깝게 누르지 않도록
        # 커버 가능한 flight만으로 95백분위 스케일을 계산하고 미커버 flight는 1로 고정함.
        uncovered = set(uncovered_flight_ids or ())
        covered_abs = sorted(
            abs(value) for flight_id, value in raw.items()
            if flight_id not in uncovered and abs(value) > 1e-12
        )
        if covered_abs:
            percentile_index = max(0, math.ceil(0.95 * len(covered_abs)) - 1)
            scale = max(covered_abs[percentile_index], 1e-8)
        else:
            scale = 1.0
        normalized = {
            flight_id: (
                1.0 if flight_id in uncovered
                else max(-1.0, min(1.0, value / scale))
            )
            for flight_id, value in raw.items()
        }
    else:
        scale = max([abs(value) for value in raw.values()] + [1.0])
        normalized = {
            flight_id: max(-1.0, min(1.0, value / scale))
            for flight_id, value in raw.items()
        }

    if mode in ("shuffled", "robust-shuffled"):
        keys = sorted(normalized)
        values = [normalized[key] for key in keys]
        random.Random(shuffle_seed).shuffle(values)
        normalized = dict(zip(keys, values))
    return normalized


def run_phase2(encoder, decoder, optimizer, n_episodes, constraint, save_dir, flight_sampler,
               global_step_offset=0, entropy_start=0.01, entropy_end=0.005,
               constraint_sampler=None, init_best=float("inf"), dual_weight_override=None,
               dual_mode="real", checkpoint_metadata=None):
    dual_mode = {"net": "real", "coverage_only": "coverage-only"}.get(dual_mode, dual_mode)
    assert dual_mode in (
        "zero", "uncovered-only", "shuffled", "real", "coverage-only",
        "robust-real", "robust-shuffled",
    ), f"unknown dual_mode: {dual_mode}"
    from evaluation.set_partition import solve_lp_relaxation

    params            = list(encoder.parameters()) + list(decoder.parameters())
    best_avg_pairings = init_best
    best_coverage_pct = -1.0
    greedy_pairings   = []
    greedy_coverages  = []
    airline_histories = defaultdict(list)
    expected_airlines = tuple((checkpoint_metadata or {}).get("airlines") or [config.AIRLINE])
    dual_vars         = {}
    dh_dual_vars      = {}
    lp_value          = None

    print(f"\n{'='*60}")
    print(f"Phase 2: CG dual feedback  "
          f"(LP interval={config.PHASE2_LP_INTERVAL}, "
          f"pool rollouts={config.PHASE2_POOL_ROLLOUTS})")
    print(f"{'='*60}")

    for ep in range(n_episodes):
        sample = flight_sampler()
        if sample is None:
            continue
        flights, origins, dests, dep_times, arr_times, fly_times, base_airport, sampled_airline = _unpack_flight_sample(
            sample, require_airline=(constraint_sampler is not None), default_airline=config.AIRLINE
        )

        base_c   = constraint_sampler(sampled_airline, base_airport) if constraint_sampler else constraint
        c        = {**base_c, "base_airport": base_airport}
        c        = _prepare_cpp_constraint(flights, c)
        c_tensor = constraint_to_tensor(c, device=DEVICE)

        with torch.no_grad():
            encoded = encoder(origins, dests, dep_times, arr_times, fly_times, c_tensor)
            # 매 episode의 local flight ID 의미가 달라 이전 dual을 재사용할 수 없음.
            # 현재 instance의 전체 flight를 artificial singleton으로 LP에 포함하여
            # pool에서 한 번도 생성되지 않은 flight에도 coverage dual을 부여함.
            pool = _collect_pool(
                flights, c, encoder, decoder, encoded,
                n_rollouts=config.PHASE2_POOL_ROLLOUTS,
            )
            lp_result = solve_lp_relaxation(
                pool,
                flight_ids=[f["id"] for f in flights],
                artificial_cost=config.PHASE2_ARTIFICIAL_COST,
            )
            coverage_duals = lp_result["dual_vars"] if lp_result is not None else {}
            raw_excess_duals = lp_result["dh_dual_vars"] if lp_result is not None else {}
            dual_vars = normalize_phase2_dual_signal(
                coverage_duals, raw_excess_duals, mode=dual_mode,
                uncovered_flight_ids=(lp_result or {}).get("artificial_flight_ids", []),
                shuffle_seed=ep,
            )
            dual_abs_values = [abs(value) for value in dual_vars.values()]
            dual_abs_mean = (
                sum(dual_abs_values) / len(dual_abs_values)
                if dual_abs_values else 0.0
            )
            dual_nonzero_fraction = (
                sum(value > 1e-8 for value in dual_abs_values) / len(dual_abs_values)
                if dual_abs_values else 0.0
            )
            dual_saturated_fraction = (
                sum(value >= 1.0 - 1e-8 for value in dual_abs_values) / len(dual_abs_values)
                if dual_abs_values else 0.0
            )
            # net 계산과 정규화를 한 번에 끝냈으므로 rollout에서 다시 차감하지 않음.
            dh_dual_vars = {}
            lp_value = lp_result["lp_value"] if lp_result is not None else None

        _base_dw = dual_weight_override if dual_weight_override is not None else config.PHASE2_DUAL_WEIGHT
        _eff_dw = _base_dw * min(1.0, (ep + 1) / max(config.PHASE2_DUAL_WARMUP, 1))
        encoded_train = encoder(origins, dests, dep_times, arr_times, fly_times, c_tensor)
        reward_s, log_probs, entropies, metrics_s = run_episode_with_dual(
            flights, c, encoder, decoder, encoded_train, dual_vars, dual_weight=_eff_dw, dh_dual_vars=dh_dual_vars
        )
        if len(log_probs) == 0:
            continue

        with torch.no_grad():
            encoded_g = encoder(origins, dests, dep_times, arr_times, fly_times, c_tensor)
            reward_g, _, _, metrics_g = run_episode_with_dual(
                flights, c, encoder, decoder, encoded_g, dual_vars, greedy=True, dual_weight=_eff_dw, dh_dual_vars=dh_dual_vars
            )

        greedy_pairings.append(metrics_g["n_pairings"])
        greedy_coverages.append(metrics_g["coverage_pct"])
        airline_histories[sampled_airline].append(metrics_g)
        advantage = (reward_s - reward_g) / (abs(reward_g) + 1e-6)

        entropy_coef = max(entropy_start * (1.0 - ep / n_episodes), entropy_end)
        loss = torch.stack([
            -lp * advantage - entropy_coef * ent
            for lp, ent in zip(log_probs, entropies)
        ]).sum()

        avg25 = sum(greedy_pairings[-25:]) / min(len(greedy_pairings), 25)
        coverage25 = sum(greedy_coverages[-25:]) / min(len(greedy_coverages), 25)
        selection_score = _airline_selection_score(airline_histories, expected_airlines)

        if selection_score is not None and _is_better_checkpoint(
            selection_score["coverage_pct"], selection_score["avg_pairings"],
            best_coverage_pct, best_avg_pairings,
        ):
            best_avg_pairings = selection_score["avg_pairings"]
            best_coverage_pct = selection_score["coverage_pct"]
            ckpt_path = os.path.join(save_dir, "phase2_best.pt")
            torch.save({
                "encoder":           encoder.state_dict(),
                "decoder":           decoder.state_dict(),
                "stage":             "phase2",
                "episode":           ep,
                "best_avg_pairings": best_avg_pairings,
                "best_coverage_pct": best_coverage_pct,
                "best_per_airline": selection_score["per_airline"],
                "time_basis":        "turkish_native" if config.AIRLINE == "turkish" else "utc",
                **(checkpoint_metadata or {}),
            }, ckpt_path)
            wandb.save(ckpt_path)

        # 위 greedy 지표를 산출한 바로 그 파라미터를 best로 저장한 뒤 업데이트함.
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        wandb.log({
            "phase2/greedy_pairings":     metrics_g["n_pairings"],
            "phase2/sample_pairings":     metrics_s["n_pairings"],
            "phase2/greedy_deadheads":    metrics_g["n_deadheads"],
            "phase2/greedy_avg_legs":     metrics_g.get("avg_legs", 0),
            "phase2/greedy_avg_overnight": metrics_g.get("avg_overnight", 0),
            "phase2/sample_reward":       reward_s,
            "phase2/avg25":               avg25,
            "phase2/coverage25":          coverage25,
            "phase2/advantage":           advantage,
            "phase2/loss":                loss.item(),
            "phase2/entropy_coef":        entropy_coef,
            "phase2/best_avg25":          best_avg_pairings if best_avg_pairings < float("inf") else avg25,
            "phase2/n_dual_keys":         len(dual_vars),
            "phase2/n_dh_dual_keys":      sum(1 for v in raw_excess_duals.values() if v > 0),
            "phase2/dual_weight":         _eff_dw,
            "phase2/gap_weight":          decoder.gap_weight.item(),
            "phase2/lp_value":            lp_value if lp_value is not None else float("nan"),
            "phase2/artificial_count":    len((lp_result or {}).get("artificial_flight_ids", [])),
            "phase2/dual_abs_mean":       dual_abs_mean,
            "phase2/dual_nonzero_fraction": dual_nonzero_fraction,
            "phase2/dual_saturated_fraction": dual_saturated_fraction,
        }, step=global_step_offset + ep)

        if ep % 25 == 0:
            _lp_str = f"{lp_value:.2f}" if lp_value is not None else "n/a"
            print(
                f"  Ep {ep:4d} | "
                f"sample: p={metrics_s['n_pairings']:3d} dh={metrics_s['n_deadheads']:3d} | "
                f"greedy: p={metrics_g['n_pairings']:3d} legs={metrics_g.get('avg_legs', 0):.2f} "
                f"(avg25={avg25:5.1f}, cov25={coverage25:5.1f}%) | "
                f"adv: {advantage:6.3f} | dw={_eff_dw:.3f} | dual keys: {len(dual_vars)} | "
                f"dual |x| mean: {dual_abs_mean:.3f} sat: {dual_saturated_fraction:.1%} | "
                f"dh dual keys: {sum(1 for v in raw_excess_duals.values() if v > 0)} | lp_value: {_lp_str}"
            )

    print(f"  → best avg pairings: {best_avg_pairings:.1f}  "
          f"(saved: checkpoints/phase2_best.pt)")
    return best_avg_pairings


def run_curriculum_stage(
    stage, encoder, decoder, optimizer,
    n_episodes, constraint_override, save_dir,
    flight_sampler, constraint_sampler=None,
    global_step_offset=0,
    entropy_start=0.05, entropy_end=0.005,
    base_stage2_constraint=None,  # Stage 3 리플레이용 베이스 파라미터 추가
    checkpoint_metadata=None,
):
    best_avg_pairings = float("inf")
    best_coverage_pct = -1.0
    greedy_pairings = []
    greedy_coverages = []
    airline_histories = defaultdict(list)
    expected_airlines = tuple((checkpoint_metadata or {}).get("airlines") or [config.AIRLINE])

    # stage1/2는 constraint_sampler가 있어도 stage1_constraint/stage2_constraint로
    # 항공사별 고정값을 주입할 뿐 실제 랜덤 샘플링이 아님 -- 진짜 랜덤 샘플링(sample_constraint,
    # STAGE3_CONSTRAINT_RANGES 사용)은 stage 3에서만 쓰임.
    if constraint_sampler and stage == 3:
        sampler_label = " [constraint 랜덤 샘플링]"
    elif constraint_sampler:
        sampler_label = " [항공사별 고정 constraint 주입]"
    else:
        sampler_label = ""

    print(f"\n{'='*60}")
    print(f"Curriculum Stage {stage}: max_duty_periods={constraint_override['max_duty_periods']}, "
          f"max_pairing_days={constraint_override['max_pairing_days']}"
          + sampler_label)
    print(f"{'='*60}")

    # [B 패턴 극복 조치] Stage 3 진입 시 파괴적 망각 차단을 위한 Backbone 가중치 보호 LR 스케일링
    if stage == 3:
        print(f"--> [CRITICAL] Stage 3 Detected: Scaling down learning rate by 30% to stabilize backbone.")
        for g in optimizer.param_groups:
            g["lr"] *= 0.3

    params = list(encoder.parameters()) + list(decoder.parameters())

    for ep in range(n_episodes):
        sample = flight_sampler()
        if sample is None:
            continue
        flights, origins, dests, dep_times, arr_times, fly_times, base_airport, sampled_airline = _unpack_flight_sample(
            sample, require_airline=(constraint_sampler is not None), default_airline=config.AIRLINE
        )

        # [B 패턴 극복 조치] Stage 3에서 30% 확률로 Stage 2 기준 제약을 주입하여 과거 환경 기억 보존 (Continual Replay)
        if stage == 3 and base_stage2_constraint is not None and random.random() < 0.3:
            replay_constraint = (base_stage2_constraint(sampled_airline, base_airport)
                                 if callable(base_stage2_constraint) else base_stage2_constraint)
            c = {
                **replay_constraint,
                "max_duty_periods": 2,
            }
        else:
            c = (constraint_sampler(sampled_airline, base_airport)
                 if constraint_sampler else constraint_override)

        c = {**c, "base_airport": base_airport}  # 에피소드별 base 주입
        c = _prepare_cpp_constraint(flights, c)
        
        # 선택된 복원/샘플링 제약조건 사전(c)을 기반으로 정확히 텐서를 빌드하여 FiLM 정렬 유지
        c_tensor = constraint_to_tensor(c, device=DEVICE)
        encoded  = encoder(origins, dests, dep_times, arr_times, fly_times, c_tensor)

        reward_s, log_probs, entropies, metrics_s = run_episode(
            flights, c, encoder, decoder, encoded, greedy=False, stage=stage
        )
        if len(log_probs) == 0:
            continue

        with torch.no_grad():
            encoded_g = encoder(origins, dests, dep_times, arr_times, fly_times, c_tensor)
            reward_g, _, _, metrics_g = run_episode(
                flights, c, encoder, decoder, encoded_g, greedy=True, stage=stage
            )

        greedy_pairings.append(metrics_g["n_pairings"])
        greedy_coverages.append(metrics_g["coverage_pct"])
        airline_histories[sampled_airline].append(metrics_g)
        advantage = (reward_s - reward_g) / (abs(reward_g) + 1e-6)

        entropy_coef = max(entropy_start * (1.0 - ep / n_episodes), entropy_end)
        loss = torch.stack([
            -lp * advantage - entropy_coef * ent
            for lp, ent in zip(log_probs, entropies)
        ]).sum()

        avg25 = sum(greedy_pairings[-25:]) / min(len(greedy_pairings), 25)
        coverage25 = sum(greedy_coverages[-25:]) / min(len(greedy_coverages), 25)
        selection_score = _airline_selection_score(airline_histories, expected_airlines)

        if selection_score is not None:
            if _is_better_checkpoint(
                selection_score["coverage_pct"], selection_score["avg_pairings"],
                best_coverage_pct, best_avg_pairings,
            ):
                best_avg_pairings = selection_score["avg_pairings"]
                best_coverage_pct = selection_score["coverage_pct"]
                ckpt_path = os.path.join(save_dir, f"stage{stage}_best.pt")
                torch.save({
                    "encoder":           encoder.state_dict(),
                    "decoder":           decoder.state_dict(),
                    "stage":             stage,
                    "episode":           ep,
                    "best_avg_pairings": best_avg_pairings,
                    "best_coverage_pct": best_coverage_pct,
                    "best_per_airline": selection_score["per_airline"],
                    "time_basis":        "turkish_native" if config.AIRLINE == "turkish" else "utc",
                    **(checkpoint_metadata or {}),
                }, ckpt_path)
                wandb.save(ckpt_path)

        # 위 greedy 지표와 동일한 pre-update 파라미터를 저장한 뒤 학습을 진행함.
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        wandb.log({
            f"stage{stage}/greedy_pairings":   metrics_g["n_pairings"],
            f"stage{stage}/sample_pairings":   metrics_s["n_pairings"],
            f"stage{stage}/greedy_deadheads":  metrics_g["n_deadheads"],
            f"stage{stage}/greedy_avg_legs":   metrics_g.get("avg_legs", 0),
            f"stage{stage}/greedy_avg_overnight": metrics_g.get("avg_overnight", 0),
            f"stage{stage}/sample_reward":     reward_s,
            f"stage{stage}/avg25":             avg25,
            f"stage{stage}/coverage25":        coverage25,
            f"stage{stage}/advantage":         advantage,
            f"stage{stage}/loss":              loss.item(),
            f"stage{stage}/entropy_coef":      entropy_coef,
            f"stage{stage}/best_avg25":        best_avg_pairings if best_avg_pairings < float("inf") else avg25,
            f"stage{stage}/gap_weight":        decoder.gap_weight.item(),
        }, step=global_step_offset + ep)

        if ep % 25 == 0:
            print(
                f"  Ep {ep:4d} | "
                f"sample: p={metrics_s['n_pairings']:3d} dh={metrics_s['n_deadheads']:3d} | "
                f"greedy: p={metrics_g['n_pairings']:3d} legs={metrics_g.get('avg_legs', 0):.2f} "
                f"(avg25={avg25:5.1f}, cov25={coverage25:5.1f}%) | "
                f"adv: {advantage:6.3f}"
            )

    print(f"  → best avg pairings: {best_avg_pairings:.1f}  "
          f"(saved: checkpoints/stage{stage}_best.pt)")
    return best_avg_pairings


def train(phase2_only=False, multi_airline=False, skip_film=False, skip_decoder_constraint=False,
          ckpt_dir=None, from_stage2=False, turkish_files=None, dual_weight=None, dual_mode="real",
          airport_universe_paths=None):
    WINDOW_DAYS = (
        max(config.AIRLINE_WINDOW_DAYS[a] for a in config.MULTI_AIRLINES)
        if multi_airline else config.AIRLINE_WINDOW_DAYS[config.AIRLINE]
    )

    # 2x2 FiLM 인과성 실험(C/D/C'/D') — 디코더의 constraint 직접 concat 경로를
    # 원천 차단할지 여부. 이 프로세스 안에서 학습·rollout 전체(train.py, rollout.py
    # 둘 다 동일한 RL/utils.py를 import하므로)에 즉시 반영된다.
    set_skip_decoder_constraint(skip_decoder_constraint)

    _select_environment("multi" if multi_airline else config.AIRLINE)

    if multi_airline:
        airlines = list(config.MULTI_AIRLINES)
        all_paths = [config.AIRLINE_DATA[a] for a in airlines]
        airport_map = build_airport_map(all_paths)
        all_base_ids = {a: bases_to_ids(config.AIRLINE_BASES[a], airport_map) for a in airlines}
        n_airports = len(airport_map)
        print(f"airports: {n_airports}개 (통합), airlines: {airlines}")
    else:
        airline_bases = config.AIRLINE_BASES[config.AIRLINE]
        if config.AIRLINE == "turkish":
            from turkish.loader_turkish import (
                parse_legs_dir, build_airport_map_turkish, load_flights_rolling_turkish,
                ZEREN_FEB_FILE, ZEREN_FEB_WINDOW,
            )
            DATA_PATH    = None  # Turkish는 단일 CSV 없음
            # turkish_files 미지정 시 Zeren Feb 벤치마크 윈도우(15,742편, 목표 15,738 대비
            # 오차 0.03%) 기본 사용
            if turkish_files is None:
                _turkish_df = parse_legs_dir(config.AIRLINE_DATA["turkish"], files=[ZEREN_FEB_FILE], date_range=ZEREN_FEB_WINDOW)
            else:
                _turkish_df = parse_legs_dir(config.AIRLINE_DATA["turkish"], files=turkish_files)
            airport_map  = build_airport_map_turkish(df=_turkish_df)
        else:
            DATA_PATH   = config.AIRLINE_DATA[config.AIRLINE]
            if airport_universe_paths:
                # 학습은 DATA_PATH만 쓰되, 공항 embedding 사전은 seasonal transfer 평가에
                # 쓸 다른 달(2월/8월 등)까지 합쳐 미리 잡는다. 학습에 안 나온 공항은 그냥
                # 학습되지 않은 embedding으로 남고, 이게 "본 적 없는 공항"의 정확한 표현이다.
                # (미리 안 잡으면 evaluate_ip.py가 unknown airport로 평가를 거부함)
                _map_paths = [DATA_PATH] + [p for p in airport_universe_paths if p != DATA_PATH]
                airport_map = build_airport_map(_map_paths)
                print(f"airport universe: {len(_map_paths)}개 경로 통합 -> {_map_paths}")
            else:
                airport_map = build_airport_map(DATA_PATH)
        base_ids   = bases_to_ids(airline_bases, airport_map)
        n_airports = len(airport_map)
        print(f"airports: {n_airports}개, airline: {config.AIRLINE}, bases: {airline_bases}")
        if config.AIRLINE == "turkish":
            # 두 Istanbul base 중 하나에서 시작하고 HB1/HB2 어느 쪽으로든 복귀함
            _CONSTRAINT_FN["turkish"] = lambda b, _hb=base_ids: get_turkish_constraints_hb(b, base_ids=_hb)

    encoder = FlightEncoder(
        n_airports=n_airports,
        constraint_dim=len(FILM_CONSTRAINT_KEYS),
        airport_emb_dim=32,
        d_model=128,
        use_film_before=not skip_film,
        use_film_after=not skip_film,
    ).to(DEVICE)
    decoder   = PointerDecoder(d_model=128, airport_emb_dim=32,
                               constraint_dim=len(FILM_CONSTRAINT_KEYS)).to(DEVICE)
    
    if skip_film:
        params    = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = optim.Adam(params, lr=1e-4)
    else:
        optimizer = optim.Adam([
            {"params": encoder.film_params(),     "lr": 1e-3},
            {"params": encoder.non_film_params(), "lr": 1e-4},
            {"params": decoder.parameters(),      "lr": 1e-4},
        ])

    tag = "multi-airline" if multi_airline else config.AIRLINE
    tag += "-nofilm" if skip_film else ""
    tag += "-nodecoderc" if skip_decoder_constraint else ""
    tag += "-nodual" if dual_weight == 0 else ""
    tag += f"-dual-{dual_mode}"
    run_name = f"phase2-{dual_mode}" if phase2_only else tag
    wandb.init(
        project="ASCP-2026-journal",
        mode="online",
        name=run_name,
        config={
            "airline":            "multi" if multi_airline else config.AIRLINE,
            "multi_airline":      multi_airline,
            "window_days":        WINDOW_DAYS,
            "phase2_lp_interval": config.PHASE2_LP_INTERVAL,
            "phase2_pool_rollouts": config.PHASE2_POOL_ROLLOUTS,
            "phase2_dual_weight": dual_weight if dual_weight is not None else config.PHASE2_DUAL_WEIGHT,
            "phase2_dual_mode":   dual_mode,
            "phase2_n_episodes":  config.PHASE2_N_EPISODES,
            "lr":                 1e-4,
            "device":             str(DEVICE),
            "skip_film":              skip_film,
            "skip_decoder_constraint": skip_decoder_constraint,
        },
        resume="allow",
    )

    save_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints", wandb.run.id)
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_metadata = {
        "checkpoint_schema_version": 2,
        "airport_map": dict(airport_map),
        "airport_map_hash": airport_map_hash(airport_map),
        "n_airports": n_airports,
        "airline": "multi" if multi_airline else config.AIRLINE,
        "airlines": list(airlines) if multi_airline else [config.AIRLINE],
        "multi_airline": bool(multi_airline),
        "skip_film": bool(skip_film),
        "skip_decoder_constraint": bool(skip_decoder_constraint),
        "window_days": WINDOW_DAYS,
        "max_time": WINDOW_DAYS * 24,
        "time_basis": "turkish_native" if (not multi_airline and config.AIRLINE == "turkish") else "utc",
    }

    import pandas as pd

    if multi_airline:
        _df_caches = {}
        _max_offsets = {}
        for a in airlines:
            p = config.AIRLINE_DATA[a]
            df = pd.read_csv(p, usecols=["ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "FL_DATE"]).dropna()
            df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="mixed")
            _df_caches[a]   = df
            _max_offsets[a] = max(0, df["FL_DATE"].nunique() - WINDOW_DAYS)

        _selected_airline = ["delta"]

        def flight_sampler():
            # airline 선택은 여기서 한 번만 확정하고, (base, offset) 조합이 빈 윈도우로
            # 실패하면 같은 airline으로 재시도한다 — 항공사별로 데이터량 차이가 커서
            # (delta 73,836편 vs alaska/jetblue 20,744~24,443편) base/offset을 airline과
            # 함께 다시 뽑으면 데이터가 적은 항공사일수록 실패→스킵이 잦아져 실제 학습에
            # 쓰이는 에피소드 비율이 delta 쪽으로 쏠린다.
            airline      = random.choice(airlines)
            _selected_airline[0] = airline
            for _ in range(20):
                base_airport = random.choice(all_base_ids[airline])
                offset_days  = random.randint(0, _max_offsets[airline])
                flights = load_flights_rolling(
                    config.AIRLINE_DATA[airline], WINDOW_DAYS, offset_days, airport_map,
                    base_airport=base_airport,
                    n_max=config.EPISODE_MAX_FLIGHTS,
                    df=_df_caches[airline],
                )
                if flights and any(f["origin"] == base_airport for f in flights):
                    origins, dests, dep_times, arr_times, fly_times = flights_to_tensors(flights, WINDOW_DAYS * 24.0, device=DEVICE)
                    return flights, origins, dests, dep_times, arr_times, fly_times, base_airport, airline
            return None

        _first_airline = airlines[0]
        _first_base = all_base_ids[_first_airline][0]
        base_constraint = _constraint_for_episode(_first_airline, _first_base)

    else:
        if config.AIRLINE == "turkish":
            _df_cache  = _turkish_df
            total_days = _df_cache["dep_date_utc"].nunique()
            max_offset = max(0, total_days - WINDOW_DAYS)

            def flight_sampler():
                base_airport = random.choice(base_ids)
                offset_days  = random.randint(0, max_offset)
                flights = load_flights_rolling_turkish(
                    WINDOW_DAYS, offset_days, airport_map,
                    base_airport=base_airport, df=_df_cache,
                    n_max=config.EPISODE_MAX_FLIGHTS,
                )
                if not flights:
                    return None
                if not any(f["origin"] == base_airport for f in flights):
                    return None
                origins, dests, dep_times, arr_times, fly_times = flights_to_tensors(flights, WINDOW_DAYS * 24.0, device=DEVICE)
                return flights, origins, dests, dep_times, arr_times, fly_times, base_airport, config.AIRLINE
        else:
            DATA_PATH = config.AIRLINE_DATA[config.AIRLINE]
            _df_cache = pd.read_csv(DATA_PATH, usecols=["ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "FL_DATE"]).dropna()
            _df_cache["FL_DATE"] = pd.to_datetime(_df_cache["FL_DATE"], format="mixed")
            total_days = _df_cache["FL_DATE"].nunique()
            max_offset = max(0, total_days - WINDOW_DAYS)

            def flight_sampler():
                base_airport = random.choice(base_ids)
                offset_days  = random.randint(0, max_offset)
                flights = load_flights_rolling(
                    DATA_PATH, WINDOW_DAYS, offset_days, airport_map,
                    base_airport=base_airport,
                    n_max=config.EPISODE_MAX_FLIGHTS,
                    df=_df_cache,
                )
                if not flights:
                    return None
                if not any(f["origin"] == base_airport for f in flights):
                    return None
                origins, dests, dep_times, arr_times, fly_times = flights_to_tensors(flights, WINDOW_DAYS * 24.0, device=DEVICE)
                return flights, origins, dests, dep_times, arr_times, fly_times, base_airport, config.AIRLINE

        base_constraint = _CONSTRAINT_FN[config.AIRLINE](base_ids[0])

    _stage3_base = {**base_constraint, "max_duty_periods": 4, "max_pairing_days": WINDOW_DAYS - 1}
    def sample_constraint(sampled_airline, base_airport):
        r = config.STAGE3_CONSTRAINT_RANGES
        if multi_airline:
            if sampled_airline != _selected_airline[0]:
                raise RuntimeError("flight sample의 항공사와 constraint 항공사가 일치하지 않음")
            base = {**_constraint_for_episode(sampled_airline, base_airport)}
        else:
            base = _stage3_base
        if random.random() < config.STAGE3_REAL_CONSTRAINT_INJECT_PROB:
            # 주입할 constraint의 항공사는 이 episode의 flight 데이터 항공사(_selected_airline)와
            # 무관하게 4개(turkish 포함) 중에서 고른다 — turkish는 flight_sampler()의 airlines
            # 풀에 애초에 없어서(로더가 달라 별도 취급) _selected_airline로만 묶으면 turkish
            # 실제값이 학습 중 한 번도 주입되지 않는다. Table3 ③/④'에서 이미 검증된 대로
            # "다른 항공사 flight + turkish constraint" 조합은 정상 동작한다.
            inject_airline = random.choice(list(_CONSTRAINT_FN.keys())) if multi_airline else config.AIRLINE
            real = _CONSTRAINT_FN[inject_airline](0)
            return {**base, **{k: real[k] for k in FILM_CONSTRAINT_KEYS}}
        return {
            **base,
            "max_duty":         random.uniform(*r["max_duty"]),
            "min_rest":         random.uniform(*r["min_rest"]),
            "min_conn":         random.uniform(*r["min_conn"]),
            "max_conn":         random.uniform(*r["max_conn"]),
            "max_legs":         random.randint(*r["max_legs"]),
            "max_duty_periods": random.randint(*r["max_duty_periods"]),
            "max_pairing_days": random.randint(*r["max_pairing_days"]),
        }


    def stage1_constraint(sampled_airline, base_airport):
        return _constraint_for_episode(
            sampled_airline, base_airport,
            max_duty_periods=1, max_pairing_days=1, base_penalty=5.0,
        )

    def stage2_constraint(sampled_airline, base_airport):
        return _constraint_for_episode(
            sampled_airline, base_airport, base_penalty=5.0,
        )

    _s3_best     = float("inf")  
    _s3_ckpt_dir = save_dir      

    if phase2_only:
        _s3_ckpt_dir = ckpt_dir if ckpt_dir else save_dir
        ckpt_path = os.path.join(_s3_ckpt_dir, "stage3_best.pt")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        ckpt_n_airports = ckpt["encoder"]["airport_emb.weight"].shape[0]
        if ckpt_n_airports != n_airports:
            raise ValueError(
                "phase2-only checkpoint의 공항 embedding 크기가 현재 airport map과 다름: "
                f"checkpoint={ckpt_n_airports}, current={n_airports}"
            )
        if ckpt.get("airport_map_hash") != checkpoint_metadata["airport_map_hash"]:
            raise ValueError("phase2-only checkpoint의 airport map이 현재 학습 설정과 다름")
        if bool(ckpt.get("skip_film", False)) != bool(skip_film):
            raise ValueError("phase2-only checkpoint와 --skip-film 설정이 다름")
        if bool(ckpt.get("skip_decoder_constraint", False)) != bool(skip_decoder_constraint):
            raise ValueError("phase2-only checkpoint와 --skip-decoder-constraint 설정이 다름")
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        _s3_best = float(ckpt.get("best_avg_pairings", float("inf")))
        print(f"stage3_best.pt 로드 완료: {ckpt_path} → Phase 2만 실행 (n_airports={n_airports})")

    if not phase2_only:
        if from_stage2:
            _s2_load_dir = ckpt_dir
            if not _s2_load_dir:
                raise ValueError("--from-stage2 사용 시 --ckpt-dir로 stage2_best.pt 폴더를 지정해야 합니다.")
            _s2_ckpt_path = os.path.join(_s2_load_dir, "stage2_best.pt")
            _s2_ckpt = torch.load(_s2_ckpt_path, map_location=DEVICE, weights_only=True)
            if _s2_ckpt.get("airport_map_hash") != checkpoint_metadata["airport_map_hash"]:
                raise ValueError("stage2 checkpoint의 airport map이 현재 학습 설정과 다름")
            if bool(_s2_ckpt.get("skip_film", False)) != bool(skip_film):
                raise ValueError("stage2 checkpoint와 --skip-film 설정이 다름")
            if bool(_s2_ckpt.get("skip_decoder_constraint", False)) != bool(skip_decoder_constraint):
                raise ValueError("stage2 checkpoint와 --skip-decoder-constraint 설정이 다름")
            encoder.load_state_dict(_s2_ckpt["encoder"])
            decoder.load_state_dict(_s2_ckpt["decoder"])
            print(f"stage2_best.pt 로드: {_s2_ckpt_path} → Stage 3부터 실행")
        else:
            # ── Stage 1: 단일 duty (overnight 없음) ──────────────────────────
            # base_penalty는 stage1/2에서 5.0(원래값) 고정 — stage3/phase2부터 config.py의
            # 현재값(500.0)을 그대로 물려받는다. x2gcdva5(stage1/2, p5)를 이어받는 기존
            # run들과 동일 조건을 신규 seed에서도 재현하기 위함.
            stage1_c = {**base_constraint, "max_duty_periods": 1, "max_pairing_days": 1,
                       "base_penalty": 5.0}
            run_curriculum_stage(1, encoder, decoder, optimizer,
                                 n_episodes=1000, constraint_override=stage1_c,
                                 save_dir=save_dir, flight_sampler=flight_sampler,
                                 constraint_sampler=stage1_constraint if multi_airline else None,
                                 global_step_offset=0,
                                 entropy_start=0.30, entropy_end=0.005,
                                 checkpoint_metadata=checkpoint_metadata)

            # ── Stage 2: full multi-day ───────────────────────────────────────
            stage2_c = {**base_constraint, "max_duty_periods": 2, "max_pairing_days": WINDOW_DAYS - 1, "base_penalty": 5.0}
            run_curriculum_stage(2, encoder, decoder, optimizer,
                                 n_episodes=2000, constraint_override=stage2_c,
                                 save_dir=save_dir, flight_sampler=flight_sampler,
                                 constraint_sampler=stage2_constraint if multi_airline else None,
                                 global_step_offset=1000,
                                 entropy_start=0.02, entropy_end=0.005,
                                 checkpoint_metadata=checkpoint_metadata)

        # ── Stage 3: 7개 constraint 전체 랜덤 augmentation (FiLM 학습) ───
        _s3_offset = 0 if from_stage2 else 3000
        # [B 패턴 극복 조치] base_stage2_constraint 인자에 base_constraint를 주입하여 Stage 2 제약 복원 유도
        _s3_best = run_curriculum_stage(3, encoder, decoder, optimizer,
                             n_episodes=2000, constraint_override=_stage3_base,
                             save_dir=save_dir, flight_sampler=flight_sampler,
                             constraint_sampler=sample_constraint,
                             global_step_offset=_s3_offset,
                             entropy_start=0.01, entropy_end=0.005,
                             base_stage2_constraint=stage2_constraint if multi_airline else base_constraint,
                             checkpoint_metadata=checkpoint_metadata)

        _s3_ckpt = torch.load(os.path.join(save_dir, "stage3_best.pt"), map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(_s3_ckpt["encoder"])
        decoder.load_state_dict(_s3_ckpt["decoder"])
        print(f"Phase 2 시작: stage3_best.pt 로드 (best_avg={_s3_ckpt.get('best_avg_pairings', 0):.1f})")

    # ── FiLM 검증 공용 data setup ─────────────────────────────────────────
    if multi_airline:
        _val_data_path     = config.AIRLINE_DATA["delta"]
        _val_df            = _df_caches["delta"]
        _val_base          = all_base_ids["delta"][0]
        _val_bases_save    = config.AIRLINE_BASES["delta"]
        _val_constraint_fn = _CONSTRAINT_FN["delta"]
    else:
        _val_data_path     = DATA_PATH
        _val_df            = _df_cache
        _val_base          = base_ids[0]
        _val_bases_save    = airline_bases
        _val_constraint_fn = _CONSTRAINT_FN[config.AIRLINE]

    if not multi_airline and config.AIRLINE == "turkish":
        val_flights = load_flights_rolling_turkish(
            WINDOW_DAYS, 0, airport_map,
            base_airport=_val_base, df=_val_df,
            n_max=config.EPISODE_MAX_FLIGHTS,
        )
    else:
        val_flights = load_flights_rolling(
            _val_data_path, WINDOW_DAYS, 0, airport_map,
            base_airport=_val_base,
            n_max=config.EPISODE_MAX_FLIGHTS,
            df=_val_df,
        )
    val_origins, val_dests, val_dep_times, val_arr_times, val_fly_times = flights_to_tensors(val_flights, WINDOW_DAYS * 24.0, device=DEVICE)

    N_FILM_ROLLOUTS = 10

    def _film_validation(label):
        """FiLM 학습 검증: constraint 변화 시 행동 변화 여부 측정.
        (1) max_duty_periods 1→4: overnight 가능 횟수 변화 → pairings 수 단조 감소 기대
        (2) max_legs 2→8: duty당 허용 leg 수 변화 → avg_legs 변화 기대
        stochastic rollout × N_FILM_ROLLOUTS 평균."""
        encoder.eval(); decoder.eval()
        print()
        print("=" * 60)
        print(f"FiLM 검증 ({label}): 같은 flights, 다른 constraints")
        print("=" * 60)
        with torch.no_grad():
            print(f"  [max_duty_periods 변화] (합격: dp=1→4 pairings ≥30% 감소)")
            for dp in [1, 2, 3, 4]:
                val_c = {**_val_constraint_fn(_val_base), "max_duty_periods": dp,
                         "max_pairing_days": WINDOW_DAYS}
                val_enc = encoder(val_origins, val_dests, val_dep_times, val_arr_times,
                                  val_fly_times, constraint_to_tensor(val_c, device=DEVICE))
                p_list, dh_list, cov_list = [], [], []
                for _ in range(N_FILM_ROLLOUTS):
                    _, _, _, m = run_episode(val_flights, val_c, encoder, decoder, val_enc, greedy=False)
                    p_list.append(m["n_pairings"])
                    dh_list.append(m["n_deadheads"])
                    cov_list.append(m["coverage_pct"])
                print(f"    max_duty_periods={dp} → "
                      f"pairings(avg{N_FILM_ROLLOUTS})={sum(p_list)/len(p_list):.1f}  "
                      f"deadheads={sum(dh_list)/len(dh_list):.1f}  "
                      f"coverage={sum(cov_list)/len(cov_list):.1f}%")

            print(f"  [max_legs 변화] (합격: legs=2→8에서 avg_legs 뚜렷이 증가)")
            for ml in [2, 4, 8]:
                val_c = {**_val_constraint_fn(_val_base), "max_legs": ml,
                         "max_duty_periods": 2, "max_pairing_days": WINDOW_DAYS}
                val_enc = encoder(val_origins, val_dests, val_dep_times, val_arr_times,
                                  val_fly_times, constraint_to_tensor(val_c, device=DEVICE))
                p_list, l_list, on_list = [], [], []
                for _ in range(N_FILM_ROLLOUTS):
                    _, _, _, m = run_episode(val_flights, val_c, encoder, decoder, val_enc, greedy=False)
                    p_list.append(m["n_pairings"])
                    l_list.append(m.get("avg_legs", 0))
                    on_list.append(m.get("avg_overnight", 0))
                print(f"    max_legs={ml} → "
                      f"pairings(avg{N_FILM_ROLLOUTS})={sum(p_list)/len(p_list):.1f}  "
                      f"avg_legs={sum(l_list)/len(l_list):.2f}  "
                      f"avg_overnight={sum(on_list)/len(on_list):.2f}")
        encoder.train(); decoder.train()

    # Stage 3 FiLM 검증 — Phase 2 전 기준점
    _film_validation("Stage 3 best")

    # ── Phase 2: CG dual feedback ──────────────────────────────────────
    phase2_c = {**base_constraint, "max_duty_periods": 2, "max_pairing_days": WINDOW_DAYS - 1}
    phase2_offset = 0 if phase2_only else (2000 if from_stage2 else 1000 + 2000 + 2000)

    # init_best을 Stage3 기록으로 주지 않고 무한대(기본값)로 둠 — Stage3를 못 넘어도
    # phase2_best.pt가 Phase2 자체의 최고점으로 항상 생기게 함(avg_pairings가 dead_time/FTC를
    # 반영 못 하는 지표라 "Stage3를 못 넘음=평가할 가치 없음"으로 단정할 수 없음). Stage3 대비
    # 비교는 아래에서 참고용으로만 출력.
    _p2_best = run_phase2(encoder, decoder, optimizer,
               n_episodes=config.PHASE2_N_EPISODES,
               constraint=phase2_c,
               save_dir=save_dir,
               flight_sampler=flight_sampler,
               global_step_offset=phase2_offset,
               constraint_sampler=sample_constraint,
               dual_weight_override=dual_weight, dual_mode=dual_mode,
               checkpoint_metadata=checkpoint_metadata)

    # ── FiLM 최종 검증: stage3_best.pt 기준 ───────────────────────────
    # (Phase 2가 FiLM 가중치를 덮어썼을 수 있으므로 검증만 stage3_best로 임시 복원해서 확인)
    _film_ckpt = torch.load(os.path.join(_s3_ckpt_dir, "stage3_best.pt"), map_location=DEVICE, weights_only=True)
    encoder.load_state_dict(_film_ckpt["encoder"])
    decoder.load_state_dict(_film_ckpt["decoder"])
    print("FiLM 최종 검증: stage3_best.pt 로드")
    _film_validation("final / stage3_best")

    # ── 최종 모델 선택: Phase 2의 avg_pairings가 Stage3보다 실제로 더 낮을 때만 phase2_best ──
    # phase2_best.pt는 이제(init_best=inf) Stage3를 못 넘어도 항상 생기므로, 파일 존재
    # 여부가 아니라 avg_pairings 값 자체를 직접 비교해야 함(존재 여부만 보면 항상 phase2가
    # 선택돼버려 이 로직이 무력화됨).
    _phase2_ckpt_path = os.path.join(save_dir, "phase2_best.pt")
    _phase2_wins = False
    if os.path.exists(_phase2_ckpt_path):
        _phase2_ckpt = torch.load(_phase2_ckpt_path, map_location=DEVICE, weights_only=True)
        _phase2_wins = _is_better_checkpoint(
            float(_phase2_ckpt.get("best_coverage_pct", -1.0)),
            float(_phase2_ckpt.get("best_avg_pairings", float("inf"))),
            float(_film_ckpt.get("best_coverage_pct", -1.0)),
            float(_film_ckpt.get("best_avg_pairings", float("inf"))),
        )
    if _phase2_wins:
        encoder.load_state_dict(_phase2_ckpt["encoder"])
        decoder.load_state_dict(_phase2_ckpt["decoder"])
        print("최종 모델: legal coverage 우선 비교에서 phase2_best.pt 사용")
    else:
        encoder.load_state_dict(_film_ckpt["encoder"])
        decoder.load_state_dict(_film_ckpt["decoder"])
        print("최종 모델: legal coverage 우선 비교에서 stage3_best.pt 사용")

    # ── 최종 모델 저장 ────────────────────────────────────────────────
    torch.save({
        "encoder":        encoder.state_dict(),
        "decoder":        decoder.state_dict(),
        "n_airports":     n_airports,
        "constraint_dim": len(FILM_CONSTRAINT_KEYS),
        "bases":          _val_bases_save,
        "window_days":    WINDOW_DAYS,
        "max_time":       WINDOW_DAYS * 24,
        "time_basis":     "turkish_native" if config.AIRLINE == "turkish" else "utc",
        **checkpoint_metadata,
    }, os.path.join(save_dir, "model_latest.pt"))
    print(f"\n모델 저장: checkpoints/model_latest.pt")

    wandb.finish(quiet=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="학습 device (예: cpu, cuda, cuda:0, cuda:1)")
    parser.add_argument("--log", default=os.path.join(os.path.dirname(__file__), "..", "log", "train_log.txt"),
                        help="로그 파일 경로 (기본: log/train_log.txt)")
    parser.add_argument("--phase2-only", action="store_true",
                        help="stage3_best.pt 로드 후 Phase 2만 실행")
    parser.add_argument("--from-stage2", action="store_true",
                        help="stage2_best.pt 로드 후 Stage 3 + Phase 2만 실행 (--ckpt-dir 필수)")
    parser.add_argument("--ckpt-dir", default=None,
                        help="--phase2-only: stage3_best.pt 폴더 / --from-stage2: stage2_best.pt 폴더")
    parser.add_argument("--multi-airline", action="store_true",
                        help="Delta/Alaska/JetBlue 세 항공사 데이터로 동시 학습 (통합 airport_map 사용)")
    parser.add_argument("--skip-film", action="store_true",
                        help="FiLM 비활성화 (use_film_before=False, use_film_after=False) — ablation B/D용")
    parser.add_argument("--skip-decoder-constraint", action="store_true",
                        help="디코더가 매 step 직접 보는 constraint_vec(7)을 0으로 고정 — "
                             "2x2 FiLM 인과성 실험(C'/D')용. state_to_vec 차원은 그대로 유지되어 "
                             "체크포인트 구조는 C/D와 동일하게 호환된다.")
    parser.add_argument("--airline", default=None,
                        help="단일 항공사 지정 (delta/alaska/jetblue/turkish). 미지정 시 config.AIRLINE 사용")
    parser.add_argument("--turkish-files", default=None,
                        help="Turkish 학습 시 사용할 .legs 파일 이름 콤마 구분 (예: tt201401.legs). 미지정 시 "
                             "Zeren Feb 벤치마크 윈도우(tt201402.legs, 2/1~3/8, 15,742편) 기본 사용")
    parser.add_argument("--dual-weight", type=float, default=None,
                        help="Phase2 CG dual reward 가중치를 config.PHASE2_DUAL_WEIGHT(기본 0.6) 대신 "
                             "이 값으로 덮어씀. 0을 주면 CG-dual 완전히 비활성화.")
    parser.add_argument(
        "--dual-mode", default="real",
        choices=[
            "zero", "uncovered-only", "shuffled", "real", "coverage-only",
            "robust-real", "robust-shuffled", "net", "coverage_only",
        ],
        help="Phase2 ablation. zero=LP 계산은 유지하고 신호 0, uncovered-only=artificial flight만 1, "
             "shuffled=real 신호를 flight 간 섞음, real=coverage-excess net dual. "
             "robust-real=미커버는 1로 두고 커버 가능한 flight만 95백분위 정규화, "
             "robust-shuffled=robust-real 값의 flight 대응을 섞는 대조군. "
             "coverage-only와 legacy net/coverage_only도 호환함.",
    )
    parser.add_argument("--data-path", default=None,
                        help="CSV 경로. 미지정 시 config.AIRLINE_DATA[airline] 사용. "
                             "delta-small 등 대체 데이터셋으로 학습/이어받기할 때 지정")
    parser.add_argument("--seed", type=int, default=None,
                        help="random/torch seed 고정 (재현 가능한 학습용). 미지정 시 시드 고정 안 함")
    parser.add_argument("--airport-universe-paths", default=None,
                        help="공항 embedding 사전을 만들 때 합칠 CSV 경로들(콤마 구분). 단일 항공사 학습 전용. "
                             "학습 rollout은 --data-path만 쓰고, 사전만 넓힌다. "
                             "예: 1월로 학습하되 2, 8월 seasonal eval까지 하려면 "
                             "--airport-universe-paths RL/data/delta_2019_02.csv,RL/data/delta_2019_08.csv")
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"seed: {args.seed}")
    if args.airline:
        config.AIRLINE = args.airline
    if args.data_path:
        # config.AIRLINE_DATA를 덮어써야 train() 안의 DATA_PATH/airport_map이 이 경로를 따라감
        config.AIRLINE_DATA[config.AIRLINE] = args.data_path
        print(f"data_path 지정: {config.AIRLINE} → {args.data_path}")
    _set_device(args.device)
    print(f"device: {DEVICE}")
    print(f"time_basis: {'Turkish 전용 시각' if config.AIRLINE == 'turkish' and not args.multi_airline else 'UTC (BTS 고정)'}")
    print(f"log: {args.log}")
    _turkish_files = [f.strip() for f in args.turkish_files.split(",")] if args.turkish_files else None
    _universe_paths = (
        [p.strip() for p in args.airport_universe_paths.split(",") if p.strip()]
        if args.airport_universe_paths else None
    )
    if _universe_paths and args.multi_airline:
        parser.error("--airport-universe-paths는 단일 항공사 학습 전용입니다 "
                     "(--multi-airline은 이미 3항공사 통합 사전을 씀)")
    train(phase2_only=args.phase2_only, multi_airline=args.multi_airline, skip_film=args.skip_film,
          skip_decoder_constraint=args.skip_decoder_constraint,
          ckpt_dir=args.ckpt_dir, from_stage2=args.from_stage2, turkish_files=_turkish_files,
          dual_weight=args.dual_weight, dual_mode=args.dual_mode,
          airport_universe_paths=_universe_paths)
