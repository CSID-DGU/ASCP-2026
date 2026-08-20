import os
import sys
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RL"))
import torch
import torch.optim as optim
from torch.distributions import Categorical
import wandb

from model import FlightEncoder, PointerDecoder
from loader import build_airport_map, bases_to_ids, load_flights_rolling
import environment as _env_default
from turkish.environment_turkish import get_mask as _get_mask_turkish, step as _step_turkish, final_reward as _final_reward_turkish
from turkish.constraints_turkish import get_turkish_constraints as get_turkish_constraints_hb
from constraints import (
    get_delta_constraints, get_alaska_constraints,
    get_jetblue_constraints,
    FILM_CONSTRAINT_KEYS,
)

get_mask, step, final_reward = _env_default.get_mask, _env_default.step, _env_default.final_reward


def _select_environment(airline):
    """airline에 맞는 get_mask/step/final_reward 구현으로 전환. run_episode 등 이 모듈의 get_mask/step/final_reward를
    참조하는 모든 호출부에 즉시 반영됨 (모듈 전역 rebind)."""
    global get_mask, step, final_reward
    if airline == "turkish":
        get_mask, step, final_reward = _get_mask_turkish, _step_turkish, _final_reward_turkish
    else:
        get_mask, step, final_reward = _env_default.get_mask, _env_default.step, _env_default.final_reward


_CONSTRAINT_FN = {
    "delta":   get_delta_constraints,
    "alaska":  get_alaska_constraints,
    "jetblue": get_jetblue_constraints,
    "turkish": get_turkish_constraints_hb,  # Turkish 규정값 사용, CPP 동일 base 복귀 계약 유지
}
from state import init_state
from base_reach import build_base_reach, can_reach_base
from utils import flights_to_tensors, constraint_to_tensor, state_to_vec, flight_gap_bias, set_skip_decoder_constraint
import config

DEVICE = torch.device("cpu")  # train() 호출 전 _set_device()로 설정
USE_UTC = False  # dep_time UTC 앵커링 여부 — --use-utc로 켬


def _set_device(device_str: str):
    global DEVICE
    DEVICE = torch.device(device_str)


def _prepare_cpp_constraint(flights, constraint):
    """모든 학습 episode에 CPP base 복귀 조건과 reachability를 구성함."""
    c = dict(constraint)
    base = c["base_airport"]
    if c.get("_base_reach") is not None and c.get("_base_reach_base") == base:
        return c
    # 같은 episode와 base에서 계산한 reachability는 sample/greedy rollout이 공유함.
    c["_base_reach"] = build_base_reach(flights, base, c)
    c["_base_reach_base"] = base
    return c


def run_episode(flights, constraint, encoder, decoder, encoded, greedy=False):
    """
    Returns:
        total_reward, log_probs, entropies, metrics dict
        metrics: {n_pairings, n_deadheads, n_uncovered, coverage_pct}
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

    max_steps = len(flights) * 20  # 무한루프 방지 (flight당 최대 20 step)
    step_count = 0
    while True:
        step_count += 1
        if step_count > max_steps:
            break
        mask_list = get_mask(state, flights, assigned, constraint)
        mask = torch.tensor(mask_list, dtype=torch.float32).to(DEVICE)

        # 합법 action이 없으면 임의 위치 이동 없이 미커버 상태로 episode를 종료함.
        no_flight     = sum(mask_list[:-2]) == 0
        no_end_duty   = mask_list[-2] == 0
        no_end_pairing = mask_list[-1] == 0
        if no_flight and no_end_duty and no_end_pairing:
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if not unassigned:
                break
            # CPP에서 합법 action이 없으면 relocation하지 않고 미커버 상태로 종료함.
            n_zero_mask += 1
            break

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
            state, r, done = step(state, action, flights, assigned, constraint)
            total_reward += r
            if done:
                break
            continue

        # flight action
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

    def flush_pairing(is_forced=False):
        if len(current_legs) < 1 or pairing_dep is None:
            return
        elapsed   = pairing_last_arr - pairing_dep
        n_legs    = len(current_legs)
        # dual pool에도 완결된 CPP pairing만 column으로 저장함.
        if flight_by_id[current_legs[0]]["origin"] != episode_base \
                or flight_by_id[current_legs[-1]]["dest"] != episode_base:
            raise ValueError("base로 복귀하지 않은 pairing은 dual pool에 저장할 수 없습니다.")
        if n_legs < constraint["min_pairing_legs"]:
            raise ValueError("최소 leg 수를 충족하지 않은 pairing은 dual pool에 저장할 수 없습니다.")
        if elapsed / 24.0 > constraint["max_pairing_days"]:
            raise ValueError("최대 pairing 기간을 초과한 pairing은 dual pool에 저장할 수 없습니다.")
        dead_time = max(elapsed - pairing_fly - pairing_rest, 0.0)
        cost      = (dead_time
                     - _LEG_BONUS_IP * max(len(current_legs) - 1, 0)
                     + (_DEADHEAD_PENALTY_IP if is_forced else 0.0)
                     + _PAIRING_FIXED_COST)
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
        base_flights = [f for f in candidates if f["origin"] == episode_base]
        # 수동 시작 flight도 decoder와 같은 복귀 가능성 검사를 통과해야 함.
        return [f for f in base_flights if can_reach_base(
            constraint["_base_reach"], f, f["dep_time"],
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

    max_steps  = len(flights) * 20
    step_count = 0

    while True:
        step_count += 1
        if step_count > max_steps:
            break

        mask_list = get_mask(state, flights, assigned, constraint)
        mask      = torch.tensor(mask_list, dtype=torch.float32).to(DEVICE)

        if sum(mask_list[:-2]) == 0 and mask_list[-2] == 0 and mask_list[-1] == 0:
            # 미복귀 partial pairing은 CPP column으로 저장하지 않음.
            break

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


def _collect_pool(flights, constraint, encoder, decoder, encoded, n_rollouts):
    constraint = _prepare_cpp_constraint(flights, constraint)
    # Exclude pairings that do not return to base -- the restricted LP of
    # Eq. (2) is defined over Omega(c), and its duals mu^cov/nu^exc (Eq. 9)
    # should not be computed from infeasible columns.
    pool = {}
    for _ in range(n_rollouts):
        for p in _rollout_with_pairings(flights, constraint, encoder, decoder, encoded):
            if not p["ends_at_base"]:
                continue
            key = tuple(sorted(p["legs"]))
            if key not in pool or p["cost"] < pool[key]["cost"]:
                pool[key] = p
    for p in _rollout_with_pairings(flights, constraint, encoder, decoder, encoded, greedy=True):
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

    max_steps  = len(flights) * 20
    step_count = 0

    while True:
        step_count += 1
        if step_count > max_steps:
            break

        mask_list  = get_mask(state, flights, assigned, constraint)
        mask       = torch.tensor(mask_list, dtype=torch.float32).to(DEVICE)

        no_flight      = sum(mask_list[:-2]) == 0
        no_end_duty    = mask_list[-2] == 0
        no_end_pairing = mask_list[-1] == 0
        if no_flight and no_end_duty and no_end_pairing:
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if not unassigned:
                break
            # dual 학습도 동일한 CPP action space에서 미커버 상태로 종료함.
            n_zero_mask += 1
            break

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
            state, r, done = step(state, action, flights, assigned, constraint)
            total_reward += r
            if done:
                break
            continue

        flight_id = flights[action]["id"]
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


def run_phase2(encoder, decoder, optimizer, n_episodes, constraint, save_dir, flight_sampler,
               global_step_offset=0, entropy_start=0.01, entropy_end=0.005,
               constraint_sampler=None, init_best=float("inf"), dual_weight_override=None,
               dual_mode="net"):
    # dual_mode: "net"(기본, 기존 동작) = π^cov - ν^exc를 그대로 씀.
    # "coverage_only" = ν^exc(deadhead dual)를 0으로 고정 — coverage dual(π^cov)만 반영.
    # dual-ablation 3분할(off/coverage_only/net) 중 off는 기존 --dual-weight 0으로 이미 커버됨.
    assert dual_mode in ("net", "coverage_only"), f"unknown dual_mode: {dual_mode}"
    from evaluation.set_partition import solve_lp_relaxation

    params            = list(encoder.parameters()) + list(decoder.parameters())
    best_avg_pairings = init_best
    greedy_pairings   = []
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
        flights, origins, dests, dep_times, arr_times, fly_times, base_airport = sample

        base_c   = constraint_sampler() if constraint_sampler else constraint
        c        = {**base_c, "base_airport": base_airport}
        c        = _prepare_cpp_constraint(flights, c)
        c_tensor = constraint_to_tensor(c, device=DEVICE)

        with torch.no_grad():
            encoded = encoder(origins, dests, dep_times, arr_times, fly_times, c_tensor)

            if ep % config.PHASE2_LP_INTERVAL == 0:
                pool      = _collect_pool(flights, c, encoder, decoder, encoded,
                                          n_rollouts=config.PHASE2_POOL_ROLLOUTS)
                lp_result = solve_lp_relaxation(pool)
                if lp_result is not None:
                    dual_vars    = lp_result["dual_vars"]
                    # coverage_only 모드: deadhead dual(ν^exc)을 아예 안 받아옴 → 아래
                    # run_episode_with_dual의 _nu_exc가 항상 0이 되어 π^cov만 반영됨.
                    dh_dual_vars = lp_result["dh_dual_vars"] if dual_mode == "net" else {}
                    lp_value     = lp_result["lp_value"]

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
        advantage = (reward_s - reward_g) / (abs(reward_g) + 1e-6)

        entropy_coef = max(entropy_start * (1.0 - ep / n_episodes), entropy_end)
        loss = torch.stack([
            -lp * advantage - entropy_coef * ent
            for lp, ent in zip(log_probs, entropies)
        ]).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        avg25 = sum(greedy_pairings[-25:]) / min(len(greedy_pairings), 25)

        if len(greedy_pairings) >= 25 and avg25 < best_avg_pairings:
            best_avg_pairings = avg25
            ckpt_path = os.path.join(save_dir, "phase2_best.pt")
            torch.save({
                "encoder":           encoder.state_dict(),
                "decoder":           decoder.state_dict(),
                "stage":             "phase2",
                "episode":           ep,
                "best_avg_pairings": best_avg_pairings,
            }, ckpt_path)
            wandb.save(ckpt_path)

        wandb.log({
            "phase2/greedy_pairings":     metrics_g["n_pairings"],
            "phase2/sample_pairings":     metrics_s["n_pairings"],
            "phase2/greedy_deadheads":    metrics_g["n_deadheads"],
            "phase2/greedy_avg_legs":     metrics_g.get("avg_legs", 0),
            "phase2/greedy_avg_overnight": metrics_g.get("avg_overnight", 0),
            "phase2/sample_reward":       reward_s,
            "phase2/avg25":               avg25,
            "phase2/advantage":           advantage,
            "phase2/loss":                loss.item(),
            "phase2/entropy_coef":        entropy_coef,
            "phase2/best_avg25":          best_avg_pairings if best_avg_pairings < float("inf") else avg25,
            "phase2/n_dual_keys":         len(dual_vars),
            "phase2/n_dh_dual_keys":      sum(1 for v in dh_dual_vars.values() if v > 0),
            "phase2/dual_weight":         _eff_dw,
            "phase2/gap_weight":          decoder.gap_weight.item(),
            "phase2/lp_value":            lp_value if lp_value is not None else float("nan"),
        }, step=global_step_offset + ep)

        if ep % 25 == 0:
            _lp_str = f"{lp_value:.2f}" if lp_value is not None else "n/a"
            print(
                f"  Ep {ep:4d} | "
                f"sample: p={metrics_s['n_pairings']:3d} dh={metrics_s['n_deadheads']:3d} | "
                f"greedy: p={metrics_g['n_pairings']:3d} legs={metrics_g.get('avg_legs', 0):.2f} (avg25={avg25:5.1f}) | "
                f"adv: {advantage:6.3f} | dw={_eff_dw:.3f} | dual keys: {len(dual_vars)} | "
                f"dh dual keys: {sum(1 for v in dh_dual_vars.values() if v > 0)} | lp_value: {_lp_str}"
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
):
    best_avg_pairings = float("inf")
    greedy_pairings = []

    print(f"\n{'='*60}")
    print(f"Curriculum Stage {stage}: max_duty_periods={constraint_override['max_duty_periods']}, "
          f"max_pairing_days={constraint_override['max_pairing_days']}"
          + (" [constraint 랜덤 샘플링]" if constraint_sampler else ""))
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
        flights, origins, dests, dep_times, arr_times, fly_times, base_airport = sample

        # [B 패턴 극복 조치] Stage 3에서 30% 확률로 Stage 2 기준 제약을 주입하여 과거 환경 기억 보존 (Continual Replay)
        if stage == 3 and base_stage2_constraint is not None and random.random() < 0.3:
            c = {
                **base_stage2_constraint,
                "max_duty_periods": 2,
                "max_pairing_days": config.WINDOW_DAYS - 1,
            }
        else:
            c = constraint_sampler() if constraint_sampler else constraint_override

        c = {**c, "base_airport": base_airport}  # 에피소드별 base 주입
        c = _prepare_cpp_constraint(flights, c)
        
        # 선택된 복원/샘플링 제약조건 사전(c)을 기반으로 정확히 텐서를 빌드하여 FiLM 정렬 유지
        c_tensor = constraint_to_tensor(c, device=DEVICE)
        encoded  = encoder(origins, dests, dep_times, arr_times, fly_times, c_tensor)

        reward_s, log_probs, entropies, metrics_s = run_episode(
            flights, c, encoder, decoder, encoded, greedy=False
        )
        if len(log_probs) == 0:
            continue

        with torch.no_grad():
            encoded_g = encoder(origins, dests, dep_times, arr_times, fly_times, c_tensor)
            reward_g, _, _, metrics_g = run_episode(
                flights, c, encoder, decoder, encoded_g, greedy=True
            )

        greedy_pairings.append(metrics_g["n_pairings"])
        advantage = (reward_s - reward_g) / (abs(reward_g) + 1e-6)

        entropy_coef = max(entropy_start * (1.0 - ep / n_episodes), entropy_end)
        loss = torch.stack([
            -lp * advantage - entropy_coef * ent
            for lp, ent in zip(log_probs, entropies)
        ]).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        avg25 = sum(greedy_pairings[-25:]) / min(len(greedy_pairings), 25)

        if len(greedy_pairings) >= 25:
            if avg25 < best_avg_pairings:
                best_avg_pairings = avg25
                ckpt_path = os.path.join(save_dir, f"stage{stage}_best.pt")
                torch.save({
                    "encoder":           encoder.state_dict(),
                    "decoder":           decoder.state_dict(),
                    "stage":             stage,
                    "episode":           ep,
                    "best_avg_pairings": best_avg_pairings,
                }, ckpt_path)
                wandb.save(ckpt_path)

        wandb.log({
            f"stage{stage}/greedy_pairings":   metrics_g["n_pairings"],
            f"stage{stage}/sample_pairings":   metrics_s["n_pairings"],
            f"stage{stage}/greedy_deadheads":  metrics_g["n_deadheads"],
            f"stage{stage}/greedy_avg_legs":   metrics_g.get("avg_legs", 0),
            f"stage{stage}/greedy_avg_overnight": metrics_g.get("avg_overnight", 0),
            f"stage{stage}/sample_reward":     reward_s,
            f"stage{stage}/avg25":             avg25,
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
                f"greedy: p={metrics_g['n_pairings']:3d} legs={metrics_g.get('avg_legs', 0):.2f} (avg25={avg25:5.1f}) | "
                f"adv: {advantage:6.3f}"
            )

    print(f"  → best avg pairings: {best_avg_pairings:.1f}  "
          f"(saved: checkpoints/stage{stage}_best.pt)")
    return best_avg_pairings


def train(phase2_only=False, multi_airline=False, skip_film=False, skip_decoder_constraint=False,
          ckpt_dir=None, from_stage2=False, turkish_files=None, dual_weight=None, dual_mode="net"):
    WINDOW_DAYS = config.WINDOW_DAYS  # config.py에서 관리 — max_pairing_days 상한과 연동

    # 2x2 FiLM 인과성 실험(C/D/C'/D') — 디코더의 constraint 직접 concat 경로를
    # 원천 차단할지 여부. 이 프로세스 안에서 학습·rollout 전체(train.py, rollout.py
    # 둘 다 동일한 RL/utils.py를 import하므로)에 즉시 반영된다.
    set_skip_decoder_constraint(skip_decoder_constraint)

    _select_environment("multi" if multi_airline else config.AIRLINE)

    if multi_airline:
        import os as _os
        airlines = [a for a in config.AIRLINE_DATA if not _os.path.isdir(config.AIRLINE_DATA[a])]
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
            airport_map = build_airport_map(DATA_PATH)
        base_ids   = bases_to_ids(airline_bases, airport_map)
        n_airports = len(airport_map)
        print(f"airports: {n_airports}개, airline: {config.AIRLINE}, bases: {airline_bases}")
        if config.AIRLINE == "turkish":
            # 두 Istanbul base 중 episode base를 선택하되 pairing은 동일 base로 복귀함
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
    tag += "-covonly" if dual_mode == "coverage_only" else ""
    run_name = "phase2-only" if phase2_only else tag
    wandb.init(
        project="ASCP-2026-paper",
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
                    use_utc=USE_UTC,
                )
                if flights and any(f["origin"] == base_airport for f in flights):
                    origins, dests, dep_times, arr_times, fly_times = flights_to_tensors(flights, WINDOW_DAYS * 24.0, device=DEVICE)
                    return flights, origins, dests, dep_times, arr_times, fly_times, base_airport
            return None

        _first_base = all_base_ids["delta"][0]
        base_constraint = _CONSTRAINT_FN["delta"](_first_base)

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
                return flights, origins, dests, dep_times, arr_times, fly_times, base_airport
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
                    use_utc=USE_UTC,
                )
                if not flights:
                    return None
                if not any(f["origin"] == base_airport for f in flights):
                    return None
                origins, dests, dep_times, arr_times, fly_times = flights_to_tensors(flights, WINDOW_DAYS * 24.0, device=DEVICE)
                return flights, origins, dests, dep_times, arr_times, fly_times, base_airport

        base_constraint = _CONSTRAINT_FN[config.AIRLINE](base_ids[0])

    _stage3_base = {**base_constraint, "max_duty_periods": 4, "max_pairing_days": WINDOW_DAYS - 1}
    def sample_constraint():
        r = config.STAGE3_CONSTRAINT_RANGES
        if multi_airline:
            base = {**_CONSTRAINT_FN[_selected_airline[0]](0),
                    "max_duty_periods": 2, "max_pairing_days": WINDOW_DAYS - 1}
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

    _s3_best     = float("inf")  
    _s3_ckpt_dir = save_dir      

    if phase2_only:
        _s3_ckpt_dir = ckpt_dir if ckpt_dir else save_dir
        ckpt_path = os.path.join(_s3_ckpt_dir, "stage3_best.pt")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        ckpt_n_airports = ckpt["encoder"]["airport_emb.weight"].shape[0]
        if ckpt_n_airports != n_airports:
            encoder = FlightEncoder(
                n_airports=ckpt_n_airports,
                constraint_dim=len(FILM_CONSTRAINT_KEYS),
                airport_emb_dim=32,
                d_model=128,
                use_film_before=not skip_film,
                use_film_after=not skip_film,
            ).to(DEVICE)
            n_airports = ckpt_n_airports
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        print(f"stage3_best.pt 로드 완료: {ckpt_path} → Phase 2만 실행 (n_airports={n_airports})")

    if not phase2_only:
        if from_stage2:
            _s2_load_dir = ckpt_dir
            if not _s2_load_dir:
                raise ValueError("--from-stage2 사용 시 --ckpt-dir로 stage2_best.pt 폴더를 지정해야 합니다.")
            _s2_ckpt_path = os.path.join(_s2_load_dir, "stage2_best.pt")
            _s2_ckpt = torch.load(_s2_ckpt_path, map_location=DEVICE, weights_only=True)
            encoder.load_state_dict(_s2_ckpt["encoder"])
            decoder.load_state_dict(_s2_ckpt["decoder"])
            print(f"stage2_best.pt 로드: {_s2_ckpt_path} → Stage 3부터 실행")
        else:
            # ── Stage 1: 단일 duty (overnight 없음) ──────────────────────────
            # base_penalty는 stage1/2에서 5.0(원래값) 고정 — stage3/phase2부터 config.py의
            # 현재값(500.0)을 그대로 물려받는다. x2gcdva5(stage1/2, p5)를 이어받는 기존
            # run들과 동일 조건을 신규 seed에서도 재현하기 위함.
            stage1_c = {**base_constraint, "max_duty_periods": 1, "max_pairing_days": 1, "base_penalty": 5.0}
            run_curriculum_stage(1, encoder, decoder, optimizer,
                                 n_episodes=1000, constraint_override=stage1_c,
                                 save_dir=save_dir, flight_sampler=flight_sampler,
                                 global_step_offset=0,
                                 entropy_start=0.30, entropy_end=0.005)

            # ── Stage 2: full multi-day ───────────────────────────────────────
            stage2_c = {**base_constraint, "max_duty_periods": 2, "max_pairing_days": WINDOW_DAYS - 1, "base_penalty": 5.0}
            run_curriculum_stage(2, encoder, decoder, optimizer,
                                 n_episodes=2000, constraint_override=stage2_c,
                                 save_dir=save_dir, flight_sampler=flight_sampler,
                                 global_step_offset=1000,
                                 entropy_start=0.02, entropy_end=0.005)

        # ── Stage 3: 7개 constraint 전체 랜덤 augmentation (FiLM 학습) ───
        _s3_offset = 0 if from_stage2 else 3000
        # [B 패턴 극복 조치] base_stage2_constraint 인자에 base_constraint를 주입하여 Stage 2 제약 복원 유도
        _s3_best = run_curriculum_stage(3, encoder, decoder, optimizer,
                             n_episodes=2000, constraint_override=_stage3_base,
                             save_dir=save_dir, flight_sampler=flight_sampler,
                             constraint_sampler=sample_constraint,
                             global_step_offset=_s3_offset,
                             entropy_start=0.01, entropy_end=0.005,
                             base_stage2_constraint=base_constraint)

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
            use_utc=USE_UTC,
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
               dual_weight_override=dual_weight, dual_mode=dual_mode)

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
    if os.path.exists(_phase2_ckpt_path) and _p2_best < _s3_best:
        _phase2_ckpt = torch.load(_phase2_ckpt_path, map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(_phase2_ckpt["encoder"])
        decoder.load_state_dict(_phase2_ckpt["decoder"])
        print(f"최종 모델: phase2_best.pt 사용 (avg_pairings={_p2_best:.1f} < stage3 {_s3_best:.1f})")
    else:
        encoder.load_state_dict(_film_ckpt["encoder"])
        decoder.load_state_dict(_film_ckpt["decoder"])
        print(f"최종 모델: stage3_best.pt 사용 (Phase 2 {_p2_best:.1f}가 {_s3_best:.1f} 기록을 못 넘김)")

    # ── 최종 모델 저장 ────────────────────────────────────────────────
    torch.save({
        "encoder":        encoder.state_dict(),
        "decoder":        decoder.state_dict(),
        "n_airports":     n_airports,
        "constraint_dim": len(FILM_CONSTRAINT_KEYS),
        "bases":          _val_bases_save,
        "window_days":    WINDOW_DAYS,
        "max_time":       WINDOW_DAYS * 24,
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
    parser.add_argument("--use-utc", action="store_true",
                        help="dep_time을 UTC 절대시간으로 앵커링. 새로 이 옵션으로 학습한 모델만 "
                             "이 옵션 켠 채로 평가해야 함 — 기존 체크포인트에 켜면 OOD")
    parser.add_argument("--dual-weight", type=float, default=None,
                        help="Phase2 CG dual reward 가중치를 config.PHASE2_DUAL_WEIGHT(기본 0.6) 대신 "
                             "이 값으로 덮어씀. 0을 주면 CG-dual 완전히 비활성화.")
    parser.add_argument("--dual-mode", default="net", choices=["net", "coverage_only"],
                        help="CG-dual ablation 3분할용. net(기본) = coverage dual(π^cov) - "
                             "deadhead dual(ν^exc) 그대로 사용(현재 버전). coverage_only = "
                             "ν^exc를 0으로 고정해 π^cov만 반영. off는 --dual-weight 0으로 커버.")
    parser.add_argument("--data-path", default=None,
                        help="CSV 경로. 미지정 시 config.AIRLINE_DATA[airline] 사용. "
                             "delta-small 등 대체 데이터셋으로 학습/이어받기할 때 지정")
    args = parser.parse_args()
    if args.airline:
        config.AIRLINE = args.airline
    if args.data_path:
        # config.AIRLINE_DATA를 덮어써야 train() 안의 DATA_PATH/airport_map이 이 경로를 따라감
        config.AIRLINE_DATA[config.AIRLINE] = args.data_path
        print(f"data_path 지정: {config.AIRLINE} → {args.data_path}")
    _set_device(args.device)
    USE_UTC = args.use_utc
    print(f"device: {DEVICE}")
    print(f"use_utc: {USE_UTC}")
    print(f"log: {args.log}")
    _turkish_files = [f.strip() for f in args.turkish_files.split(",")] if args.turkish_files else None
    train(phase2_only=args.phase2_only, multi_airline=args.multi_airline, skip_film=args.skip_film,
          skip_decoder_constraint=args.skip_decoder_constraint,
          ckpt_dir=args.ckpt_dir, from_stage2=args.from_stage2, turkish_files=_turkish_files,
          dual_weight=args.dual_weight, dual_mode=args.dual_mode)
