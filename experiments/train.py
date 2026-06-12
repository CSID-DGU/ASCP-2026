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
from environment import get_mask, step, final_reward
from constraints import get_delta_constraints, get_alaska_constraints, get_jetblue_constraints, FILM_CONSTRAINT_KEYS

_CONSTRAINT_FN = {
    "delta":   get_delta_constraints,
    "alaska":  get_alaska_constraints,
    "jetblue": get_jetblue_constraints,
}
from state import init_state
import config

DEVICE = torch.device("cpu")  # train() 호출 전 _set_device()로 설정


def _set_device(device_str: str):
    global DEVICE
    DEVICE = torch.device(device_str)



def constraint_to_tensor(constraint):
    return torch.tensor(
        [constraint[k] / config.CONSTRAINT_NORMS[k] for k in FILM_CONSTRAINT_KEYS],
        dtype=torch.float32,
    ).to(DEVICE)

def flights_to_tensors(flights, window_days=5):
    origins   = torch.tensor([f["origin"]   for f in flights]).to(DEVICE)
    dests     = torch.tensor([f["dest"]     for f in flights]).to(DEVICE)
    max_time  = window_days * 24.0
    dep_raw   = torch.tensor([f["dep_time"] for f in flights], dtype=torch.float32)
    arr_raw   = torch.tensor([f["arr_time"] for f in flights], dtype=torch.float32)
    dep_times = (dep_raw / max_time).to(DEVICE)
    arr_times = (arr_raw / max_time).to(DEVICE)
    fly_times = ((arr_raw - dep_raw) / max_time).to(DEVICE)
    return origins, dests, dep_times, arr_times, fly_times


def state_to_vec(state, encoder, constraint):
    """혜린 state dict → 찬주 decoder 입력 tensor 변환

    state_vec(71,) = current_emb(32) + base_emb(32) + scalars(7)
    7개 스칼라: time_of_day, day_norm, duty_elapsed/max, legs/max, duty_period/max, is_resting, rest_remaining
    """
    current_emb = encoder.airport_emb(torch.tensor(state["current_airport"]).to(DEVICE))
    base_emb    = encoder.airport_emb(torch.tensor(constraint["base_airport"]).to(DEVICE))

    time_of_day      = (state["current_time"] % 24.0) / 24.0
    day_norm         = (state["current_time"] // 24.0) / config.CONSTRAINT_NORMS["max_pairing_days"]
    duty_period_norm = state.get("duty_period", 0) / config.CONSTRAINT_NORMS["max_duty_periods"]

    # duty_elapsed: 비행 시간만이 아닌 FAA 기준 실제 경과 시간 (비행 + 대기)
    # is_resting/pairing_start 중이면 새 duty 아직 시작 안 함 → 0
    if state.get("is_resting", False) or state.get("pairing_start", False):
        duty_elapsed = 0.0
    else:
        duty_elapsed = max(0.0, state["current_time"] - state.get("duty_start_time", state["current_time"]))

    # rest_remaining: is_resting=True일 때 남은 rest 시간 비율 (0~1), 아니면 0.0
    if state.get("is_resting", False) and state.get("rest_end_time") is not None:
        rest_remaining = max(0.0, state["rest_end_time"] - state["current_time"]) / config.CONSTRAINT_NORMS["min_rest"]
    else:
        rest_remaining = 0.0

    # 고정 분모(CONSTRAINT_NORMS) 사용 이유:
    # constraint 값으로 직접 나누면 constraint 정보가 state_vec에 인코딩되어 FiLM gradient가 약해짐.
    # CONSTRAINT_NORMS는 훈련 범위 최대값으로 고정 → FiLM이 constraint 정보의 유일한 경로가 됨.
    return torch.cat([
        current_emb,
        base_emb,
        torch.tensor([
            time_of_day,
            day_norm,
            duty_elapsed / config.CONSTRAINT_NORMS["max_duty"],
            state["legs"] / config.CONSTRAINT_NORMS["max_legs"],
            duty_period_norm,
            1.0 if state.get("is_resting", False) else 0.0,
            rest_remaining,
        ], dtype=torch.float32).to(DEVICE)
    ])


def run_episode(flights, constraint, encoder, decoder, encoded, greedy=False):
    """
    혜린 environment + 찬주 model로 에피소드 진행

    Returns:
        total_reward, log_probs, entropies, metrics dict
        metrics: {n_pairings, n_deadheads, n_uncovered, coverage_pct}
    """
    assigned = {f["id"]: False for f in flights}
    state = init_state(flights, constraint)

    log_probs = []
    entropies = []
    total_reward = 0
    n_pairings = 0
    n_deadheads = 0  # 강제 시작된 pairing 수 (connection 못 찾아서)

    max_steps = len(flights) * 20  # 무한루프 방지 (flight당 최대 20 step)
    step_count = 0
    while True:
        step_count += 1
        if step_count > max_steps:
            break
        # 혜린 mask
        mask_list = get_mask(state, flights, assigned, constraint)
        mask = torch.tensor(mask_list, dtype=torch.float32).to(DEVICE)

        # flight도 없고 END_DUTY/END_PAIRING도 불가 → 강제로 새 pairing 시작 (deadhead)
        no_flight     = sum(mask_list[:-2]) == 0
        no_end_duty   = mask_list[-2] == 0
        no_end_pairing = mask_list[-1] == 0
        if no_flight and no_end_duty and no_end_pairing:
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if len(unassigned) == 0:
                break

            # base 출발 편 우선, 없으면 가장 이른 편으로 강제 이동 (deadhead)
            base = constraint["base_airport"]
            base_unassigned = [f for f in unassigned if f["origin"] == base]
            earliest = sorted(base_unassigned or unassigned, key=lambda x: x["dep_time"])[0]

            if not state.get("pairing_start", False):
                n_pairings += 1
                n_deadheads += 1
                # BASE_PENALTY, PAIRING_COST는 environment step()과 중복되지 않도록
                # deadhead 강제이동 시에만 직접 차감
                total_reward -= config.DEFAULT_CONSTRAINTS["pairing_cost"]
                if state["current_airport"] != base:
                    total_reward -= config.DEFAULT_CONSTRAINTS["base_penalty"]

            state = {
                "current_airport":    earliest["origin"],
                "current_time":       earliest["dep_time"],
                "duty_time":          0.0,
                "duty_start_time":    earliest["dep_time"],
                "legs":               0,
                "remaining":          sum(1 for v in assigned.values() if not v),
                "pairing_start":      True,
                "duty_period":        0,
                "pairing_start_time": earliest["dep_time"],
                "is_resting":         False,
                "rest_end_time":      None,
            }
            continue

        # 찬주 decoder
        state_vec = state_to_vec(state, encoder, constraint)
        probs = decoder(encoded, state_vec, mask)

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
            state, r, done = step(state, action, flights, assigned, constraint)
            total_reward += r
            continue

        # END_PAIRING (index N+1): step()이 BASE_PENALTY + PAIRING_COST 처리
        if action == n_flights + 1:
            n_pairings += 1
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
        "n_pairings": n_pairings,
        "n_deadheads": n_deadheads,
        "n_uncovered": n_uncovered,
        "coverage_pct": coverage_pct,
    }
    return total_reward, log_probs, entropies, metrics


# ── Phase 2 helpers ──────────────────────────────────────────────────────────
"""
phase2는 LP dual variable을 RL reward에 피드백하는 구조임. 
이를 위해 4개 함수가 필요함
1. pairing 구조체를 수집하는 rollout 함수 
2. 중복 제거 pool 수집 함수 
3. dual feedback이 포함된 에피소드 함수
4. Phase 2 학습 루프
"""
# evaluate_ip.py의 rollout_with_pairings()를 참고해 train.py 내부 함수로 구현.

# pairing cost = dead_time - LEG_BONUS*(n_legs-1) + DEADHEAD_PENALTY*(강제종료여부)
# dead_time: 실제 비행 외 대기 시간 (elapsed - fly - rest)
# LEG_BONUS: leg 추가될수록 cost 감소 → 효율적 연결 장려
# DEADHEAD_PENALTY: 강제 deadhead 발생 시 가산
_LEG_BONUS_IP        = 1.5
_DEADHEAD_PENALTY_IP = 5.0
_PAIRING_FIXED_COST  = 1.5  # IP/LP cost와 동일 — Phase 2 dual signal 일관성 유지


def _rollout_with_pairings(flights, constraint, encoder, decoder, encoded, greedy=False):
    """
    run_episode()와 동일한 rollout이지만 pairing 구조체(legs, cost 등)를 반환.
    LP relaxation에 넣을 pool 수집 전용. gradient 추적 불필요 → no_grad 하에서 호출.

    feat/ip-multibase evaluate_ip.py의 rollout_with_pairings() 참고.
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
        dead_time = max(elapsed - pairing_fly - pairing_rest, 0.0)
        cost      = (dead_time
                     - _LEG_BONUS_IP * max(len(current_legs) - 1, 0)
                     + (_DEADHEAD_PENALTY_IP if is_forced else 0.0)
                     + _PAIRING_FIXED_COST)
        pairings.append({"legs": list(current_legs), "fly": pairing_fly,
                         "elapsed": elapsed, "cost": cost})

    def start_new(f):
        nonlocal pairing_dep, pairing_fly, pairing_last_arr, pairing_rest
        current_legs.clear()
        current_legs.append(f["id"])
        pairing_dep      = f["dep_time"]
        pairing_fly      = f["arr_time"] - f["dep_time"]
        pairing_last_arr = f["arr_time"]
        pairing_rest     = 0.0

    episode_base = constraint.get("base_airport", 0)

    # 첫 flight 수동 시작 — base 출발 편 우선
    unassigned   = [f for f in flights if not assigned[f["id"]]]
    base_flights = [f for f in unassigned if f["origin"] == episode_base]
    first        = sorted(base_flights or unassigned, key=lambda f: f["dep_time"])[0]
    assigned[first["id"]] = True
    start_new(first)
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
    }

    max_steps  = len(flights) * 20
    step_count = 0

    while True:
        step_count += 1
        if step_count > max_steps:
            flush_pairing(is_forced=False)
            break

        mask_list = get_mask(state, flights, assigned, constraint)
        mask      = torch.tensor(mask_list, dtype=torch.float32).to(DEVICE)

        # mask 전부 0 → 강제 deadhead
        if sum(mask_list[:-2]) == 0 and mask_list[-2] == 0 and mask_list[-1] == 0:
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if not unassigned:
                flush_pairing(is_forced=False)
                break
            flush_pairing(is_forced=True)
            base_flights = [f for f in unassigned if f["origin"] == episode_base]
            nxt = sorted(base_flights or unassigned, key=lambda f: f["dep_time"])[0]
            assigned[nxt["id"]] = True
            start_new(nxt)
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
            }
            continue

        state_vec = state_to_vec(state, encoder, constraint)
        probs     = decoder(encoded, state_vec, mask)
        action    = probs.argmax().item() if greedy else Categorical(probs).sample().item()

        if action == len(flights):          # END_DUTY
            pairing_rest += constraint.get("min_rest", 9.5)
            state, _, _ = step(state, action, flights, assigned, constraint)
            continue

        if action == len(flights) + 1:      # END_PAIRING → 새 pairing 시작
            flush_pairing(is_forced=False)
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if not unassigned:
                break
            base_flights = [f for f in unassigned if f["origin"] == episode_base]
            nxt = sorted(base_flights or unassigned, key=lambda f: f["dep_time"])[0]
            assigned[nxt["id"]] = True
            start_new(nxt)
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
            }
            continue

        # flight action — pairing에 leg 추가
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
    """
    stochastic rollout × n_rollouts + greedy rollout × 1 → 중복 제거 pairing pool.
    중복 기준: legs 집합이 동일하면 cost가 낮은 쪽 유지.
    pool은 solve_lp_relaxation()의 입력으로 사용된다.
    """
    pool = {}
    for _ in range(n_rollouts):
        for p in _rollout_with_pairings(flights, constraint, encoder, decoder, encoded):
            key = tuple(sorted(p["legs"]))
            if key not in pool or p["cost"] < pool[key]["cost"]:
                pool[key] = p
    # greedy 1번 추가 — 실현 가능한 고품질 pairing 항상 보장
    for p in _rollout_with_pairings(flights, constraint, encoder, decoder, encoded, greedy=True):
        key = tuple(sorted(p["legs"]))
        if key not in pool or p["cost"] < pool[key]["cost"]:
            pool[key] = p
    return list(pool.values())


def run_episode_with_dual(flights, constraint, encoder, decoder, encoded, dual_vars, greedy=False):
    """
    Phase 2 전용 run_episode — flight 배정 시 LP dual variable π[f]를 reward에 추가.

    π[f] > 0: LP에서 이 flight 커버 가치 높음 → RL이 적극적으로 포함하도록 유도
    π[f] ≈ 0: 이미 여러 pairing이 커버 → 굳이 포함 안 해도 됨
    dual_vars: {flight_id: π[f]} — solve_lp_relaxation()["dual_vars"]

    greedy=True: stochastic 샘플링 없이 argmax 선택 (baseline 계산용)
                 stochastic/greedy 모두 동일한 dual_vars 적용 → advantage가 순수 policy 차이만 반영
    """
    assigned = {f["id"]: False for f in flights}
    state    = init_state(flights, constraint)

    log_probs    = []
    entropies    = []
    total_reward = 0
    n_pairings   = 0
    n_deadheads  = 0
    base         = constraint["base_airport"]

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
            base_unassigned = [f for f in unassigned if f["origin"] == base]
            earliest = sorted(base_unassigned or unassigned, key=lambda x: x["dep_time"])[0]
            if not state.get("pairing_start", False):
                n_pairings  += 1
                n_deadheads += 1
                total_reward -= config.DEFAULT_CONSTRAINTS["pairing_cost"]
                if state["current_airport"] != base:
                    total_reward -= config.DEFAULT_CONSTRAINTS["base_penalty"]
            state = {
                "current_airport":    earliest["origin"],
                "current_time":       earliest["dep_time"],
                "duty_time":          0.0,
                "duty_start_time":    earliest["dep_time"],
                "legs":               0,
                "remaining":          sum(1 for v in assigned.values() if not v),
                "pairing_start":      True,
                "duty_period":        0,
                "pairing_start_time": earliest["dep_time"],
                "is_resting":         False,
                "rest_end_time":      None,
            }
            continue

        state_vec = state_to_vec(state, encoder, constraint)
        probs     = decoder(encoded, state_vec, mask)
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
            state, r, done = step(state, action, flights, assigned, constraint)
            total_reward += r
            continue

        if action == n_flights + 1:     # END_PAIRING
            n_pairings += 1
            state, r, done = step(state, action, flights, assigned, constraint)
            total_reward += r
            if done:
                break
            continue

        # flight action — π[flight_id] 추가 (CG dual feedback)
        flight_id = flights[action]["id"]
        state, r, done = step(state, action, flights, assigned, constraint)
        total_reward += r + dual_vars.get(flight_id, 0.0) * config.PHASE2_DUAL_WEIGHT
        if done:
            break

    total_reward += final_reward(assigned)
    n_uncovered  = sum(1 for v in assigned.values() if not v)
    coverage_pct = (len(flights) - n_uncovered) / len(flights) * 100
    return total_reward, log_probs, entropies, {
        "n_pairings":   n_pairings,
        "n_deadheads":  n_deadheads,
        "n_uncovered":  n_uncovered,
        "coverage_pct": coverage_pct,
    }


def run_phase2(encoder, decoder, optimizer, n_episodes, constraint, save_dir, flight_sampler,
               global_step_offset=0, entropy_start=0.01, entropy_end=0.005,
               constraint_sampler=None):
    """
    Phase 2 — Column Generation dual feedback 학습.

    매 PHASE2_LP_INTERVAL 에피소드마다:
      1. pool 수집 (_collect_pool, no_grad)
      2. LP relaxation → dual variable π[f] 추출 (set_partition.solve_lp_relaxation)
      3. π[f]를 reward에 반영해 REINFORCE 업데이트 (run_episode_with_dual)

    set_partition.py는 혜린 담당 — import만 사용, 파일 수정 없음.
    """
    from set_partition import solve_lp_relaxation

    params            = list(encoder.parameters()) + list(decoder.parameters())
    best_avg_pairings = float("inf")
    greedy_pairings   = []
    dual_vars         = {}  # {flight_id: π[f]}, LP interval마다 갱신, 사이에는 캐싱

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

        # constraint_sampler가 있으면 에피소드마다 샘플링 — FiLM이 Phase 2에서도 constraint 변화 학습
        # 없으면 고정 constraint 사용 (기존 동작 유지)
        base_c   = constraint_sampler() if constraint_sampler else constraint
        c        = {**base_c, "base_airport": base_airport}
        c_tensor = constraint_to_tensor(c)

        with torch.no_grad():
            encoded = encoder(origins, dests, dep_times, arr_times, fly_times, c_tensor)

            # LP interval마다 pool 수집 → LP relaxation → dual vars 갱신
            if ep % config.PHASE2_LP_INTERVAL == 0:
                pool      = _collect_pool(flights, c, encoder, decoder, encoded,
                                          n_rollouts=config.PHASE2_POOL_ROLLOUTS)
                lp_result = solve_lp_relaxation(pool)
                if lp_result is not None:
                    dual_vars = lp_result["dual_vars"]  # {flight_id: π[f]}

        # dual feedback 포함 REINFORCE
        encoded_train = encoder(origins, dests, dep_times, arr_times, fly_times, c_tensor)
        reward_s, log_probs, entropies, metrics_s = run_episode_with_dual(
            flights, c, encoder, decoder, encoded_train, dual_vars
        )
        if len(log_probs) == 0:
            continue

        with torch.no_grad():
            encoded_g = encoder(origins, dests, dep_times, arr_times, fly_times, c_tensor)
            reward_g, _, _, metrics_g = run_episode_with_dual(
                flights, c, encoder, decoder, encoded_g, dual_vars, greedy=True
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
            "phase2/greedy_pairings":  metrics_g["n_pairings"],
            "phase2/sample_pairings":  metrics_s["n_pairings"],
            "phase2/greedy_deadheads": metrics_g["n_deadheads"],
            "phase2/sample_reward":    reward_s,
            "phase2/avg25":            avg25,
            "phase2/advantage":        advantage,
            "phase2/loss":             loss.item(),
            "phase2/entropy_coef":     entropy_coef,
            "phase2/best_avg25":       best_avg_pairings if best_avg_pairings < float("inf") else avg25,
            "phase2/n_dual_keys":      len(dual_vars),
        }, step=global_step_offset + ep)

        if ep % 25 == 0:
            print(
                f"  Ep {ep:4d} | "
                f"sample: p={metrics_s['n_pairings']:3d} dh={metrics_s['n_deadheads']:3d} | "
                f"greedy: p={metrics_g['n_pairings']:3d} (avg25={avg25:5.1f}) | "
                f"adv: {advantage:6.3f} | dual keys: {len(dual_vars)}"
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
):
    """
    커리큘럼 1단계 실행.

    flight_sampler: () → (flights, origins, dests, dep_times, arr_times, fly_times, base_airport)
                    에피소드마다 호출 — (base, window) 쌍 랜덤 선택 + flight 로드
    constraint_sampler: () → constraint dict. None이면 constraint_override 고정 사용.
    """
    best_avg_pairings = float("inf")
    greedy_pairings = []

    print(f"\n{'='*60}")
    print(f"Curriculum Stage {stage}: max_duty_periods={constraint_override['max_duty_periods']}, "
          f"max_pairing_days={constraint_override['max_pairing_days']}"
          + (" [constraint 랜덤 샘플링]" if constraint_sampler else ""))
    print(f"{'='*60}")

    params = list(encoder.parameters()) + list(decoder.parameters())

    for ep in range(n_episodes):
        # 에피소드마다 (base, window) 랜덤 선택
        sample = flight_sampler()
        if sample is None:
            continue
        flights, origins, dests, dep_times, arr_times, fly_times, base_airport = sample

        c = constraint_sampler() if constraint_sampler else constraint_override
        c = {**c, "base_airport": base_airport}  # 에피소드별 base 주입
        c_tensor = constraint_to_tensor(c)
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

        # best checkpoint: greedy pairings 25ep 이동평균 기준
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
            f"stage{stage}/greedy_pairings":  metrics_g["n_pairings"],
            f"stage{stage}/sample_pairings":  metrics_s["n_pairings"],
            f"stage{stage}/greedy_deadheads": metrics_g["n_deadheads"],
            f"stage{stage}/sample_reward":    reward_s,
            f"stage{stage}/avg25":            avg25,
            f"stage{stage}/advantage":        advantage,
            f"stage{stage}/loss":             loss.item(),
            f"stage{stage}/entropy_coef":     entropy_coef,
            f"stage{stage}/best_avg25":       best_avg_pairings if best_avg_pairings < float("inf") else avg25,
        }, step=global_step_offset + ep)

        if ep % 25 == 0:
            print(
                f"  Ep {ep:4d} | "
                f"sample: p={metrics_s['n_pairings']:3d} dh={metrics_s['n_deadheads']:3d} | "
                f"greedy: p={metrics_g['n_pairings']:3d} (avg25={avg25:5.1f}) | "
                f"adv: {advantage:6.3f}"
            )

    print(f"  → best avg pairings: {best_avg_pairings:.1f}  "
          f"(saved: checkpoints/stage{stage}_best.pt)")
    return best_avg_pairings


def train(phase2_only=False, multi_airline=False, skip_film=False):
    WINDOW_DAYS = config.WINDOW_DAYS  # config.py에서 관리 — max_pairing_days 상한과 연동

    if multi_airline:
        # 세 항공사 CSV 전체에서 통합 공항 ID 공간 구성
        # Delta/Alaska/JetBlue의 LAX 등 공유 공항이 동일 ID를 갖도록 보장
        all_paths = list(config.AIRLINE_DATA.values())
        airport_map = build_airport_map(all_paths)
        airlines = list(config.AIRLINE_DATA.keys())
        all_base_ids = {a: bases_to_ids(config.AIRLINE_BASES[a], airport_map) for a in airlines}
        # n_airports는 통합 맵 기준 — encoder embedding을 충분히 크게
        n_airports = len(airport_map)
        print(f"airports: {n_airports}개 (통합), airlines: {airlines}")
    else:
        DATA_PATH   = config.AIRLINE_DATA[config.AIRLINE]
        airline_bases = config.AIRLINE_BASES[config.AIRLINE]
        airport_map = build_airport_map(DATA_PATH)
        base_ids    = bases_to_ids(airline_bases, airport_map)
        n_airports  = len(airport_map)
        print(f"airports: {n_airports}개, airline: {config.AIRLINE}, bases: {airline_bases}")

    encoder = FlightEncoder(
        n_airports=n_airports,
        constraint_dim=len(FILM_CONSTRAINT_KEYS),
        airport_emb_dim=32,
        d_model=128,
        use_film_before=not skip_film,
        use_film_after=not skip_film,
    ).to(DEVICE)
    decoder   = PointerDecoder(d_model=128, airport_emb_dim=32).to(DEVICE)
    # FiLM lr 분리: FiLM은 constraint 변화에 빠르게 반응해야 하므로 lr=1e-3
    # skip_film=True(ablation)일 때는 FiLM이 identity → 분리 불필요, 단일 lr 유지
    if skip_film:
        params    = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = optim.Adam(params, lr=1e-4)
    else:
        optimizer = optim.Adam([
            {"params": encoder.film_params(),     "lr": 1e-3},
            {"params": encoder.non_film_params(), "lr": 1e-4},
            {"params": decoder.parameters(),      "lr": 1e-4},
        ])

    tag = "multi-airline" if multi_airline else "delta"
    tag += "-nofilm" if skip_film else ""
    run_name = "phase2-only" if phase2_only else tag
    wandb.init(
        project="ASCP-2026",
        name=run_name,
        config={
            "airline":            "multi" if multi_airline else config.AIRLINE,
            "multi_airline":      multi_airline,
            "window_days":        WINDOW_DAYS,
            "episode_max_flights": config.EPISODE_MAX_FLIGHTS,
            "phase2_lp_interval": config.PHASE2_LP_INTERVAL,
            "phase2_pool_rollouts": config.PHASE2_POOL_ROLLOUTS,
            "phase2_dual_weight": config.PHASE2_DUAL_WEIGHT,
            "phase2_n_episodes":  config.PHASE2_N_EPISODES,
            "lr":                 1e-4,
            "device":             str(DEVICE),
        },
        resume="allow",
    )

    save_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints", wandb.run.id)
    os.makedirs(save_dir, exist_ok=True)

    # CSV 한 번만 로드 → flight_sampler 클로저에 캐싱 (에피소드마다 재로딩 방지)
    import pandas as pd

    if multi_airline:
        # 항공사별 DataFrame + max_offset 사전 계산
        _df_caches = {}
        _max_offsets = {}
        for a in airlines:
            p = config.AIRLINE_DATA[a]
            df = pd.read_csv(p, usecols=["ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "FL_DATE"]).dropna()
            df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="mixed")
            _df_caches[a]   = df
            _max_offsets[a] = max(0, df["FL_DATE"].nunique() - WINDOW_DAYS)

        # 에피소드마다 선택된 항공사를 공유 — constraint_sampler가 읽기 위함
        _selected_airline = ["delta"]

        def flight_sampler():
            """에피소드마다 항공사 + (base, window) 랜덤 선택"""
            airline      = random.choice(airlines)
            _selected_airline[0] = airline
            base_airport = random.choice(all_base_ids[airline])
            offset_days  = random.randint(0, _max_offsets[airline])
            flights = load_flights_rolling(
                config.AIRLINE_DATA[airline], WINDOW_DAYS, offset_days, airport_map,
                base_airport=base_airport,
                n_max=config.EPISODE_MAX_FLIGHTS,
                df=_df_caches[airline],
            )
            if not flights:
                return None
            if not any(f["origin"] == base_airport for f in flights):
                return None
            origins, dests, dep_times, arr_times, fly_times = flights_to_tensors(flights, WINDOW_DAYS)
            return flights, origins, dests, dep_times, arr_times, fly_times, base_airport

        _first_base = all_base_ids["delta"][0]
        base_constraint = _CONSTRAINT_FN["delta"](_first_base)
    else:
        DATA_PATH = config.AIRLINE_DATA[config.AIRLINE]
        _df_cache = pd.read_csv(DATA_PATH, usecols=["ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "FL_DATE"]).dropna()
        _df_cache["FL_DATE"] = pd.to_datetime(_df_cache["FL_DATE"], format="mixed")
        total_days = _df_cache["FL_DATE"].nunique()
        max_offset = max(0, total_days - WINDOW_DAYS)

        def flight_sampler():
            """에피소드마다 (base, window) 쌍 랜덤 선택 → base-first sampling으로 flight 로드"""
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
            origins, dests, dep_times, arr_times, fly_times = flights_to_tensors(flights, WINDOW_DAYS)
            return flights, origins, dests, dep_times, arr_times, fly_times, base_airport

        base_constraint = _CONSTRAINT_FN[config.AIRLINE](base_ids[0])  # base는 에피소드마다 교체됨

    # sample_constraint: Stage 3 + Phase 2 공용 constraint 샘플러
    # phase2_only=True일 때도 Phase 2에서 사용하므로 두 블록 밖에서 정의
    _stage3_base = {**base_constraint, "max_duty_periods": 2, "max_pairing_days": WINDOW_DAYS - 1}
    def sample_constraint():
        r = config.STAGE3_CONSTRAINT_RANGES
        if multi_airline:
            airline_base = _CONSTRAINT_FN[_selected_airline[0]](0)
            base = {**airline_base, "max_duty_periods": 2, "max_pairing_days": WINDOW_DAYS - 1}
        else:
            base = _stage3_base
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

    if phase2_only:
        ckpt_path = os.path.join(save_dir, "stage3_best.pt")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        print(f"stage3_best.pt 로드 완료 → Phase 2만 실행")

    if not phase2_only:
        # ── Stage 1: 단일 duty (overnight 없음) ──────────────────────────
        # max_duty_periods=1 → END_DUTY 불가 → 당일 connection만 학습
        stage1_c = {**base_constraint, "max_duty_periods": 1, "max_pairing_days": 1}
        run_curriculum_stage(1, encoder, decoder, optimizer,
                             n_episodes=1000, constraint_override=stage1_c,
                             save_dir=save_dir, flight_sampler=flight_sampler,
                             global_step_offset=0,
                             entropy_start=0.05, entropy_end=0.005)

        # ── Stage 2: full multi-day ───────────────────────────────────────
        # overnight connection 포함 전체 multi-day pairing 학습
        # max_pairing_days를 WINDOW_DAYS로 제한 — window 밖 pairing은 데이터 없어 deadhead만 유발
        stage2_c = {**base_constraint, "max_duty_periods": 2, "max_pairing_days": WINDOW_DAYS - 1}
        run_curriculum_stage(2, encoder, decoder, optimizer,
                             n_episodes=2000, constraint_override=stage2_c,
                             save_dir=save_dir, flight_sampler=flight_sampler,
                             global_step_offset=1000,
                             entropy_start=0.02, entropy_end=0.005)

        # ── Stage 3: 7개 constraint 전체 랜덤 augmentation (FiLM 학습) ───
        # 매 에피소드 7개 constraint 전부 랜덤 샘플링 → FiLM이 다양한 constraint에 적응
        # max_pairing_days 상한도 WINDOW_DAYS-1로 제한
        # sample_constraint, _stage3_base는 Phase 2와 공용 — if not phase2_only 블록 밖에서 정의됨
        run_curriculum_stage(3, encoder, decoder, optimizer,
                             n_episodes=2000, constraint_override=_stage3_base,
                             save_dir=save_dir, flight_sampler=flight_sampler,
                             constraint_sampler=sample_constraint,
                             global_step_offset=3000,
                             entropy_start=0.01, entropy_end=0.005)

    # ── Phase 2: CG dual feedback ──────────────────────────────────────
    # Stage 3 이후 동일 모델 이어서 학습 (Phase 1 → Phase 2 연속)
    phase2_c = {**base_constraint, "max_duty_periods": 2, "max_pairing_days": WINDOW_DAYS - 1}
    phase2_offset = 0 if phase2_only else 1000 + 2000 + 2000

    run_phase2(encoder, decoder, optimizer,
               n_episodes=config.PHASE2_N_EPISODES,
               constraint=phase2_c,
               save_dir=save_dir,
               flight_sampler=flight_sampler,
               global_step_offset=phase2_offset,
               constraint_sampler=sample_constraint)

    # ── FiLM 검증: constraint별 greedy 결과 비교 ─────────────────────
    print()
    print("=" * 60)
    print("FiLM 검증: 같은 flights, 다른 max_legs")
    print("=" * 60)

    encoder.eval()
    decoder.eval()
    # 검증용 고정 데이터 (offset=0, base=ATL) — n_max 지정으로 수천 개 로드 방지
    val_base    = base_ids[0]
    val_flights = load_flights_rolling(
        DATA_PATH, WINDOW_DAYS, 0, airport_map,
        base_airport=val_base, n_max=config.EPISODE_MAX_FLIGHTS,
        df=_df_cache,
    )
    val_origins, val_dests, val_dep_times, val_arr_times, val_fly_times = flights_to_tensors(val_flights, WINDOW_DAYS)

    # max_legs=[2,4,6,8]: 4배 변동폭 → FiLM이 constraint에 따라 다르게 행동하는지 명확하게 확인
    # max_duty 12~14h(14% 변동)보다 훨씬 큰 신호 차이를 가짐
    with torch.no_grad():
        for legs in [2, 4, 6, 8]:
            c = {**_CONSTRAINT_FN[config.AIRLINE](val_base), "max_legs": legs,
                 "max_duty_periods": 2, "max_pairing_days": WINDOW_DAYS}
            enc = encoder(val_origins, val_dests, val_dep_times, val_arr_times, val_fly_times, constraint_to_tensor(c))
            _, _, _, metrics = run_episode(val_flights, c, encoder, decoder, enc, greedy=True)
            print(f"  max_legs={legs} → pairings: {metrics['n_pairings']:3d}  "
                  f"deadheads: {metrics['n_deadheads']:3d}  "
                  f"coverage: {metrics['coverage_pct']:5.1f}%")

    # ── 최종 모델 저장 ────────────────────────────────────────────────
    torch.save({
        "encoder":        encoder.state_dict(),
        "decoder":        decoder.state_dict(),
        "n_airports":     n_airports,
        "constraint_dim": len(FILM_CONSTRAINT_KEYS),
        "bases":          airline_bases,
        "window_days":    WINDOW_DAYS,
        "max_time":       WINDOW_DAYS * 24,  # evaluate_ip.py가 ckpt["max_time"]으로 직접 읽음
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
    parser.add_argument("--multi-airline", action="store_true",
                        help="Delta/Alaska/JetBlue 세 항공사 데이터로 동시 학습 (통합 airport_map 사용)")
    parser.add_argument("--skip-film", action="store_true",
                        help="FiLM 비활성화 (use_film_before=False, use_film_after=False) — ablation B/D용")
    args = parser.parse_args()
    _set_device(args.device)
    print(f"device: {DEVICE}")
    print(f"log: {args.log}")
    train(phase2_only=args.phase2_only, multi_airline=args.multi_airline, skip_film=args.skip_film)
