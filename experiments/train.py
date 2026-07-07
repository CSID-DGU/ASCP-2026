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
    """airline에 맞는 get_mask/step/final_reward 구현으로 전환 (turkish는 HB1/HB2 비대칭
    종료 허용). run_episode 등 이 모듈의 get_mask/step/final_reward를
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
    "turkish": get_turkish_constraints_hb,  # HB1/HB2 비대칭 종료 허용 (base_ids는 train()에서 주입)
}
from state import init_state
from utils import flights_to_tensors, constraint_to_tensor, state_to_vec, flight_gap_bias
import config

DEVICE = torch.device("cpu")  # train() 호출 전 _set_device()로 설정
USE_UTC = False  # dep_time UTC 앵커링 여부 — --use-utc로 켬


def _set_device(device_str: str):
    global DEVICE
    DEVICE = torch.device(device_str)


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
    n_end_duties = 0
    total_legs_sum = 0

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
                total_legs_sum += state.get("total_legs", 0) 
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
                "total_legs":         0,
                "remaining":          sum(1 for v in assigned.values() if not v),
                "pairing_start":      True,
                "duty_period":        0,
                "pairing_start_time": earliest["dep_time"],
                "is_resting":         False,
                "rest_end_time":      None,
            }
            continue

        # 찬주 decoder
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
    }
    return total_reward, log_probs, entropies, metrics


# ── Phase 2 helpers ──────────────────────────────────────────────────────────
_LEG_BONUS_IP        = config.IP_LEG_BONUS           
_DEADHEAD_PENALTY_IP = config.IP_DEADHEAD_PENALTY     
_PAIRING_FIXED_COST  = config.IP_PAIRING_FIXED_COST  


def _rollout_with_pairings(flights, constraint, encoder, decoder, encoded, greedy=False):
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
            flush_pairing(is_forced=False)
            break

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

        state_vec = state_to_vec(state, encoder, constraint, device=DEVICE)
        gap_bias  = flight_gap_bias(state, flights, constraint, device=DEVICE)
        probs     = decoder(encoded, state_vec, mask, gap_bias=gap_bias)
        action    = probs.argmax().item() if greedy else Categorical(probs).sample().item()

        if action == len(flights):          # END_DUTY
            pairing_rest += constraint.get("min_rest", 10.0)
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
    pool = {}
    for _ in range(n_rollouts):
        for p in _rollout_with_pairings(flights, constraint, encoder, decoder, encoded):
            key = tuple(sorted(p["legs"]))
            if key not in pool or p["cost"] < pool[key]["cost"]:
                pool[key] = p
    for p in _rollout_with_pairings(flights, constraint, encoder, decoder, encoded, greedy=True):
        key = tuple(sorted(p["legs"]))
        if key not in pool or p["cost"] < pool[key]["cost"]:
            pool[key] = p
    return list(pool.values())


def run_episode_with_dual(flights, constraint, encoder, decoder, encoded, dual_vars, greedy=False, dual_weight=None):
    assigned = {f["id"]: False for f in flights}
    state    = init_state(flights, constraint)

    log_probs    = []
    entropies    = []
    total_reward = 0
    n_pairings    = 0
    n_deadheads   = 0
    n_end_duties  = 0
    total_legs_sum = 0
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
            base_unassigned = [f for f in unassigned if f["origin"] == base]
            earliest = sorted(base_unassigned or unassigned, key=lambda x: x["dep_time"])[0]
            if not state.get("pairing_start", False):
                total_legs_sum += state.get("total_legs", 0)  # 측정 버그 수정: deadhead 시 legs 분자에 포함
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
                "total_legs":         0,
                "remaining":          sum(1 for v in assigned.values() if not v),
                "pairing_start":      True,
                "duty_period":        0,
                "pairing_start_time": earliest["dep_time"],
                "is_resting":         False,
                "rest_end_time":      None,
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
            state, r, done = step(state, action, flights, assigned, constraint)
            total_reward += r
            if done:
                break
            continue

        flight_id = flights[action]["id"]
        _dw = dual_weight if dual_weight is not None else config.PHASE2_DUAL_WEIGHT
        state, r, done = step(state, action, flights, assigned, constraint)
        total_reward += r + dual_vars.get(flight_id, 0.0) * _dw
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
    }


def run_phase2(encoder, decoder, optimizer, n_episodes, constraint, save_dir, flight_sampler,
               global_step_offset=0, entropy_start=0.01, entropy_end=0.005,
               constraint_sampler=None, init_best=float("inf")):
    from set_partition import solve_lp_relaxation

    params            = list(encoder.parameters()) + list(decoder.parameters())
    best_avg_pairings = init_best
    greedy_pairings   = []
    dual_vars         = {}  

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
        c_tensor = constraint_to_tensor(c, device=DEVICE)

        with torch.no_grad():
            encoded = encoder(origins, dests, dep_times, arr_times, fly_times, c_tensor)

            if ep % config.PHASE2_LP_INTERVAL == 0:
                pool      = _collect_pool(flights, c, encoder, decoder, encoded,
                                          n_rollouts=config.PHASE2_POOL_ROLLOUTS)
                lp_result = solve_lp_relaxation(pool)
                if lp_result is not None:
                    dual_vars = lp_result["dual_vars"]  

        _eff_dw = config.PHASE2_DUAL_WEIGHT * min(1.0, (ep + 1) / max(config.PHASE2_DUAL_WARMUP, 1))
        encoded_train = encoder(origins, dests, dep_times, arr_times, fly_times, c_tensor)
        reward_s, log_probs, entropies, metrics_s = run_episode_with_dual(
            flights, c, encoder, decoder, encoded_train, dual_vars, dual_weight=_eff_dw
        )
        if len(log_probs) == 0:
            continue

        with torch.no_grad():
            encoded_g = encoder(origins, dests, dep_times, arr_times, fly_times, c_tensor)
            reward_g, _, _, metrics_g = run_episode_with_dual(
                flights, c, encoder, decoder, encoded_g, dual_vars, greedy=True, dual_weight=_eff_dw
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
            "phase2/dual_weight":         _eff_dw,
            "phase2/gap_weight":          decoder.gap_weight.item(),
        }, step=global_step_offset + ep)

        if ep % 25 == 0:
            print(
                f"  Ep {ep:4d} | "
                f"sample: p={metrics_s['n_pairings']:3d} dh={metrics_s['n_deadheads']:3d} | "
                f"greedy: p={metrics_g['n_pairings']:3d} legs={metrics_g.get('avg_legs', 0):.2f} (avg25={avg25:5.1f}) | "
                f"adv: {advantage:6.3f} | dw={_eff_dw:.3f} | dual keys: {len(dual_vars)}"
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


def train(phase2_only=False, multi_airline=False, skip_film=False, ckpt_dir=None, from_stage2=False, turkish_files=None):
    WINDOW_DAYS = config.WINDOW_DAYS  # config.py에서 관리 — max_pairing_days 상한과 연동

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
            # HB1/HB2 비대칭 종료 허용 — base_ids를 클로저로 캡처해 get_turkish_constraints_hb에 주입
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
    run_name = "phase2-only" if phase2_only else tag
    wandb.init(
        project="ASCP-2026-chanju",
        name=run_name,
        config={
            "airline":            "multi" if multi_airline else config.AIRLINE,
            "multi_airline":      multi_airline,
            "window_days":        WINDOW_DAYS,
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
            airline      = random.choice(airlines)
            _selected_airline[0] = airline
            base_airport = random.choice(all_base_ids[airline])
            offset_days  = random.randint(0, _max_offsets[airline])
            flights = load_flights_rolling(
                config.AIRLINE_DATA[airline], WINDOW_DAYS, offset_days, airport_map,
                base_airport=base_airport,
                n_max=config.EPISODE_MAX_FLIGHTS,
                df=_df_caches[airline],
                use_utc=USE_UTC,
            )
            if not flights:
                return None
            if not any(f["origin"] == base_airport for f in flights):
                return None
            origins, dests, dep_times, arr_times, fly_times = flights_to_tensors(flights, WINDOW_DAYS * 24.0, device=DEVICE)
            return flights, origins, dests, dep_times, arr_times, fly_times, base_airport

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
            airline_base = _CONSTRAINT_FN[_selected_airline[0]](0) # 혹은 기존 찬주님 코드 방식대로 매칭
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
            stage1_c = {**base_constraint, "max_duty_periods": 1, "max_pairing_days": 1}
            run_curriculum_stage(1, encoder, decoder, optimizer,
                                 n_episodes=1000, constraint_override=stage1_c,
                                 save_dir=save_dir, flight_sampler=flight_sampler,
                                 global_step_offset=0,
                                 entropy_start=0.30, entropy_end=0.005)

            # ── Stage 2: full multi-day ───────────────────────────────────────
            stage2_c = {**base_constraint, "max_duty_periods": 2, "max_pairing_days": WINDOW_DAYS - 1}
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
               constraint_sampler=sample_constraint)

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
    parser.add_argument("--airline", default=None,
                        help="단일 항공사 지정 (delta/alaska/jetblue/turkish). 미지정 시 config.AIRLINE 사용")
    parser.add_argument("--turkish-files", default=None,
                        help="Turkish 학습 시 사용할 .legs 파일 이름 콤마 구분 (예: tt201401.legs). 미지정 시 "
                             "Zeren Feb 벤치마크 윈도우(tt201402.legs, 2/1~3/8, 15,742편) 기본 사용")
    parser.add_argument("--use-utc", action="store_true",
                        help="dep_time을 UTC 절대시간으로 앵커링. 새로 이 옵션으로 학습한 모델만 "
                             "이 옵션 켠 채로 평가해야 함 — 기존 체크포인트에 켜면 OOD")
    args = parser.parse_args()
    if args.airline:
        config.AIRLINE = args.airline
    _set_device(args.device)
    USE_UTC = args.use_utc
    print(f"device: {DEVICE}")
    print(f"use_utc: {USE_UTC}")
    print(f"log: {args.log}")
    _turkish_files = [f.strip() for f in args.turkish_files.split(",")] if args.turkish_files else None
    train(phase2_only=args.phase2_only, multi_airline=args.multi_airline, skip_film=args.skip_film,
          ckpt_dir=args.ckpt_dir, from_stage2=args.from_stage2, turkish_files=_turkish_files)
