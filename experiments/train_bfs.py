"""train_bfs.py — BFS connectivity-aware sampling으로 훈련하는 실험 버전

train.py와의 유일한 차이:
  flight_sampler()에서 n_max 랜덤 추출 대신 BFS sampling 사용
  → 에피소드마다 base 출발편 중심으로 연결 가능한 600편을 선택
  → 훈련 중 multi-leg pairing 형성 밀도 향상 목적

기존 train.py는 그대로 유지
"""

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
from utils import constraint_to_tensor, flights_to_tensors, state_to_vec
from rollout import rollout_with_pairings as _rollout_with_pairings

_CONSTRAINT_FN = {
    "delta":   get_delta_constraints,
    "alaska":  get_alaska_constraints,
    "jetblue": get_jetblue_constraints,
}
from state import init_state
import config

DEVICE = torch.device("cpu")


def _set_device(device_str: str):
    global DEVICE
    DEVICE = torch.device(device_str)


# ── BFS connectivity-aware subset sampling ────────────────────────────────────

def sample_connected_subset(window_flights, subset_size, base_id, min_conn=0.65, max_conn=9.0):
    """BFS로 base 출발편에서 시작해 연결 가능한 편을 우선 선택.

    train.py의 랜덤 n_max 추출 대비 subset 내 연결 밀도가 높아
    RL이 multi-leg pairing을 형성할 수 있다.
    """
    by_origin = {}
    for f in window_flights:
        by_origin.setdefault(f["origin"], []).append(f)

    selected_ids = set()
    selected = []

    base_departs = [f for f in window_flights if f["origin"] == base_id]
    random.shuffle(base_departs)
    queue = list(base_departs)

    while queue and len(selected) < subset_size:
        f = queue.pop(0)
        if f["id"] in selected_ids:
            continue
        selected_ids.add(f["id"])
        selected.append(f)

        nexts = [
            g for g in by_origin.get(f["dest"], [])
            if g["id"] not in selected_ids
            and min_conn <= g["dep_time"] - f["arr_time"] <= max_conn
        ]
        random.shuffle(nexts)
        queue.extend(nexts)

    # BFS로 못 채웠으면 base 인접편으로 보충
    if len(selected) < subset_size:
        others = [f for f in window_flights
                  if f["id"] not in selected_ids
                  and (f["origin"] == base_id or f["dest"] == base_id)]
        random.shuffle(others)
        for f in others[:subset_size - len(selected)]:
            selected_ids.add(f["id"])
            selected.append(f)

    # 그래도 부족하면 나머지 임의 보충
    if len(selected) < subset_size:
        others = [f for f in window_flights if f["id"] not in selected_ids]
        random.shuffle(others)
        for f in others[:subset_size - len(selected)]:
            selected.append(f)

    selected = sorted(selected, key=lambda f: f["dep_time"])
    for local_id, f in enumerate(selected):
        f["id"] = local_id

    return selected


# ── 이하 train.py와 동일 ──────────────────────────────────────────────────────

def run_episode(flights, constraint, encoder, decoder, encoded, greedy=False):
    assigned = {f["id"]: False for f in flights}
    state = init_state(flights, constraint)

    log_probs = []
    entropies = []
    total_reward = 0
    n_pairings = 0
    n_deadheads = 0

    max_steps = len(flights) * 20
    step_count = 0
    while True:
        step_count += 1
        if step_count > max_steps:
            break
        mask_list = get_mask(state, flights, assigned, constraint)
        mask = torch.tensor(mask_list, dtype=torch.float32).to(DEVICE)

        no_flight      = sum(mask_list[:-2]) == 0
        no_end_duty    = mask_list[-2] == 0
        no_end_pairing = mask_list[-1] == 0
        if no_flight and no_end_duty and no_end_pairing:
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if len(unassigned) == 0:
                break

            base = constraint["base_airport"]
            base_unassigned = [f for f in unassigned if f["origin"] == base]
            earliest = sorted(base_unassigned or unassigned, key=lambda x: x["dep_time"])[0]

            if not state.get("pairing_start", False):
                n_pairings += 1
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

        if action == n_flights:
            state, r, done = step(state, action, flights, assigned, constraint)
            total_reward += r
            continue

        if action == n_flights + 1:
            n_pairings += 1
            state, r, done = step(state, action, flights, assigned, constraint)
            total_reward += r
            if done:
                break
            continue

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


def _collect_pool(flights, constraint, encoder, decoder, encoded, n_rollouts):
    pool = {}
    for _ in range(n_rollouts):
        for p in _rollout_with_pairings(flights, constraint, encoder, decoder, encoded, device=DEVICE):
            key = tuple(sorted(p["legs"]))
            if key not in pool or p["cost"] < pool[key]["cost"]:
                pool[key] = p
    for p in _rollout_with_pairings(flights, constraint, encoder, decoder, encoded, greedy=True, device=DEVICE):
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

        if action == n_flights:
            state, r, done = step(state, action, flights, assigned, constraint)
            total_reward += r
            continue

        if action == n_flights + 1:
            n_pairings += 1
            state, r, done = step(state, action, flights, assigned, constraint)
            total_reward += r
            if done:
                break
            continue

        flight_id = flights[action]["id"]
        state, r, done = step(state, action, flights, assigned, constraint)
        _dw = dual_weight if dual_weight is not None else config.PHASE2_DUAL_WEIGHT
        total_reward += r + dual_vars.get(flight_id, 0.0) * _dw
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

    print(f"  → best avg pairings: {best_avg_pairings:.1f}")
    return best_avg_pairings


def run_curriculum_stage(
    stage, encoder, decoder, optimizer,
    n_episodes, constraint_override, save_dir,
    flight_sampler, constraint_sampler=None,
    global_step_offset=0,
    entropy_start=0.05, entropy_end=0.005,
):
    best_avg_pairings = float("inf")
    greedy_pairings = []

    print(f"\n{'='*60}")
    print(f"Curriculum Stage {stage}: max_duty_periods={constraint_override['max_duty_periods']}, "
          f"max_pairing_days={constraint_override['max_pairing_days']}"
          + (" [constraint 랜덤 샘플링]" if constraint_sampler else ""))
    print(f"{'='*60}")

    params = list(encoder.parameters()) + list(decoder.parameters())

    for ep in range(n_episodes):
        sample = flight_sampler()
        if sample is None:
            continue
        flights, origins, dests, dep_times, arr_times, fly_times, base_airport = sample

        c = constraint_sampler() if constraint_sampler else constraint_override
        c = {**c, "base_airport": base_airport}
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

    print(f"  → best avg pairings: {best_avg_pairings:.1f}")
    return best_avg_pairings


def train(phase2_only=False, multi_airline=False, skip_film=False, ckpt_dir=None, from_stage2=False):
    WINDOW_DAYS = config.WINDOW_DAYS

    if multi_airline:
        all_paths = list(config.AIRLINE_DATA.values())
        airport_map = build_airport_map(all_paths)
        airlines = list(config.AIRLINE_DATA.keys())
        all_base_ids = {a: bases_to_ids(config.AIRLINE_BASES[a], airport_map) for a in airlines}
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

    tag = "multi-airline" if multi_airline else "delta"
    tag += "-nofilm" if skip_film else ""
    tag += "-bfs"  # BFS sampling 버전임을 wandb에 표시
    run_name = "phase2-only-bfs" if phase2_only else tag
    wandb.init(
        project="ASCP-2026-chanju",
        name=run_name,
        config={
            "airline":             "multi" if multi_airline else config.AIRLINE,
            "multi_airline":       multi_airline,
            "window_days":         WINDOW_DAYS,
            "episode_max_flights": config.EPISODE_MAX_FLIGHTS,
            "sampling":            "bfs_connected",  # 기존 random과 구분
            "phase2_lp_interval":  config.PHASE2_LP_INTERVAL,
            "phase2_pool_rollouts": config.PHASE2_POOL_ROLLOUTS,
            "phase2_dual_weight":  config.PHASE2_DUAL_WEIGHT,
            "phase2_n_episodes":   config.PHASE2_N_EPISODES,
            "lr":                  1e-4,
            "device":              str(DEVICE),
        },
        resume="allow",
    )

    save_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints", wandb.run.id)
    os.makedirs(save_dir, exist_ok=True)

    import pandas as pd

    # ── BFS flight_sampler ─────────────────────────────────────────────────────
    # train.py와 다른 유일한 부분: n_max 랜덤 추출 → 전체 로드 후 BFS 선택

    if multi_airline:
        _df_caches = {}
        _max_offsets = {}
        for a in airlines:
            p = config.AIRLINE_DATA[a]
            df = pd.read_csv(p, usecols=["ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "FL_DATE"]).dropna()
            df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="mixed")
            _df_caches[a]   = df
            _max_offsets[a] = max(0, df["FL_DATE"].nunique() - WINDOW_DAYS)

        _selected_airline = ["delta"]

        def flight_sampler():
            airline      = random.choice(airlines)
            _selected_airline[0] = airline
            base_airport = random.choice(all_base_ids[airline])
            offset_days  = random.randint(0, _max_offsets[airline])

            # 전체 윈도우 로드 후 BFS로 EPISODE_MAX_FLIGHTS편 선택
            flights_all = load_flights_rolling(
                config.AIRLINE_DATA[airline], WINDOW_DAYS, offset_days, airport_map,
                base_airport=base_airport,
                n_max=None,
                df=_df_caches[airline],
            )
            if not flights_all:
                return None
            flights = sample_connected_subset(
                flights_all, config.EPISODE_MAX_FLIGHTS, base_airport
            )
            if not flights or not any(f["origin"] == base_airport for f in flights):
                return None
            origins, dests, dep_times, arr_times, fly_times = flights_to_tensors(
                flights, WINDOW_DAYS * 24.0, device=DEVICE
            )
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
            base_airport = random.choice(base_ids)
            offset_days  = random.randint(0, max_offset)

            # 전체 윈도우 로드 후 BFS로 EPISODE_MAX_FLIGHTS편 선택
            flights_all = load_flights_rolling(
                DATA_PATH, WINDOW_DAYS, offset_days, airport_map,
                base_airport=base_airport,
                n_max=None,
                df=_df_cache,
            )
            if not flights_all:
                return None
            flights = sample_connected_subset(
                flights_all, config.EPISODE_MAX_FLIGHTS, base_airport
            )
            if not flights or not any(f["origin"] == base_airport for f in flights):
                return None
            origins, dests, dep_times, arr_times, fly_times = flights_to_tensors(
                flights, WINDOW_DAYS * 24.0, device=DEVICE
            )
            return flights, origins, dests, dep_times, arr_times, fly_times, base_airport

        base_constraint = _CONSTRAINT_FN[config.AIRLINE](base_ids[0])

    _stage3_base = {**base_constraint, "max_duty_periods": 4, "max_pairing_days": WINDOW_DAYS - 1}
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
        load_dir  = ckpt_dir if ckpt_dir else save_dir
        ckpt_path = os.path.join(load_dir, "stage3_best.pt")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        print(f"stage3_best.pt 로드 완료: {ckpt_path} → Phase 2만 실행")

    _s3_best     = float("inf")
    _s3_ckpt_dir = save_dir

    if phase2_only:
        _s3_ckpt_dir = ckpt_dir if ckpt_dir else save_dir
        ckpt_path = os.path.join(_s3_ckpt_dir, "stage3_best.pt")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        print(f"stage3_best.pt 로드 완료: {ckpt_path} → Phase 2만 실행")

    if not phase2_only:
        if from_stage2:
            _s2_load_dir = ckpt_dir
            if not _s2_load_dir:
                raise ValueError("--from-stage2 사용 시 --ckpt-dir로 stage2_best.pt 폴더를 지정해야 합니다.")
            _s2_ckpt = torch.load(os.path.join(_s2_load_dir, "stage2_best.pt"), map_location=DEVICE, weights_only=True)
            encoder.load_state_dict(_s2_ckpt["encoder"])
            decoder.load_state_dict(_s2_ckpt["decoder"])
            print(f"stage2_best.pt 로드: {_s2_load_dir} → Stage 3부터 실행")
        else:
            stage1_c = {**base_constraint, "max_duty_periods": 1, "max_pairing_days": 1}
            run_curriculum_stage(1, encoder, decoder, optimizer,
                                 n_episodes=1000, constraint_override=stage1_c,
                                 save_dir=save_dir, flight_sampler=flight_sampler,
                                 global_step_offset=0,
                                 entropy_start=0.15, entropy_end=0.005)

            stage2_c = {**base_constraint, "max_duty_periods": 2, "max_pairing_days": WINDOW_DAYS - 1}
            run_curriculum_stage(2, encoder, decoder, optimizer,
                                 n_episodes=2000, constraint_override=stage2_c,
                                 save_dir=save_dir, flight_sampler=flight_sampler,
                                 global_step_offset=1000,
                                 entropy_start=0.02, entropy_end=0.005)

        _s3_best = run_curriculum_stage(3, encoder, decoder, optimizer,
                             n_episodes=2000, constraint_override=_stage3_base,
                             save_dir=save_dir, flight_sampler=flight_sampler,
                             constraint_sampler=sample_constraint,
                             global_step_offset=0 if from_stage2 else 3000,
                             entropy_start=0.01, entropy_end=0.005)

        # Phase 2 시작 전 stage3_best.pt 로드 — 마지막 epoch이 아닌 best checkpoint에서 시작
        _s3_ckpt = torch.load(os.path.join(save_dir, "stage3_best.pt"), map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(_s3_ckpt["encoder"])
        decoder.load_state_dict(_s3_ckpt["decoder"])
        print(f"Phase 2 시작: stage3_best.pt 로드 (best_avg={_s3_ckpt.get('best_avg_pairings', 0):.1f})")

    # ── FiLM 검증 공용 셋업 (Phase 2 전/후 공용) ──────────────────────────
    N_FILM_ROLLOUTS = 10
    val_base = base_ids[0] if not multi_airline else all_base_ids["delta"][0]
    val_flights_all = load_flights_rolling(
        DATA_PATH if not multi_airline else config.AIRLINE_DATA["delta"],
        WINDOW_DAYS, 0, airport_map,
        base_airport=val_base, n_max=None,
        df=_df_cache if not multi_airline else _df_caches["delta"],
    )
    val_flights = sample_connected_subset(val_flights_all, config.EPISODE_MAX_FLIGHTS, val_base)
    val_origins, val_dests, val_dep_times, val_arr_times, val_fly_times = flights_to_tensors(
        val_flights, WINDOW_DAYS * 24.0, device=DEVICE
    )
    _val_constraint_fn = _CONSTRAINT_FN[config.AIRLINE if not multi_airline else "delta"]

    def _film_validation(label):
        encoder.eval(); decoder.eval()
        print()
        print("=" * 60)
        print(f"FiLM 검증 ({label}): 같은 flights, 다른 max_duty_periods (stochastic×{N_FILM_ROLLOUTS})")
        print("=" * 60)
        with torch.no_grad():
            for dp in [1, 2, 3, 4]:
                val_c = {**_val_constraint_fn(val_base), "max_duty_periods": dp,
                         "max_pairing_days": WINDOW_DAYS}
                enc = encoder(val_origins, val_dests, val_dep_times, val_arr_times, val_fly_times,
                              constraint_to_tensor(val_c, device=DEVICE))
                p_list, dh_list, cov_list = [], [], []
                for _ in range(N_FILM_ROLLOUTS):
                    _, _, _, m = run_episode(val_flights, val_c, encoder, decoder, enc, greedy=False)
                    p_list.append(m["n_pairings"])
                    dh_list.append(m["n_deadheads"])
                    cov_list.append(m["coverage_pct"])
                print(f"  max_duty_periods={dp} → "
                      f"pairings(avg{N_FILM_ROLLOUTS})={sum(p_list)/len(p_list):.1f}  "
                      f"deadheads={sum(dh_list)/len(dh_list):.1f}  "
                      f"coverage={sum(cov_list)/len(cov_list):.1f}%")
        encoder.train(); decoder.train()

    # Stage 3 best 기준 FiLM 검증 (Phase 2 전)
    if not phase2_only:
        _film_validation("Stage 3 best")

    phase2_c = {**base_constraint, "max_duty_periods": 2, "max_pairing_days": WINDOW_DAYS - 1}
    phase2_offset = 0 if phase2_only else (2000 if from_stage2 else 1000 + 2000 + 2000)

    run_phase2(encoder, decoder, optimizer,
               n_episodes=config.PHASE2_N_EPISODES,
               constraint=phase2_c,
               save_dir=save_dir,
               flight_sampler=flight_sampler,
               global_step_offset=phase2_offset,
               constraint_sampler=sample_constraint,
               init_best=_s3_best)

    # Phase 2 후 FiLM 최종 검증 — stage3_best.pt 복원 후 (Phase 2가 FiLM 덮어쓸 수 있으므로)
    _film_ckpt = torch.load(os.path.join(_s3_ckpt_dir, "stage3_best.pt"), map_location=DEVICE, weights_only=True)
    encoder.load_state_dict(_film_ckpt["encoder"])
    decoder.load_state_dict(_film_ckpt["decoder"])
    print("FiLM 최종 검증: stage3_best.pt 로드")
    _film_validation("final / stage3_best")

    torch.save({
        "encoder":        encoder.state_dict(),
        "decoder":        decoder.state_dict(),
        "n_airports":     n_airports,
        "constraint_dim": len(FILM_CONSTRAINT_KEYS),
        "bases":          airline_bases if not multi_airline else list(config.AIRLINE_BASES.values()),
        "window_days":    WINDOW_DAYS,
        "max_time":       WINDOW_DAYS * 24,
    }, os.path.join(save_dir, "model_latest.pt"))
    print(f"\n모델 저장 완료")

    wandb.finish(quiet=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log", default=os.path.join(os.path.dirname(__file__), "..", "log", "train_bfs_log.txt"))
    parser.add_argument("--phase2-only", action="store_true")
    parser.add_argument("--from-stage2", action="store_true",
                        help="stage2_best.pt 로드 후 Stage 3 + Phase 2만 실행 (--ckpt-dir 필수)")
    parser.add_argument("--ckpt-dir", default=None,
                        help="--phase2-only: stage3_best.pt 폴더 / --from-stage2: stage2_best.pt 폴더")
    parser.add_argument("--multi-airline", action="store_true")
    parser.add_argument("--skip-film", action="store_true")
    args = parser.parse_args()
    _set_device(args.device)
    print(f"[train_bfs] device: {DEVICE}  sampling: BFS connected")
    train(phase2_only=args.phase2_only, multi_airline=args.multi_airline, skip_film=args.skip_film,
          ckpt_dir=args.ckpt_dir, from_stage2=args.from_stage2)
