import os
import sys
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RL"))
import torch
import torch.optim as optim
from torch.distributions import Categorical

from model import FlightEncoder, PointerDecoder
from loader import build_airport_map, bases_to_ids, load_flights_rolling
from environment import get_mask, step, final_reward
from constraints import get_delta_constraints, FILM_CONSTRAINT_KEYS
from state import init_state
import config



def constraint_to_tensor(constraint):
    """constraint dict → FiLM 입력 tensor (정규화 적용)

    정규화 기준값은 config.CONSTRAINT_NORMS에서 관리.
    evaluate_ip.py도 동일한 값을 써야 checkpoint 호환됨.
    """
    return torch.tensor(
        [constraint[k] / config.CONSTRAINT_NORMS[k] for k in FILM_CONSTRAINT_KEYS],
        dtype=torch.float32,
    )

def flights_to_tensors(flights, window_days=5):
    """혜린 flight dict → 찬주 encoder 입력 tensor 변환

    dep/arr/fly_time을 window_days * 24 기준으로 정규화.
    fly_time = arr - dep (비행 시간) — encoder input_dim이 airport_emb*2 + 3이므로 필수.
    """
    origins = torch.tensor([f["origin"] for f in flights])
    dests = torch.tensor([f["dest"] for f in flights])
    max_time = window_days * 24.0
    dep_raw = torch.tensor([f["dep_time"] for f in flights], dtype=torch.float32)
    arr_raw = torch.tensor([f["arr_time"] for f in flights], dtype=torch.float32)
    dep_times = dep_raw / max_time
    arr_times = arr_raw / max_time
    fly_times = (arr_raw - dep_raw) / max_time
    return origins, dests, dep_times, arr_times, fly_times


def state_to_vec(state, encoder, constraint):
    """혜린 state dict → 찬주 decoder 입력 tensor 변환

    state_vec(71,) = current_emb(32) + base_emb(32) + scalars(7)
    7개 스칼라: time_of_day, day_norm, duty_elapsed/max, legs/max, duty_period/max, is_resting, rest_remaining
    """
    current_emb = encoder.airport_emb(torch.tensor(state["current_airport"]))
    base_emb    = encoder.airport_emb(torch.tensor(constraint["base_airport"]))

    max_pairing_days = constraint.get("max_pairing_days", 5)
    time_of_day      = (state["current_time"] % 24.0) / 24.0
    day_norm         = (state["current_time"] // 24.0) / max(max_pairing_days, 1)
    duty_period_norm = state.get("duty_period", 0) / max(constraint.get("max_duty_periods", 4), 1)

    # duty_elapsed: 비행 시간만이 아닌 FAA 기준 실제 경과 시간 (비행 + 대기)
    # is_resting/pairing_start 중이면 새 duty 아직 시작 안 함 → 0
    if state.get("is_resting", False) or state.get("pairing_start", False):
        duty_elapsed = 0.0
    else:
        duty_elapsed = max(0.0, state["current_time"] - state.get("duty_start_time", state["current_time"]))

    # rest_remaining: is_resting=True일 때 남은 rest 시간 비율 (0~1), 아니면 0.0
    min_rest = constraint.get("min_rest", 10.0)
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
            duty_elapsed / constraint["max_duty"],
            state["legs"] / constraint["max_legs"],
            duty_period_norm,
            1.0 if state.get("is_resting", False) else 0.0,
            rest_remaining,
        ], dtype=torch.float32)
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
        mask = torch.tensor(mask_list, dtype=torch.float32)

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


def run_curriculum_stage(
    stage, encoder, decoder, optimizer,
    n_episodes, constraint_override, save_dir,
    flight_sampler, constraint_sampler=None,
):
    """
    커리큘럼 1단계 실행.

    flight_sampler: () → (flights, origins, dests, dep_times, arr_times, base_airport)
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
        flights, origins, dests, dep_times, arr_times, base_airport = sample

        c = constraint_sampler() if constraint_sampler else constraint_override
        c = {**c, "base_airport": base_airport}  # 에피소드별 base 주입
        c_tensor = constraint_to_tensor(c)
        encoded  = encoder(origins, dests, dep_times, arr_times, c_tensor)

        reward_s, log_probs, entropies, metrics_s = run_episode(
            flights, c, encoder, decoder, encoded, greedy=False
        )
        if len(log_probs) == 0:
            continue

        with torch.no_grad():
            encoded_g = encoder(origins, dests, dep_times, arr_times, c_tensor)
            reward_g, _, _, metrics_g = run_episode(
                flights, c, encoder, decoder, encoded_g, greedy=True
            )

        greedy_pairings.append(metrics_g["n_pairings"])
        advantage = (reward_s - reward_g) / (abs(reward_g) + 1e-6)

        loss = torch.stack([
            -lp * advantage - 0.01 * ent
            for lp, ent in zip(log_probs, entropies)
        ]).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # best checkpoint: greedy pairings 25ep 이동평균 기준
        if len(greedy_pairings) >= 25:
            recent_avg = sum(greedy_pairings[-25:]) / 25
            if recent_avg < best_avg_pairings:
                best_avg_pairings = recent_avg
                torch.save({
                    "encoder":   encoder.state_dict(),
                    "decoder":   decoder.state_dict(),
                    "stage":     stage,
                    "episode":   ep,
                    "best_avg_pairings": best_avg_pairings,
                }, os.path.join(save_dir, f"stage{stage}_best.pt"))

        if ep % 25 == 0:
            avg25 = sum(greedy_pairings[-25:]) / len(greedy_pairings[-25:])
            print(
                f"  Ep {ep:4d} | "
                f"sample: p={metrics_s['n_pairings']:3d} dh={metrics_s['n_deadheads']:3d} | "
                f"greedy: p={metrics_g['n_pairings']:3d} (avg25={avg25:5.1f}) | "
                f"adv: {advantage:6.3f}"
            )

    print(f"  → best avg pairings: {best_avg_pairings:.1f}  "
          f"(saved: checkpoints/stage{stage}_best.pt)")
    return best_avg_pairings


def train():
    DATA_PATH   = "RL/data/T_ONTIME_MARKETING.csv"
    # TODO: 협의 필요 — 우선 max_pairing_days=4(Delta CBA) 기준으로 4일 설정
    # window가 pairing 최대 기간과 맞아야 한 에피소드 안에서 완성된 pairing 학습 가능
    WINDOW_DAYS = 4

    # 항공사 base 설정 — config.py에서 AIRLINE 바꾸면 자동 반영
    airline_bases = config.AIRLINE_BASES[config.AIRLINE]

    # 전체 CSV 기준 공항 ID 고정 (에피소드 간 ID 일관성 보장)
    airport_map = build_airport_map(DATA_PATH)
    base_ids    = bases_to_ids(airline_bases, airport_map)
    n_airports  = len(airport_map)

    print(f"airports: {n_airports}개, airline: {config.AIRLINE}, bases: {airline_bases}")

    encoder = FlightEncoder(
        n_airports=n_airports,
        constraint_dim=len(FILM_CONSTRAINT_KEYS),
        airport_emb_dim=32,
        d_model=128,
    )
    decoder   = PointerDecoder(d_model=128, airport_emb_dim=32)
    params    = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=1e-4)

    save_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # 전체 날짜 수 파악 → offset_days 범위 결정
    import pandas as pd
    df_dates = pd.read_csv(DATA_PATH, usecols=["FL_DATE"])
    df_dates["FL_DATE"] = pd.to_datetime(df_dates["FL_DATE"], format="mixed")
    total_days = df_dates["FL_DATE"].nunique()
    max_offset = max(0, total_days - WINDOW_DAYS)

    def flight_sampler():
        """에피소드마다 (base, window) 쌍 랜덤 선택 → flight 로드"""
        base_airport = random.choice(base_ids)
        offset_days  = random.randint(0, max_offset)
        flights = load_flights_rolling(DATA_PATH, WINDOW_DAYS, offset_days, airport_map)
        if not flights:
            return None
        # base 출발 편 없는 window는 스킵 (state.py fallback 방지)
        if not any(f["origin"] == base_airport for f in flights):
            return None
        origins, dests, dep_times, arr_times = flights_to_tensors(flights, WINDOW_DAYS)
        return flights, origins, dests, dep_times, arr_times, base_airport

    base_constraint = get_delta_constraints(base_ids[0])  # base는 에피소드마다 교체됨

    # ── Stage 1: 단일 duty (overnight 없음) ──────────────────────────
    # max_duty_periods=1 → END_DUTY 불가 → 당일 connection만 학습
    stage1_c = {**base_constraint, "max_duty_periods": 1, "max_pairing_days": 1}
    run_curriculum_stage(1, encoder, decoder, optimizer,
                         n_episodes=1000, constraint_override=stage1_c,
                         save_dir=save_dir, flight_sampler=flight_sampler)

    # ── Stage 2: full multi-day ───────────────────────────────────────
    # overnight connection 포함 전체 multi-day pairing 학습
    # max_pairing_days를 WINDOW_DAYS로 제한 — window 밖 pairing은 데이터 없어 deadhead만 유발
    stage2_c = {**base_constraint, "max_duty_periods": 4, "max_pairing_days": WINDOW_DAYS}
    run_curriculum_stage(2, encoder, decoder, optimizer,
                         n_episodes=2000, constraint_override=stage2_c,
                         save_dir=save_dir, flight_sampler=flight_sampler)

    # ── Stage 3: 7개 constraint 전체 랜덤 augmentation (FiLM 학습) ───
    # 매 에피소드 7개 constraint 전부 랜덤 샘플링 → FiLM이 다양한 constraint에 적응
    # max_pairing_days 상한도 WINDOW_DAYS로 제한 (config.STAGE3_CONSTRAINT_RANGES 확인)
    stage3_base = {**base_constraint, "max_duty_periods": 4, "max_pairing_days": WINDOW_DAYS}
    def sample_constraint():
        # 범위는 config.STAGE3_CONSTRAINT_RANGES에서 관리
        # TODO: 범위 확정 후 config.py 범위 수정 
        r = config.STAGE3_CONSTRAINT_RANGES
        return {
            **stage3_base,
            "max_duty":         random.uniform(*r["max_duty"]),
            "min_rest":         random.uniform(*r["min_rest"]),
            "min_conn":         random.uniform(*r["min_conn"]),
            "max_conn":         random.uniform(*r["max_conn"]),
            "max_legs":         random.randint(*r["max_legs"]),
            "max_duty_periods": random.randint(*r["max_duty_periods"]),
            "max_pairing_days": random.randint(*r["max_pairing_days"]),
        }

    run_curriculum_stage(3, encoder, decoder, optimizer,
                         n_episodes=2000, constraint_override=stage3_base,
                         save_dir=save_dir, flight_sampler=flight_sampler,
                         constraint_sampler=sample_constraint)

    # ── TODO: Phase 2 — CG dual feedback ─────────────────────────────
    # LP relaxation으로 μ[f] (dual variable) 추출 후 RL reward로 피드백
    # set_partition.py IP 결과를 활용한 양방향 학습 구조
    # 혜린 set_partition.py 완성 후 구현 예정

    # ── FiLM 검증: constraint별 greedy 결과 비교 ─────────────────────
    print()
    print("=" * 60)
    print("FiLM 검증: 같은 flights, 다른 max_duty")
    print("=" * 60)

    encoder.eval()
    decoder.eval()
    # 검증용 고정 데이터 (offset=0, base=ATL)
    val_flights = load_flights_rolling(DATA_PATH, WINDOW_DAYS, 0, airport_map)
    val_origins, val_dests, val_dep_times, val_arr_times = flights_to_tensors(val_flights, WINDOW_DAYS)
    val_base = base_ids[0]

    with torch.no_grad():
        for duty in [12.0, 12.5, 13.0, 13.5, 14.0]:
            c = {**get_delta_constraints(val_base), "max_duty": duty,
                 "max_duty_periods": 4, "max_pairing_days": 5}
            enc = encoder(val_origins, val_dests, val_dep_times, val_arr_times, constraint_to_tensor(c))
            _, _, _, metrics = run_episode(val_flights, c, encoder, decoder, enc, greedy=True)
            print(f"  max_duty={duty:4.1f}h → pairings: {metrics['n_pairings']:3d}  "
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
    }, os.path.join(save_dir, "model_latest.pt"))
    print(f"\n모델 저장: checkpoints/model_latest.pt")


if __name__ == "__main__":
    train()
