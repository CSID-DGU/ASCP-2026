"""
찬주 model/ + 혜린 RL/environment 통합 학습 스크립트
- encoder: 찬주 (FlightEncoder — Embedding + FiLM + Transformer)
- decoder: 찬주 (PointerDecoder — Pointer Attention + hard masking)
- environment: 혜린 (mask + step + reward)
- loader: 혜린 (BTS CSV → flight dict)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RL"))
import torch
import torch.optim as optim
from torch.distributions import Categorical

from model import FlightEncoder, PointerDecoder
from loader import load_flights
from environment import get_mask, step, step_end_duty, final_reward, END_DUTY
from constraints import get_delta_constraints, FILM_CONSTRAINT_KEYS
from state import init_state

PAIRING_COST      = 5.0   # pairing 완성/강제종료 시 -5 (deadhead 억제)
BASE_PENALTY      = 5.0   # base 미복귀 시 -5 (feasibility 강제)
UNCOVERED_PENALTY = 10.0  # 미배정 flight당 -10 (coverage 강제)
OVERNIGHT_PENALTY = 0.5   # END_DUTY 1회당 -0.5 (overnight rest 허용)
LEG_BONUS         = 1.5   # 2번째 leg부터 +1.5 (연결 장려)


def constraint_to_tensor(constraint):
    """constraint dict → FiLM 입력 tensor"""
    return torch.tensor([constraint[k] for k in FILM_CONSTRAINT_KEYS], dtype=torch.float32)

def flights_to_tensors(flights):
    """혜린 flight dict → 찬주 encoder 입력 tensor 변환"""
    origins = torch.tensor([f["origin"] for f in flights])
    dests = torch.tensor([f["dest"] for f in flights])
    dep_times = torch.tensor([f["dep_time"] for f in flights], dtype=torch.float32)
    arr_times = torch.tensor([f["arr_time"] for f in flights], dtype=torch.float32)
    return origins, dests, dep_times, arr_times


def state_to_vec(state, encoder, constraint):
    """혜린 state dict → 찬주 decoder 입력 tensor 변환"""
    airport_emb = encoder.airport_emb(torch.tensor(state["current_airport"]))

    max_pairing_days = constraint.get("max_pairing_days", 5)
    time_of_day  = (state["current_time"] % 24.0) / 24.0
    day_norm     = (state["current_time"] // 24.0) / max(max_pairing_days, 1)
    duty_period_norm = state.get("duty_period", 0) / max(constraint.get("max_duty_periods", 4), 1)

    return torch.cat([
        airport_emb,
        torch.tensor([
            time_of_day,
            day_norm,
            state["duty_time"] / constraint["max_duty"],
            state["legs"] / constraint["max_legs"],
            duty_period_norm,
            1.0 if state.get("is_resting", False) else 0.0,
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

        # flight도 없고 END_DUTY도 불가 → 다음 미배정 flight로 강제 이동
        no_flight = sum(mask_list[:-2]) == 0
        no_end_duty = mask_list[-2] == 0
        if no_flight and no_end_duty:
            # pairing_start=False (진행 중 막힌 경우)만 deadhead 패널티
            if not state.get("pairing_start", False):
                n_pairings += 1
                n_deadheads += 1
                if state["current_airport"] != constraint["base_airport"]:
                    total_reward -= BASE_PENALTY
                total_reward -= PAIRING_COST

            unassigned = [f for f in flights if not assigned[f["id"]]]
            if len(unassigned) == 0:
                break

            earliest = sorted(unassigned, key=lambda x: x["dep_time"])[0]
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

        # END_DUTY (index N): 현재 duty 종료 → rest period 진입
        if action == n_flights:
            state = step_end_duty(state, constraint)
            total_reward -= OVERNIGHT_PENALTY
            continue

        # END_PAIRING (index N+1): pairing 전체 종료
        if action == n_flights + 1:
            n_pairings += 1

            if state["current_airport"] != constraint["base_airport"]:
                total_reward -= BASE_PENALTY
            total_reward -= PAIRING_COST

            unassigned = [f for f in flights if not assigned[f["id"]]]
            if len(unassigned) == 0:
                break

            earliest = sorted(unassigned, key=lambda x: x["dep_time"])[0]
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

        # flight action
        prev_legs = state.get("legs", 0)
        state, r, done = step(state, action, flights, assigned, constraint)
        total_reward += r
        if prev_legs >= 1:              # 2번째 leg부터 보너스 (연결 장려)
            total_reward += LEG_BONUS

        if done:
            break

    total_reward += final_reward(assigned, uncovered_penalty=UNCOVERED_PENALTY)

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
    stage, flights, encoder, decoder, optimizer, origins, dests, dep_times, arr_times,
    n_episodes, constraint_override, save_dir
):
    """
    커리큘럼 1단계 실행.
    constraint_override: 이 단계에서 사용할 constraint dict.
    """
    best_avg_pairings = float("inf")
    greedy_pairings = []

    print(f"\n{'='*60}")
    print(f"Curriculum Stage {stage}: max_duty_periods={constraint_override['max_duty_periods']}, "
          f"max_pairing_days={constraint_override['max_pairing_days']}")
    print(f"{'='*60}")

    params = list(encoder.parameters()) + list(decoder.parameters())

    for ep in range(n_episodes):
        c_tensor = constraint_to_tensor(constraint_override)
        encoded  = encoder(origins, dests, dep_times, arr_times, c_tensor)

        reward_s, log_probs, entropies, metrics_s = run_episode(
            flights, constraint_override, encoder, decoder, encoded, greedy=False
        )
        if len(log_probs) == 0:
            continue

        with torch.no_grad():
            encoded_g = encoder(origins, dests, dep_times, arr_times, c_tensor)
            reward_g, _, _, metrics_g = run_episode(
                flights, constraint_override, encoder, decoder, encoded_g, greedy=True
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
    flights    = load_flights("RL/data/T_ONTIME_MARKETING.csv", limit=200, hub_only=True, n_days_max=4)
    n_airports = max(max(f["origin"], f["dest"]) for f in flights) + 1

    print(f"flights: {len(flights)}개, airports: {n_airports}개")

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

    origins, dests, dep_times, arr_times = flights_to_tensors(flights)

    base = get_delta_constraints()

    # ── Stage 1: 단일 duty (overnight 없음) ──────────────────────────
    # max_duty_periods=1 → END_DUTY 불가 → 당일 connection만 학습
    stage1_c = {**base, "max_duty_periods": 1, "max_pairing_days": 1}
    run_curriculum_stage(1, flights, encoder, decoder, optimizer,
                         origins, dests, dep_times, arr_times,
                         n_episodes=1000, constraint_override=stage1_c,
                         save_dir=save_dir)

    # ── Stage 2: 1박 overnight ────────────────────────────────────────
    stage2_c = {**base, "max_duty_periods": 2, "max_pairing_days": 2}
    run_curriculum_stage(2, flights, encoder, decoder, optimizer,
                         origins, dests, dep_times, arr_times,
                         n_episodes=1000, constraint_override=stage2_c,
                         save_dir=save_dir)

    # ── Stage 3: full multi-day ───────────────────────────────────────
    stage3_c = {**base, "max_duty_periods": 4, "max_pairing_days": 5}
    run_curriculum_stage(3, flights, encoder, decoder, optimizer,
                         origins, dests, dep_times, arr_times,
                         n_episodes=2000, constraint_override=stage3_c,
                         save_dir=save_dir)

    # ── FiLM 검증: constraint별 greedy 결과 비교 ─────────────────────
    print()
    print("=" * 60)
    print("FiLM 검증: 같은 flights, 다른 max_duty")
    print("=" * 60)

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for duty in [6.0, 8.0, 10.0, 12.0, 14.0]:
            c = {**stage3_c, "max_duty": duty}
            enc = encoder(origins, dests, dep_times, arr_times, constraint_to_tensor(c))
            _, _, _, metrics = run_episode(flights, c, encoder, decoder, enc, greedy=True)
            print(f"  max_duty={duty:4.0f}h → pairings: {metrics['n_pairings']:3d}  "
                  f"deadheads: {metrics['n_deadheads']:3d}  "
                  f"coverage: {metrics['coverage_pct']:5.1f}%")

    # ── 최종 모델 저장 ────────────────────────────────────────────────
    torch.save({
        "encoder":        encoder.state_dict(),
        "decoder":        decoder.state_dict(),
        "n_airports":     n_airports,
        "constraint_dim": len(FILM_CONSTRAINT_KEYS),
    }, os.path.join(save_dir, "model_latest.pt"))
    print(f"\n모델 저장: checkpoints/model_latest.pt")


if __name__ == "__main__":
    train()
