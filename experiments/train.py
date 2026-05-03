"""
찬주 model/ + 혜린 RL/environment 통합 학습 스크립트
- encoder: 찬주 (FlightEncoder — Embedding + FiLM + Transformer)
- decoder: 찬주 (PointerDecoder — Pointer Attention + hard masking)
- environment: 혜린 (mask + step + reward)
- loader: 혜린 (BTS CSV → flight dict)
"""

import os
import sys
import torch
import torch.optim as optim
from torch.distributions import Categorical

# 찬주 모델 (model/ 폴더 먼저 잡히도록)
from model import FlightEncoder, PointerDecoder

# 혜린 코드 (RL/ 폴더)
sys.path.insert(0, "RL")
from loader import load_flights
from environment import get_mask, step, step_end_duty, final_reward, END_DUTY
from RL.constraints import get_delta_constraints, FILM_CONSTRAINT_KEYS
from RL.state import init_state

PAIRING_COST = 1.0        # pairing 완성/강제종료 시 -1 (최소화 대상)
BASE_PENALTY = 5.0        # base 미복귀 시 -5 (feasibility 강제)
UNCOVERED_PENALTY = 10.0  # 미배정 flight당 -10 (coverage 강제)


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

    while True:
        # 혜린 mask
        mask_list = get_mask(state, flights, assigned, constraint)
        mask = torch.tensor(mask_list, dtype=torch.float32)

        # flight도 없고 END_DUTY도 불가 → END_PAIRING 강제 (deadhead)
        no_flight = sum(mask_list[:-2]) == 0
        no_end_duty = mask_list[-2] == 0
        if no_flight and no_end_duty and not state.get("pairing_start", False):
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
        state, r, done = step(state, action, flights, assigned, constraint)
        total_reward += r

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


def train():
    # 데이터 로드 (혜린 loader)
    flights = load_flights("RL/data/T_ONTIME_MARKETING.csv", limit=50)
    n_airports = max(max(f["origin"], f["dest"]) for f in flights) + 1

    print(f"flights: {len(flights)}개, airports: {n_airports}개")
    print()

    # 찬주 모델 생성
    encoder = FlightEncoder(
        n_airports=n_airports,
        constraint_dim=len(FILM_CONSTRAINT_KEYS),
        airport_emb_dim=32,
        d_model=128,
    )
    decoder = PointerDecoder(d_model=128, airport_emb_dim=32)

    # 전체 파라미터 합쳐서 optimizer
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=1e-4)  # lr 낮춤 (안정성)

    # flight → tensor (1번만)
    origins, dests, dep_times, arr_times = flights_to_tensors(flights)

    for episode in range(1000):
        # constraint 번갈아 (FiLM 검증용)
        constraint = get_delta_constraints()
        c_tensor = constraint_to_tensor(constraint)

        # encode (에피소드당 1번)
        encoded = encoder(origins, dests, dep_times, arr_times, c_tensor)

        # ── sample rollout (학습용) ──
        reward_s, log_probs, entropies, metrics_s = run_episode(
            flights, constraint, encoder, decoder, encoded, greedy=False
        )

        if len(log_probs) == 0:
            continue

        # ── greedy rollout (baseline) ──
        with torch.no_grad():
            encoded_g = encoder(origins, dests, dep_times, arr_times, c_tensor)
            reward_g, _, _, metrics_g = run_episode(
                flights, constraint, encoder, decoder, encoded_g, greedy=True
            )

        # scaling (reward 범위에 따라 advantage가 너무 커지는 문제 방지)
        # advantage = sample - greedy
        advantage = (reward_s - reward_g) / (abs(reward_g) + 1e-6)

        # REINFORCE loss + entropy bonus
        loss = torch.stack([
            -lp * advantage - 0.005 * ent
            for lp, ent in zip(log_probs, entropies)
        ]).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)  # gradient clipping
        optimizer.step()

        if episode % 20 == 0:
            print(
                f"Ep {episode:4d} | "
                f"sample: {reward_s:7.1f} (p={metrics_s['n_pairings']:2d} dh={metrics_s['n_deadheads']:2d} cov={metrics_s['coverage_pct']:5.1f}%) | "
                f"greedy: {reward_g:7.1f} (p={metrics_g['n_pairings']:2d}) | "
                f"adv: {advantage:6.2f} | "
                f"duty: {constraint['max_duty']:2.0f}h | "
                f"loss: {loss.item():8.3f}"
            )

    # ── FiLM 검증: constraint별 greedy 결과 비교 ──
    print()
    print("=" * 60)
    print("FiLM 검증: 같은 flights, 다른 constraint")
    print("=" * 60)

    for duty in [6.0, 8.0, 10.0, 12.0, 14.0]:
        with torch.no_grad():
            test_constraint = get_delta_constraints()
            test_constraint["max_duty"] = duty
            c = constraint_to_tensor(test_constraint)

            enc = encoder(origins, dests, dep_times, arr_times, c)
            reward, _, _, metrics = run_episode(
                flights,
                test_constraint,
                encoder,
                decoder,
                enc,
                greedy=True
            )
        print(f"  max_duty={duty:4.0f}h → pairings: {metrics['n_pairings']:3d}  "
              f"deadheads: {metrics['n_deadheads']:2d}  "
              f"coverage: {metrics['coverage_pct']:5.1f}%  "
              f"reward: {reward:7.1f}")

    # ── 모델 저장 ──
    save_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "n_airports": n_airports,
        "constraint_dim": len(FILM_CONSTRAINT_KEYS),
    }, os.path.join(save_dir, "model_latest.pt"))
    print(f"\n모델 저장: checkpoints/model_latest.pt")


if __name__ == "__main__":
    train()
