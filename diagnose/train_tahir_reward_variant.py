"""
train_tahir_reward_variant.py — Tahir의 실제 목적함수(duty당 4시간 최소보장 pay,
Eq.2)를 모사하는 보너스를 RL reward에 얹어서, "RL 보상 신호를 바꾸면 Tahir 대비
구조적 격차(+171.3%, pairing 수 +59~61%)가 줄어드는지"를 판단하기 위한 별도 실험
스크립트. 기존 experiments/train.py, RL/environment.py는 전혀 수정하지 않는다 —
런타임에 RL/environment.py의 step() 함수만 monkey-patch해서 END_DUTY/END_PAIRING
시점에 추가 페널티를 얹은 뒤, 기존 train.py의 커리큘럼/Phase2 로직을 그대로
재사용한다.

가설(log/0709/tahir_모델_구조_차이_및_공정비교_계획.md §3, tahir_비교_계획.md §6):
  Tahir는 duty 하나가 최소 4시간(240분) 비행한 것으로 간주해 비용을 매긴다
  (Tahir/solver/constraints.py::pairing_cost, T_p = max(elapsed/4,
  sum_d max(240, fly_d + 0.5*dh_d))) → 짧은 duty를 많이 만들수록 "낭비된 최소보장"
  비용이 쌓여서, Tahir 최적화는 자연히 duty를 길게(적게) 묶는 방향으로 간다. 우리
  RL reward는 이런 유인이 없어서(연결 gap만 직접 패널티) pairing/duty 수가
  구조적으로 더 많이 나온다는 게 지금까지의 결론(`tahir_비교_계획.md` §9).

이 실험이 하는 일:
  END_DUTY 또는 END_PAIRING으로 duty가 닫힐 때, 그 duty의 실제 비행시간
  (state["duty_time"], 시간 단위)이 4시간 미만이면 그 부족분에 비례한 페널티를
  추가로 부과한다:
      extra_penalty = tahir_duty_lambda * max(0, 4.0 - duty_fly_hours)
  tahir_duty_lambda=0.0이면 기존과 완전히 동일(no-op).

판단 기준(사용자 지시, log/0711 참고):
  학습 후 이 체크포인트를 기존(무수정) baselines/tahir/eval_cross_objective.py로
  평가해서 +171.3% 격차가 줄어드는지 확인한다.
  - 줄어들면 → 보상 수정이 유효, 정식 조건으로 채택할지 검토.
  - 안 줄어들면 → "보상 신호 문제"가 아니라 "평가 방식(coverage/objective 정의)
    자체의 문제"일 가능성이 커지므로, 그 쪽을 재점검하는 방향으로 전환.

Usage:
    cd /home/hyrn/ASCP-2026
    source ascp/bin/activate
    python -u diagnose/train_tahir_reward_variant.py \
        --tahir-duty-lambda 2.0 --device cuda:0 --use-utc \
        --log log/0711/train_tahir_reward_variant.log
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RL"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

import argparse
import torch

import environment  # RL/environment.py — train.py도 동일 모듈 이름으로 import하므로 같은 객체 공유

TAHIR_MIN_DUTY_PAY_H = 4.0  # Tahir Eq.2의 240분(4시간) 최소보장 pay 기준

_orig_step = environment.step


def _make_wrapped_step(tahir_duty_lambda):
    def wrapped_step(state, action, flights, assigned, constraint=None):
        N = len(flights)
        duty_fly_hours = state.get("duty_time", 0.0)  # 이번 duty에서 지금까지 쌓인 비행시간(h)
        next_state, reward, done = _orig_step(state, action, flights, assigned, constraint)
        if tahir_duty_lambda > 0.0 and (action == N or action == N + 1):
            shortfall = max(0.0, TAHIR_MIN_DUTY_PAY_H - duty_fly_hours)
            reward -= tahir_duty_lambda * shortfall
        return next_state, reward, done
    return wrapped_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tahir-duty-lambda", type=float, required=True,
                         help="Tahir 4시간 최소보장 pay 모사 페널티 강도. 0.0=기존과 동일(no-op)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log", default=os.path.join(
        os.path.dirname(__file__), "..", "log", "train_tahir_reward_variant_log.txt"))
    parser.add_argument("--use-utc", action="store_true",
                         help="z2db089m과 비교 가능하게 하려면 켜야 함(eval_cross_objective.py 기본값과 일치)")
    parser.add_argument("--airline", default="delta")
    args = parser.parse_args()

    environment.step = _make_wrapped_step(args.tahir_duty_lambda)

    import train as train_mod  # experiments/train.py 재사용
    train_mod.config.AIRLINE = args.airline
    train_mod._set_device(args.device)
    train_mod.USE_UTC = args.use_utc

    print(f"device: {train_mod.DEVICE}")
    print(f"use_utc: {train_mod.USE_UTC}")
    print(f"tahir_duty_lambda: {args.tahir_duty_lambda}  (0.0=baseline과 동일)")
    print(f"log: {args.log}")

    train_mod.train(phase2_only=False, multi_airline=False, skip_film=False)


if __name__ == "__main__":
    main()
