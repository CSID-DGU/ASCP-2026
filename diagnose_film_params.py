"""
diagnose_film_params.py — FiLM(gamma, beta)이 항공사별 constraint 벡터에 실제로
다르게 반응하는지, 정책 행동(rollout)을 거치지 않고 파라미터 레벨에서 직접 확인한다.

기존 diagnose_film_overnight.py는 Delta 하나에서 max_duty_periods 성분 하나만 흔들어
pairings 수 변화로 간접 추론했음(confound 있음, log/0704/film_학습판단_방법론_분석.md
§3 참고). 이 스크립트는 FiLM.mlp(constraint) -> gamma, beta를 항공사별로 직접 계산해서
identity(gamma=1, beta=0)에서 얼마나 벗어났는지, 항공사 간에 서로 얼마나 다른지를 잰다.
"""
import sys
import argparse

import torch

sys.path.insert(0, "RL")

from model import FlightEncoder
from constraints import (
    get_delta_constraints, get_alaska_constraints,
    get_jetblue_constraints, get_turkish_constraints,
    FILM_CONSTRAINT_KEYS,
)
from utils import constraint_to_tensor

AIRLINES = {
    "delta":   get_delta_constraints,
    "alaska":  get_alaska_constraints,
    "jetblue": get_jetblue_constraints,
    "turkish": get_turkish_constraints,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    n_airports = ckpt.get("n_airports", ckpt["encoder"]["airport_emb.weight"].shape[0])

    encoder = FlightEncoder(n_airports=n_airports, constraint_dim=len(FILM_CONSTRAINT_KEYS)).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    gammas_before, betas_before = {}, {}
    gammas_after, betas_after = {}, {}

    with torch.no_grad():
        for name, get_fn in AIRLINES.items():
            c = get_fn(0)
            c_t = constraint_to_tensor(c, device=device)

            params_b = encoder.film_before.mlp(c_t)
            g_b, b_b = params_b.chunk(2, dim=-1)
            gammas_before[name] = g_b
            betas_before[name] = b_b

            params_a = encoder.film_after.mlp(c_t)
            g_a, b_a = params_a.chunk(2, dim=-1)
            gammas_after[name] = g_a
            betas_after[name] = b_a

    print(f"checkpoint: {args.checkpoint}")
    print(f"constraint 값 (정규화 전): { {k: {kk: AIRLINES[k](0)[kk] for kk in FILM_CONSTRAINT_KEYS} for k in AIRLINES} }")
    print()

    for label, gammas, betas in [("film_before", gammas_before, betas_before),
                                  ("film_after", gammas_after, betas_after)]:
        print(f"[{label}] identity(gamma=1,beta=0) 이탈도")
        for name in AIRLINES:
            g, b = gammas[name], betas[name]
            g_dev = (g - 1.0).norm().item()
            b_dev = b.norm().item()
            print(f"  {name:8s}  ||gamma-1||={g_dev:.4f}  ||beta||={b_dev:.4f}")

        print(f"[{label}] 항공사 쌍 간 gamma 차이 (||gamma_A - gamma_B||)")
        names = list(AIRLINES.keys())
        header = "          " + "".join(f"{n:>10s}" for n in names)
        print(header)
        for a in names:
            row = f"  {a:8s}"
            for b_name in names:
                d = (gammas[a] - gammas[b_name]).norm().item()
                row += f"{d:10.4f}"
            print(row)
        print()


if __name__ == "__main__":
    main()
