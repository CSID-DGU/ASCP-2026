"""
diagnose_film_params.py -- directly check, at the parameter level (without going through
a policy rollout), whether FiLM (gamma, beta) actually responds differently to each
airline's constraint vector.

diagnose_film_overnight.py only perturbed a single max_duty_periods component within
Delta and inferred the effect indirectly from the change in pairing count (which has a
confound; see log/0704/film_training_judgment_methodology_analysis.md §3). This script
instead computes FiLM.mlp(constraint) -> gamma, beta directly per airline and measures
how far each deviates from identity (gamma=1, beta=0), and how much airlines differ
from one another.
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
    print(f"constraint values (before normalization): { {k: {kk: AIRLINES[k](0)[kk] for kk in FILM_CONSTRAINT_KEYS} for k in AIRLINES} }")
    print()

    for label, gammas, betas in [("film_before", gammas_before, betas_before),
                                  ("film_after", gammas_after, betas_after)]:
        print(f"[{label}] deviation from identity (gamma=1,beta=0)")
        for name in AIRLINES:
            g, b = gammas[name], betas[name]
            g_dev = (g - 1.0).norm().item()
            b_dev = b.norm().item()
            print(f"  {name:8s}  ||gamma-1||={g_dev:.4f}  ||beta||={b_dev:.4f}")

        print(f"[{label}] gamma difference between airline pairs (||gamma_A - gamma_B||)")
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
