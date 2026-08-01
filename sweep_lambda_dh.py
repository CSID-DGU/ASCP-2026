"""
sweep_lambda_dh.py -- lambda_dh (DH penalty) sweep experiment on the delta dataset

Collects the pool once, then repeatedly runs solve_set_covering with different
lambda_dh values to check the ManDays/DH/avg_legs tradeoff.
"""

import sys
import math
import argparse

import torch

sys.path.insert(0, "RL")

import evaluate_ip as eip
from set_partition import solve_set_covering
import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--airline", default="delta")
    parser.add_argument("--subset-size", type=int, default=1200)
    parser.add_argument("--window-days", type=int, default=5)
    parser.add_argument("--n-rollouts-per-chunk", type=int, default=5)
    parser.add_argument("--ip-time-limit", type=int, default=300)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lambda-dh", type=float, nargs="+",
                         default=[1.0, 3.0, 5.0, 10.0, 20.0, 40.0])
    args = parser.parse_args()

    eip.DEVICE = torch.device(args.device)
    DEVICE = eip.DEVICE

    data_path = config.AIRLINE_DATA[args.airline]
    bases = config.AIRLINE_BASES[args.airline]

    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    n_airports = ckpt.get("n_airports", ckpt["encoder"]["airport_emb.weight"].shape[0])

    if n_airports > 145:
        map_paths = [v for k, v in config.AIRLINE_DATA.items() if k != "turkish"]
    else:
        map_paths = data_path
    airport_map = eip.build_airport_map(map_paths)
    base_ids = eip.bases_to_ids(list(bases), airport_map)

    encoder = eip.FlightEncoder(n_airports=n_airports, constraint_dim=len(eip.FILM_CONSTRAINT_KEYS)).to(DEVICE)
    airport_emb_dim = encoder.airport_emb.embedding_dim
    ckpt_state_dim = ckpt["decoder"]["state_mlp.0.weight"].shape[1]
    n_scalars = ckpt_state_dim - airport_emb_dim * 2 - len(eip.FILM_CONSTRAINT_KEYS)
    decoder = eip.PointerDecoder(constraint_dim=len(eip.FILM_CONSTRAINT_KEYS),
                                  airport_emb_dim=airport_emb_dim, n_scalars=n_scalars).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    constraint = eip._GET_CONSTRAINT[args.airline](base_ids[0])

    print(f"\nLoading full dataset ({args.airline}, window_days={args.window_days})...", flush=True)
    windows, n_total = eip.load_windows_with_global_ids(data_path, airport_map, args.window_days)
    print(f"{n_total} flights total, {len(windows)} windows", flush=True)

    connected_sampler = eip.sample_connected_subnet_std
    print(f"\nCollecting pool (rollouts/chunk={args.n_rollouts_per_chunk}, subset={args.subset_size})... (done once)", flush=True)
    with torch.no_grad():
        pool, covered = eip.collect_pool_full(
            windows, base_ids, constraint, encoder, decoder,
            n_rollouts_per_chunk=args.n_rollouts_per_chunk,
            subset_size=args.subset_size,
            connected_sampler=connected_sampler,
        )
    print(f"pool size: {len(pool)}, covered: {len(covered)}/{n_total}", flush=True)

    print("\n" + "=" * 90)
    print(f"{'lambda_dh':>10} {'pairing':>8} {'ManDays':>9} {'DH':>7} {'dead_h':>10} {'FTC%':>8} {'avg_legs':>9} {'status':>10}")
    print("=" * 90)

    results = []
    for lam in args.lambda_dh:
        result = solve_set_covering(pool, n_flights=n_total, lambda_dh=lam, time_limit=args.ip_time_limit)
        sel = result["selected"]
        fly_total = sum(p.get("fly", 0.0) for p in sel) if sel else 0.0
        dead_total = sum(p.get("dead_time", p["cost"]) for p in sel) if sel else 0.0
        legs_total = sum(p.get("n_legs", len(p["legs"])) for p in sel) if sel else 0
        man_days = sum(math.ceil(p.get("elapsed", 0.0) / 24.0) for p in sel) if sel else 0
        avg_legs = legs_total / len(sel) if sel else 0.0
        ftc = dead_total / fly_total * 100 if fly_total > 0 else 0.0

        print(f"{lam:>10.1f} {result['n_pairings']:>8} {man_days:>9} {result['deadhead_count']:>7} "
              f"{dead_total:>10.1f} {ftc:>8.2f} {avg_legs:>9.2f} {result['status']:>10}", flush=True)
        results.append({
            "lambda_dh": lam, "pairing": result["n_pairings"], "man_days": man_days,
            "dh": result["deadhead_count"], "dead_h": dead_total, "ftc": ftc,
            "avg_legs": avg_legs, "status": result["status"],
        })

    print("=" * 90)


if __name__ == "__main__":
    main()
