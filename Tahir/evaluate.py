"""
Phase 4: Evaluation pipeline
Runs the full Tahir I²CGp baseline on CPP instances and reports:
  - Number of pairings in solution
  - Coverage rate (% flights covered)
  - Solve time
  - Gap vs. reference (greedy reference cost)

Also compares:
  - Baseline (greedy without DNN): use all D_ib successors
  - I²CGp (DNN-guided): use reduced D^+_ib

Usage:
  python evaluate.py [--model_dir experiments/loto] [--beam 10]
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from dnn.cpp_loader import (
    load_instance, discover_instances,
    load_cppsc_instance, discover_all_instances,
)
from dnn.dataset import (
    build_successor_sets, build_encoders, build_dataset,
    filter_successors_by_pattern,
)
from dnn.reference import generate_reference_pairings
from solver.constraints import is_feasible_pairing, pairing_cost
from solver.setpart import solve_set_partitioning, compute_coverage_gap


def run_greedy_baseline(inst: Dict, ref_pairings: List[List[int]]) -> Dict:
    """Run greedy solver (no DNN) as lower-bound baseline."""
    leg_map = {leg["flight_id"]: leg for leg in inst["legs"]}
    flights = [leg["flight_id"] for leg in inst["legs"]]
    bases   = inst["bases"]

    # Use reference pairings directly as columns
    costs   = [pairing_cost([leg_map[f] for f in p]) for p in ref_pairings]
    t0      = time.time()
    sel, obj, status, solve_time = solve_set_partitioning(flights, ref_pairings, costs)
    gap = compute_coverage_gap(flights, sel, ref_pairings)

    return {
        "method":      "Greedy+SP",
        "n_columns":   len(ref_pairings),
        "n_selected":  len(sel),
        "coverage":    1.0 - gap,
        "obj":         obj,
        "solve_time":  solve_time,
        "status":      status,
    }


def run_dnn_baseline(
    inst: Dict,
    ref_pairings: List[List[int]],
    enc: Dict,
    model_dir: Path,
    aircraft_type: str,
    beam_width: int = 10,
) -> Dict:
    """Run DNN-guided column generation + set partitioning."""
    import tensorflow as tf
    from solver.column_gen import build_probability_matrix, compute_psi, generate_columns_beam

    weights_path = model_dir / f"weights_AT_{aircraft_type}.h5"
    cfg_path     = model_dir / f"model_config_AT_{aircraft_type}.json"
    norm_path    = model_dir / f"norm_AT_{aircraft_type}.json"

    if not weights_path.exists():
        return {"method": "DNN+SP", "error": f"Weights not found: {weights_path}"}

    with open(cfg_path) as f:
        cfg = json.load(f)

    from dnn.model import build_model
    model = build_model(
        n_airports=cfg["n_airports"],
        n_aircraft=cfg["n_aircraft"],
        **cfg.get("hparams", {}),
    )
    # Build model with dummy input to initialise weights
    _np = np
    dummy = _np.zeros((1, 1, 27), dtype=_np.float32)
    model(tf.constant(dummy))
    model.load_weights(str(weights_path))

    norm_mean, norm_std = None, None
    if norm_path.exists():
        with open(norm_path) as f:
            nd = json.load(f)
        norm_mean = np.array(nd["mean"], dtype=np.float32)
        norm_std  = np.array(nd["std"],  dtype=np.float32)

    leg_map  = {leg["flight_id"]: leg for leg in inst["legs"]}
    flights  = [leg["flight_id"] for leg in inst["legs"]]
    succ_raw = build_successor_sets(inst["legs"])
    succ_flt = filter_successors_by_pattern(inst["legs"], succ_raw, ref_pairings)

    all_columns = []
    for base in inst["bases"]:
        # Build P^b matrix
        P = build_probability_matrix(
            inst["legs"], succ_flt, model, enc, base, norm_mean, norm_std
        )
        Psi, class_max = compute_psi(P)

        # Generate columns via beam search on reduced graph
        cols = generate_columns_beam(
            inst["legs"], [base], P, Psi, class_max, beam_width=beam_width
        )
        all_columns.extend(cols)

    if not all_columns:
        return {"method": "DNN+SP", "error": "No columns generated"}

    costs = [pairing_cost([leg_map[f] for f in col]) for col in all_columns]

    t0 = time.time()
    sel, obj, status, solve_time = solve_set_partitioning(flights, all_columns, costs)
    gap = compute_coverage_gap(flights, sel, all_columns)

    return {
        "method":      "DNN+SP (I2CGp-like)",
        "n_columns":   len(all_columns),
        "n_selected":  len(sel),
        "coverage":    1.0 - gap,
        "obj":         obj,
        "solve_time":  solve_time,
        "status":      status,
    }


def evaluate_instance(
    aircraft_type: str,
    instance_id: int,
    enc: Dict,
    model_dir: Path,
    beam_width: int = 10,
    source: str = "CPP",
) -> Dict:
    """Run both methods on one instance, return combined results."""
    if source == "CPPSC":
        inst = load_cppsc_instance(aircraft_type, instance_id)
    else:
        inst = load_instance(aircraft_type, instance_id)
    ref  = generate_reference_pairings(inst)

    print(f"\n[{source} AT_{aircraft_type} instance_{instance_id}] "
          f"{len(inst['legs'])} legs, {len(ref)} ref pairings, bases={inst['bases']}")

    greedy_result = run_greedy_baseline(inst, ref)
    dnn_result    = run_dnn_baseline(inst, ref, enc, model_dir, aircraft_type, beam_width)

    print(f"  Greedy+SP : pairings={greedy_result['n_selected']} "
          f"coverage={greedy_result['coverage']:.3f} "
          f"time={greedy_result['solve_time']:.1f}s")
    if "error" not in dnn_result:
        print(f"  DNN+SP    : pairings={dnn_result['n_selected']} "
              f"coverage={dnn_result['coverage']:.3f} "
              f"cols={dnn_result['n_columns']} "
              f"time={dnn_result['solve_time']:.1f}s")
    else:
        print(f"  DNN+SP    : {dnn_result['error']}")

    return {
        "aircraft_type": aircraft_type,
        "instance_id":   instance_id,
        "source":        source,
        "n_legs":        len(inst["legs"]),
        "greedy":        greedy_result,
        "dnn":           dnn_result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="experiments/loto")
    parser.add_argument("--beam",      type=int, default=10)
    parser.add_argument("--at",        default=None, help="Filter by aircraft type, e.g. 320")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    meta      = discover_all_instances()   # (at, iid, source)
    if args.at:
        meta = [(at, iid, src) for at, iid, src in meta if at == args.at]

    # Build global encoders from all instances
    all_instances = []
    for at, iid, src in meta:
        if src == "CPPSC":
            all_instances.append(load_cppsc_instance(at, iid))
        else:
            all_instances.append(load_instance(at, iid))
    enc = build_encoders(all_instances)

    all_results = []
    for at, iid, src in meta:
        result = evaluate_instance(at, iid, enc, model_dir, args.beam, source=src)
        all_results.append(result)

    # Aggregate
    print("\n=== Summary ===")
    print(f"{'Instance':<20} {'Greedy pairings':>17} {'Greedy cov':>12} "
          f"{'DNN pairings':>14} {'DNN cov':>9}")
    for r in all_results:
        g = r["greedy"]
        d = r["dnn"]
        dnn_n = d.get("n_selected", "ERR")
        dnn_c = f"{d.get('coverage', 0):.3f}" if "error" not in d else "ERR"
        print(f"  {r['source']} AT_{r['aircraft_type']} inst_{r['instance_id']}"
              f"  {g['n_selected']:>8}  {g['coverage']:>12.3f}"
              f"  {dnn_n!s:>14}  {dnn_c:>9}")

    out_path = Path("experiments") / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
