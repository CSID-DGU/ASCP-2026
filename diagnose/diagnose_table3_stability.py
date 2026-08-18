"""
diagnose_table3_stability.py — ④' Table3 IP 단계만 같은 체크포인트로 N회
반복해서, stochastic pool 수집 때문에 결과(특히 jetblue vs delta 순위)가
실행마다 얼마나 흔들리는지 확인한다.

배경(log/0717/지금_확정된_결론.md 부록 A·C): ③ greedy는 argmax 기반이라
결정론적이지만, ④' Table3는 evaluation/evaluate_ip.py::collect_pool_full()이
random.shuffle/random.choice로 pool을 매번 다르게 모으기 때문에 실행마다
IP solve 결과가 달라질 수 있다. pws5cjlz(C)는 4단계 중 ④'에서만 jetblue
pairings/avg_legs가 delta에 근소하게 뒤져 예측과 반대로 나온 적이 있는데
(log/0709/table3_C.out), 이게 안정적으로 재현되는 패턴인지 pool 노이즈인지
가리려는 목적. ③은 반복할 필요가 없어 이 스크립트에서는 안 돈다.

diagnose_film_inference_ablation.py의 load_model/run_table3_stage를 그대로
재사용(로직 중복 없음), FiLM 무력화("bypassed") 모드는 이 목적에 필요 없어
빼고 "정상" 모드만 N회 반복한다.

Usage:
    cd /home/hyrn/ASCP-2026
    source ascp/bin/activate
    python -u diagnose/diagnose_table3_stability.py checkpoints/pws5cjlz/stage3_best.pt \
        --device cuda:0 --use-utc --n-runs 2
"""
import os
import sys
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from loader import build_airport_map, bases_to_ids
import config
from rollout import set_environment
from evaluation import evaluate_ip

from diagnose.legacy.diagnose_film_inference_ablation import load_model, run_table3_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window-days", type=int, default=5)
    parser.add_argument("--subset-size", type=int, default=config.EPISODE_MAX_FLIGHTS)
    parser.add_argument("--n-rollouts-per-chunk", type=int, default=5)
    parser.add_argument("--ip-time-limit", type=int, default=1800)
    parser.add_argument("--lambda-dh", type=float, default=1.0)
    parser.add_argument("--use-utc", action="store_true")
    parser.add_argument("--n-runs", type=int, default=2, help="④' Table3를 몇 회 반복할지")
    args = parser.parse_args()

    device = torch.device(args.device)
    evaluate_ip.DEVICE = device
    set_environment("delta")

    map_paths = [v for k, v in config.AIRLINE_DATA.items() if k != "turkish"]
    airport_map = build_airport_map(map_paths)
    base_ids = bases_to_ids(list(config.AIRLINE_BASES["delta"]), airport_map)
    base = base_ids[0]

    encoder, decoder, n_airports = load_model(args.checkpoint, device)
    if n_airports <= 145:
        print(f"경고: n_airports={n_airports} — multi-airline 체크포인트가 아닐 수 있음.")

    all_runs = []
    for run_idx in range(1, args.n_runs + 1):
        label = f"실행 {run_idx}/{args.n_runs}"
        result = run_table3_stage(encoder, decoder, airport_map, base_ids, base, device, args, label)
        all_runs.append(result)

    print(f"\n\n{'='*70}\n{args.n_runs}회 반복 비교 (pws5cjlz 기준 사전 예측: "
          f"jetblue pairings 최저 / avg_legs·avg_duties 최고 / FTC 최저)\n{'='*70}")
    for airline in ["delta", "alaska", "jetblue"]:
        print(f"\n[{airline}]")
        print(f"{'실행':<10}{'pairings':>10}{'ManDays':>10}{'deadhead':>10}{'FTC':>8}{'avg_legs':>10}{'avg_duties':>12}")
        for run_idx, result in enumerate(all_runs, 1):
            r = result[airline]
            print(f"{run_idx:<10}{r['n_pairings']:>10}{r['man_days']:>10}{r['deadhead']:>10}"
                  f"{r['ftc']:>7.2f}%{r['avg_legs']:>10.2f}{r['avg_duties']:>12.2f}")

    print(f"\n{'='*70}\n예측-일치 여부(각 실행에서 jetblue가 최저/최고였는지)\n{'='*70}")
    for run_idx, result in enumerate(all_runs, 1):
        jb, dl, ak = result["jetblue"], result["delta"], result["alaska"]
        pairings_min = jb["n_pairings"] == min(result[a]["n_pairings"] for a in ("delta", "alaska", "jetblue"))
        legs_max = jb["avg_legs"] == max(result[a]["avg_legs"] for a in ("delta", "alaska", "jetblue"))
        duties_max = jb["avg_duties"] == max(result[a]["avg_duties"] for a in ("delta", "alaska", "jetblue"))
        ftc_min = jb["ftc"] == min(result[a]["ftc"] for a in ("delta", "alaska", "jetblue"))
        n_pass = sum([pairings_min, legs_max, duties_max, ftc_min])
        print(f"  실행 {run_idx}: pairings최저={pairings_min} avg_legs최고={legs_max} "
              f"avg_duties최고={duties_max} FTC최저={ftc_min}  → {n_pass}/4")


if __name__ == "__main__":
    main()
