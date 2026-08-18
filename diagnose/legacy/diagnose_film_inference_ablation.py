"""
diagnose_film_inference_ablation.py — [legacy] 이미 학습된 체크포인트(기본 C=pws5cjlz)를
재학습 없이, 추론 시점에만 FiLM 출력을 identity(gamma=1, beta=0)로 강제해서
"C인데 FiLM만 무력화된 상태"를 만든 뒤, 정상 C와 같은 검증 배터리(③ greedy,
④' Table3 IP)로 비교한다.

배경(log/0711/paper/03_FiLM_CD_다단계비교.md, 08_범용성_증명_C자체근거.md):
  처음부터 FiLM 없이 학습한 D는 "학습 중에 다른 경로(디코더의 직접 constraint
  concat)로 우회해서 비슷하게 학습될 수 있다"만 보여줬을 뿐, "우리가 실제로 배포하는
  C 안에서 FiLM이 하는 일"에 대한 답은 아니었다(C vs D 비교로는 순효과 분리 불가,
  §3 결론). 이 스크립트는 재학습 없이 훨씬 싸게 그 질문에 바로 답한다 — 같은
  가중치(C)에서 FiLM 경로만 인과적으로 꺼봐서, 학습된 FiLM 출력이 실제로 C의
  항공사별 적응 행동에 얼마나 기여하는지 직접 측정한다(activation ablation).

무력화 방법: encoder.film_before/film_after 인스턴스의 forward를 통째로
`lambda flight_vecs, constraint: flight_vecs`로 교체 — 즉 "gamma=1, beta=0을
강제로 대입"과 수학적으로 완전히 동일한 identity 변환이다. 가중치 파일은 전혀
안 건드리고(model/film.py, model/encoder.py 무수정), 이 프로세스 안에서만
일시적으로 우회한다.

같은 실행 안에서 정상(normal) vs 무력화(bypassed) 두 모드를 순서대로 돌려서
바로 비교 가능하게 출력한다.

Usage:
    cd /home/hyrn/ASCP-2026
    source ascp/bin/activate
    python -u diagnose/legacy/diagnose_film_inference_ablation.py checkpoints/pws5cjlz/stage3_best.pt \
        --device cuda:0 --use-utc
"""
import os
import sys
import math
import argparse

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments"))

from model import FlightEncoder, PointerDecoder
from loader import build_airport_map, bases_to_ids, load_flights_rolling
from constraints import (
    get_delta_constraints, get_alaska_constraints, get_jetblue_constraints,
    get_turkish_constraints, FILM_CONSTRAINT_KEYS,
)
from utils import flights_to_tensors, constraint_to_tensor
from evaluation.set_partition import solve_set_covering
from rollout import set_environment
import config

import train as train_mod
from train import run_episode

from evaluation import evaluate_ip
from evaluation.evaluate_ip import collect_pool_full, sample_connected_subnet_std

GREEDY_AIRLINES = {
    "delta": get_delta_constraints, "alaska": get_alaska_constraints,
    "jetblue": get_jetblue_constraints, "turkish": get_turkish_constraints,
}
TABLE3_AIRLINES = {
    "delta": get_delta_constraints, "alaska": get_alaska_constraints,
    "jetblue": get_jetblue_constraints,
}


def bypass_film(encoder):
    """encoder.film_before/film_after를 identity로 영구 대체(이 프로세스 안에서만).
    가중치는 안 지움 — forward만 우회, 원한다면 언제든 encoder를 새로 로드해서 복원 가능."""
    identity = lambda flight_vecs, constraint: flight_vecs
    encoder.film_before.forward = identity
    encoder.film_after.forward = identity


def run_greedy_stage(encoder, decoder, airport_map, base, device, label):
    print(f"\n{'='*70}\n③ greedy 단계 ({label}) — delta 600편 고정, constraint만 교체\n{'='*70}")
    flights = load_flights_rolling(
        config.AIRLINE_DATA["delta"], window_days=5, offset_days=0, airport_map=airport_map,
        base_airport=base, n_max=config.EPISODE_MAX_FLIGHTS,
    )
    origins, dests, dep_times, arr_times, fly_times = flights_to_tensors(flights, 5 * 24.0, device=device)
    train_mod.DEVICE = device
    results = {}
    print(f"{'항공사':<10}{'pairings':>10}{'deadheads':>11}{'avg_overnight':>15}{'avg_legs':>10}")
    with torch.no_grad():
        for airline, get_fn in GREEDY_AIRLINES.items():
            val_c = get_fn(base)
            val_enc = encoder(origins, dests, dep_times, arr_times, fly_times,
                               constraint_to_tensor(val_c, device=device))
            _, _, _, m = run_episode(flights, val_c, encoder, decoder, val_enc, greedy=True)
            results[airline] = dict(pairings=m["n_pairings"], deadheads=m["n_deadheads"],
                                     avg_overnight=m.get("avg_overnight", 0), avg_legs=m.get("avg_legs", 0))
            print(f"{airline:<10}{m['n_pairings']:>10}{m['n_deadheads']:>11}"
                  f"{m.get('avg_overnight', 0):>15.3f}{m.get('avg_legs', 0):>10.3f}")
    return results


def run_table3_stage(encoder, decoder, airport_map, base_ids, base, device, args, label):
    print(f"\n{'='*70}\n④' Table3 단계 ({label}) — delta {args.subset_size and ''}편 고정, constraint만 교체\n{'='*70}")
    window_flights = load_flights_rolling(
        config.AIRLINE_DATA["delta"], window_days=args.window_days, offset_days=0,
        airport_map=airport_map, base_airport=base, n_max=None, use_utc=args.use_utc,
    )
    for f in window_flights:
        f["global_id"] = f["id"]
    n_total = len(window_flights)
    print(f"고정 flight 수: {n_total}편 (base={base})")

    results = {}
    for airline, get_fn in TABLE3_AIRLINES.items():
        constraint = get_fn(base)
        windows = [[dict(f) for f in window_flights]]
        with torch.no_grad():
            pool, covered = collect_pool_full(
                windows, base_ids, constraint, encoder, decoder,
                n_rollouts_per_chunk=args.n_rollouts_per_chunk,
                subset_size=args.subset_size,
                connected_sampler=sample_connected_subnet_std,
            )
        result = solve_set_covering(pool, n_flights=n_total, time_limit=args.ip_time_limit, lambda_dh=args.lambda_dh)
        sel = result["selected"]
        fly_total  = sum(p["fly"] for p in sel) if sel else 0.0
        legs_total = sum(p.get("n_legs", len(p["legs"])) for p in sel) if sel else 0
        duties_total = sum(p.get("n_duties", 1) for p in sel) if sel else 0
        man_days   = sum(math.ceil(p["elapsed"] / 24.0) for p in sel) if sel else 0
        intra_gap_total = sum(p.get("intra_duty_gap", 0.0) for p in sel) if sel else 0.0
        ftc = intra_gap_total / fly_total * 100 if fly_total > 0 else 0.0
        results[airline] = dict(
            n_pairings=result["n_pairings"], man_days=man_days, deadhead=result["deadhead_count"],
            ftc=ftc, avg_legs=legs_total / len(sel) if sel else 0.0,
            avg_duties=duties_total / len(sel) if sel else 0.0, status=result["status"],
        )
        print(f"  constraint={airline:<8} pairing={result['n_pairings']:>6} ManDays={man_days:>6} "
              f"deadhead={result['deadhead_count']:>6} FTC={ftc:>6.2f}% "
              f"avg_legs={results[airline]['avg_legs']:.2f} avg_duties={results[airline]['avg_duties']:.2f} "
              f"status={result['status']}")
    return results


def check_predictions(greedy_res, table3_res, label):
    print(f"\n{'='*70}\n예측-일치 체크 ({label})\n{'='*70}")
    checks = []
    g = greedy_res
    checks.append(("greedy jetblue pairings 최저", g["jetblue"]["pairings"] == min(v["pairings"] for v in g.values())))
    checks.append(("greedy jetblue avg_legs 최고", g["jetblue"]["avg_legs"] == max(v["avg_legs"] for v in g.values())))
    checks.append(("greedy jetblue avg_overnight 최고", g["jetblue"]["avg_overnight"] == max(v["avg_overnight"] for v in g.values())))
    checks.append(("greedy turkish pairings 최다", g["turkish"]["pairings"] == max(v["pairings"] for v in g.values())))
    checks.append(("greedy turkish avg_legs 최저", g["turkish"]["avg_legs"] == min(v["avg_legs"] for v in g.values())))
    t = table3_res
    checks.append(("table3 jetblue pairings 최저", t["jetblue"]["n_pairings"] == min(v["n_pairings"] for v in t.values())))
    checks.append(("table3 jetblue avg_legs 최고", t["jetblue"]["avg_legs"] == max(v["avg_legs"] for v in t.values())))
    checks.append(("table3 jetblue avg_duties 최고", t["jetblue"]["avg_duties"] == max(v["avg_duties"] for v in t.values())))
    checks.append(("table3 jetblue FTC 최저", t["jetblue"]["ftc"] == min(v["ftc"] for v in t.values())))
    n_pass = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  합계: {n_pass}/{len(checks)} 통과")
    return n_pass, len(checks)


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    n_airports = ckpt.get("n_airports", ckpt["encoder"]["airport_emb.weight"].shape[0])
    encoder = FlightEncoder(n_airports=n_airports, constraint_dim=len(FILM_CONSTRAINT_KEYS)).to(device)
    airport_emb_dim = encoder.airport_emb.embedding_dim
    ckpt_state_dim = ckpt["decoder"]["state_mlp.0.weight"].shape[1]
    n_scalars = ckpt_state_dim - airport_emb_dim * 2 - len(FILM_CONSTRAINT_KEYS)
    decoder = PointerDecoder(constraint_dim=len(FILM_CONSTRAINT_KEYS),
                              airport_emb_dim=airport_emb_dim, n_scalars=n_scalars).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval(); decoder.eval()
    return encoder, decoder, n_airports


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
    args = parser.parse_args()

    device = torch.device(args.device)
    evaluate_ip.DEVICE = device
    set_environment("delta")

    map_paths = [v for k, v in config.AIRLINE_DATA.items() if k != "turkish"]
    airport_map = build_airport_map(map_paths)
    base_ids = bases_to_ids(list(config.AIRLINE_BASES["delta"]), airport_map)
    base = base_ids[0]

    summary = {}
    for mode, label in [("normal", "정상 C(FiLM 켜짐)"), ("bypassed", "C, FiLM 무력화(identity 강제)")]:
        encoder, decoder, n_airports = load_model(args.checkpoint, device)
        if n_airports <= 145:
            print(f"경고: n_airports={n_airports} — multi-airline 체크포인트가 아닐 수 있음.")
        if mode == "bypassed":
            bypass_film(encoder)

        greedy_res = run_greedy_stage(encoder, decoder, airport_map, base, device, label)
        table3_res = run_table3_stage(encoder, decoder, airport_map, base_ids, base, device, args, label)
        n_pass, n_total = check_predictions(greedy_res, table3_res, label)
        summary[mode] = (n_pass, n_total, greedy_res, table3_res)

    print(f"\n\n{'='*70}\n최종 비교\n{'='*70}")
    for mode, label in [("normal", "정상 C"), ("bypassed", "FiLM 무력화 C")]:
        n_pass, n_total, _, _ = summary[mode]
        print(f"  {label:<20} 예측-일치 {n_pass}/{n_total}")


if __name__ == "__main__":
    main()
