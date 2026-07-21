"""
train_delta.py — I2CGp DNN을 실제 BTS Delta 도메인 데이터로 학습.

기존 dnn/train.py는 CPPSC 합성 벤치마크(7개 항공기 타입)로만 leave-one-type-out
학습을 한다 — 실제 BTS Delta 데이터로 학습된 가중치가 없어서 eval_delta.py의
I2CGp가 항상 스킵됐다(log/0709/tahir_비교_계획.md §10-2). 이 스크립트는
dnn/delta_loader.py(이미 존재하는 BTS→Tahir 인스턴스 변환기)로 여러 주(week) 단위
Delta 실제 데이터를 인스턴스로 만들어 같은 학습 파이프라인(dataset.py/reference.py/
model.py)에 그대로 태운다 — 인스턴스 스키마가 동일해서 별도 데이터 파이프라인 수정은
필요 없다.

주의: RL/data/small-scale 비교에 쓴 평가 윈도우(2019-01-01~01-07)는 학습에서 제외한다
(train/eval 겹침 방지).

Usage:
    python -m dnn.train_delta --epochs 30
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from dnn.delta_loader import load_bts_instance, discover_bts_instances
from dnn.dataset import build_successor_sets, build_encoders, build_dataset, filter_successors_by_pattern
from dnn.reference import generate_reference_pairings
from dnn.train import pad_batch, train_model

EVAL_WINDOW = ("2019-01-01", "2019-01-07")  # RL/data/small-scale 비교와 겹치는 구간 — 학습 제외


def build_training_windows(carrier: str, step_days: int, csv_path: str = None) -> List[Dict]:
    """discover_bts_instances로 주 단위 윈도우를 찾고, 평가 윈도우와 겹치는 것만 제외."""
    windows = discover_bts_instances(carrier=carrier, csv_path=csv_path, step_days=step_days)
    kept = [w for w in windows if not (w["date_start"] == EVAL_WINDOW[0] and w["date_end"] == EVAL_WINDOW[1])]
    print(f"발견된 윈도우 {len(windows)}개, 평가 윈도우(0107) 제외 후 학습용 {len(kept)}개:")
    for w in kept:
        print(f"  {w['date_start']} ~ {w['date_end']}  ({w['n_legs']} legs)")
    return kept


def load_all_delta_data(
    windows: List[Dict], carrier: str, enc: Dict, ref_method: str = "cg", csv_path: str = None,
    max_legs: int = 3000,
) -> Tuple[List[np.ndarray], List[int]]:
    X_all, y_all = [], []
    for i, w in enumerate(windows):
        inst = load_bts_instance(
            carrier=carrier, date_start=w["date_start"], date_end=w["date_end"], csv_path=csv_path,
            max_legs=max_legs,
        )
        print(f"  [{i+1}/{len(windows)}] {w['date_start']}~{w['date_end']} "
              f"({len(inst['legs'])} legs, bases={inst['bases']}) reference 생성 중...", flush=True)
        ref = generate_reference_pairings(inst, method=ref_method)
        succ = build_successor_sets(inst["legs"])
        fsucc = filter_successors_by_pattern(inst["legs"], succ, ref)

        for base in inst["bases"]:
            Xb, yb = build_dataset(inst, fsucc, ref, enc, base)
            X_all.extend(Xb)
            y_all.extend(yb)
        print(f"         → 누적 샘플 {len(X_all)}개 (reference pairings {len(ref)}개)", flush=True)

    return X_all, y_all


def main():
    parser = argparse.ArgumentParser(description="I2CGp DNN을 BTS Delta 실제 데이터로 학습")
    parser.add_argument("--carrier", default="DL")
    parser.add_argument("--step_days", type=int, default=7)
    parser.add_argument("--ref", default="cg", choices=["cg", "greedy"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--emb_dim", type=int, default=10)
    parser.add_argument("--conv", type=int, default=1)
    parser.add_argument("--filters", type=int, default=128)
    parser.add_argument("--fsize", type=int, default=3)
    parser.add_argument("--dense", type=int, default=2)
    parser.add_argument("--neurons", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--out_dir", default=None,
                         help="가중치 저장 위치 (기본: Tahir/experiments/delta_dnn)")
    parser.add_argument("--max_legs_per_window", type=int, default=3000,
                         help="윈도우당 최대 leg 수 — CG reference 생성 시간이 인스턴스"
                              "크기에 초선형으로 늘어나서(14,439legs=300s+ 타임아웃 확인됨,"
                              "3,000legs=6.6s) 제한 필요")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    hparams = dict(
        embedding_dim=args.emb_dim, num_conv_layers=args.conv, num_filters=args.filters,
        filter_size=args.fsize, num_dense_layers=args.dense, neurons_per_layer=args.neurons,
        dropout_rate=args.dropout,
    )

    windows = build_training_windows(args.carrier, args.step_days, csv_path=args.csv)
    if len(windows) < 2:
        raise RuntimeError(f"학습용 윈도우가 {len(windows)}개뿐 — 검증 세트를 못 나눔")

    # 마지막 윈도우(시간상 가장 나중)를 검증용으로 홀드아웃 — 나머지로 학습
    val_window = windows[-1]
    train_windows = windows[:-1]
    print(f"\n검증 윈도우(홀드아웃): {val_window['date_start']}~{val_window['date_end']}")
    print(f"학습 윈도우: {len(train_windows)}개\n")

    # 인코더는 전체(학습+검증) 윈도우의 공항 집합으로 통일해서 구성
    print("인스턴스 로드 후 encoder 구성 중...")
    all_insts = []
    for w in windows:
        all_insts.append(load_bts_instance(
            carrier=args.carrier, date_start=w["date_start"], date_end=w["date_end"], csv_path=args.csv,
            max_legs=args.max_legs_per_window,
        ))
    enc = build_encoders(all_insts)
    print(f"encoder: airports={len(enc['airport'])}, aircraft={len(enc['aircraft'])}\n")

    print("=== 학습 데이터 구성 ===")
    X_train, y_train = load_all_delta_data(train_windows, args.carrier, enc, args.ref, args.csv,
                                            max_legs=args.max_legs_per_window)
    print(f"\n학습 샘플 총 {len(X_train)}개")

    print("\n=== 검증 데이터 구성 ===")
    X_val, y_val = load_all_delta_data([val_window], args.carrier, enc, args.ref, args.csv,
                                        max_legs=args.max_legs_per_window)
    print(f"\n검증 샘플 총 {len(X_val)}개")

    X_tr_pad, y_tr, m_tr = pad_batch(X_train, y_train)
    X_va_pad, y_va, m_va = pad_batch(X_val, y_val)

    num_cols = list(range(4, 9)) + list(range(13, 18)) + list(range(22, 27))
    mean = X_tr_pad[:, :, num_cols].mean(axis=(0, 1))
    std = X_tr_pad[:, :, num_cols].std(axis=(0, 1)) + 1e-9
    X_tr_pad[:, :, num_cols] = (X_tr_pad[:, :, num_cols] - mean) / std
    X_va_pad[:, :, num_cols] = (X_va_pad[:, :, num_cols] - mean) / std

    print(f"\n=== 학습 시작 (train={len(X_tr_pad)}, val={len(X_va_pad)}, max_K={X_tr_pad.shape[1]}) ===")
    model, history = train_model(
        X_tr_pad, y_tr, m_tr, X_va_pad, y_va, m_va, enc,
        epochs=args.epochs, batch_size=args.batch, lr=args.lr, **hparams,
    )
    best_acc = max(history["val_acc"])
    print(f"\nBest val accuracy: {best_acc:.3f}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent.parent / "experiments" / "delta_dnn"
    out_dir.mkdir(parents=True, exist_ok=True)

    at_tag = args.carrier  # eval_delta.py --model_at DL 로 조회
    model.save_weights(str(out_dir / f"weights_AT_{at_tag}.h5"))
    with open(out_dir / f"model_config_AT_{at_tag}.json", "w") as f:
        json.dump({"n_airports": len(enc["airport"]), "n_aircraft": len(enc["aircraft"]), "hparams": hparams}, f)
    with open(out_dir / f"norm_AT_{at_tag}.json", "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
    # eval_delta.py가 평가 인스턴스 하나만으로 encoder를 재구성하면 학습 시 encoder와
    # airport→index 매핑이 어긋날 수 있어(§10-2 재검토 과정에서 발견), encoder 자체도
    # 저장해서 평가 시 재사용하도록 한다(eval_delta.py 쪽 로딩 수정과 짝을 이룸).
    with open(out_dir / f"enc_AT_{at_tag}.json", "w") as f:
        json.dump({"airport": enc["airport"], "aircraft": enc["aircraft"]}, f)

    with open(out_dir / f"train_meta_AT_{at_tag}.json", "w") as f:
        json.dump({
            "carrier": args.carrier,
            "train_windows": [(w["date_start"], w["date_end"]) for w in train_windows],
            "val_window": (val_window["date_start"], val_window["date_end"]),
            "excluded_eval_window": EVAL_WINDOW,
            "n_train_samples": len(X_train),
            "n_val_samples": len(X_val),
            "best_val_acc": best_acc,
        }, f, indent=2)

    print(f"\n저장 완료: {out_dir}")


if __name__ == "__main__":
    main()
