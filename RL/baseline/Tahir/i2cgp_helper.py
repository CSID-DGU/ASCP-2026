"""
i2cgp_helper.py — eval_same_subset.py / eval_cross_objective_full_coverage.py에
I²CGp(DNN-guided) 경로를 추가하기 위한 공용 헬퍼.

지금까지 Tahir 3버전 비교(부분집합만/강제100%커버리지/각각그자체로)는 전부
I²CG(비-DNN baseline)만 썼다 — I²CGp(논문이 실제 제안하는 메인 방법)는 별도
윈도우(eval_delta.py 자체 인스턴스 로딩, 2019-01-01~01-07/4,000편 cap)에서
"I²CG와 mip_obj가 동일하다"는 것만 1회 확인했을 뿐, 3버전 비교와 정확히 같은
subset으로는 실행된 적이 없다(log/0713/세션_요약_0713.md §1-10 참고).

이 모듈은 그 갭을 메우기 위해, Tahir/eval_delta.py의 I²CGp 실행 로직
(_load_dnn/_build_p_psi/run_i2cgp)을 그대로 재사용해서 임의의 inst(Tahir
인스턴스 dict)에 대해 I²CGp를 돌려주는 함수 하나만 제공한다. eval_delta.py,
solver/icg.py는 전혀 수정하지 않았다 — 함수만 import해서 재사용.

Delta 도메인 전용 학습 가중치(Tahir/experiments/delta_dnn/, val_acc 77.2%,
2019-01-08~02-04로 학습되어 3버전 비교의 평가 윈도우 2019-01-01~01-07과 겹치지
않음 확인됨 — log/0709/tahir_비교_계획.md §10-2-1)를 기본값으로 쓴다.
"""
import os
import sys

_THIS_DIR  = os.path.dirname(os.path.realpath(__file__))            # .../RL/baseline/Tahir
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_TAHIR_DIR = os.path.join(_REPO_ROOT, "Tahir")
for p in (_REPO_ROOT, _TAHIR_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_MODEL_DIR = os.path.join(_TAHIR_DIR, "experiments", "delta_dnn")
DEFAULT_MODEL_AT  = "DL"


def run_tahir_i2cgp(
    inst, ref,
    model_dir=DEFAULT_MODEL_DIR, model_at=DEFAULT_MODEL_AT,
    max_fail=3, max_iter=100, time_limit_mip=300,
    max_labels=300, max_pricing_cols=500, verbose=True,
):
    """inst(Tahir 인스턴스 dict) + ref(reference pairings)로 I²CGp를 실행.

    eval_delta.py::run_bts_instance()의 I²CGp 분기(§287-320)와 완전히 동일한
    절차 — enc(학습 시점 encoder, enc_AT_DL.json) 로드 → DNN 가중치 로드 →
    P/Psi 행렬 구성 → run_i2cgp() 호출. eval_delta.py/solver/icg.py는 무수정,
    함수만 재사용.

    반환값은 run_i2cg()와 동일한 shape(mip_obj/coverage/selected_pairings/
    n_iters/total_time/status 등)이라 기존 run_i2cg() 호출부와 그대로
    바꿔 끼울 수 있다.
    """
    import json
    from pathlib import Path

    import eval_delta as tahir_eval_delta   # Tahir, 재사용 (_load_dnn/_build_p_psi)
    from solver.icg import run_i2cgp

    model_dir = Path(model_dir)
    enc_path = model_dir / f"enc_AT_{model_at}.json"
    if not enc_path.exists():
        raise FileNotFoundError(
            f"학습 시점 encoder({enc_path})가 없음 — Tahir/dnn/train_delta.py로 "
            f"먼저 도메인 전용 I2CGp를 학습해야 함(log/0709/tahir_비교_계획.md §10-2-1 참고)."
        )
    with open(enc_path) as f:
        enc = json.load(f)
    if verbose:
        print(f"[I2CGp] 학습 시점 encoder 재사용: {enc_path}")

    model, norm_mean, norm_std = tahir_eval_delta._load_dnn(model_at, model_dir, enc)
    if model is None:
        raise FileNotFoundError(f"AT_{model_at} DNN 가중치를 {model_dir}에서 못 찾음")

    P, Psi, class_max = tahir_eval_delta._build_p_psi(
        inst, ref, model, enc, norm_mean, norm_std
    )

    r = run_i2cgp(
        inst, P, Psi, class_max,
        initial_columns=[list(p) for p in ref],
        max_fail=max_fail, max_iter=max_iter,
        time_limit_mip=time_limit_mip,
        max_labels=max_labels, max_pricing_cols=max_pricing_cols,
        verbose=verbose,
    )
    return r
