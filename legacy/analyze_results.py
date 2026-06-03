"""
Skip Connection Ablation 결과 분석 스크립트

experiments/results/skip_ablation_results.json 을 읽어:
  1. 학습 곡선 비교 (reward & pairings, sample/greedy)
  2. 최종 성능 비교 (constraint별 pairings 히트맵)
  3. 수렴 속도 비교 (처음으로 특정 threshold에 도달하는 에피소드)
  4. 요약 테이블 출력

실행:
  source venv/bin/activate
  python experiments/analyze_results.py

출력:
  experiments/results/learning_curves.png
  experiments/results/final_performance.png
  experiments/results/convergence_speed.png
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import matplotlib
    matplotlib.use("Agg")   # 화면 없는 서버에서도 동작
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[경고] matplotlib 없음 — 텍스트 요약만 출력합니다.")

RESULTS_PATH = os.path.join(ROOT, "experiments", "results", "skip_ablation_results.json")
OUT_DIR      = os.path.join(ROOT, "experiments", "results")

# 구성별 색상 & 스타일
STYLE = {
    "baseline":         dict(color="#555555", linestyle="-",  linewidth=2,   marker="o",  label="baseline (no skip)"),
    "film_skip":        dict(color="#E07B39", linestyle="--", linewidth=1.8, marker="s",  label="film_skip"),
    "transformer_skip": dict(color="#3A8CC1", linestyle="--", linewidth=1.8, marker="^",  label="transformer_skip"),
    "decoder_skip":     dict(color="#5AB96E", linestyle="--", linewidth=1.8, marker="D",  label="decoder_skip"),
    "all_skip":         dict(color="#B347BF", linestyle="-",  linewidth=2,   marker="*",  label="all_skip"),
}

DUTY_LABELS = ["6h", "8h", "10h", "12h", "14h"]
DUTY_KEYS   = ["6.0", "8.0", "10.0", "12.0", "14.0"]


def load_results():
    if not os.path.exists(RESULTS_PATH):
        print(f"[오류] 결과 파일 없음: {RESULTS_PATH}")
        print("  먼저 실험을 실행하세요: python experiments/skip_ablation.py")
        sys.exit(1)
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── 1. 학습 곡선 ─────────────────────────────────────────────────────────────
def plot_learning_curves(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Skip Connection Ablation — Learning Curves", fontsize=14, fontweight="bold")

    metrics = [
        ("reward_sample",   "Sample Reward",  axes[0, 0]),
        ("reward_greedy",   "Greedy Reward",  axes[0, 1]),
        ("pairings_sample", "Sample Pairings (↓ better)", axes[1, 0]),
        ("pairings_greedy", "Greedy Pairings (↓ better)", axes[1, 1]),
    ]

    for key, title, ax in metrics:
        for name, res in results.items():
            h   = res["history"]
            eps = [r["episode"] for r in h]
            val = [r[key]       for r in h]
            st  = STYLE[name]
            ax.plot(eps, val, color=st["color"], linestyle=st["linestyle"],
                    linewidth=st["linewidth"], marker=st["marker"],
                    markersize=4, markevery=3, label=st["label"])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "learning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {path}")


# ── 2. 최종 성능 비교 (constraint별) ─────────────────────────────────────────
def plot_final_performance(results):
    config_names = list(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Skip Connection Ablation — Final Performance (greedy, after training)",
                 fontsize=13, fontweight="bold")

    # 2-a. constraint별 pairings 라인 플롯
    ax = axes[0]
    duty_vals = [float(k) for k in DUTY_KEYS]
    for name, res in results.items():
        pairings = [res["final_eval"][k]["pairings"] for k in DUTY_KEYS]
        st = STYLE[name]
        ax.plot(duty_vals, pairings, color=st["color"], linestyle=st["linestyle"],
                linewidth=st["linewidth"], marker=st["marker"], markersize=7,
                label=st["label"])
    ax.set_title("Pairings vs max_duty (↓ better)")
    ax.set_xlabel("max_duty (hours)")
    ax.set_ylabel("# Pairings")
    ax.set_xticks(duty_vals)
    ax.set_xticklabels(DUTY_LABELS)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # 2-b. max_duty=14h 기준 bar chart
    ax2 = axes[1]
    bar_data = {name: results[name]["final_eval"]["14.0"]["pairings"] for name in config_names}
    colors   = [STYLE[n]["color"] for n in config_names]
    bars     = ax2.bar(config_names, bar_data.values(), color=colors, edgecolor="black", linewidth=0.8)
    ax2.bar_label(bars, fmt="%d", padding=3, fontsize=11)
    ax2.set_title("Final Pairings at max_duty=14h (↓ better)")
    ax2.set_ylabel("# Pairings")
    ax2.set_ylim(0, max(bar_data.values()) * 1.25)
    ax2.set_xticklabels(config_names, rotation=20, ha="right")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "final_performance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {path}")


# ── 3. 수렴 속도: greedy pairings가 처음으로 threshold 이하가 되는 에피소드 ──
def plot_convergence_speed(results):
    # baseline의 최종 greedy pairings를 기준으로 threshold 설정
    baseline_final = results["baseline"]["final_eval"]["14.0"]["pairings"]
    # threshold = baseline 최종 + 일부 여유 (도달하기 어렵지 않도록)
    thresholds = sorted(set([
        baseline_final + 5,
        baseline_final + 2,
        baseline_final,
        max(baseline_final - 2, 1),
    ]))

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Convergence Speed — Episodes to First Reach Threshold (greedy pairings, ↓ faster)",
                 fontsize=12, fontweight="bold")

    x     = list(range(len(thresholds)))
    width = 0.15
    config_names = list(results.keys())

    for i, name in enumerate(config_names):
        h = results[name]["history"]
        reach_eps = []
        for thr in thresholds:
            found = next(
                (r["episode"] for r in h if r["pairings_greedy"] <= thr),
                None  # 도달 못 했으면 None
            )
            reach_eps.append(found if found is not None else 9999)
        st = STYLE[name]
        bars = ax.bar(
            [xi + i * width for xi in x], reach_eps,
            width=width, color=st["color"], edgecolor="black", linewidth=0.7,
            label=st["label"]
        )
        for bar, ep in zip(bars, reach_eps):
            label = str(ep) if ep < 9999 else "N/A"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    label, ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks([xi + width * 2 for xi in x])
    ax.set_xticklabels([f"≤{t} pairings" for t in thresholds])
    ax.set_ylabel("First Episode to Reach Threshold")
    ax.set_ylim(0, max(
        ep
        for res in results.values()
        for r in res["history"]
        for ep in [r["pairings_greedy"]]
        if False  # dummy — just set a safe limit
    ) if False else None)
    ax.set_ylim(0, 700)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "convergence_speed.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {path}")


# ── 4. 요약 테이블 ────────────────────────────────────────────────────────────
def print_summary_table(results):
    print("\n" + "=" * 90)
    print("SKIP CONNECTION ABLATION — 결과 요약")
    print("=" * 90)

    # 헤더
    header_duty = "  ".join(f"{d:>5}" for d in DUTY_LABELS)
    print(f"{'구성':<22}  {'pairings @ duty (greedy)':^40}  {'최종 reward':>11}  {'학습시간':>9}")
    print(f"{'':22}  {header_duty}  {'(14h)':>11}  {'':>9}")
    print("-" * 90)

    baseline_p14 = results["baseline"]["final_eval"]["14.0"]["pairings"]

    for name, res in results.items():
        p_vals = [res["final_eval"][k]["pairings"] for k in DUTY_KEYS]
        p_str  = "  ".join(f"{p:>5d}" for p in p_vals)
        reward = res["final_eval"]["14.0"]["reward"]
        t      = res["total_time_sec"]

        # baseline 대비 개선 표시
        p14  = p_vals[-1]
        diff = p14 - baseline_p14
        diff_str = f"({diff:+d})" if name != "baseline" else "     "

        print(f"{name:<22}  {p_str}  {reward:>8.1f} {diff_str:>4}  {t:>8.1f}s")

    print("=" * 90)
    print("  baseline 대비 pairings 차이: (+) 증가 (악화), (-) 감소 (개선)")

    # 최적 구성 추천
    print("\n[분석]")
    best_name = min(
        results.keys(),
        key=lambda n: results[n]["final_eval"]["14.0"]["pairings"]
    )
    best_p = results[best_name]["final_eval"]["14.0"]["pairings"]
    print(f"  최소 pairings (max_duty=14h): {best_name} → {best_p}개")

    best_reward_name = max(
        results.keys(),
        key=lambda n: results[n]["final_eval"]["14.0"]["reward"]
    )
    best_r = results[best_reward_name]["final_eval"]["14.0"]["reward"]
    print(f"  최고 reward (max_duty=14h):   {best_reward_name} → {best_r:.1f}")

    # 수렴 속도 (greedy pairings 기준 마지막 10% 에피소드 평균)
    print("\n[수렴 후 안정성 — 마지막 20% 에피소드 greedy pairings 평균]")
    for name, res in results.items():
        h = res["history"]
        tail = h[int(len(h) * 0.8):]
        if tail:
            avg_p = sum(r["pairings_greedy"] for r in tail) / len(tail)
            avg_r = sum(r["reward_greedy"]   for r in tail) / len(tail)
            print(f"  {name:<22}  avg_pairings: {avg_p:5.1f}   avg_reward: {avg_r:7.1f}")

    print()


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    results = load_results()
    print(f"결과 로드 완료: {len(results)}개 구성")

    if HAS_MPL:
        print("\n플롯 생성 중...")
        plot_learning_curves(results)
        plot_final_performance(results)
        plot_convergence_speed(results)
    else:
        print("matplotlib 없음 — 플롯 생략")

    print_summary_table(results)


if __name__ == "__main__":
    main()
