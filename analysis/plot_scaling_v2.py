"""
plot_scaling_v2.py — eval_scaling_v2(순차, seed 0/1/2, gpu2 단독 실행) 3-seed 결과를
파싱해 scaling 그래프 2장을 그린다. seed 디렉토리 구조:
    log/p500_new/scaling_v2/seed{0,1,2}/eval_scaling_<n>.out
    log/p500_new/scaling_v2/seed{0,1,2}/runtimes.txt

runtime 그래프는 mean±std 음영(3-seed 변동폭)으로, coverage/FTC/deadhead/ManDays
4패널은 mean 선만(음영 없음) 그린다.
"""
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

EVALDIR = "log/p500_new/scaling_v2"
OUTDIR = "log/supplementary"
SIZES = [3500, 7000, 14000, 28000, 56000]
SEEDS = [0, 1, 2]


def load_runtimes(seed):
    runtimes = {}
    with open(os.path.join(EVALDIR, f"seed{seed}", "runtimes.txt")) as f:
        for line in f:
            n, sec = line.split()
            runtimes[int(n)] = int(sec)
    return runtimes


def parse_out(seed, n):
    with open(os.path.join(EVALDIR, f"seed{seed}", f"eval_scaling_{n}.out")) as f:
        text = f.read()
    coverage = float(re.search(r"coverage:\s+([\d.]+)%", text).group(1))
    mandays = int(re.search(r"ManDays:\s+(\d+)", text).group(1))
    deadhead = int(re.search(r"deadhead:\s+(\d+)개 flight", text).group(1))
    ftc = float(re.search(r"FTC:\s+([\d.]+)%", text).group(1))
    return dict(coverage=coverage, mandays=mandays, deadhead=deadhead, ftc=ftc)


def collect():
    """seed별로 다 모아서 {n: {metric: [values per seed]}} 형태로 반환."""
    raw = {n: {"runtime": [], "coverage": [], "mandays": [], "deadhead": [], "ftc": []} for n in SIZES}
    for seed in SEEDS:
        rt = load_runtimes(seed)
        for n in SIZES:
            s = parse_out(seed, n)
            raw[n]["runtime"].append(rt[n])
            raw[n]["coverage"].append(s["coverage"])
            raw[n]["mandays"].append(s["mandays"])
            raw[n]["deadhead"].append(s["deadhead"])
            raw[n]["ftc"].append(s["ftc"])
    return raw


def mean_std(raw, metric):
    means = np.array([np.mean(raw[n][metric]) for n in SIZES])
    stds = np.array([np.std(raw[n][metric]) for n in SIZES])
    return means, stds


def main():
    raw = collect()
    xs = SIZES

    rt_mean, rt_std = mean_std(raw, "runtime")

    # runtime vs N — N이 등비수열(3500→7000→...→56000, 매번 ×2)이라 순수 linear 축에
    # raw value 그대로 찍으면 간격이 등차가 아니라 뒤로 갈수록 벌어지는 모양이 그대로 드러남
    # (인덱스 기반 균등간격은 이 특성을 지워버려서 부적절 — log-scale도 동일 이유로 부적절).
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(xs, rt_mean, "o-", color="#C44E52", linewidth=2, markersize=7, label="mean (n=3 seed)")
    ax.fill_between(xs, rt_mean - rt_std, rt_mean + rt_std, color="#C44E52", alpha=0.2, label="±1 std")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n:,}" for n in xs], rotation=45, ha="right")
    ax.set_xlabel("N (flights)")
    ax.set_ylabel("runtime (s)")
    ax.set_title("Scaling: runtime vs. instance size\n(seed 0/1/2 mean±std)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    for x, y in zip(xs, rt_mean):
        ax.annotate(f"{y:.0f}s", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "Figure_scaling_runtime_v2.png"), dpi=150)
    plt.close(fig)

    # coverage / FTC / deadhead / ManDays — 4개 서브플롯, mean±std 음영
    cov_mean, cov_std = mean_std(raw, "coverage")
    ftc_mean, ftc_std = mean_std(raw, "ftc")
    dh_mean, dh_std = mean_std(raw, "deadhead")
    md_mean, md_std = mean_std(raw, "mandays")

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    panels = [
        (axes[0], cov_mean, cov_std, "coverage (%)", "#4C72B0", "coverage vs. N", "{:.1f}%", (0, 100)),
        (axes[1], ftc_mean, ftc_std, "FTC (%)", "#C44E52", "FTC vs. N", "{:.1f}%", (0, 100)),
        (axes[2], dh_mean, dh_std, "deadhead (count)", "#55A868", "deadhead vs. N", "{:.0f}", None),
        (axes[3], md_mean, md_std, "ManDays", "#8172B2", "ManDays vs. N", "{:.0f}", None),
    ]
    for ax, ys, ystd, ylabel, color, title, fmt, ylim in panels:
        ax.plot(xs, ys, "o-", color=color, linewidth=2, markersize=7)
        ax.fill_between(xs, ys - ystd, ys + ystd, color=color, alpha=0.2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{n:,}" for n in xs], rotation=45, ha="right")
        ax.set_xlabel("N (flights)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)
        for x, y in zip(xs, ys):
            ax.annotate(fmt.format(y), (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    fig.suptitle("Scaling: coverage / FTC / deadhead / ManDays vs. instance size (3-seed mean±std)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "Figure_scaling_coverage_v2.png"), dpi=150)
    plt.close(fig)

    print("저장 완료:")
    print(f"  {OUTDIR}/Figure_scaling_runtime_v2.png")
    print(f"  {OUTDIR}/Figure_scaling_coverage_v2.png")
    print()
    header = f"{'N':>8} {'runtime(mean±std)':>20} {'coverage%':>12} {'ManDays':>12} {'deadhead':>12} {'FTC%':>12}"
    print(header)
    for i, n in enumerate(xs):
        rt_vals = raw[n]["runtime"]
        cov_vals = raw[n]["coverage"]
        md_vals = raw[n]["mandays"]
        dh_vals = raw[n]["deadhead"]
        ftc_vals = raw[n]["ftc"]
        print(f"{n:>8} "
              f"{f'{np.mean(rt_vals):.0f}±{np.std(rt_vals):.0f}':>20} "
              f"{f'{np.mean(cov_vals):.1f}±{np.std(cov_vals):.1f}':>12} "
              f"{f'{np.mean(md_vals):.0f}±{np.std(md_vals):.0f}':>12} "
              f"{f'{np.mean(dh_vals):.0f}±{np.std(dh_vals):.0f}':>12} "
              f"{f'{np.mean(ftc_vals):.1f}±{np.std(ftc_vals):.1f}':>12}")


if __name__ == "__main__":
    main()
