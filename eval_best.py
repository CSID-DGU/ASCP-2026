"""
eval_best.py -- evaluates the stage3/phase2 checkpoints in parallel and prints
only the result for the better one.

Criterion (user-specified): if one side is <= the other on both dead time
(within-duty gaps only, excluding overnight) and deadhead, and strictly lower
on at least one (Pareto dominance), only that side is adopted. If neither side
dominates (e.g. one has better dead time while the other has better deadhead),
both are printed and the user makes the call.

Usage example:
  python eval_best.py checkpoints/meze3bec --airline turkish --subset-size 1200 \
      --lambda-dh 10 --ip-time-limit 1800 --device-a cuda:0 --device-b cuda:1

Extra args meant for evaluate_ip.py can just be appended after these (--airline,
--subset-size, etc.).
"""
import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor
import subprocess


def run_eval(checkpoint, extra_args, device):
    cmd = [sys.executable, "-u", "evaluate_ip.py", checkpoint, "--device", device] + extra_args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.stdout, proc.stderr, proc.returncode


def parse_metrics(output):
    dead_time = re.search(r"dead time \(within-duty gaps only, excl\. overnight\):\s*([\d.]+)h", output)
    deadhead  = re.search(r"deadhead:\s*(\d+) legs", output)
    mandays   = re.search(r"ManDays:\s*(\d+)", output)
    ftc       = re.search(r"FTC:\s*([\d.]+)%", output)
    return dict(
        dead_time=float(dead_time.group(1)) if dead_time else None,
        deadhead=int(deadhead.group(1)) if deadhead else None,
        mandays=int(mandays.group(1)) if mandays else None,
        ftc=float(ftc.group(1)) if ftc else None,
    )


def dominates(a, b):
    """Whether a Pareto-dominates b: dead_time and deadhead are both <=, and at least one is <."""
    if a["dead_time"] is None or b["dead_time"] is None:
        return False
    le_dead = a["dead_time"] <= b["dead_time"]
    le_dh   = a["deadhead"] <= b["deadhead"]
    strict  = a["dead_time"] < b["dead_time"] or a["deadhead"] < b["deadhead"]
    return le_dead and le_dh and strict


def main():
    parser = argparse.ArgumentParser(description="Evaluate stage3/phase2 in parallel -- print only the side that's better on dead_time/deadhead")
    parser.add_argument("checkpoint_dir", help="e.g. checkpoints/meze3bec")
    parser.add_argument("--stage-a", default="stage3_best.pt")
    parser.add_argument("--stage-b", default="phase2_best.pt")
    parser.add_argument("--device-a", default="cuda:0")
    parser.add_argument("--device-b", default="cuda:1")
    args, extra = parser.parse_known_args()

    ckpt_a = f"{args.checkpoint_dir}/{args.stage_a}"
    ckpt_b = f"{args.checkpoint_dir}/{args.stage_b}"

    print(f"[eval_best] Starting parallel evaluation: {args.stage_a}(device={args.device_a}) / {args.stage_b}(device={args.device_b})", flush=True)

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_a = ex.submit(run_eval, ckpt_a, extra, args.device_a)
        fut_b = ex.submit(run_eval, ckpt_b, extra, args.device_b)
        out_a, err_a, rc_a = fut_a.result()
        out_b, err_b, rc_b = fut_b.result()

    if rc_a != 0:
        print(f"[eval_best] {args.stage_a} run failed (rc={rc_a}):\n{err_a}", file=sys.stderr)
    if rc_b != 0:
        print(f"[eval_best] {args.stage_b} run failed (rc={rc_b}):\n{err_b}", file=sys.stderr)

    m_a = parse_metrics(out_a)
    m_b = parse_metrics(out_b)

    print(f"\n[eval_best] {args.stage_a}: dead_time={m_a['dead_time']}h deadhead={m_a['deadhead']} ManDays={m_a['mandays']} FTC={m_a['ftc']}%")
    print(f"[eval_best] {args.stage_b}: dead_time={m_b['dead_time']}h deadhead={m_b['deadhead']} ManDays={m_b['mandays']} FTC={m_b['ftc']}%")

    if m_a["dead_time"] is None or m_b["dead_time"] is None:
        print("\n[eval_best] At least one side failed to parse -- printing both raw outputs\n")
        print(f"=== {args.stage_a} ===\n{out_a}")
        print(f"=== {args.stage_b} ===\n{out_b}")
        return

    if dominates(m_a, m_b):
        print(f"\n[eval_best] {args.stage_a} dominates on both dead_time and deadhead -- adopting {args.stage_a} results only\n")
        print(out_a)
    elif dominates(m_b, m_a):
        print(f"\n[eval_best] {args.stage_b} dominates on both dead_time and deadhead -- adopting {args.stage_b} results only\n")
        print(out_b)
    else:
        print("\n[eval_best] No dominance (one is better on dead_time, the other on deadhead) -- printing both\n")
        print(f"=== {args.stage_a} ===\n{out_a}")
        print(f"=== {args.stage_b} ===\n{out_b}")


if __name__ == "__main__":
    main()
