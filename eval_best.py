"""
eval_best.py — stage3/phase2 체크포인트를 병렬로 평가해서 더 나은 쪽만 결과로 출력.

기준(사용자 지정): dead time(duty-내부 gap만, overnight 제외)과 deadhead 둘 다
한쪽이 다른 쪽보다 같거나 낮고 적어도 하나는 진짜 낮으면(Pareto dominance) 그 쪽만
채택. 어느 쪽도 우세하지 않으면(한쪽은 dead time이 낫고 다른 쪽은 deadhead가 나은
경우) 둘 다 출력해서 판단은 사용자가 하게 한다.

사용례:
  python eval_best.py checkpoints/meze3bec --airline turkish --subset-size 1200 \
      --lambda-dh 10 --ip-time-limit 1800 --device-a cuda:0 --device-b cuda:1

evaluate_ip.py에 전달할 추가 인자는 그대로 뒤에 붙이면 된다(--airline, --subset-size 등).
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
    dead_time = re.search(r"dead time\(duty-내부 gap만, overnight 제외\):\s*([\d.]+)h", output)
    deadhead  = re.search(r"deadhead:\s*(\d+)개 flight", output)
    mandays   = re.search(r"ManDays:\s*(\d+)", output)
    ftc       = re.search(r"FTC:\s*([\d.]+)%", output)
    return dict(
        dead_time=float(dead_time.group(1)) if dead_time else None,
        deadhead=int(deadhead.group(1)) if deadhead else None,
        mandays=int(mandays.group(1)) if mandays else None,
        ftc=float(ftc.group(1)) if ftc else None,
    )


def dominates(a, b):
    """a가 b를 Pareto dominate하는가: dead_time·deadhead 둘 다 <=, 적어도 하나는 <."""
    if a["dead_time"] is None or b["dead_time"] is None:
        return False
    le_dead = a["dead_time"] <= b["dead_time"]
    le_dh   = a["deadhead"] <= b["deadhead"]
    strict  = a["dead_time"] < b["dead_time"] or a["deadhead"] < b["deadhead"]
    return le_dead and le_dh and strict


def main():
    parser = argparse.ArgumentParser(description="stage3/phase2 병렬 평가 → dead_time·deadhead 기준 더 나은 쪽만 출력")
    parser.add_argument("checkpoint_dir", help="예: checkpoints/meze3bec")
    parser.add_argument("--stage-a", default="stage3_best.pt")
    parser.add_argument("--stage-b", default="phase2_best.pt")
    parser.add_argument("--device-a", default="cuda:0")
    parser.add_argument("--device-b", default="cuda:1")
    args, extra = parser.parse_known_args()

    ckpt_a = f"{args.checkpoint_dir}/{args.stage_a}"
    ckpt_b = f"{args.checkpoint_dir}/{args.stage_b}"

    print(f"[eval_best] {args.stage_a}(device={args.device_a}) / {args.stage_b}(device={args.device_b}) 병렬 평가 시작", flush=True)

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_a = ex.submit(run_eval, ckpt_a, extra, args.device_a)
        fut_b = ex.submit(run_eval, ckpt_b, extra, args.device_b)
        out_a, err_a, rc_a = fut_a.result()
        out_b, err_b, rc_b = fut_b.result()

    if rc_a != 0:
        print(f"[eval_best] {args.stage_a} 실행 실패(rc={rc_a}):\n{err_a}", file=sys.stderr)
    if rc_b != 0:
        print(f"[eval_best] {args.stage_b} 실행 실패(rc={rc_b}):\n{err_b}", file=sys.stderr)

    m_a = parse_metrics(out_a)
    m_b = parse_metrics(out_b)

    print(f"\n[eval_best] {args.stage_a}: dead_time={m_a['dead_time']}h deadhead={m_a['deadhead']} ManDays={m_a['mandays']} FTC={m_a['ftc']}%")
    print(f"[eval_best] {args.stage_b}: dead_time={m_b['dead_time']}h deadhead={m_b['deadhead']} ManDays={m_b['mandays']} FTC={m_b['ftc']}%")

    if m_a["dead_time"] is None or m_b["dead_time"] is None:
        print("\n[eval_best] 한쪽 이상 파싱 실패 — 둘 다 원본 출력\n")
        print(f"=== {args.stage_a} ===\n{out_a}")
        print(f"=== {args.stage_b} ===\n{out_b}")
        return

    if dominates(m_a, m_b):
        print(f"\n[eval_best] {args.stage_a}가 dead_time·deadhead 둘 다 우세 → {args.stage_a} 결과만 채택\n")
        print(out_a)
    elif dominates(m_b, m_a):
        print(f"\n[eval_best] {args.stage_b}가 dead_time·deadhead 둘 다 우세 → {args.stage_b} 결과만 채택\n")
        print(out_b)
    else:
        print("\n[eval_best] 우열 없음(한쪽은 dead_time↓, 다른 쪽은 deadhead↓) → 둘 다 출력\n")
        print(f"=== {args.stage_a} ===\n{out_a}")
        print(f"=== {args.stage_b} ===\n{out_b}")


if __name__ == "__main__":
    main()
