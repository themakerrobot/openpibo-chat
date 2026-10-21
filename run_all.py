"""처음부터 끝까지 한 번에: 데이터 → 원래 모델 채점 → 미세조정 → 채점 → GGUF.

    python run_all.py            # 다시 돌리면 끝난 단계는 건너뛰고 이어서
    python run_all.py --status

단계마다 로그는 logs/<단계>.log, 끝나면 state/<단계>.done. 로그가
--stall 초 동안 멈추면 그 단계를 멈추고 다시 합니다 (세 번까지). 실패한
단계가 있어도 다음 단계로 가고(필요한 단계가 끝나지 않았으면 건너뜀),
REPORT.md는 단계마다 새로 씁니다.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

#: (이름, 학습 시작점, 학습률, 배치, 누적)
STUDENTS = [("qwen3-0.6b", "Qwen/Qwen3-0.6B", 2e-5, 16, 2),
            ("qwen3-1.7b", "Qwen/Qwen3-1.7B", 1e-5, 8, 4)]


def steps():
    s = [("data-train", ["make_data.py", "--out", "data/train.jsonl", "--chat", "24000",
                         "--tool", "8000", "--seed", "0"], []),
         ("data-heldout", ["make_data.py", "--out", "data/heldout.jsonl", "--chat", "400",
                           "--tool", "300", "--seed", "999", "--avoid", "data/train.jsonl"],
          ["data-train"]),
         # 지금 파이보에서 쓰는 모델 (도구 형식이 없어 대화만 채점)
         ("eval-gemma3-1b", ["evaluate.py", "gen", "--model", "google/gemma-3-1b-it",
                             "--name", "gemma3-1b-원본", "--no-tools"], ["data-heldout"])]
    for name, base, *_ in STUDENTS:
        s.append((f"eval-{name}-원본", ["evaluate.py", "gen", "--model", base,
                                       "--name", f"{name}-원본"], ["data-heldout"]))
    for name, base, lr, batch, accum in STUDENTS:
        s.append((f"train-{name}", ["train.py", "--base", base, "--data", "data/train.jsonl",
                                    "--out", f"runs/{name}", "--lr", str(lr),
                                    "--batch", str(batch), "--accum", str(accum)],
                  ["data-train"]))
        s.append((f"eval-{name}-파이보", ["evaluate.py", "gen", "--model",
                                        f"runs/{name}/final", "--name", f"{name}-파이보"],
                  [f"train-{name}", "data-heldout"]))
    s.append(("judge", ["evaluate.py", "judge"], ["data-heldout"]))
    for name, *_ in STUDENTS:
        s.append((f"gguf-{name}", ["export.py", "--model", f"runs/{name}/final",
                                   "--name", f"pibo-{name}"], [f"train-{name}"]))
    return s


def step_env() -> dict:
    """가상환경을 켜지 않고 .venv/bin/python으로 돌려도 그 안의 도구(ninja 등)를
    찾도록: vLLM은 처음 뜰 때 ninja로 GPU 코드를 컴파일합니다."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env.setdefault("VIRTUAL_ENV", str(Path(sys.executable).parents[1]))
    return env


def run_step(name, cmd, stall) -> bool:
    log = HERE / "logs" / f"{name}.log"
    log.parent.mkdir(exist_ok=True)
    with log.open("a") as out:
        out.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 시작\n")
        out.flush()
        proc = subprocess.Popen([PY, "-u", *cmd], cwd=HERE, stdout=out, stderr=subprocess.STDOUT,
                                env=step_env(),
                                start_new_session=True)
        while proc.poll() is None:
            time.sleep(30)
            if time.time() - log.stat().st_mtime > stall:
                out.write(f"\n===== {stall}초 동안 멈춰 있어 중단\n")
                out.flush()
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                return False
        out.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 끝 (코드 {proc.returncode})\n")
    return proc.returncode == 0


def last_error(log: Path, n: int = 3) -> list[str]:
    """실패한 단계의 마지막 시도에서 원인이 적힌 줄 몇 개 (상태 화면에 바로 보이게)."""
    if not log.exists():
        return ["(로그 없음)"]
    text = log.read_text(errors="replace")
    tail = text.split("===== ")[-2] if text.count("===== ") >= 2 else text
    keys = ("Error", "error:", "Exception", "SystemExit", "못함", "없어요", "401", "403")
    hits = [ln.strip() for ln in tail.splitlines() if any(k in ln for k in keys)]
    return [h[:160] for h in (hits[-n:] or tail.strip().splitlines()[-n:])]


def report():
    state = HERE / "state"
    lines = [f"# openpibo-chat 결과 ({time.strftime('%Y-%m-%d %H:%M')})", ""]
    for name, _, _ in steps():
        mark = ("끝" if (state / f"{name}.done").exists() else
                "실패" if (state / f"{name}.failed").exists() else "대기")
        lines.append(f"- {name}: {mark}")
        if mark == "실패":
            lines += [f"    {ln}" for ln in last_error(HERE / "logs" / f"{name}.log")]
    if any((HERE / "results").glob("answers-*.json")):
        sys.path.insert(0, str(HERE))
        os.chdir(HERE)
        from evaluate import report as table

        lines += ["", "## 채점 (채점용 질문, 선생 모델이 대화 채점)", "", table()]
    for check in sorted((HERE / "out").glob("*-check.json")):
        d = json.loads(check.read_text())
        lines += ["", f"## {Path(d['file']).name} ({d['MB']} MB)", "", "```",
                  json.dumps(d["check"], ensure_ascii=False, indent=1), "```"]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--status", action="store_true")
    p.add_argument("--stall", type=int, default=2 * 3600)
    p.add_argument("--tries", type=int, default=4)
    args = p.parse_args()
    if args.status:
        report()
        print((HERE / "REPORT.md").read_text())
        return
    lock = open(HERE / ".lock", "w")  # noqa: SIM115 — 끝날 때까지 쥐고 있는 잠금
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("이미 돌고 있어요") from None
    state = HERE / "state"
    state.mkdir(exist_ok=True)
    for rnd in range(2):
        for name, cmd, needs in steps():
            done, failed = state / f"{name}.done", state / f"{name}.failed"
            if done.exists():
                continue
            if any(not (state / f"{n}.done").exists() for n in needs):
                print(f"{name}: {', '.join(needs)} 기다림", flush=True)
                continue
            for attempt in range(1, args.tries + 1):
                print(f"{time.strftime('%m-%d %H:%M')} {name} ({attempt}번째)", flush=True)
                if run_step(name, cmd, args.stall):
                    done.touch()
                    failed.unlink(missing_ok=True)
                    break
                time.sleep(60)
            else:
                failed.touch()
                print(f"{name}: {args.tries}번 실패, 다음으로", flush=True)
            report()
        if all((state / f"{n}.done").exists() for n, _, _ in steps()):
            break
        if rnd == 0:
            print("실패한 단계를 30분 뒤 한 번 더", flush=True)
            time.sleep(1800)
    report()
    print((HERE / "REPORT.md").read_text())


if __name__ == "__main__":
    main()
