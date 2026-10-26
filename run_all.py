"""최종 대화 모델(themaker) 한 번에: 데이터 → 검사 → 합치기 → Qwen3.5-0.8B 학습 → 어휘 줄이기 → Q4_K_M.

    python run_all.py            # 다시 돌리면 끝난 단계는 건너뛰고 이어서
    python run_all.py --status

결과: out/themaker-0.8b-Q4_K_M.gguf (모든 기기 공통 대화 모델, 도구 없음).
단계마다 로그는 logs/<단계>.log, 끝나면 state/<단계>.done. 로그가 --stall 초 동안 멈추면
그 단계를 멈추고 다시 합니다 (네 번까지). 실패한 단계가 있어도 다음 단계로 가고(필요한
단계가 끝나지 않았으면 건너뜀), REPORT.md는 단계마다 새로 씁니다.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

BASE = "Qwen/Qwen3.5-0.8B"   # 2026-03, Apache-2.0
NAME = "themaker-0.8b"
DATA = "data/train-themaker.jsonl"
GGUF = f"out/{NAME}-Q4_K_M.gguf"


def steps():
    """(단계 이름, 명령, 먼저 끝나야 하는 단계들)."""
    run = f"runs/{NAME}"
    return [
        # 초·중·고 친구 대화 6만 개 (학년 고르게, 초등 70자·중고 120자) → 사실·존댓말 검사
        ("data-school", ["make_data.py", "--out", "data/school.jsonl", "--chat", "60000",
                         "--seed", "1"], []),
        ("data-school-clean", ["filter_chat.py", "--data", "data/school.jsonl",
                               "--out", "data/train-school.jsonl"], ["data-school"]),
        # 영어 대화 6천 (학생이 영어로 말하면 쉬운 영어로) + 한국어로 묻는 영어 공부 3천
        ("data-english", ["make_data.py", "--out", "data/english.jsonl", "--english", "6000",
                          "--learn", "3000", "--seed", "3"], []),
        ("data-english-clean", ["filter_chat.py", "--data", "data/english.jsonl",
                                "--out", "data/english-clean.jsonl",
                                "--log", "data/chat-check-english.jsonl"], ["data-english"]),
        # 파이보가 대화를 이어 가는 일상 대화 4~8번 주고받기 1만 2천 개
        ("data-lead", ["make_data.py", "--out", "data/lead.jsonl", "--lead", "12000",
                       "--seed", "4"], []),
        ("data-lead-clean", ["filter_chat.py", "--data", "data/lead.jsonl",
                             "--out", "data/lead-clean.jsonl",
                             "--log", "data/chat-check-lead.jsonl"], ["data-lead"]),
        # 합치기: 대화만, 같은 대답 줄이기, 기기 이름·소개·자기소개 (merge_data.py, devices.py)
        ("data-themaker", ["merge_data.py", "--out", DATA, "data/train-school.jsonl",
                           "data/english-clean.jsonl", "data/lead-clean.jsonl"],
         ["data-school-clean", "data-english-clean", "data-lead-clean"]),
        ("train", ["train.py", "--base", BASE, "--data", DATA, "--out", run, "--lr", "2e-5",
                   "--batch", "2", "--accum", "16"], ["data-themaker"]),
        # 어휘 24만 → 약 3만 (한글·학습 데이터에 나온 토큰은 그대로): Q4_K_M 505 → 약 323MiB
        ("small", ["prune_vocab.py", "--model", f"{run}/final", "--out", f"{run}/small",
                   "--data", DATA], ["train"]),
        # 학습 데이터로 imatrix를 재서 Q4_K_M
        ("gguf", ["export.py", "--model", f"{run}/small", "--name", NAME, "--quant", "Q4_K_M",
                  "--imatrix", DATA], ["small"]),
    ]


def step_env() -> dict:
    """가상환경을 켜지 않고 .venv/bin/python으로 돌려도 그 안의 도구(ninja 등)를
    찾도록: vLLM은 처음 뜰 때 ninja로 GPU 코드를 컴파일합니다."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("VIRTUAL_ENV", str(Path(sys.executable).parents[1]))
    return env


def gpu_holders() -> list[tuple[int, str, int]]:
    """GPU 메모리를 쥔 계산 프로세스들: (pid, 이름, MiB)."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30, check=False).stdout
    rows = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            rows.append((int(parts[0]), parts[1], int(parts[2]) if parts[2].isdigit() else 0))
    return rows


def parent_pid(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def kill_orphan_engines() -> list[int]:
    """부모가 죽어 혼자 남은 vLLM 엔진(VLLM::EngineCore, 부모 pid 1)을 끕니다: 단계를 밖에서
    멈추면(pkill) 엔진만 남아 GPU 메모리를 쥔 채 아무 일도 하지 않고, 다음 단계는 메모리가
    없어 뜨지 못합니다."""
    killed = []
    for pid, name, _ in gpu_holders():
        if "VLLM" in name.upper() and parent_pid(pid) == 1:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    break
                time.sleep(10)
            killed.append(pid)
    if killed:
        print(f"혼자 남은 vLLM 엔진을 껐어요: {killed}", flush=True)
    return killed


def wait_gpu_free(limit_mb: int = 2000, wait_s: int = 600) -> None:
    """앞 단계의 GPU 프로세스(vLLM)가 다 내려갈 때까지 기다립니다: 바로 시작하면
    아직 잡혀 있는 메모리 때문에 다음 단계가 메모리 부족으로 죽습니다."""
    end = time.time() + wait_s
    while time.time() < end:
        try:
            kill_orphan_engines()
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30, check=False).stdout
            used = max(int(x) for x in out.split())
        except (OSError, ValueError, subprocess.SubprocessError):
            return                                   # GPU가 없거나 nvidia-smi가 없음
        if used < limit_mb:
            return
        time.sleep(10)
    holders = ", ".join(f"{name}(pid {pid}, {mb}MiB)" for pid, name, mb in gpu_holders())
    print(f"GPU 메모리가 {wait_s}초 동안 비지 않았지만 시작합니다. 쥐고 있는 것: {holders}",
          flush=True)


def run_step(name, cmd, stall) -> bool:
    wait_gpu_free()
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


def runner_alive() -> bool:
    """실행기가 지금 돌고 있는지: .lock을 다른 누가 쥐고 있으면 돌고 있습니다."""
    with open(HERE / ".lock", "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
    return False


def progress(log: Path) -> str | None:
    """마지막 시도가 아직 끝나지 않았으면 그 로그의 마지막 줄(진행률), 끝났으면 None."""
    if not log.exists():
        return None
    last = log.read_text(errors="replace").rsplit("===== ", 1)[-1]
    if not last.split("\n", 1)[0].endswith("시작"):
        return None
    lines = [ln.strip() for ln in re.split(r"[\r\n]", last)[1:] if ln.strip()]
    return lines[-1][:100] if lines else ""


def report():
    state = HERE / "state"
    lines = [f"# themaker 만들기 ({time.strftime('%Y-%m-%d %H:%M')})", ""]
    alive, shown = runner_alive(), False
    for name, _, _ in steps():
        mark = ("끝" if (state / f"{name}.done").exists() else
                "실패" if (state / f"{name}.failed").exists() else "대기")
        if mark != "끝" and alive and not shown:
            now = progress(HERE / "logs" / f"{name}.log")
            if now is not None:
                mark, shown = f"진행 중 {now}".rstrip(), True
        lines.append(f"- {name}: {mark}")
        if mark == "실패":
            lines += [f"    {ln}" for ln in last_error(HERE / "logs" / f"{name}.log")]
    check = HERE / GGUF.replace(".gguf", "-check.json")
    if check.exists():
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
