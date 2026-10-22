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
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

#: 만들어서 같은 질문으로 채점해 볼 양자화 (작은 모델이라 Q4에서 많이 나빠집니다)
QUANTS = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
BASE = "Qwen/Qwen3-0.6B"
#: 학생 모델. 1.7B는 KV 캐시까지 합치면 파이보에서 젬마3 1B보다 램을 450MB쯤 더 써서 뺐습니다.
#: data: (학습 데이터, 그걸 만드는 단계), quants: 만들어서 채점할 양자화,
#: set: 채점용 질문 묶음 (kids: 처음 만든 미취학~초저, school: 초·중·고)
QWEN35 = "Qwen/Qwen3.5-0.8B"
STUDENTS = [
    # 초·중·고 친구 대화(학년 고르게, 초등 70자·중고 120자, 호기심 질문 15%)를 선생이
    # 생각을 켜고 사실·한국어를 다시 본 데이터 + 거른 도구 예시 (filter_chat.py).
    # 증류는 어린이 데이터에서 v1보다 나빠서(대화 4.64, 도구 92%) 하지 않습니다.
    {"name": "qwen3-0.6b-school", "base": BASE, "lr": 2e-5, "batch": 4, "accum": 8,
     "data": ("data/train-school.jsonl", "data-school-clean"),
     "quants": ["Q4_K_M", "Q6_K", "Q8_0"], "no_kd": True, "set": "school"},
    # Qwen3.5-0.8B (2026-03, Apache-2.0): 한국어가 낫고(글자당 2.73비트, Qwen3-0.6B 3.27)
    # 한국어 토큰이 23% 적고 KV 캐시가 작음. 어휘가 커서(24만) 배치는 2×16. 선생(Qwen3)과
    # 토크나이저가 달라 증류는 못 합니다. Q4_K_M은 505MiB로 KV 캐시까지 치면 가장 작음.
    {"name": "qwen3.5-0.8b-school", "base": QWEN35, "lr": 2e-5, "batch": 2, "accum": 16,
     "data": ("data/train-school.jsonl", "data-school-clean"),
     "quants": ["Q4_K_M", "Q6_K", "Q8_0"], "no_kd": True, "set": "school"},
    # 처음 만든 어린이 데이터 (원래 데이터, 도구 예시를 거른 데이터)
    {"name": "qwen3-0.6b", "base": BASE, "lr": 2e-5, "batch": 4, "accum": 8,
     "data": ("data/train.jsonl", "data-train"), "quants": QUANTS},
    {"name": "qwen3-0.6b-v2", "base": BASE, "lr": 2e-5, "batch": 4, "accum": 8,
     "data": ("data/train-clean.jsonl", "data-clean"), "quants": ["Q6_K", "Q8_0"]},
]
KD = "data/kd"
#: 채점용 질문 묶음마다 (그 질문을 만드는 단계, 답을 두는 곳, 채점 단계)
SETS = {"kids": ("data-heldout", "results", "judge"),
        "school": ("data-school-heldout", "results-school", "judge-school")}
#: 원본 그대로 채점해 둘 모델 (묶음마다): (단계 이름, 답 이름, 모델, 도구 형식이 없는지)
ORIGINALS = [("gemma3-1b", "gemma3-1b-원본", "google/gemma-3-1b-it", True),  # 지금 파이보의 모델
             ("qwen3-0.6b-원본", "qwen3-0.6b-원본", BASE, False),
             ("qwen3.5-0.8b-원본", "qwen3.5-0.8b-원본", QWEN35, False)]


#: 채점 단계 → 그 답이 있는 곳
JUDGES = {judge: results for _, results, judge in SETS.values()}


def students():
    """state/kd.on 파일이 있으면 학생마다 증류로도 학습해 나란히 채점합니다
    (선생 확률 구하기 distill.py → train.py --kd). 선생 확률은 원래 데이터로 한 번
    구하고, 거른 데이터는 그 일부라 그대로 씁니다."""
    plain = [dict(s, kd=None) for s in STUDENTS]
    if not (HERE / "state" / "kd.on").exists():
        return plain
    return plain + [dict(s, name=f"{s['name']}-kd", kd=KD) for s in STUDENTS
                    if not s.get("no_kd")]


def eval_args(st: dict) -> tuple[list[str], list[str]]:
    """학생을 채점할 때 붙일 인자와 기다릴 단계."""
    which = st.get("set", "kids")
    made_by = SETS[which][0]
    return (["--set", which] if which != "kids" else []), \
        (["data-heldout", made_by] if made_by != "data-heldout" else ["data-heldout"])


def steps():
    s = [("data-train", ["make_data.py", "--out", "data/train.jsonl", "--chat", "24000",
                         "--tool", "8000", "--seed", "0"], []),
         ("data-heldout", ["make_data.py", "--out", "data/heldout.jsonl", "--chat", "400",
                           "--tool", "300", "--seed", "999", "--avoid", "data/train.jsonl"],
          ["data-train"]),
         ("data-clean", ["filter_tools.py", "--train", "data/train.jsonl", "--heldout",
                         "data/heldout.jsonl", "--out", "data/train-clean.jsonl",
                         "--marks", "data/heldout-marks.json"], ["data-train", "data-heldout"]),
         # 초·중·고: 대화만 새로 만들고, 채점용 대화 질문 400개(어린이 채점용과 같은 수)도
         ("data-school", ["make_data.py", "--audience", "school", "--out", "data/school.jsonl",
                          "--chat", "24000", "--tool", "0", "--seed", "1"], []),
         ("data-school-heldout", ["make_data.py", "--audience", "school", "--out",
                                  "data/heldout-school.jsonl", "--chat", "400", "--tool", "0",
                                  "--seed", "998", "--avoid", "data/school.jsonl"],
          ["data-school"])]
    for which, (made_by, _, _) in SETS.items():
        extra = ["--set", which] if which != "kids" else []
        needs = sorted({"data-heldout", made_by})
        tag = "" if which == "kids" else f"{which}-"
        for step, name, model, no_tools in ORIGINALS:
            s.append((f"eval-{tag}{step}", ["evaluate.py", "gen", "--model", model, "--name", name,
                                           *extra, *(["--no-tools"] if no_tools else [])], needs))
    s.append(("data-school-clean", ["filter_chat.py", "--data", "data/school.jsonl",
                                    "--tools-from", "data/train-clean.jsonl",
                                    "--out", "data/train-school.jsonl"],
              ["data-school", "data-clean"]))
    for st in students():
        name, (data, made_by) = st["name"], st["data"]
        cmd = ["train.py", "--base", st["base"], "--data", data, "--out", f"runs/{name}",
               "--lr", str(st["lr"]), "--batch", str(st["batch"]), "--accum", str(st["accum"])]
        needs = [made_by]
        if st["kd"]:
            if not any(n == "kd-teacher" for n, _, _ in s):
                s.append(("kd-teacher", ["distill.py", "--data", "data/train.jsonl", "--base",
                                         st["base"], "--out", st["kd"]], ["data-train"]))
            cmd += ["--kd", st["kd"]]
            needs.append("kd-teacher")
        s.append((f"train-{name}", cmd, needs))
        extra, wait = eval_args(st)
        s.append((f"eval-{name}-파이보", ["evaluate.py", "gen", "--model",
                                        f"runs/{name}/final", "--name", f"{name}-파이보", *extra],
                  [f"train-{name}", *wait]))
    for st in students():
        name = st["name"]
        extra, wait = eval_args(st)
        for quant in st["quants"]:   # GGUF는 CPU만 씁니다
            s.append((f"gguf-{name}-{quant}", ["export.py", "--model", f"runs/{name}/final",
                                               "--name", f"pibo-{name}", "--quant", quant],
                      [f"train-{name}"]))
            s.append((f"eval-{name}-{quant}", ["evaluate.py", "gguf", "--gguf",
                                               f"out/pibo-{name}-{quant}.gguf",
                                               "--name", f"{name}-{quant}", *extra],
                      [f"gguf-{name}-{quant}", *wait]))
    # 모은 답을 한꺼번에 채점 (나중에 새로 모은 답이 있으면 다시 돌아 그것만 채점)
    for which, (made_by, _, judge) in SETS.items():
        extra = ["--set", which] if which != "kids" else []
        s.append((judge, ["evaluate.py", "judge", *extra], sorted({"data-heldout", made_by})))
    return s


def step_env() -> dict:
    """가상환경을 켜지 않고 .venv/bin/python으로 돌려도 그 안의 도구(ninja 등)를
    찾도록: vLLM은 처음 뜰 때 ninja로 GPU 코드를 컴파일합니다."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("VIRTUAL_ENV", str(Path(sys.executable).parents[1]))
    return env


def wait_gpu_free(limit_mb: int = 2000, wait_s: int = 600) -> None:
    """앞 단계의 GPU 프로세스(vLLM)가 다 내려갈 때까지 기다립니다: 바로 시작하면
    아직 잡혀 있는 메모리 때문에 다음 단계가 메모리 부족으로 죽습니다."""
    end = time.time() + wait_s
    while time.time() < end:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30, check=False).stdout
            used = max(int(x) for x in out.split())
        except (OSError, ValueError, subprocess.SubprocessError):
            return                                   # GPU가 없거나 nvidia-smi가 없음
        if used < limit_mb:
            return
        time.sleep(10)
    print(f"GPU 메모리가 {wait_s}초 동안 비지 않았지만 시작합니다", flush=True)


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


def unscored(results: str = "results") -> list[str]:
    """답은 모았는데 선생 모델이 아직 채점(또는 사실 검사)하지 않은 모델: judge가 한 번
    끝난 뒤에 새로 모은 답(젬마, 학습한 모델)도 채점하게 judge를 다시 돌립니다."""
    answers = sorted((HERE / results).glob("answers-*.json"))
    return [f.stem.removeprefix("answers-") for f in answers
            if not all(f.with_name(f.name.replace("answers-", k)).exists()
                       for k in ("scores-", "facts-"))]


def report():
    state = HERE / "state"
    lines = [f"# openpibo-chat 결과 ({time.strftime('%Y-%m-%d %H:%M')})", ""]
    alive, shown = runner_alive(), False
    for name, _, _ in steps():
        mark = ("끝" if (state / f"{name}.done").exists() else
                "실패" if (state / f"{name}.failed").exists() else "대기")
        if name in JUDGES and unscored(JUDGES[name]):
            mark = f"대기 (아직 채점 안 한 답: {', '.join(unscored(JUDGES[name]))})"
        if mark != "끝" and alive and not shown:
            now = progress(HERE / "logs" / f"{name}.log")
            if now is not None:
                mark, shown = f"진행 중 {now}".rstrip(), True
        lines.append(f"- {name}: {mark}")
        if mark == "실패":
            lines += [f"    {ln}" for ln in last_error(HERE / "logs" / f"{name}.log")]
    titles = {"school": "초·중·고 채점용 질문", "kids": "어린이 채점용 질문 (처음 데이터)"}
    for which in ("school", "kids"):
        if not any((HERE / SETS[which][1]).glob("answers-*.json")):
            continue
        sys.path.insert(0, str(HERE))
        os.chdir(HERE)
        import evaluate

        evaluate.use_set(which)
        lines += ["", f"## 채점: {titles[which]} (선생 모델이 대화 채점)", "", evaluate.report()]
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
            if name in JUDGES and unscored(JUDGES[name]):
                done.unlink(missing_ok=True)
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
