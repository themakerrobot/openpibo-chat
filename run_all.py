"""처음부터 끝까지 한 번에: 데이터 → 원래 모델 채점 → 미세조정 → 채점 → GGUF.

    python run_all.py            # 다시 돌리면 끝난 단계는 건너뛰고 이어서
    python run_all.py --status
    python run_all.py --first 'qwen3\.5-0\.8b-school(-|$)|judge-school'   # 이 단계들부터

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
#: 어휘를 줄인 모델(prune_vocab.py)로도 만들 양자화: 한국어는 원래와 똑같이 나뉘고 이모지 등
#: 안 쓰는 토큰만 빠져, Q4_K_M이 Qwen3-0.6B 379→약 280MiB, Qwen3.5 505→322MiB.
#: 줄인 Q4_K_M은 학습 데이터로 imatrix를 재서 양자화합니다 (줄이기는 결과를 거의 안 바꾸므로
#: 그냥 Q4_K_M과의 차이가 imatrix의 효과).
#: 초·중·고 결과(2026-10-25): Qwen3.5 줄인 Q4_K_M이 bf16과 같은 점수(4.83, 도구 96%)라 이것만 만듭니다.
#: 원래 어휘의 Q4/Q6/Q8과 줄인 Q6는 그때 비교용이었습니다 (results-school/에 점수가 남아 있음).
SMALL = ["Q4_K_M"]
IMATRIX = {"Q4_K_M"}
STUDENTS = [
    # 위의 초·중·고 데이터 + 초·중·고 학생 말의 도구 예시, 이야기하다가 중간에 도구를 부탁하는
    # 대화, 도구와 비슷한 말이지만 부탁은 아닌 이야기, 인사만 한 말에 도구를 부른 예시와 덤으로
    # 눈 색을 바꾼 예시를 뺀 것 (merge_data.py). 어린이 도구 예시는 두 번씩 넣던 것을 한 번만.
    # 영어: 학생이 영어로 말하면 쉬운 영어로, 한국어로 묻는 영어 공부는 한국어로 설명하며 영어를,
    # "영어로 이야기하자"면 그다음은 영어로 (영어 대화 6천, 영어 도구 부탁 2천, 영어 공부 3천).
    {"name": "qwen3.5-0.8b-school2", "base": QWEN35, "lr": 2e-5, "batch": 2, "accum": 16,
     "data": ("data/train-school2.jsonl", "data-school2"),
     "quants": [], "no_kd": True, "set": "school", "small": SMALL, "english": True},
    # 초·중·고 친구 대화(학년 고르게, 초등 70자·중고 120자, 호기심 질문 15%)를 선생이
    # 생각을 켜고 사실·한국어를 다시 본 데이터 + 거른 도구 예시 (filter_chat.py).
    # 증류는 어린이 데이터에서 v1보다 나빠서(대화 4.64, 도구 92%) 하지 않습니다.
    # kv8: 줄인 Q4_K_M을 KV 캐시 q8_0으로도 채점 (-c 2048에서 KV 캐시 224→119MiB)
    {"name": "qwen3-0.6b-school", "base": BASE, "lr": 2e-5, "batch": 4, "accum": 8,
     "data": ("data/train-school.jsonl", "data-school-clean"),
     "quants": [], "no_kd": True, "set": "school", "small": SMALL,
     "kv8": True},
    # Qwen3.5-0.8B (2026-03, Apache-2.0): 한국어가 낫고(글자당 2.73비트, Qwen3-0.6B 3.27)
    # 한국어 토큰이 23% 적고 KV 캐시가 작음. 어휘가 커서(24만) 배치는 2×16. 선생(Qwen3)과
    # 토크나이저가 달라 증류는 못 합니다. Q4_K_M은 505MiB로 KV 캐시까지 치면 가장 작음.
    {"name": "qwen3.5-0.8b-school", "base": QWEN35, "lr": 2e-5, "batch": 2, "accum": 16,
     "data": ("data/train-school.jsonl", "data-school-clean"),
     "quants": [], "no_kd": True, "set": "school", "small": SMALL},
    # 처음 만든 어린이 데이터 (원래 데이터, 도구 예시를 거른 데이터)
    {"name": "qwen3-0.6b", "base": BASE, "lr": 2e-5, "batch": 4, "accum": 8,
     "data": ("data/train.jsonl", "data-train"), "quants": QUANTS},
    {"name": "qwen3-0.6b-v2", "base": BASE, "lr": 2e-5, "batch": 4, "accum": 8,
     "data": ("data/train-clean.jsonl", "data-clean"), "quants": ["Q6_K", "Q8_0"]},
]
KD = "data/kd"
#: 채점용 질문 묶음마다 (그 질문을 만드는 단계, 답을 두는 곳, 채점 단계)
SETS = {"kids": ("data-heldout", "results", "judge"),
        "school": ("data-school-heldout", "results-school", "judge-school"),
        # 영어 대화·영어 도구 부탁·한국어로 묻는 영어 공부 (채점용 질문과 도구 표시는 검사 단계가 씀)
        "english": ("data-english-check", "results-english", "judge-english")}
#: 초·중·고 도구 질문 (evaluate.py --set tools): 그 질문을 거르는 단계, 답을 두는 곳.
#: 선생이 채점하지 않고 도구 호출만 셉니다
TOOL_SET = ("data-school-tools-check", "results-tools")
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
         # 초·중·고: 대화만 새로 6만 개 (주제 약 200 × 학년 12 × 장면 22 × 말 꺼내는 방식 14 ×
         # 흐름), 채점용 대화 질문 400개(어린이 채점용과 같은 수)도
         ("data-school", ["make_data.py", "--audience", "school", "--out", "data/school.jsonl",
                          "--chat", "60000", "--tool", "0", "--seed", "1"], []),
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
                                    "--tools-repeat", "2",   # 대화 6만 개에 도구 약 1만 6천
                                    "--out", "data/train-school.jsonl"],
              ["data-school", "data-clean"]))
    # 초·중·고 도구: 학생 말의 도구 예시 8천, 대화 중 도구 8천, 비슷한 말 3천 → 도구 예시는
    # 선생이 다시 보고(filter_tools), 파이보의 말은 사실·존댓말 검사(filter_chat) → 합치기
    s += [("data-school-tools", ["make_data.py", "--audience", "school", "--out",
                                 "data/school-tools.jsonl", "--chat", "0", "--tool", "8000",
                                 "--mixed", "8000", "--near", "3000", "--seed", "2"], []),
          ("data-school-tools-heldout", ["make_data.py", "--audience", "school", "--out",
                                         "data/heldout-school-tools.jsonl", "--chat", "0",
                                         "--tool", "300", "--mixed", "300", "--near", "200",
                                         "--seed", "997", "--avoid", "data/school-tools.jsonl"],
           ["data-school-tools"]),
          ("data-school-tools-check", ["filter_tools.py", "--train", "data/school-tools.jsonl",
                                       "--heldout", "data/heldout-school-tools.jsonl",
                                       "--out", "data/school-tools-checked.jsonl",
                                       "--marks", "data/heldout-school-tools-marks.json",
                                       "--log", "data/tool-check-school.jsonl"],
           ["data-school-tools", "data-school-tools-heldout"]),
          ("data-school-tools-clean", ["filter_chat.py", "--data", "data/school-tools-checked.jsonl",
                                       "--out", "data/school-tools-clean.jsonl",
                                       "--log", "data/chat-check-school-tools.jsonl"],
           ["data-school-tools-check"]),
          ("data-english", ["make_data.py", "--audience", "school", "--out", "data/english.jsonl",
                            "--chat", "0", "--tool", "0", "--english", "6000",
                            "--english-tool", "2000", "--learn", "3000", "--seed", "3"], []),
          ("data-english-heldout", ["make_data.py", "--audience", "school", "--out",
                                    "data/heldout-english.jsonl", "--chat", "0", "--tool", "0",
                                    "--english", "300", "--english-tool", "150", "--learn", "150",
                                    "--seed", "996", "--avoid", "data/english.jsonl"],
           ["data-english"]),
          ("data-english-check", ["filter_tools.py", "--train", "data/english.jsonl",
                                  "--heldout", "data/heldout-english.jsonl",
                                  "--out", "data/english-checked.jsonl",
                                  "--marks", "data/heldout-english-marks.json",
                                  "--log", "data/tool-check-english.jsonl"],
           ["data-english", "data-english-heldout"]),
          ("data-english-clean", ["filter_chat.py", "--data", "data/english-checked.jsonl",
                                  "--out", "data/english-clean.jsonl",
                                  "--log", "data/chat-check-english.jsonl"],
           ["data-english-check"]),
          ("data-school2", ["merge_data.py", "--out", "data/train-school2.jsonl",
                            "data/train-school.jsonl", "data/school-tools-clean.jsonl",
                            "data/english-clean.jsonl"],
           ["data-school-clean", "data-school-tools-clean", "data-english-clean"])]
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
        if st.get("small"):
            data = st["data"][0]
            s.append((f"small-{name}", ["prune_vocab.py", "--model", f"runs/{name}/final",
                                        "--out", f"runs/{name}/small", "--data", data,
                                        *(["data/school.jsonl"] if st.get("set") == "school" else []),
                                        "--check", *(["data/heldout-school.jsonl"]
                                                     if st.get("set") == "school" else
                                                     ["data/heldout.jsonl"]),
                                        # 영어를 배운 모델은 영어 글도 그대로 나뉘는지
                                        *(["data/heldout-english.jsonl"]
                                          if st.get("english") else [])],
                      [f"train-{name}", *wait]))
        for quant in st.get("small", []):
            imat = ["--imatrix", st["data"][0]] if quant in IMATRIX else []
            gguf = f"out/pibo-{name}-small-{quant}.gguf"
            s.append((f"gguf-{name}-small-{quant}", ["export.py", "--model", f"runs/{name}/small",
                                                     "--name", f"pibo-{name}-small",
                                                     "--quant", quant, *imat], [f"small-{name}"]))
            s.append((f"eval-{name}-small-{quant}", ["evaluate.py", "gguf", "--gguf", gguf,
                                                     "--name", f"{name}-small-{quant}", *extra],
                      [f"gguf-{name}-small-{quant}", *wait]))
            if st.get("set") == "school":   # 초·중·고 도구 질문, 영어 질문으로도
                s.append((f"eval-tools-{name}-small-{quant}", [
                    "evaluate.py", "gguf", "--gguf", gguf, "--name", f"{name}-small-{quant}",
                    "--set", "tools"], [f"gguf-{name}-small-{quant}", TOOL_SET[0]]))
                s.append((f"eval-english-{name}-small-{quant}", [
                    "evaluate.py", "gguf", "--gguf", gguf, "--name", f"{name}-small-{quant}",
                    "--set", "english"], [f"gguf-{name}-small-{quant}", SETS["english"][0]]))
            if st.get("kv8") and quant == "Q4_K_M":
                s.append((f"eval-{name}-small-{quant}-kv8", [
                    "evaluate.py", "gguf", "--gguf", gguf, "--name", f"{name}-small-{quant}-kv8",
                    "--kv", "q8_0", *extra], [f"gguf-{name}-small-{quant}", *wait]))
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
    titles = {"school": "초·중·고 채점용 질문 (선생 모델이 대화 채점)",
              "tools": "초·중·고 도구 질문 (도구 호출만 셈)",
              "english": "영어 질문 (선생 모델이 대화 채점)",
              "kids": "어린이 채점용 질문 (처음 데이터, 선생 모델이 대화 채점)"}
    where = {**{k: v[1] for k, v in SETS.items()}, "tools": TOOL_SET[1]}
    for which in ("school", "tools", "english", "kids"):
        if not any((HERE / where[which]).glob("answers-*.json")):
            continue
        sys.path.insert(0, str(HERE))
        os.chdir(HERE)
        import evaluate

        evaluate.use_set(which)
        try:
            lines += ["", f"## 채점: {titles[which]}", "", evaluate.report()]
        finally:
            evaluate.use_set("kids")
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
    p.add_argument("--first", metavar="REGEX",
                   help="이름이 맞는 단계를 먼저 (재부팅 등으로 끊긴 학습을 새 단계보다 먼저 끝낼 때)")
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
    def ordered():
        s = steps()
        if not args.first:
            return s
        pick = re.compile(args.first)
        return [x for x in s if pick.search(x[0])] + [x for x in s if not pick.search(x[0])]

    def left() -> list[str]:
        """아직 끝나지 않은 단계 (채점 안 한 답이 남은 채점 단계도)."""
        return [n for n, _, _ in steps() if not (state / f"{n}.done").exists()
                or (n in JUDGES and unscored(JUDGES[n]))]

    for rnd in range(2):
        for name, cmd, needs in ordered():
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
        if not left():
            break
        # 실패한 단계가 있으면 30분 뒤, 순서 때문에 못 한 단계(--first로 앞당긴 단계가 기다린
        # 단계, 나중에 생긴 답의 채점)만 남았으면 바로 한 번 더
        if rnd == 0:
            failed_any = any((state / f"{n}.failed").exists() for n in left())
            print("남은 단계를 " + ("30분 뒤 " if failed_any else "") + "한 번 더", flush=True)
            if failed_any:
                time.sleep(1800)
    report()
    print((HERE / "REPORT.md").read_text())


if __name__ == "__main__":
    main()
