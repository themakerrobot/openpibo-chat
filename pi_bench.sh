#!/bin/bash
# 파이보에서: GGUF 모델마다 llama-server를 실제로 쓸 설정으로 띄워 메모리와 속도를 잽니다.
#   bash pi_bench.sh ~/models/gemma-3-1b-it-Q4_K_M.gguf pibo-qwen3-0.6b-Q4_K_M.gguf
#
# 모델마다 서버를 새로 띄워서 잽니다.
#   최대메모리  질문을 다 마친 뒤 llama-server가 가장 많이 쓴 램 (VmHWM, KV 캐시 포함)
#   대화        도구 없이 짧은 질문 하나에 대답이 끝날 때까지 걸린 초
#   도구첫턴    도구 목록(약 800토큰)을 처음 읽고 도구를 부르기까지 걸린 초
#   도구다음    도구 목록을 서버가 기억한 상태에서 다음 말에 걸린 초
#   읽기/말하기 토큰/초 (llama-server가 알려 주는 값)
# 도구 형식이 없는 모델(젬마3)은 도구 칸이 "-"입니다.
#
# llama-server는 PATH, ~/llama.cpp/build/bin 순서로 찾습니다. 다른 곳이면 LLAMA_BIN=/경로.
# 서버 설정은 SERVER_ARGS로 바꿉니다. 예: SERVER_ARGS="--jinja -c 2048 -t 4 -np 1 -ctk q8_0 -ctv q8_0 -fa on"
# 이미 돌고 있는 llama-server는 먼저 멈춰 주세요: 램 2GB에 둘을 올리면 측정이 틀어집니다.
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -z "$LLAMA_BIN" ]; then
  if command -v llama-server >/dev/null; then LLAMA_BIN="$(dirname "$(command -v llama-server)")"
  else LLAMA_BIN="$HOME/llama.cpp/build/bin"; fi
fi
if [ ! -x "$LLAMA_BIN/llama-server" ]; then
  echo "llama-server가 없어요: $LLAMA_BIN (LLAMA_BIN=/경로 로 알려 주세요)"; exit 1
fi
if pgrep -x llama-server >/dev/null; then
  echo "돌고 있는 llama-server가 있어요 (pid $(pgrep -x llama-server | tr '\n' ' ')). 멈춘 뒤 다시 실행하세요"
  exit 1
fi
SERVER_ARGS="${SERVER_ARGS:---jinja -c 2048 -t 4 -np 1}"
# 새 llama.cpp는 지난 질문을 램에 8GB까지 쌓아 두므로 끕니다 (옛 버전에는 이 설정이 없음)
if "$LLAMA_BIN/llama-server" --help 2>&1 | grep -q -- "--cache-ram"; then
  SERVER_ARGS="$SERVER_ARGS --cache-ram 0"
fi
echo "llama.cpp: $("$LLAMA_BIN/llama-server" --version 2>&1 | grep -m1 version)"
echo "서버 설정: $SERVER_ARGS"
echo "램: $(awk '/MemTotal/ {t=$2} /MemAvailable/ {a=$2} END {printf "전체 %d MB, 지금 남은 것 %d MB", t/1024, a/1024}' /proc/meminfo)"
exec python3 - "$LLAMA_BIN/llama-server" "$SERVER_ARGS" "$HERE" "$@" <<'EOF'
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

server_bin, server_args, here, *models = sys.argv[1:]
sys.path.insert(0, here)
from tools import SYSTEM, TOOLS  # noqa: E402

PORT = 8398


def call(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def peak_mb(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) // 1024
    return 0


def ask(text, tools):
    body = {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": text}],
            "temperature": 0, "max_tokens": 64}
    if tools:
        body["tools"] = TOOLS
    t = time.time()
    r = call("/v1/chat/completions", body)
    sec = time.time() - t
    msg = r["choices"][0]["message"]
    said = " ".join(f"[{c['function']['name']} {c['function']['arguments']}]"
                    for c in msg.get("tool_calls") or [])
    said = (said + " " + (msg.get("content") or "")).strip()
    if msg.get("reasoning_content"):
        said = "[생각함: 느려짐] " + said
    return sec, r.get("timings", {}), said


rows, said = [], []
for m in models:
    name = Path(m).name
    log = Path(f"/tmp/pi_bench-{Path(m).stem}.log")
    srv = subprocess.Popen([server_bin, "-m", m, "--port", str(PORT), *server_args.split()],
                           stdout=log.open("w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            if srv.poll() is not None:
                break
            try:
                call("/health")
                break
            except OSError:
                time.sleep(1)
        if srv.poll() is not None:
            print(f"{name}: llama-server가 꺼졌어요. 로그: {log}")
            continue
        tools_ok = "<tool_call>" in (call("/props").get("chat_template") or "")
        chat = ask("공룡은 왜 사라졌어?", False)
        first = ask("파이보야 춤춰 줘", True) if tools_ok else None
        again = ask("눈 파랗게 해 줘", True) if tools_ok else None
        read = (first or chat)[1].get("prompt_per_second", 0)
        rows.append((name, peak_mb(srv.pid), chat[0], first and first[0], again and again[0],
                     read, chat[1].get("predicted_per_second", 0)))
        said.append((name, chat[2], first and first[2], again and again[2]))
    finally:
        srv.terminate()
        srv.wait(timeout=60)


def sec(x):
    return "-" if x is None else f"{x:.1f}"


print(f"\n{'모델':<40} {'최대메모리':>8} {'대화':>6} {'도구첫턴':>8} {'도구다음':>8} "
      f"{'읽기':>7} {'말하기':>7}")
for name, mb, a, b, c, pp, tg in rows:
    print(f"{name:<40} {mb:>6} MB {sec(a):>5}초 {sec(b):>7}초 {sec(c):>7}초 "
          f"{pp:>7.1f} {tg:>7.1f}")
print("(읽기·말하기는 토큰/초)")
for name, a, b, c in said:
    print(f"\n{name}\n  대화: {a}")
    if b is not None:
        print(f"  춤춰 줘: {b}\n  눈 파랗게: {c}")
EOF
