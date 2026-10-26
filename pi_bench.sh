#!/bin/bash
# 파이보에서: GGUF 모델마다 llama-server를 실제로 쓸 설정으로 띄워 메모리와 속도를 잽니다.
#   bash pi_bench.sh /home/pi/.model/llm/llm-model.gguf themaker-0.8b-Q4_K_M.gguf
#
# 모델마다 서버를 새로 띄워서 잽니다.
#   최대메모리  질문을 다 마친 뒤 llama-server가 가장 많이 쓴 램 (VmHWM, KV 캐시 포함)
#   첫 대화     서버를 띄우고 첫 질문에 대답이 끝날 때까지 걸린 초 (시스템 글 처음 읽기 포함)
#   다음 대화   시스템 글을 서버가 기억한 상태에서 다음 질문에 걸린 초
#   읽기/말하기 토큰/초 (llama-server가 알려 주는 값)
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
from tools import SYSTEM  # noqa: E402

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


def ask(text):
    body = {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": text}],
            "temperature": 0, "max_tokens": 64}
    t = time.time()
    r = call("/v1/chat/completions", body)
    sec = time.time() - t
    msg = r["choices"][0]["message"]
    said = (msg.get("content") or "").strip()
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
        first = ask("학교에서 친구랑 싸웠어")
        again = ask("날씨가 춥네")
        rows.append((name, peak_mb(srv.pid), first[0], again[0],
                     first[1].get("prompt_per_second", 0), again[1].get("predicted_per_second", 0)))
        said.append((name, first[2], again[2]))
    finally:
        srv.terminate()
        srv.wait(timeout=60)


print(f"\n{'모델':<40} {'최대메모리':>8} {'첫 대화':>7} {'다음 대화':>8} {'읽기':>7} {'말하기':>7}")
for name, mb, a, b, pp, tg in rows:
    print(f"{name:<40} {mb:>6} MB {a:>6.1f}초 {b:>7.1f}초 {pp:>7.1f} {tg:>7.1f}")
print("(읽기·말하기는 토큰/초)")
for name, a, b in said:
    print(f"\n{name}\n  친구랑 싸웠어: {a}\n  날씨가 춥네: {b}")
EOF
