"""학습한 모델을 GGUF Q4로 바꾸고, llama-server로 대화와 도구 호출이 되는지 봅니다.

    python export.py --model runs/qwen3-0.6b/final --name pibo-qwen3-0.6b

결과: out/<name>-Q4_K_M.gguf (파이보로 복사할 파일)과 out/<name>-check.json.
llama.cpp는 ~/llama.cpp 에 빌드돼 있어야 합니다 (setup 단계가 해 둡니다).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from tools import SYSTEM, TOOLS

LLAMA = Path(os.environ.get("LLAMA_CPP", "~/llama.cpp")).expanduser()
OUT = Path("out")


def post(url, body, timeout=300):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def check(gguf: Path) -> dict:
    """CPU의 llama-server로 그냥 대화 한 번, 도구를 불러야 하는 말 두 번."""
    port = 8399
    server = subprocess.Popen([str(LLAMA / "build" / "bin" / "llama-server"), "-m", str(gguf),
                               "--jinja", "-c", "4096", "--port", str(port), "-t", "8"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                break
            except OSError:
                time.sleep(1)
        out = {}
        for key, text in (("대화", "공룡은 왜 없어졌어?"), ("춤", "파이보야 춤춰 줘"),
                          ("눈", "눈 빨갛게 해 봐")):
            r = post(f"http://127.0.0.1:{port}/v1/chat/completions", {
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": text}],
                "tools": TOOLS, "temperature": 0})
            msg = r["choices"][0]["message"]
            out[key] = {"말": text, "대답": msg.get("content"),
                        "도구": [c["function"] for c in msg.get("tool_calls") or []]}
        return out
    finally:
        server.terminate()
        server.wait(timeout=30)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--quant", default="Q4_K_M")
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)
    f16 = OUT / f"{args.name}-f16.gguf"
    q = OUT / f"{args.name}-{args.quant}.gguf"
    if not q.exists():
        subprocess.run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), args.model,
                        "--outfile", str(f16), "--outtype", "f16"], check=True)
        subprocess.run([str(LLAMA / "build" / "bin" / "llama-quantize"), str(f16), str(q),
                        args.quant], check=True)
        f16.unlink()
    result = {"file": str(q), "MB": round(q.stat().st_size / 2**20), "check": check(q)}
    (OUT / f"{args.name}-check.json").write_text(json.dumps(result, ensure_ascii=False,
                                                            indent=1))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    calls = [c["name"] for k in ("춤", "눈") for c in result["check"][k]["도구"]]
    if calls != ["set_motion", "eye_on"]:
        print("주의: 도구 호출이 기대와 다릅니다 (위 결과를 보세요)", flush=True)


if __name__ == "__main__":
    main()
