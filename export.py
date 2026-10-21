"""학습한 모델을 GGUF Q4로 바꾸고, llama-server로 대화와 도구 호출이 되는지 봅니다.

    python export.py --model runs/qwen3-0.6b/final --name pibo-qwen3-0.6b

결과: out/<name>-Q4_K_M.gguf (파이보로 복사할 파일)과 out/<name>-check.json.
llama.cpp는 ~/llama.cpp 에 빌드돼 있어야 합니다 (setup 단계가 해 둡니다).

바꾸기 전에 대화 템플릿을 '늘 생각 없이'로 고칩니다. 학습은 늘 빈 생각 칸
(<think></think>) 뒤의 대답만 했는데, llama-server는 따로 말하지 않으면 생각하는
쪽으로 템플릿을 펼쳐서 대답 전에 영어로 한참 생각합니다. 템플릿에 박아 두면
파이보 앱이나 서버 설정을 따로 고치지 않아도 됩니다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from tools import SYSTEM, TOOLS

LLAMA = Path(os.environ.get("LLAMA_CPP", "~/llama.cpp")).expanduser()
OUT = Path("out")

#: Qwen3 템플릿에서 생각을 끌지 말지 고르는 곳: 고르지 말고 늘 빈 생각 칸을 붙이게
THINK_IF = "enable_thinking is defined and enable_thinking is false"
NO_THINK = re.compile(r"\{%-?\s*if " + THINK_IF + r"\s*-?%\}\s*"
                      r"(\{\{-?\s*'<think>\\n\\n</think>\\n\\n'\s*-?\}\})\s*"
                      r"\{%-?\s*endif\s*-?%\}")


def force_no_think(template: str) -> str:
    if THINK_IF not in template:
        if "<think>\\n\\n</think>\\n\\n" in template:
            return template                  # 이미 고쳤음
        raise SystemExit("대화 템플릿이 Qwen3와 달라 생각 끄기를 넣지 못했어요")
    fixed, n = NO_THINK.subn(lambda m: m.group(1), template)
    if n != 1:
        raise SystemExit(f"대화 템플릿에서 생각 고르는 곳을 {n}개 찾았어요 (1개여야 함)")
    return fixed


def fix_template(model: Path) -> None:
    """chat_template.jinja(transformers 5.x)와 tokenizer_config.json 둘 다 고칩니다."""
    jinja = model / "chat_template.jinja"
    if jinja.exists():
        jinja.write_text(force_no_think(jinja.read_text()))
    config = model / "tokenizer_config.json"
    data = json.loads(config.read_text())
    if isinstance(data.get("chat_template"), str):
        data["chat_template"] = force_no_think(data["chat_template"])
        config.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    elif not jinja.exists():
        raise SystemExit(f"{model}에 대화 템플릿이 없어요")


def gguf_template(path: Path) -> str:
    from gguf import GGUFReader

    field = GGUFReader(path).fields.get("tokenizer.chat_template")
    return bytes(field.parts[field.data[0]]).decode() if field else ""


def post(url, body, timeout=300):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def check(gguf: Path) -> dict:
    """CPU의 llama-server로 그냥 대화 한 번, 도구를 불러야 하는 말 두 번.
    파이보 앱처럼 생각 끄기를 따로 보내지 않고 묻습니다."""
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
                "tools": TOOLS, "temperature": 0, "max_tokens": 150})
            msg = r["choices"][0]["message"]
            out[key] = {"말": text, "대답": msg.get("content"),
                        "도구": [c["function"] for c in msg.get("tool_calls") or []]}
            if msg.get("reasoning_content"):
                out[key]["생각"] = msg["reasoning_content"][:200]
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
    if q.exists() and THINK_IF in gguf_template(q):
        print(f"{q}: 생각 끄기 전 템플릿이라 다시 만듭니다", flush=True)
        q.unlink()
    if not q.exists():
        fix_template(Path(args.model))
        subprocess.run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), args.model,
                        "--outfile", str(f16), "--outtype", "f16"], check=True)
        subprocess.run([str(LLAMA / "build" / "bin" / "llama-quantize"), str(f16), str(q),
                        args.quant], check=True)
        f16.unlink()
    if THINK_IF in gguf_template(q):
        raise SystemExit(f"{q}: 템플릿에 생각 끄기가 들어가지 않았어요")
    result = {"file": str(q), "MB": round(q.stat().st_size / 2**20), "check": check(q)}
    (OUT / f"{args.name}-check.json").write_text(json.dumps(result, ensure_ascii=False,
                                                            indent=1))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    calls = [c["name"] for k in ("춤", "눈") for c in result["check"][k]["도구"]]
    if calls != ["set_motion", "eye_on"]:
        print("주의: 도구 호출이 기대와 다릅니다 (위 결과를 보세요)", flush=True)
    if any("생각" in v for v in result["check"].values()):
        print("주의: 대답 전에 생각을 합니다 (파이보에서 아주 느려집니다)", flush=True)


if __name__ == "__main__":
    main()
