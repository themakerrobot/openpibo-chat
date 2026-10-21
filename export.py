"""학습한 모델을 GGUF로 바꾸고, llama-server로 대화와 도구 호출이 되는지 봅니다.

    python export.py --model runs/qwen3-0.6b/final --name pibo-qwen3-0.6b
    python export.py --model runs/qwen3-0.6b/final --name pibo-qwen3-0.6b --quant Q8_0

결과: out/<name>-<quant>.gguf (파이보로 복사할 파일)과 out/<name>-<quant>-check.json.
llama.cpp는 ~/llama.cpp 에 빌드돼 있어야 합니다 (setup 단계가 해 둡니다).

바꾸기 전에 대화 템플릿을 '늘 생각 없이'로 고칩니다. 학습은 늘 빈 생각 칸
(<think></think>) 뒤의 대답만 했는데, llama-server는 따로 말하지 않으면 생각하는
쪽으로 템플릿을 펼쳐서 대답 전에 영어로 한참 생각합니다. 템플릿에 박아 두면
파이보 앱이나 서버 설정을 따로 고치지 않아도 됩니다.
"""

from __future__ import annotations

import argparse
import contextlib
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

#: 템플릿에서 생각을 할지 고르는 곳: 고르지 말고 늘 빈 생각 칸을 붙이게 바꿉니다.
#: Qwen3: "false면 빈 칸", Qwen3.5: "true면 <think>, 아니면 빈 칸"
EMPTY = r"(\{\{-?\s*'<think>\\n\\n</think>\\n\\n'\s*-?\}\})"
THINK_IFS = {
    "enable_thinking is defined and enable_thinking is false": re.compile(
        r"\{%-?\s*if enable_thinking is defined and enable_thinking is false\s*-?%\}\s*"
        + EMPTY + r"\s*\{%-?\s*endif\s*-?%\}"),
    "enable_thinking is defined and enable_thinking is true": re.compile(
        r"\{%-?\s*if enable_thinking is defined and enable_thinking is true\s*-?%\}\s*"
        r"\{\{-?\s*'<think>\\n'\s*-?\}\}\s*\{%-?\s*else\s*-?%\}\s*"
        + EMPTY + r"\s*\{%-?\s*endif\s*-?%\}"),
}


def thinks(template: str) -> bool:
    """템플릿이 아직 생각을 고를 수 있는지."""
    return any(cond in template for cond in THINK_IFS)


def force_no_think(template: str) -> str:
    if not thinks(template):
        if "<think>\\n\\n</think>\\n\\n" in template:
            return template                  # 이미 고쳤음
        raise SystemExit("대화 템플릿이 Qwen3·Qwen3.5와 달라 생각 끄기를 넣지 못했어요")
    for cond, pattern in THINK_IFS.items():
        if cond in template:
            template, n = pattern.subn(lambda m: m.group(1), template)
            if n != 1:
                raise SystemExit(f"대화 템플릿에서 생각 고르는 곳을 {n}개 찾았어요 (1개여야 함)")
    return template


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


def mtp_flags(model: Path) -> list[str]:
    """Qwen3.5 설정에는 다음 토큰 미리 맞히기(MTP) 층이 있다고 적혀 있지만, 학습한 뒤 저장한
    글 모델에는 그 가중치가 없습니다. 새 llama.cpp 변환기는 그 층을 찾다 실패하므로
    --no-mtp를 줍니다 (그 설정이 없는 옛 변환기는 MTP를 다루지 않아 그냥 됩니다)."""
    from safetensors import safe_open

    if not json.loads((model / "config.json").read_text()).get("mtp_num_hidden_layers"):
        return []
    for f in model.glob("*.safetensors"):
        with safe_open(str(f), "pt") as st:
            if any(k.startswith(("mtp.", "model.mtp.")) for k in st.keys()):
                return []
    converter = (LLAMA / "convert_hf_to_gguf.py").read_text()
    return ["--no-mtp"] if "--no-mtp" in converter else []


def gguf_template(path: Path) -> str:
    from gguf import GGUFReader

    field = GGUFReader(path).fields.get("tokenizer.chat_template")
    return bytes(field.parts[field.data[0]]).decode() if field else ""


def post(url, body, timeout=300):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


@contextlib.contextmanager
def server(gguf: Path, port: int):
    """PC의 CPU로 llama-server를 띄웁니다 (파이보처럼 --jinja, 한 자리)."""
    proc = subprocess.Popen([str(LLAMA / "build" / "bin" / "llama-server"), "-m", str(gguf),
                             "--jinja", "-c", "4096", "-np", "1", "--port", str(port),
                             "-t", "8"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            if proc.poll() is not None:
                raise SystemExit(f"llama-server가 꺼졌어요: {gguf}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                break
            except OSError:
                time.sleep(1)
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def check(gguf: Path) -> dict:
    """그냥 대화 한 번, 도구를 불러야 하는 말 두 번.
    파이보 앱처럼 생각 끄기를 따로 보내지 않고 묻습니다."""
    out = {}
    with server(gguf, 8399) as url:
        for key, text in (("대화", "공룡은 왜 없어졌어?"), ("춤", "파이보야 춤춰 줘"),
                          ("눈", "눈 빨갛게 해 봐")):
            r = post(f"{url}/v1/chat/completions", {
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": text}],
                "tools": TOOLS, "temperature": 0, "max_tokens": 150})
            msg = r["choices"][0]["message"]
            out[key] = {"말": text, "대답": msg.get("content"),
                        "도구": [c["function"] for c in msg.get("tool_calls") or []]}
            if msg.get("reasoning_content"):
                out[key]["생각"] = msg["reasoning_content"][:200]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--quant", default="Q4_K_M")
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)
    f16 = OUT / f"{args.name}-f16.gguf"
    q = OUT / f"{args.name}-{args.quant}.gguf"
    if q.exists() and thinks(gguf_template(q)):
        print(f"{q}: 생각 끄기 전 템플릿이라 다시 만듭니다", flush=True)
        q.unlink()
    if not q.exists():
        fix_template(Path(args.model))
        subprocess.run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), args.model,
                        "--outfile", str(f16), "--outtype", "f16",
                        *mtp_flags(Path(args.model))], check=True)
        subprocess.run([str(LLAMA / "build" / "bin" / "llama-quantize"), str(f16), str(q),
                        args.quant], check=True)
        f16.unlink()
    if thinks(gguf_template(q)):
        raise SystemExit(f"{q}: 템플릿에 생각 끄기가 들어가지 않았어요")
    result = {"file": str(q), "MB": round(q.stat().st_size / 2**20), "check": check(q)}
    (OUT / f"{q.stem}-check.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    calls = [c["name"] for k in ("춤", "눈") for c in result["check"][k]["도구"]]
    if calls != ["set_motion", "eye_on"]:
        print("주의: 도구 호출이 기대와 다릅니다 (위 결과를 보세요)", flush=True)
    if any("생각" in v for v in result["check"].values()):
        print("주의: 대답 전에 생각을 합니다 (파이보에서 아주 느려집니다)", flush=True)


if __name__ == "__main__":
    main()
