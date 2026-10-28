"""학습한 모델을 GGUF로 바꾸고, llama-server로 대화가 되는지 봅니다.

    python export.py --model runs/themaker-0.8b/small --name themaker-0.8b --quant Q4_K_M \
        --imatrix data/train-themaker.jsonl
    python export.py --retemplate out/themaker-0.8b-Q4_K_M.gguf   # 템플릿만 다시

결과: out/<name>-<quant>.gguf (기기로 복사할 파일)과 out/<name>-<quant>-check.json.
llama.cpp는 ~/llama.cpp 에 빌드돼 있어야 합니다 (llama-quantize, llama-server, llama-imatrix).

바꾸기 전에 대화 템플릿을 '늘 생각 없이'로 고칩니다. 학습은 늘 빈 생각 칸
(<think></think>) 뒤의 대답만 했는데, llama-server는 따로 말하지 않으면 생각하는
쪽으로 템플릿을 펼쳐서 대답 전에 영어로 한참 생각합니다. 템플릿에 박아 두면
앱이나 서버 설정을 따로 고치지 않아도 됩니다.

시스템 글 없이 물으면(llama-server 웹 화면, 시스템 글을 안 보내는 앱) 학습 때와 같은 모양의
시스템 글(tools.SYSTEM, 모든 기기 공통)을 넣도록 템플릿 앞에 한 줄을 더합니다: 모델은 늘
시스템 글과 함께 배워서, 없으면 엉뚱한 말을 합니다. 앱이 시스템 글을 보내면 그걸 씁니다.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from tools import SYSTEM

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


DEFAULT_MARK = "{#- pibo: 시스템 글이 없으면 학습 때의 시스템 글 #}"


def default_system(template: str, system: str | None = None) -> str:
    """시스템 글로 시작하지 않는 대화에 학습 때의 시스템 글을 앞에 넣는 템플릿."""
    if DEFAULT_MARK in template:
        return template
    system = system or SYSTEM
    return (DEFAULT_MARK + "\n{%- if not messages or messages[0].role != 'system' %}"
            "{%- set messages = [{'role': 'system', 'content': "
            + json.dumps(system, ensure_ascii=False) + "}] + messages %}{%- endif %}\n" + template)


def pibo_template(template: str) -> str:
    """기기용 템플릿: 늘 생각 없이, 시스템 글이 없으면 더메이커로."""
    return default_system(force_no_think(template))


def fix_template(model: Path) -> None:
    """chat_template.jinja(transformers 5.x)와 tokenizer_config.json 둘 다 고칩니다."""
    jinja = model / "chat_template.jinja"
    if jinja.exists():
        jinja.write_text(pibo_template(jinja.read_text()))
    config = model / "tokenizer_config.json"
    data = json.loads(config.read_text())
    if isinstance(data.get("chat_template"), str):
        data["chat_template"] = pibo_template(data["chat_template"])
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


def convert(model: Path, f16: Path) -> None:
    """HF 모델 → f16 GGUF. 어휘를 줄인 모델(prune_vocab.py)은 토크나이저 해시가 달라
    변환기가 토크나이저 종류를 못 알아보므로, 줄이기 전의 종류(pruned.json)를 알려 줍니다."""
    argv = [str(model), "--outfile", str(f16), "--outtype", "f16", *mtp_flags(model)]
    pruned = model / "pruned.json"
    if not pruned.exists():
        subprocess.run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), *argv], check=True)
        return
    pre = json.loads(pruned.read_text())["pre"]
    if not pre:
        raise SystemExit(f"{pruned}: 토크나이저 종류를 몰라요 (prune_vocab.PRE에 추가)")
    code = ("import sys; sys.path.insert(0, sys.argv[1]); pre = sys.argv[2]; "
            "sys.argv = ['convert_hf_to_gguf.py', *sys.argv[3:]]; "
            "import convert_hf_to_gguf as c; from conversion import base; "
            "base.TextModel.get_vocab_base_pre = lambda self, tokenizer: pre; c.main()")
    subprocess.run([sys.executable, "-c", code, str(LLAMA), pre, *argv], check=True)


def calibration(model: Path, data: str, out: Path, n: int = 400) -> Path:
    """imatrix용 글: 학습 데이터 n개를 모델이 실제로 보는 모양(대화 템플릿)으로."""
    import random

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    lines = Path(data).read_text().splitlines()
    picked = random.Random(0).sample(lines, min(n, len(lines)))
    from train import tools_of

    exs = [json.loads(x) for x in picked]
    text = "\n".join(tok.apply_chat_template(ex["messages"], tools=tools_of(ex), tokenize=False,
                                             enable_thinking=False) for ex in exs)
    out.write_text(text)
    return out


def imatrix(f16: Path, model: Path, data: str) -> Path:
    """양자화할 때 중요한 가중치를 덜 깎도록 학습 데이터로 활성값 크기를 재 둡니다
    (llama-imatrix). 작은 모델의 Q4에서 특히 차이가 납니다."""
    bin_ = LLAMA / "build" / "bin" / "llama-imatrix"
    if not bin_.exists():
        raise SystemExit(f"{bin_}이 없어요. 먼저 빌드하세요: "
                         f"cmake --build {LLAMA / 'build'} -j --target llama-imatrix")
    calib = calibration(model, data, f16.with_suffix(".calib.txt"))
    out = f16.with_suffix(".imatrix.gguf")
    special = "--parse-special" in subprocess.run([str(bin_), "--help"], capture_output=True,
                                                  text=True, check=False).stdout
    subprocess.run([str(bin_), "-m", str(f16), "-f", str(calib), "-o", str(out), "-c", "512",
                    "-t", "8", "--no-ppl", *(["--parse-special"] if special else []),
                    *gpu_flags()], check=True)
    calib.unlink()
    return out


def gguf_template(path: Path) -> str:
    from gguf import GGUFReader

    field = GGUFReader(path).fields.get("tokenizer.chat_template")
    return bytes(field.parts[field.data[0]]).decode() if field else ""


def retemplate(path: Path) -> bool:
    """GGUF를 다시 만들지 않고 대화 템플릿만 기기용으로 바꿉니다. 바꿨으면 True."""
    old = gguf_template(path)
    new = pibo_template(old)
    if new == old:
        return False
    jinja = path.with_name(path.name + ".jinja")
    tmp = path.with_name(path.name + ".tmp")
    jinja.write_text(new)
    try:
        subprocess.run([sys.executable, "-m", "gguf.scripts.gguf_new_metadata", str(path),
                        str(tmp), "--chat-template-file", str(jinja), "--force"], check=True)
        os.replace(tmp, path)
    finally:
        jinja.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
    print(f"{path}: 대화 템플릿을 바꿨어요 (생각 끄기, 기본 시스템 글)", flush=True)
    return True


def post(url, body, timeout=300):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


@functools.cache
def gpu_flags() -> list[str]:
    """llama.cpp가 CUDA로 빌드돼 있으면 모델을 GPU에 올립니다: 채점용 질문 700개를 CPU로
    물으면 양자화 하나에 몇 시간이 걸립니다. 계산 순서가 달라 글자가 조금 다를 수는
    있지만 양자화끼리 비교하는 데는 상관없습니다. PIBO_EVAL_CPU=1이면 CPU로."""
    if os.environ.get("PIBO_EVAL_CPU"):
        return []
    try:
        out = subprocess.run([str(LLAMA / "build" / "bin" / "llama-server"), "--list-devices"],
                             capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    return ["-ngl", "99"] if "CUDA" in out.stdout + out.stderr else []


def kv_flags(kind: str | None) -> list[str]:
    """KV 캐시를 kind(q8_0 등)로: 파이보에서 KV 캐시 메모리가 반으로 줍니다. 양자화한 V
    캐시는 flash attention이 있어야 해서 함께 켭니다 (옛 llama.cpp는 -fa만, 새것은 -fa on)."""
    if not kind:
        return []
    help_ = subprocess.run([str(LLAMA / "build" / "bin" / "llama-server"), "--help"],
                           capture_output=True, text=True, check=False).stdout
    fa = ["-fa", "on"] if "[on|off|auto]" in help_ else ["-fa"]
    return ["-ctk", kind, "-ctv", kind, *fa]


@contextlib.contextmanager
def server(gguf: Path, port: int, kv: str | None = None):
    """PC에서 llama-server를 띄웁니다 (파이보처럼 --jinja, 한 자리)."""
    proc = subprocess.Popen([str(LLAMA / "build" / "bin" / "llama-server"), "-m", str(gguf),
                             "--jinja", "-c", "4096", "-np", "1", "--port", str(port),
                             "-t", "8", *gpu_flags(), *kv_flags(kv)],
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


#: 만든 파일을 바로 확인하는 질문: (이름, 시스템 글 (None이면 보내지 않음), 말)
#: (이름, 시스템 글, 학생 말, 도구 목록을 보낼지)
CHECKS = [("대화", SYSTEM, "학교에서 친구랑 싸웠어", False),
          ("이어 묻기", SYSTEM, "날씨가 춥네", False),
          ("시스템 글 없음", None, "너는 누구야?", False),
          ("자기소개", SYSTEM, "너 이름이 뭐야?", False),
          ("기기 사양", SYSTEM, "너 키가 몇이야?", False),
          ("도구 있을 때 대화", SYSTEM, "오늘 학교에서 발표했어", True),
          ("불빛", SYSTEM, "빨간 불 켜 줘", True),
          ("움직이기", SYSTEM, "춤 좀 춰 봐", True),
          ("보기", SYSTEM, "앞에 뭐 있어?", True),
          ("부르면 안 됨", SYSTEM, "난 빨간색이 좋아", True)]


def check(gguf: Path) -> dict:
    """대화 몇 개를 llama-server로 물어봅니다 (생각 끄기를 따로 보내지 않고, 앱처럼)."""
    from tools import TOOLS, calls_of

    out = {}
    with server(gguf, 8399) as url:
        for key, system, text, tools in CHECKS:
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                [{"role": "user", "content": text}]
            body = {"messages": msgs, "temperature": 0, "max_tokens": 150}
            if tools:
                body["tools"] = TOOLS
            msg = post(f"{url}/v1/chat/completions", body)["choices"][0]["message"]
            out[key] = {"말": text, "대답": msg.get("content")}
            if calls_of(msg):
                out[key]["도구"] = [f"{n}({json.dumps(a, ensure_ascii=False)})" for n, a in calls_of(msg)]
            if msg.get("reasoning_content"):
                out[key]["생각"] = msg["reasoning_content"][:200]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--name")
    p.add_argument("--quant", default="Q4_K_M")
    p.add_argument("--imatrix", metavar="DATA",
                   help="이 학습 데이터로 imatrix를 재서 양자화 (llama-imatrix)")
    p.add_argument("--retemplate", metavar="GGUF", nargs="+",
                   help="이미 만든 GGUF의 대화 템플릿만 기기용으로 (다시 변환하지 않음)")
    args = p.parse_args()
    if args.retemplate:
        for path in args.retemplate:
            if not retemplate(Path(path)):
                print(f"{path}: 이미 기기용 템플릿이에요", flush=True)
        return
    if not (args.model and args.name):
        p.error("--model과 --name이 필요해요")
    OUT.mkdir(exist_ok=True)
    f16 = OUT / f"{args.name}-f16.gguf"
    q = OUT / f"{args.name}-{args.quant}.gguf"
    if q.exists():
        retemplate(q)                        # 예전에 만든 파일: 템플릿만 바꿈
    if not q.exists():
        fix_template(Path(args.model))
        convert(Path(args.model), f16)
        imat = imatrix(f16, Path(args.model), args.imatrix) if args.imatrix else None
        subprocess.run([str(LLAMA / "build" / "bin" / "llama-quantize"),
                        *(["--imatrix", str(imat)] if imat else []), str(f16), str(q),
                        args.quant], check=True)
        f16.unlink()
        if imat:
            imat.unlink()
    if thinks(gguf_template(q)) or DEFAULT_MARK not in gguf_template(q):
        raise SystemExit(f"{q}: 템플릿에 생각 끄기나 기본 시스템 글이 들어가지 않았어요")
    result = {"file": str(q), "MB": round(q.stat().st_size / 2**20), "check": check(q)}
    (OUT / f"{q.stem}-check.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if any("생각" in v for v in result["check"].values()):
        print("주의: 대답 전에 생각을 합니다 (파이보에서 아주 느려집니다)", flush=True)


if __name__ == "__main__":
    main()
