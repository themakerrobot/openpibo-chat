"""선생 모델(vLLM, GPU 하나): 파이보 말 바꿔 쓰기·검사(rewrite.py)와 채점(score.py)에 씁니다."""

from __future__ import annotations

import json
import re
from pathlib import Path

TEACHER = "Qwen/Qwen3-32B-AWQ"   # Apache-2.0. 다른 것으로 바꾸지 않습니다 (--teacher로만)
CHOSEN = Path("data/teacher.txt")   # 실제로 쓴 선생 모델 (기록용)


def gpu_free_fraction() -> float | None:
    """GPU 메모리 중 비어 있는 몫 (nvidia-smi가 없으면 None). vLLM보다 먼저 CUDA를 건드리지
    않도록 nvidia-smi로 봅니다."""
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=30, check=False).stdout.splitlines()[0]
        free, total = (float(x) for x in out.split(","))
        return free / total
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def load_teacher(name: str | None = None, memory: float = 0.92):
    """선생 모델(기본 Qwen3-32B-AWQ)을 띄웁니다. 뜨지 않으면 까닭을 보이고 끝냅니다
    (더 작은 모델로 몰래 바꾸지 않음)."""
    import os
    import sys

    # 가상환경을 켜지 않고 .venv/bin/python으로 돌려도 그 안의 ninja를 찾도록
    # (vLLM·flashinfer가 처음 뜰 때 ninja로 GPU 코드를 컴파일함)
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from vllm import LLM

    name = name or TEACHER
    free = gpu_free_fraction()
    if free is not None and free < memory:
        raise SystemExit(f"GPU 메모리가 {free:.0%}만 비어 있어 선생 모델을 띄우지 않아요 "
                         f"({memory:.0%} 필요): nvidia-smi로 GPU를 쥔 프로세스를 확인하세요")
    try:
        # 여섯 번 주고받는 대화 + 지시 + 생각 3천 토큰이 들어가게
        llm = LLM(name, max_model_len=8192, gpu_memory_utilization=memory,
                  enable_prefix_caching=True)
    except Exception as exc:
        raise SystemExit(f"선생 모델 {name}을 띄우지 못함: {exc}") from exc
    CHOSEN.parent.mkdir(parents=True, exist_ok=True)
    CHOSEN.write_text(name)
    print(f"선생 모델: {name}", flush=True)
    return llm


def thinking_params(max_tokens: int):
    """선생이 생각을 켜고 답할 때의 설정 (Qwen3 권장값: 생각할 때 탐욕 디코딩은 되풀이에 빠짐)."""
    from vllm import SamplingParams

    return SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=max_tokens, seed=0)


def plain_params(max_tokens: int):
    """생각 없이 답할 때의 설정 (Qwen3 권장값)."""
    from vllm import SamplingParams

    return SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=max_tokens, seed=0)


def ask(llm, prompts: list[str], max_tokens: int, think: bool = True) -> list[str]:
    """여러 개를 한 번에 묻습니다 (생각 뒤의 글만 돌려줌). think가 거짓이면 생각 없이: 번역처럼
    생각이 필요 없는 일은 답이 수백 토큰 짧아져 몇 배 빠릅니다."""
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts],
                    thinking_params(max_tokens) if think else plain_params(max_tokens),
                    chat_template_kwargs={"enable_thinking": think}, use_tqdm=False)
    return [after_think(o.outputs[0].text) for o in outs]


def first_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            try:
                return json.loads(text[start:i + 1])
            except json.JSONDecodeError:
                return None
    return None


def after_think(text: str) -> str:
    """생각을 켜고 받은 답에서 생각 뒤의 글만. 생각이 길어 잘렸으면 빈 글."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return "" if "<think>" in text else text
