"""선생 모델로 학습 예시를 만듭니다 (vLLM, GPU 하나).

    python make_data.py --out data/train.jsonl --chat 24000 --tool 8000 --seed 0
    python make_data.py --out data/heldout.jsonl --chat 300 --tool 300 --seed 999 \\
        --avoid data/train.jsonl

이미 쓴 줄은 두고 모자란 만큼만 더 만들어 붙입니다 (중간에 끊겨도 이어서).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from prompts import TEACHER_SYSTEM, build_example, requests

TEACHER = "Qwen/Qwen3-32B-AWQ"   # Apache-2.0
#: 앞의 것이 이 GPU에서 뜨지 않으면 다음 것으로 (모두 Apache-2.0)
TEACHERS = [TEACHER, "Qwen/Qwen3-14B-AWQ", "Qwen/Qwen3-8B"]
CHOSEN = Path("data/teacher.txt")   # 실제로 쓴 선생 모델: 채점도 같은 것으로


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
    """선생 모델을 띄웁니다. 정한 게 없으면 목록에서 아직 실패하지 않은 첫 번째.

    뜨지 않으면 그 이름을 data/teacher-failed.txt에 적고 끝냅니다. 같은
    프로세스에서 다음 것을 띄우면 GPU 메모리가 덜 풀려 있을 수 있어, 다음
    후보는 run_all.py가 다시 시도할 때 새 프로세스에서 띄웁니다.
    """
    from vllm import LLM

    failed_file = CHOSEN.with_name("teacher-failed.txt")
    failed = set(failed_file.read_text().split()) if failed_file.exists() else set()
    if name is None and CHOSEN.exists():
        name = CHOSEN.read_text().strip()
    if name is None:
        left = [n for n in TEACHERS if n not in failed]
        if not left:
            raise SystemExit(f"선생 모델 후보가 모두 실패했어요: {TEACHERS}")
        name = left[0]
    free = gpu_free_fraction()
    if free is not None and free < memory:
        # 모델 탓이 아니므로 실패로 적지 않습니다 (적으면 다음에는 더 작은 선생으로 갑니다)
        raise SystemExit(f"GPU 메모리가 {free:.0%}만 비어 있어 선생 모델을 띄우지 않아요 "
                         f"({memory:.0%} 필요): nvidia-smi로 GPU를 쥔 프로세스를 확인하세요")
    try:
        llm = LLM(name, max_model_len=4096, gpu_memory_utilization=memory,
                  enable_prefix_caching=True)
    except Exception as exc:
        CHOSEN.parent.mkdir(parents=True, exist_ok=True)
        with failed_file.open("a") as f:
            f.write(name + "\n")
        raise SystemExit(f"선생 모델 {name}을 띄우지 못함 (다음 시도는 다음 후보로): {exc}") \
            from exc
    CHOSEN.parent.mkdir(parents=True, exist_ok=True)
    CHOSEN.write_text(name)
    print(f"선생 모델: {name}", flush=True)
    return llm


def thinking_params(max_tokens: int):
    """선생이 생각을 켜고 답할 때의 설정 (Qwen3 권장값: 생각할 때 탐욕 디코딩은 되풀이에 빠짐)."""
    from vllm import SamplingParams

    return SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=max_tokens, seed=0)


def first_user(ex: dict) -> str:
    return next(m["content"] for m in ex["messages"] if m["role"] == "user")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--chat", type=int, required=True)
    p.add_argument("--tool", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--teacher", help="정해서 쓸 선생 모델 (없으면 목록에서 뜨는 것)")
    p.add_argument("--avoid", help="이 파일에 있는 아이 말은 빼고 (채점용 데이터)")
    p.add_argument("--batch", type=int, default=4000)
    p.add_argument("--audience", choices=["kids", "school"], default="kids",
                   help="kids: 미취학~초저(처음 데이터), school: 초·중·고 학년 고르게")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rejected = out.with_name(f"rejected-{out.stem}.jsonl")   # 버린 답 몇 개씩 (훑어보기용)
    have = ([json.loads(line) for line in out.read_text().splitlines()]
            if out.exists() else [])
    seen = {first_user(e) for e in have}
    if args.avoid and Path(args.avoid).exists():
        seen |= {first_user(json.loads(line))
                 for line in Path(args.avoid).read_text().splitlines()}
    count = {"chat": sum(e["kind"] == "chat" for e in have),
             "tool": sum(e["kind"] == "tool" for e in have)}
    want = {"chat": args.chat, "tool": args.tool}
    print(f"이미 {count}, 목표 {want}", flush=True)
    if all(count[k] >= want[k] for k in want):
        return

    from vllm import SamplingParams

    llm = load_teacher(args.teacher)
    params = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=1000)  # 중·고 3번 주고받기도 들어가게
    rng = random.Random(args.seed)
    for rnd in range(50):
        need = {k: max(0, want[k] - count[k]) for k in want}
        if not any(need.values()):
            break
        # 버려지는 답이 있으니 모자란 것보다 넉넉히 청합니다
        n_chat = min(args.batch, int(need["chat"] * 1.4) + 8) if need["chat"] else 0
        n_tool = min(args.batch, int(need["tool"] * 1.6) + 8) if need["tool"] else 0
        reqs = requests(n_chat, n_tool, seed=args.seed * 1000 + rnd, audience=args.audience)
        convs = [[{"role": "system", "content": TEACHER_SYSTEM},
                  {"role": "user", "content": r["prompt"]}] for r in reqs]
        outs = llm.chat(convs, params, chat_template_kwargs={"enable_thinking": False},
                        use_tqdm=False)
        kept, why, shown = 0, Counter(), Counter()
        with out.open("a") as f, rejected.open("a") as rej:
            for r, o in zip(reqs, outs, strict=True):
                if count[r["kind"]] >= want[r["kind"]]:
                    continue
                ex, reason = build_example(r, o.outputs[0].text, rng)
                if reason is None and first_user(ex) in seen:
                    reason = "아이 말이 앞의 것과 같음"
                if reason is not None and o.outputs[0].finish_reason == "length":
                    reason = f"{params.max_tokens}토큰에서 잘림"
                if reason is not None:
                    why[reason] += 1
                    key = re.sub(r"'.*'", "", reason)
                    if shown[key] < 20:        # 까닭마다 몇 개씩 남겨 사람이 봅니다
                        shown[key] += 1
                        rej.write(json.dumps({"why": reason, "reply": o.outputs[0].text},
                                             ensure_ascii=False) + "\n")
                    continue
                seen.add(first_user(ex))
                count[ex["kind"]] += 1
                kept += 1
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"라운드 {rnd}: {len(reqs)}개 청해서 {kept}개 남김, 지금 {count}", flush=True)
        print("  버린 까닭: " + ", ".join(f"{k} {v}" for k, v in why.most_common(8)), flush=True)
    if any(count[k] < want[k] for k in want):
        raise SystemExit(f"목표에 못 미침: {count} / {want}")


if __name__ == "__main__":
    main()
