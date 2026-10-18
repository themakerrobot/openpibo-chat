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
from pathlib import Path

from prompts import TEACHER_SYSTEM, requests, to_example

TEACHER = "Qwen/Qwen3-32B-AWQ"   # Apache-2.0


def first_user(ex: dict) -> str:
    return next(m["content"] for m in ex["messages"] if m["role"] == "user")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--chat", type=int, required=True)
    p.add_argument("--tool", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--teacher", default=TEACHER)
    p.add_argument("--avoid", help="이 파일에 있는 아이 말은 빼고 (채점용 데이터)")
    p.add_argument("--batch", type=int, default=4000)
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
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

    from vllm import LLM, SamplingParams

    llm = LLM(args.teacher, max_model_len=4096, gpu_memory_utilization=0.92,
              enable_prefix_caching=True)
    params = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=700)
    rng = random.Random(args.seed)
    for rnd in range(50):
        need = {k: max(0, want[k] - count[k]) for k in want}
        if not any(need.values()):
            break
        # 버려지는 답이 있으니 모자란 것보다 넉넉히 청합니다
        n_chat = min(args.batch, int(need["chat"] * 1.4) + 8) if need["chat"] else 0
        n_tool = min(args.batch, int(need["tool"] * 1.6) + 8) if need["tool"] else 0
        reqs = requests(n_chat, n_tool, seed=args.seed * 1000 + rnd)
        convs = [[{"role": "system", "content": TEACHER_SYSTEM},
                  {"role": "user", "content": r["prompt"]}] for r in reqs]
        outs = llm.chat(convs, params, chat_template_kwargs={"enable_thinking": False},
                        use_tqdm=False)
        kept = 0
        with out.open("a") as f:
            for r, o in zip(reqs, outs, strict=True):
                if count[r["kind"]] >= want[r["kind"]]:
                    continue
                ex = to_example(r, o.outputs[0].text, rng)
                if ex is None or first_user(ex) in seen:
                    continue
                seen.add(first_user(ex))
                count[ex["kind"]] += 1
                kept += 1
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"라운드 {rnd}: {len(reqs)}개 청해서 {kept}개 남김, 지금 {count}", flush=True)
    if any(count[k] < want[k] for k in want):
        raise SystemExit(f"목표에 못 미침: {count} / {want}")


if __name__ == "__main__":
    main()
