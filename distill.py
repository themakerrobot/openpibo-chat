"""증류 준비: 선생 모델이 학습 데이터의 파이보 대답을 토큰마다 어떻게 보는지(상위 K개
토큰과 그 로그확률)를 미리 구해 둡니다 (vLLM, GPU 하나).

    python distill.py --data data/train.jsonl --base Qwen/Qwen3-0.6B --out data/kd
    python train.py ... --kd data/kd

학습 차례는 train.py와 똑같이 만들고(학생 토크나이저), 그 토큰 그대로 선생 모델에
읽혀서 맞힐 자리마다 상위 K개를 남깁니다. 그래서 선생과 학생의 토크나이저가 같아야
합니다 (Qwen3는 0.6B부터 32B까지 같은 토크나이저). 선생 모델은 데이터를 만든 것
(data/teacher.txt)을 씁니다. 끊겼다가 다시 돌리면 끝난 묶음은 건너뜁니다.

결과: data/kd/part-*.npz (keys: 차례 이름, counts: 차례마다 맞힐 토큰 수,
ids: [토큰, K] int32, lp: [토큰, K] float16)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from make_data import CHOSEN, TEACHER
from train import build_rows, kd_key


def top_k(prompt_logprobs, labels, k: int):
    """vLLM의 prompt_logprobs(자리마다 {토큰: Logprob})에서 맞힐 자리의 상위 k개.
    선생 값이 없는 자리(맨 앞 등)는 토큰 -1."""
    pos = [j for j, x in enumerate(labels) if x != -100]
    ids = np.full((len(pos), k), -1, dtype=np.int32)
    lp = np.full((len(pos), k), -1e4, dtype=np.float16)
    for r, j in enumerate(pos):
        entry = prompt_logprobs[j] if j > 0 else None
        if not entry:
            continue
        best = sorted(((t, v.logprob) for t, v in entry.items()), key=lambda x: -x[1])[:k]
        ids[r, :len(best)] = [t for t, _ in best]
        lp[r, :len(best)] = [v for _, v in best]
    return ids, lp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--base", required=True, help="학생 모델 (토크나이저와 학습 차례를 맞추려고)")
    p.add_argument("--out", required=True)
    p.add_argument("--teacher", help="없으면 데이터를 만든 선생 모델")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--max-len", type=int, default=1536, help="train.py와 같게")
    p.add_argument("--batch", type=int, default=256, help="한 번에 선생에게 주는 차례 수")
    p.add_argument("--memory", type=float, default=0.85)
    p.add_argument("--chunk", type=int, help="한 걸음에 읽는 토큰 수 (메모리가 모자라면 줄이기)")
    p.add_argument("--limit", type=int, help="시험용: 앞의 몇 차례만")
    args = p.parse_args()

    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    teacher = args.teacher or (CHOSEN.read_text().strip() if CHOSEN.exists() else TEACHER)
    tok = AutoTokenizer.from_pretrained(args.base)
    if AutoTokenizer.from_pretrained(teacher).get_vocab() != tok.get_vocab():
        raise SystemExit(f"선생 {teacher}와 학생 {args.base}의 토크나이저가 달라 증류할 수 없어요")

    todo, seen = [], set()
    for part in out.glob("part-*.npz"):
        seen |= set(np.load(part)["keys"].tolist())
    have = len(seen)
    for r in build_rows(tok, args.data, args.max_len):   # 직접 쓴 대화의 세 번 반복은 한 번만
        key = kd_key(r["input_ids"])
        if key not in seen:
            seen.add(key)
            todo.append((key, r))
    if args.limit:
        todo = todo[:args.limit]
    print(f"선생 {teacher}: 이미 {have}차례, 이번에 {len(todo)}차례", flush=True)
    if not todo:
        return

    from vllm import LLM, SamplingParams

    # 맞힐 자리만 고를 수는 없어 모든 자리의 상위 K개를 받습니다 (vLLM이 1024자리씩 나눠
    # 계산). flat_logprobs: 자리마다 dict를 만들지 않아 파이썬 쪽 부담이 적음
    llm = LLM(teacher, max_model_len=args.max_len + 64, gpu_memory_utilization=args.memory,
              enable_prefix_caching=False, max_logprobs=args.k,
              **({"max_num_batched_tokens": args.chunk} if args.chunk else {}))
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=args.k,
                            detokenize=False, flat_logprobs=True)
    n_part = len(list(out.glob("part-*.npz")))
    start = time.time()
    for b in range(0, len(todo), args.batch):
        chunk = todo[b:b + args.batch]
        outs = llm.generate([{"prompt_token_ids": r["input_ids"]} for _, r in chunk], params,
                            use_tqdm=False)
        keys, counts, ids, lps = [], [], [], []
        for (key, r), o in zip(chunk, outs, strict=True):
            i, lp = top_k(o.prompt_logprobs, r["labels"], args.k)
            keys.append(key)
            counts.append(len(i))
            ids.append(i)
            lps.append(lp)
        part = out / f"part-{n_part:05d}.npz"
        tmp = part.with_name(part.stem + ".tmp.npz")
        np.savez(tmp, keys=np.array(keys), counts=np.array(counts, dtype=np.int32),
                 ids=np.concatenate(ids), lp=np.concatenate(lps))
        tmp.rename(part)
        n_part += 1
        done = b + len(chunk)
        left = (time.time() - start) / done * (len(todo) - done)
        print(f"{done}/{len(todo)} 차례, 남은 시간 약 {left / 60:.0f}분", flush=True)
    (out / "meta.json").write_text(json.dumps(
        {"teacher": teacher, "student": args.base, "k": args.k, "max_len": args.max_len},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
