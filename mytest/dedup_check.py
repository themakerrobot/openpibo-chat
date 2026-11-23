"""시험 파일의 학생 말이 학습 데이터와 겹치는지 봅니다 (학습 PC에서, 바깥으로 보내는 것 없음).

    python dedup_check.py --test mytest/mytest-chat.jsonl mytest/mytest-tools.jsonl \
        --train data/*.jsonl data/*.txt

- 학생 말만 견줍니다 (모델이 받는 쪽). 띄어쓰기·문장부호·대소문자는 무시합니다.
- 똑같음: 글자가 그대로 같음. 비슷함: 3글자 덩어리가 --sim 이상 겹침 (6글자 이상인 말만).
- 학습 쪽 문장은 출력하지 않습니다. 시험 쪽 문장과 겹친 학습 파일 이름, 비슷한 정도만 나옵니다.
- 이름에 heldout·mytest가 들어간 파일은 학습 파일로 치지 않습니다.
- 학습 파일이 .txt면 "아이:" / "Kid:" / "학생:" / "Student:"로 시작하는 줄을 학생 말로 읽습니다.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

KID = re.compile(r"^\s*(아이|학생|Kid|Student|User)\s*:\s*(.+)$", re.I)


def norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def grams(text: str) -> set[str]:
    return {text[i:i + 3] for i in range(len(text) - 2)}


def users_of(path: Path) -> list[str]:
    out = []
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            for m in ex.get("messages") or []:
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    out.append(m["content"])
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            m = KID.match(line)
            if m:
                out.append(m.group(2))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", nargs="+", required=True)
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--sim", type=float, default=0.6)
    args = p.parse_args()

    exact: dict[str, set[str]] = defaultdict(set)
    index: dict[str, list[int]] = defaultdict(list)
    pool: list[tuple[set[str], str]] = []
    for f in args.train:
        path = Path(f)
        if not path.is_file() or "heldout" in path.name or "mytest" in path.name:
            continue
        n = 0
        for u in users_of(path):
            t = norm(u)
            if not t:
                continue
            exact[t].add(path.name)
            if len(t) >= 6:
                g = grams(t)
                for x in g:
                    index[x].append(len(pool))
                pool.append((g, path.name))
            n += 1
        print(f"학습 {path.name}: 학생 말 {n}개")

    total = same = near = 0
    for f in args.test:
        seen = set()
        for u in users_of(Path(f)):
            t = norm(u)
            if not t or t in seen:
                continue
            seen.add(t)
            total += 1
            if t in exact:
                same += 1
                print(f"[똑같음] {Path(f).name}: {u}  ← {', '.join(sorted(exact[t]))}")
                continue
            if len(t) < 6:
                continue
            g = grams(t)
            hits = defaultdict(int)
            for x in g:
                for i in index.get(x, ()):
                    hits[i] += 1
            best, where = 0.0, ""
            for i, k in hits.items():
                if k * 2 < len(g):                    # 덩어리 절반도 안 겹치면 볼 필요 없음
                    continue
                pg, name = pool[i]
                j = k / len(g | pg)
                if j > best:
                    best, where = j, name
            if best >= args.sim:
                near += 1
                print(f"[비슷함 {best:.2f}] {Path(f).name}: {u}  ← {where}")
    print(f"\n시험 학생 말 {total}개 중 똑같음 {same}개, 비슷함 {near}개")


if __name__ == "__main__":
    main()
