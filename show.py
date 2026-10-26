"""만든 데이터를 사람이 읽기 좋게 몇 개 보여 줍니다.

    python show.py                                       # 최종 학습 데이터에서 무작위 20개
    python show.py 200 all data/try.jsonl > ~/try.txt    # 해 본 결과를 파일로
    python show.py 30 all data/source.jsonl              # 원본도 됨

두 번째 값: 원본 이름(source) 하나만 보거나 all
"""

import json
import random
import sys
from pathlib import Path

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
kind = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "all" else None
path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data/train-themaker.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
rows = [r for r in rows if kind is None or r.get("source") == kind]
for r in random.sample(rows, min(n, len(rows))):
    device = f" ({r['device']})" if r.get("device") else ""
    print(f"--- {r.get('source', '')} [{r.get('topic') or ''}]{device}")
    if "turns" in r:                                  # 원본
        for t in r["turns"]:
            print(f"A: {t['user']}")
            print(f"B: {t['other']}")
        continue
    for m in r["messages"]:
        if m["role"] == "user":
            print(f"학생:   {m['content']}")
        elif m["role"] == "assistant":
            print(f"파이보: {m['content']}")
print(f"\n{path}: 전체 {len(rows)}개 중 {min(n, len(rows))}개")
