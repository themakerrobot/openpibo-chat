"""만든 학습 데이터를 사람이 읽기 좋게 몇 개 보여 줍니다.

    python show.py                                 # 최종 학습 데이터에서 무작위 20개
    python show.py 20 lead data/lead-clean.jsonl   # 대화 이어 가기만 (다른 파일에서)

종류: chat(친구 대화), lead(대화 이어 가기), en(영어 대화), learn(영어 공부), all
"""

import json
import random
import sys
from pathlib import Path

from make_data import slot

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
kind = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "all" else None
path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data/train-themaker.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
rows = [r for r in rows if kind is None or slot(r) == kind]
for r in random.sample(rows, min(n, len(rows))):
    grade = f" {r['grade']}" if r.get("grade") else ""
    device = f" ({r['device']})" if r.get("device") else ""
    print(f"--- {slot(r)}{grade}{device}")
    for m in r["messages"]:
        if m["role"] == "user":
            print(f"학생:   {m['content']}")
        elif m["role"] == "assistant":
            print(f"파이보: {m['content']}")
print(f"\n{path}: 전체 {len(rows)}개 중 {min(n, len(rows))}개")
