"""만든 학습 데이터를 사람이 읽기 좋게 몇 개 보여 줍니다.

    python show.py            # 무작위 20개
    python show.py 50 tool    # 도구 호출 예시만 50개
"""

import json
import random
import sys
from pathlib import Path

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
kind = sys.argv[2] if len(sys.argv) > 2 else None
rows = [json.loads(line) for line in Path("data/train.jsonl").read_text().splitlines()]
rows = [r for r in rows if kind is None or r["kind"] == kind]
for r in random.sample(rows, min(n, len(rows))):
    print(f"--- {r['kind']}{' (도구 목록 있음)' if r.get('tools') else ''}")
    for m in r["messages"]:
        if m["role"] == "user":
            print(f"아이:   {m['content']}")
        elif m["role"] == "assistant" and m.get("tool_calls"):
            for c in m["tool_calls"]:
                f = c["function"]
                print(f"[도구]  {f['name']}({json.dumps(f['arguments'], ensure_ascii=False)})")
        elif m["role"] == "assistant":
            print(f"파이보: {m['content']}")
        elif m["role"] == "tool":
            print(f"[결과]  {m['content']}")
print(f"\n전체 {len(rows)}개 중 {min(n, len(rows))}개")
