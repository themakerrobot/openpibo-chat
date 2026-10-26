"""만든 학습 데이터를 사람이 읽기 좋게 몇 개 보여 줍니다.

    python show.py            # 무작위 20개
    python show.py 50 tool    # 도구 호출 예시만 50개
    python show.py 20 mixed data/school-tools-clean.jsonl   # 대화 중 도구 (다른 파일에서)
    python show.py 20 near data/school-tools-clean.jsonl    # 도구와 비슷한 이야기

종류: chat, tool, mixed(이야기하다가 중간에 도구), near(도구와 비슷한 이야기), all
"""

import json
import random
import sys
from pathlib import Path

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
kind = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "all" else None
path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data/train.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
rows = [r for r in rows if kind is None or ("near" if r.get("near") else r["kind"]) == kind]
for r in random.sample(rows, min(n, len(rows))):
    grade = f" {r['grade']}" if r.get("grade") else ""
    print(f"--- {r['kind']}{grade}{' (도구 목록 있음)' if r.get('tools') else ''}")
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
print(f"\n{path}: 전체 {len(rows)}개 중 {min(n, len(rows))}개")
