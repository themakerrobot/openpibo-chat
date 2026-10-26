"""학습 데이터 여러 개를 하나로 합치며 도구 예시의 나쁜 버릇을 거릅니다.

    python merge_data.py --out data/train-school2.jsonl \\
        data/train-school.jsonl data/school-tools-clean.jsonl

- 똑같은 줄은 한 번만 (train-school.jsonl은 어린이 도구 예시를 두 번씩 넣어 두었는데, 초·중·고
  도구 예시가 생겨 그럴 필요가 없음).
- 인사만 한 말("안녕!", "파이보 반가워")에 도구를 부른 예시는 뺍니다: 학습한 모델이 "안녕"에
  인사 동작과 눈 색을 같이 불러, 파이보에서 대답이 늦어지고 하지 않아도 될 일을 합니다.
- 눈이나 색 이야기가 없는 말에 눈 색(eye_on)을 부른 예시도 뺍니다 (춤춰 달라는데 눈 색을 덤으로).
- 대화 중 도구 예시(mixed)는 도구를 부른 차례마다 같은 규칙으로 봅니다.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

#: 인사만 한 말 (앞뒤의 '파이보야', 문장부호는 무시)
GREETING = re.compile(
    r"^(파이보(야|씨|님)?\s*)?"
    r"(안녕(하세요|하십니까|안녕)?|반가워(요)?|반갑다|반갑습니다|하이|헬로|하잉|ㅎㅇ|"
    r"좋은\s*(아침|저녁|밤)(이야|이에요|입니다)?|굿모닝|잘\s*(잤어|잤니|있었어)(요)?|"
    r"(나|저)\s*왔어(요)?|다녀왔(어|습니다)(요)?|오랜만이(야|에요|네)|또\s*만났네)"
    r"(\s*파이보(야)?)?[\s!.~?ㅎㅋ^]*$")
#: 눈 색을 바라는 말에 들어가는 낱말
EYE_WORDS = re.compile(r"눈|색|빛|불|빨|주황|노랑|노란|초록|파랑|파란|파래|하늘|남색|보라|분홍|"
                       r"핑크|하양|하얀|흰|연두|청록|갈색|무지개|레드|블루|그린|컬러|반짝|밝|깜깜")


def tool_turns(ex: dict):
    """(아이 말, 부른 도구 이름들) — 도구를 부른 차례마다."""
    msgs = ex["messages"]
    for j, m in enumerate(msgs):
        if m["role"] == "assistant" and m.get("tool_calls"):
            yield msgs[j - 1]["content"], [c["function"]["name"] for c in m["tool_calls"]]


def problem(ex: dict) -> str | None:
    """이 예시를 뺄 까닭 (괜찮으면 None)."""
    for user, names in tool_turns(ex):
        if GREETING.match(user.strip()):
            return "인사만 했는데 도구를 부름"
        if "eye_on" in names and not EYE_WORDS.search(user):
            return "눈 이야기가 없는데 눈 색을 바꿈"
    return None


def merge(paths: list[str]) -> tuple[list[str], Counter, Counter]:
    """(합친 줄들, 칸마다 남은 수, 뺀 까닭마다 수)."""
    out, seen, kept, why = [], set(), Counter(), Counter()
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            if line in seen:
                why["같은 줄이 또 나옴"] += 1
                continue
            seen.add(line)
            ex = json.loads(line)
            if reason := problem(ex):
                why[reason] += 1
                continue
            kept["near" if ex.get("near") else ex["kind"]] += 1
            out.append(line)
    return out, kept, why


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("data", nargs="+")
    args = p.parse_args()
    lines, kept, why = merge(args.data)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"{args.out}: {len(lines)}줄 ({', '.join(f'{k} {v}' for k, v in sorted(kept.items()))})",
          flush=True)
    print("뺀 것: " + (", ".join(f"{k} {v}" for k, v in why.most_common()) or "없음"), flush=True)


if __name__ == "__main__":
    main()
