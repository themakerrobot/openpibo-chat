"""파이보 학습 대화를 하나로 합칩니다 (최종 학습 데이터 data/train-themaker.jsonl).

    python merge_data.py --out data/train-themaker.jsonl data/pibo.jsonl data/tools.jsonl

- 똑같은 줄은 한 번만.
- 도구 대화(tool_data.py)는 그대로. 대화의 TOOL_SHARE(30%)에는 도구 목록을 함께 보여 줘서, 도구가
  있을 때도 이야기에는 도구를 부르지 않고 대화하게 합니다 (기기 앱은 늘 도구 목록을 보냄).
- 똑같은 파이보 대답(띄어쓰기·문장부호 무시)은 REPEAT_MAX번까지만: 같은 대답이 많으면 모델이
  비슷한 말에 늘 같은 말을 합니다. 가장 흔한 첫 문장들을 찍어 사람이 볼 수 있게 합니다.
- 모든 기기 공통 정체 (persona.py): 시스템 글은 모두 tools.SYSTEM, 파이보 말 속 이름은 더메이커,
  "저는 로봇이라"는 "저는 AI라", 교육용 AI 더메이커 자기소개 1천 개. --no-persona면 하지 않습니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from persona import apply
from prompts import same

#: 똑같은 파이보 대답을 남기는 최대 횟수 (도구 대화는 빼고)
REPEAT_MAX = 3
#: 도구 목록을 함께 보여 줄 대화의 몫
TOOL_SHARE = 0.3


def gets_tools(line: str) -> bool:
    """줄마다 늘 같은 결정 (다시 돌려도 같은 데이터)."""
    return int(hashlib.md5(line.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < TOOL_SHARE


def first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]


def merge(paths: list[str], chat_limit: int | None = None) -> tuple[list[str], Counter, Counter]:
    """(합친 줄들, 데이터마다 남은 수, 뺀 까닭마다 수). chat_limit을 주면 코딩·도구 대화가 아닌 일반 대화는
    이만큼만 (줄마다 늘 같은 순서로 골라서): 코딩 도우미에 말투만 남길 때."""
    out, seen, kept, why, replies = [], set(), Counter(), Counter(), Counter()
    for path in paths:
        lines = [x for x in Path(path).read_text().splitlines() if x.strip()]
        if chat_limit is not None:
            is_special = [(lambda e: e.get("source") == "code" or e.get("kind") == "tool")(json.loads(x)) for x in lines]
            special = [x for x, s in zip(lines, is_special, strict=True) if s]
            chats = sorted((x for x, s in zip(lines, is_special, strict=True) if not s),
                           key=lambda x: hashlib.md5(x.encode()).hexdigest())[:chat_limit]
            lines = special + chats
        for line in lines:
            if line in seen:
                why["같은 줄이 또 나옴"] += 1
                continue
            seen.add(line)
            ex = json.loads(line)
            if ex.get("kind") == "tool":
                kept["tools"] += 1
                out.append(line)
                continue
            if ex.get("source") != "code" and gets_tools(line):
                ex["tools"] = True
                line = json.dumps(ex, ensure_ascii=False)
            mine = [same(m["content"]) for m in ex["messages"]
                    if m["role"] == "assistant" and m.get("content")]
            if any(replies[r] >= REPEAT_MAX for r in mine):
                why[f"같은 파이보 대답이 {REPEAT_MAX}번 넘음"] += 1
                continue
            replies.update(mine)
            kept[ex.get("source", "?")] += 1
            out.append(line)
    return out, kept, why


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("data", nargs="+")
    p.add_argument("--lang", choices=["ko", "en"], default="ko", help="en이면 영어 모델용")
    p.add_argument("--chat-limit", type=int, help="일반 대화는 이만큼만 (코딩 도우미에 말투만 남길 때)")
    p.add_argument("--no-persona", action="store_true",
                   help="시스템 글·이름 맞추기와 자기소개를 하지 않음")
    args = p.parse_args()
    lines, kept, why = merge(args.data, args.chat_limit)
    print(f"합침: {len(lines)}개 ({', '.join(f'{k} {v}' for k, v in sorted(kept.items()))})",
          flush=True)
    print("뺀 것: " + (", ".join(f"{k} {v}" for k, v in why.most_common()) or "없음"), flush=True)
    firsts = Counter(first_sentence(m["content"]) for line in lines
                     for m in json.loads(line)["messages"]
                     if m["role"] == "assistant" and m.get("content"))
    topics = Counter(json.loads(line).get("topic") or "?" for line in lines)
    print(f"주제 {len(topics)}가지: " + ", ".join(f"{t} {n}" for t, n in topics.most_common(20)),
          flush=True)
    print("가장 흔한 첫 문장: " + " / ".join(f"{t} ({n})" for t, n in firsts.most_common(10)),
          flush=True)
    if not args.no_persona:
        lines, count = apply(lines, lang=args.lang)
        print("교육용 AI 더메이커: " + ", ".join(f"{k} {v}" for k, v in count.items()), flush=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"{args.out}: {len(lines)}줄", flush=True)


if __name__ == "__main__":
    main()
