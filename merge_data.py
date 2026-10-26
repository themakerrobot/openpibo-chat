"""학습 데이터 여러 개를 하나로 합칩니다 (최종 학습 데이터 data/train-themaker.jsonl).

    python merge_data.py --out data/train-themaker.jsonl \\
        data/train-school.jsonl data/english-clean.jsonl data/lead-clean.jsonl

- 대화만 남깁니다: 도구를 부르는 대화(예전 데이터의 tool, mixed)는 빼고, 남은 대화도 도구
  목록 없이 학습하게 표시를 지웁니다. 모델은 대화만 하고 동작은 앱이 정합니다.
- 똑같은 줄은 한 번만.
- 똑같은 파이보 대답(띄어쓰기·문장부호 무시)은 REPEAT_MAX번까지만: 같은 대답이 많으면 모델이
  비슷한 말에 늘 같은 말을 합니다. 가장 흔한 첫 문장들을 찍어 사람이 볼 수 있게 합니다.
- 여러 기기용 (devices.py): 일부 대화의 이름을 다른 이름(더메이커, 파이브레인 …)으로, 기기
  소개를 시스템 글에, 시스템 글을 보고 하는 자기소개 3천 개. --no-devices면 하지 않습니다.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from devices import vary
from make_data import slot

#: 똑같은 파이보 대답을 남기는 최대 횟수
REPEAT_MAX = 3


def said(text: str) -> str:
    """대답을 견줄 모양: 띄어쓰기와 문장부호를 뺌."""
    return re.sub(r"[\s.,!?~…'\"]", "", text)


def first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]


def chat_only(ex: dict) -> dict | None:
    """도구를 부르는 대화는 None, 나머지는 도구 목록 없이."""
    if ex.get("kind") in ("tool", "mixed") or any(m.get("tool_calls") for m in ex["messages"]):
        return None
    for key in ("tools", "tool_names", "motions", "near"):
        ex.pop(key, None)
    ex["kind"] = "chat"
    return ex


def merge(paths: list[str]) -> tuple[list[str], Counter, Counter]:
    """(합친 줄들, 칸마다 남은 수, 뺀 까닭마다 수)."""
    out, seen, kept, why, replies = [], set(), Counter(), Counter(), Counter()
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            if line in seen:
                why["같은 줄이 또 나옴"] += 1
                continue
            seen.add(line)
            ex = chat_only(json.loads(line))
            if ex is None:
                why["도구를 부르는 대화"] += 1
                continue
            mine = [said(m["content"]) for m in ex["messages"]
                    if m["role"] == "assistant" and m.get("content")]
            if any(replies[r] >= REPEAT_MAX for r in mine):
                why[f"같은 파이보 대답이 {REPEAT_MAX}번 넘음"] += 1
                continue
            replies.update(mine)
            kept[slot(ex)] += 1
            out.append(json.dumps(ex, ensure_ascii=False))
    return out, kept, why


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("data", nargs="+")
    p.add_argument("--no-devices", action="store_true",
                   help="이름 바꾸기·기기 소개·자기소개를 하지 않음")
    args = p.parse_args()
    lines, kept, why = merge(args.data)
    print(f"합침: {len(lines)}개 ({', '.join(f'{k} {v}' for k, v in sorted(kept.items()))})",
          flush=True)
    print("뺀 것: " + (", ".join(f"{k} {v}" for k, v in why.most_common()) or "없음"), flush=True)
    firsts = Counter(first_sentence(m["content"]) for line in lines
                     for m in json.loads(line)["messages"]
                     if m["role"] == "assistant" and m.get("content"))
    topics = Counter(json.loads(line).get("topic") for line in lines)
    unknown = topics.pop(None, 0)
    print(f"주제 {len(topics)}가지 (주제를 적지 않은 예전 대화 {unknown}개): " + ", ".join(
        f"{t} {n}" for t, n in topics.most_common(15)) + (" …" if len(topics) > 15 else ""),
        flush=True)
    print("가장 흔한 첫 문장: " + " / ".join(f"{t} ({n})" for t, n in firsts.most_common(10)),
          flush=True)
    if not args.no_devices:
        lines, count = vary(lines)
        print("여러 기기용: " + ", ".join(f"{k} {v}" for k, v in count.items()), flush=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"{args.out}: {len(lines)}줄", flush=True)


if __name__ == "__main__":
    main()
