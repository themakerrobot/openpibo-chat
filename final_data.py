"""아이 말투 최종 학습 데이터 (data/train-themaker-kid.jsonl, 영어는 -en): 일반 대화 + 도구 대화 + 자기소개.

    python final_data.py --lang ko
    python final_data.py --lang en

구성 (한국어; 영어는 같은 대화를 쉬운 아이 영어로 옮긴 것):
- Claude가 쓴 아이 대화 (data/claude.jsonl / claude-en.jsonl): 전부, 학습에서 CLAUDE_REPEAT번씩.
  직접 쓴 대화(data/extra.txt)처럼 모델이 말투를 확실히 배우게 많이 봄.
- 기존 대화(data/pibo-kid.jsonl): claude_gen.py kid가 어른 말투·어색한 대답을 거른 것 중 KEEP개만
  (줄마다 늘 같은 순서로 골라서). 영어는 같은 id의 pibo-en.jsonl 대화를 한국어 쪽 길이에 맞춰.
- 도구 대화 (data/tools-kid.jsonl / tools-kid-en.jsonl): tool_data.py가 아이 말 모양과 아이 대화 앞부분으로 만든 것.
- 합치기는 merge_data.py와 persona.py 그대로 (똑같은 대답은 3번까지만, 대화의 30%에 도구 목록, 자기소개 1천).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from merge_data import merge
from persona import apply

DATA = Path("data")
#: 기존 대화에서 섞는 수
KEEP = 0
#: 도구 대화 수 (전체의 20~30%쯤)
TOOLS = 4000
#: Claude가 쓴 대화를 학습에서 보는 횟수
CLAUDE_REPEAT = 1


#: 교육용이라 더메이커가 학생이나 남을 깎아내리거나 낙담시키는 말을 하면 안 됨.
#: 부정(아니에요·않아요·없어요)이나 학생 말 옮기기(~다는/~라고)로 안심시키는 문장은 허용.
_NEG_KO = re.compile(r"한심|창피|부끄|아까워|아깝|소용없|망했|망할|최악|꼴찌|못난|게으|바보|멍청|못생|재능이 없|쓸모없|포기해|포기하세요|포기하는 게 낫|틀렸어|못해요")
_NEG_KO_OK = re.compile(r"아니|않|없어요|말아|마요|\s*(다는|다고|라는|라고)|저는.*못해요|직접.*못해요|수는 없|할 수 없")
_NEG_EN = re.compile(r"(stupid|dumb|idiot|loser|lazy|ugly|pathetic|hopeless|worthless|useless|shameful|embarrassw*|ashamed|give up|your fault|you failed|too bad|silly)", re.I)
_NEG_EN_OK = re.compile(r"(not|never|no one|isn't|aren't|don't|doesn't|wasn't|nobody)|n't", re.I)


def negative(row: dict, en: bool) -> bool:
    bad, ok = (_NEG_EN, _NEG_EN_OK) if en else (_NEG_KO, _NEG_KO_OK)
    for m in row["messages"]:
        c = m.get("content")
        if m["role"] != "assistant" or not isinstance(c, str):
            continue
        for sent in re.split(r"(?<=[.!?])\s+", c):
            if bad.search(sent) and not ok.search(sent):
                return True
    return False


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def dump(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def pick_old(keep: int, daily: int = 0) -> list[dict]:
    """pibo-kid.jsonl에서 Claude가 쓴 것이 아닌 대화: emotion·empathy(학교·가족·친구 이야기)는 keep개까지,
    daily(어른 카카오톡 대화라 표본 200개 평가에서 아이 대화로 쓸 만한 것이 7%뿐)는 daily개만 (기본 0).
    줄마다 늘 같은 순서."""
    old = [r for r in load(DATA / "pibo-kid.jsonl") if not str(r.get("source", "")).startswith("claude")]
    old.sort(key=lambda r: hashlib.md5(r["id"].encode()).hexdigest())
    kid = [r for r in old if r.get("source") in ("emotion", "empathy")][:keep]
    return kid + [r for r in old if r.get("source") == "daily"][:daily]


def english_old(ko_rows: list[dict]) -> list[dict]:
    """고른 한국어 대화와 같은 id의 영어 대화 (한국어에서 잘랐으면 같은 길이로 자름)."""
    en = {r["id"]: r for r in load(DATA / "pibo-en.jsonl")}
    out = []
    for ko in ko_rows:
        row = en.get(ko["id"])
        if row is None:
            continue
        n = len(ko["messages"])
        if len(row["messages"]) < n:
            continue
        out.append(dict(row, messages=row["messages"][:n]))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lang", choices=["ko", "en"], default="ko")
    p.add_argument("--keep", type=int, default=KEEP, help="기존 emotion·empathy 대화 수")
    p.add_argument("--daily", type=int, default=0, help="기존 daily(어른 대화) 수. 기본 0")
    p.add_argument("--tools", type=int, default=TOOLS,
                   help="도구 대화 수 (일반 대화가 적어서 4천 개를 다 쓰면 도구 비율이 너무 커짐)")
    p.add_argument("--tool-ratio", type=float, default=0.35,
                   help="도구 대화가 전체에서 차지할 몫(상한). 일상 대화(자기소개 1천 포함)가 늘면 도구도 --tools까지 같이 늘어남")
    p.add_argument("--out")
    args = p.parse_args()
    en = args.lang == "en"
    sfx = "-en" if en else ""

    old = pick_old(args.keep, args.daily)
    if en:
        old = english_old(old)
    claude = load(DATA / f"claude{sfx}.jsonl")
    dropped = {}
    old, dropped["기존"] = [r for r in old if not negative(r, en)], sum(negative(r, en) for r in old)
    claude, dropped["Claude"] = [r for r in claude if not negative(r, en)], sum(negative(r, en) for r in claude)
    tool_rows = sorted(load(DATA / f"tools-kid{sfx}.jsonl"),
                       key=lambda r: hashlib.md5(json.dumps(r, ensure_ascii=False).encode()).hexdigest())
    dropped["도구"] = sum(negative(r, en) for r in tool_rows)
    chat_n = len(old) + len(claude) + 1000
    n_tools = min(args.tools, int(args.tool_ratio / (1 - args.tool_ratio) * chat_n))
    tool_rows = [r for r in tool_rows if not negative(r, en)][:n_tools]
    print("부정적인 말로 뺀 대화: " + ", ".join(f"{k} {v}" for k, v in dropped.items()))
    old_path, claude_path = DATA / f"_old-kid{sfx}.jsonl", DATA / f"_claude-once{sfx}.jsonl"
    tools = DATA / f"_tools-kid{sfx}.jsonl"
    dump(old_path, old)
    dump(claude_path, claude)
    dump(tools, tool_rows)
    lines, kept, why = merge([str(old_path), str(claude_path), str(tools)])
    print(f"합침: {len(lines)}개 ({', '.join(f'{k} {v}' for k, v in sorted(kept.items()))})")
    print("뺀 것: " + (", ".join(f"{k} {v}" for k, v in why.most_common()) or "없음"))
    lines, count = apply(lines, lang=args.lang)
    print("교육용 AI 더메이커: " + ", ".join(f"{k} {v}" for k, v in count.items()))
    mine = [x for x in lines if json.loads(x).get("source") in ("claude", "claude-rewrite")]
    lines += mine * (CLAUDE_REPEAT - 1)
    out = Path(args.out or DATA / f"train-themaker-kid{sfx}.jsonl")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    old_path.unlink()
    claude_path.unlink()
    tools.unlink()
    print(f"{out}: {len(lines)}줄 (Claude 대화 {len(mine)}개 × {CLAUDE_REPEAT}, 기존 {len(old)}개, "
          f"도구 {kept.get('tools', 0)}개)")


if __name__ == "__main__":
    main()
