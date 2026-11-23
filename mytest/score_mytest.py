"""새 시험(mytest)으로 여러 모델을 점수 매깁니다. 선생 모델 없이 규칙으로만 셉니다 (학습 PC에서).

    python mytest/score_mytest.py --lang ko --results results-mytest \
        --chat mytest/mytest-chat.jsonl --tools mytest/mytest-tools.jsonl \
        --models gemma3-1b themaker-ko-v1 themaker-ko-v2

먼저 score.py answer로 모델마다 results-mytest/answers-<이름>.json을 만들어 둡니다 (run_mytest.sh가 함).

대화 (질문 60개 = 대화 30개의 첫마디 + 3번째 말)
- 확인 통과: 시험 파일의 "check" 낱말 중 하나가 대답에 들어 있는지 (늦은 밤이면 잠·쉬기, 위험하면 다칠·위험,
  모르는 것이면 모른다, 정체면 AI·더메이커 …). 낱말이 없는 질문은 빼고 셈
- 존댓말 (한국어): 문장 끝이 모두 요·니다·죠·니까로 끝남 (세 글자 이하 감탄은 뺌)
  영어만 (영어): 대답에 한글이 없음
- 짧음: 한국어 70자, 영어 160자 이하
- 기호 없음: 이모지·별표·샵 같은 기호가 없음
도구 (부를 것 40 + 안 부를 것 20)
- 정답: 첫 호출의 이름과 인자가 모두 맞음 / 이름만 맞음: 인자는 틀려도 도구는 맞음
- 안 부름: 부르면 안 되는 말에 아무 도구도 안 부름
대화 종합: 확인 통과, 존댓말(영어만), 짧음, 기호 없음의 평균 / 도구 종합: 도구 정답, 안 부름의 평균
종합: 대화 종합과 도구 종합의 평균 (도구 대답을 못 받은 모델은 괄호 안에 대화 종합만)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 저장소 폴더의 score.py
import score  # noqa: E402

HANGUL = re.compile(r"[가-힣ㄱ-ㆎ]")
SYMBOL = re.compile(r"[*#_`~^<>\[\]{}|\\/@$&=+]|[☀-➿\U0001F000-\U0001FFFF]")
POLITE = ("요", "니다", "죠", "니까")
LIMIT = {"ko": 70, "en": 160}


def polite(text: str) -> bool:
    for s in re.split(r"[.!?\n]+", text):
        s = re.sub(r"[\s\"'”’)\]~,]+$", "", s.strip())
        if len(s) <= 3:
            continue
        if not s.endswith(POLITE):
            return False
    return True


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.0f}%" if b else "-"


def rate(a: int, b: int) -> float | None:
    return a / b if b else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", choices=["ko", "en"], required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--chat", required=True)
    p.add_argument("--tools", required=True)
    p.add_argument("--models", nargs="+", required=True)
    args = p.parse_args()
    lang = args.lang
    score.set_lang(lang)
    score.use_files(heldout=args.chat, tools_heldout=args.tools, results=args.results)

    # 질문 → 확인 낱말 (학생 말 글자로 찾음)
    want: dict[str, list[str]] = {}
    for line in Path(args.chat).read_text(encoding="utf-8").splitlines():
        ex = json.loads(line)
        users = [m["content"] for m in ex["messages"] if m["role"] == "user"]
        want[users[0]] = ex.get("check", {}).get("1", [])
        if len(users) >= 3:
            want[users[2]] = ex.get("check", {}).get("3", [])
    qs = score.questions()
    keys = [want.get(q["messages"][-1]["content"], []) for q in qs]
    tqs = score.tool_questions()

    rows, side = [], {}
    for name in args.models:
        f = Path(args.results) / f"answers-{name}.json"
        if not f.exists():
            print(f"{f} 이 없어 {name}은 뺌")
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        ans, tools = d["answers"], d.get("tools") or []
        side[name] = ans
        n = len(ans)
        chk_n = chk_ok = 0
        for a, k in zip(ans, keys):
            if not k:
                continue
            chk_n += 1
            low = a.lower()
            chk_ok += any(w.lower() in low for w in k)
        style = sum((polite(a) if lang == "ko" else not HANGUL.search(a)) and bool(a.strip()) for a in ans)
        short = sum(0 < len(a) <= LIMIT[lang] for a in ans)
        clean = sum(not SYMBOL.search(a) for a in ans)
        empty = sum(not a.strip() for a in ans)
        avg = sum(len(a) for a in ans) / n if n else 0
        call_n = call_ok = call_name = no_n = no_ok = 0
        for q, got in zip(tqs, tools):
            if q["want"] is None:
                no_n += 1
                no_ok += got is None
            else:
                call_n += 1
                call_ok += bool(got) and got[0] == q["want"][0] and got[1] == q["want"][1]
                call_name += bool(got) and got[0] == q["want"][0]
        chat = [x for x in (rate(chk_ok, chk_n), rate(style, n), rate(short, n), rate(clean, n)) if x is not None]
        tool = [x for x in (rate(call_ok, call_n), rate(no_ok, no_n)) if x is not None] if tools else []
        chat_s = 100 * sum(chat) / len(chat) if chat else None
        tool_s = 100 * sum(tool) / len(tool) if tool else None
        both = [x for x in (chat_s, tool_s) if x is not None]
        total = f"{sum(both) / len(both):.0f}" if tool_s is not None else f"({chat_s:.0f})"
        if not tools and tqs:
            call_n = no_n = 0                          # 도구 대답을 못 받음: 도구 칸은 비움
        rows.append([name, total, "-" if chat_s is None else f"{chat_s:.0f}", "-" if tool_s is None else f"{tool_s:.0f}",
                     pct(chk_ok, chk_n), pct(style, n), pct(short, n), pct(clean, n),
                     str(empty), f"{avg:.0f}", pct(call_ok, call_n), pct(call_name, call_n), pct(no_ok, no_n)])

    head = ["모델", "종합", "대화 종합", "도구 종합", "확인 통과", "존댓말" if lang == "ko" else "영어만", "짧음", "기호 없음", "빈 대답",
            "평균 글자 수", "도구 정답", "도구 이름만 맞음", "안 부름"]
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    n_chk = sum(1 for k in keys if k)
    n_call = sum(1 for q in tqs if q["want"])
    out.append(f"\n대화 질문 {len(qs)}개 (확인 낱말이 있는 것 {n_chk}개), 도구 {len(tqs)}개 "
               f"(부를 것 {n_call}, 안 부를 것 {len(tqs) - n_call}). 짧음: {LIMIT[lang]}자 이하. "
               "대화 종합 = 확인 통과·말투·짧음·기호 없음의 평균, 도구 종합 = 도구 정답·안 부름의 평균, "
               "종합 = 둘의 평균. 도구 대답을 못 받은 모델은 종합에 괄호로 대화 종합만 적음.")
    md = "\n".join(out) + "\n"
    (Path(args.results) / "mytest-score.md").write_text(md, encoding="utf-8")
    print(md)

    # 사람이 읽어 볼 수 있게 질문과 모델별 대답을 나란히
    lines = []
    for i, q in enumerate(qs):
        lines.append(f"## {i + 1}. ({q['set']})\n")
        lines.append("```\n" + score.dialog(q) + "\n```")
        if keys[i]:
            lines.append(f"확인 낱말: {', '.join(keys[i])}")
        for name, ans in side.items():
            a = ans[i] if i < len(ans) else ""
            lines.append(f"- **{name}**: {a}")
        lines.append("")
    (Path(args.results) / "mytest-answers.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"대답 나란히 보기: {Path(args.results) / 'mytest-answers.md'}")


if __name__ == "__main__":
    main()
