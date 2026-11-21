"""Qwen 선생(5090 PC)이 Claude가 쓴 아이 대화를 본보기로 같은 말투의 새 아이 대화를 늘립니다 (data/qwen-kid.jsonl).

    python expand_kid.py selftest                         # 검사가 Claude 대화를 통과시키는지 (PC 어디서나)
    python expand_kid.py dry-run                          # 만들 글(프롬프트) 두 개 보기 (선생 없이)
    python expand_kid.py gen --n 25000                    # 선생으로 만들기 (5090 PC, vLLM)
    python expand_kid.py audit --n 200 --out samples/qwen-audit.txt   # 감사용 표본 (Claude가 읽고 통과율을 정함)
    python expand_kid.py export --start 0 --n 80 --out samples/qwen-review-0.txt   # 고쳐 쓸 묶음
    python expand_kid.py import-fixed samples/qwen-review-0.txt ...  # 고친 묶음을 되돌려 합침 (source qwen-kid-fixed)

품질은 이렇게 지킵니다 (최종 책임은 Claude):
1. 본보기: 같은 주제 그룹·비슷한 학년의 Claude 대화 4개를 보여 주고, 쓰는 규칙(docs/dialogue-guide.md)을 줍니다.
2. 글자 검사: claude_gen의 모든 검사(반말·높임말·"내"·감정·사람 경험·길이·기호) + 학생 말이 어른 말투인지,
   되받아 읊기, 같은 대답 되풀이, Claude 대화와 너무 비슷한 것을 뺍니다. 걸러진 이유를 모두 세어 보임.
3. 감사: audit 표본을 Claude가 엄격하게 읽어 통과율을 정합니다. 통과율이 낮으면 final_data는 Qwen 대화를 쓰지 않거나
   (data/qwen-approved.txt가 없으면 안 씀) 통과한 것만 씁니다.
4. 고쳐 쓰기: 통과 못 한 묶음은 export → Claude가 고침 → import-fixed.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import claude_gen as cg
from prompts import overlap, pibo_problem, same, to_messages

SEEDS = Path("data/claude.jsonl")
OUT = Path("data/qwen-kid.jsonl")
FIXED = Path("data/qwen-kid-fixed.jsonl")
APPROVED = Path("data/qwen-approved.txt")

#: 학생 말에 있으면 어른 대화로 보는 낱말 (고등학생의 대학·입시 이야기는 괜찮아서 claude_gen.ADULT보다 좁음)
ADULT_STRICT = re.compile(r"여자 ?친구|남자 ?친구|소개팅|회사|직장|출근|퇴근|월급|연봉|상사|결혼|신혼|남편|아내|와이프|"
                          r"육아|임신|술자리|소주|맥주|와인|담배|대출|주식|부동산|키키|#@")

RULES = """- 학생 말: 그 학년 아이가 실제로 쓰는 말. 대부분 반말, 초1~2는 아주 짧고 단순(맞춤법도 가끔 틀림), 초3~4는 짧고 수다스럽게,
  초5~6과 중고등학생은 조금 길고 줄임말·은어·놀림도 괜찮음. 전부 질문으로 하지 않기. 욕설·개인정보·어른 같은 말투 금지.
- 더메이커 말: 한두 문장, 80자 안쪽, 열네 살 친구처럼 밝고 다정한 해요체. 모든 문장이 요/니다/니까/죠로 끝남.
  학생을 너/네가로 부르지 않고, 학생을 높이는 -시-(하셨, 좋아하세요)와 -세요 요청형(해 주세요)을 쓰지 않음(해 줘요, 말아요).
  자기는 제가. 내/내가 쓰지 않음. 이모지·기호·ㅋㅋ 금지.
- 더메이커는 AI라서 사람 경험(먹기, 자기, 학교, 가족, 여행)과 감정·몸 느낌·취향(기뻐요, 좋아해요, 오싹해요, 보고 싶어요, 자신 있어요)을
  말하지 않음. 모르는 건 모른다고 함. 학생이 말하지 않은 나이를 단정하지 않음. 부모에게 비밀을 지켜 준다고 약속하지 않음.
- 학생 말을 되받아 읊지 않고 구체적이고 맞는 사실·방법으로 답함. 틀린 사실 금지(숫자·규칙·낱말 이음·정답 확인).
- 슬픔·괴롭힘·위험은 마음을 먼저 알아주고 믿을 만한 어른께 알리게 함. 진단·약 권하기 금지."""

PROMPT = """아이와 교육용 AI 더메이커가 나누는 한국어 대화를 하나 써요.

규칙:
{rules}

본보기 (같은 규칙으로 쓴 좋은 대화들. 문장을 베끼지 말고 말투와 수준만 배워요):
{shots}

이제 새 대화를 써요.
- 학년: {grade} / 주제: {topic} / 아이 성격: {persona} / 시작: {opener}
{note}- 주고받기는 {turns}번, 마지막은 더메이커 말. 본보기와 다른 내용으로.
JSON 하나로만 답해요: {{"turns": [{{"user": "학생 말", "pibo": "더메이커 말"}}, ...]}}"""


def load_seeds() -> list[dict]:
    seeds = []
    for line in SEEDS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        msgs = [m for m in r["messages"] if m["role"] in ("user", "assistant")]
        turns = [{"user": msgs[i]["content"], "pibo": msgs[i + 1]["content"]} for i in range(0, len(msgs) - 1, 2)]
        if turns and r.get("grade"):
            seeds.append({"id": r["id"], "grade": r["grade"], "topic": r.get("topic"), "turns": turns,
                          "group": cg.TOPIC_GROUP.get(r.get("topic"), "")})
    return seeds


def band(grade: str) -> str:
    return {"초": "초", "중": "중", "고": "고"}[grade[0]] + ("저" if grade in ("초1", "초2", "초3") else "")


def pick_shots(rng: random.Random, seeds: list[dict], group: str, grade: str, k: int) -> list[dict]:
    same_group = [s for s in seeds if s["group"] == group]
    near = [s for s in same_group if band(s["grade"]) == band(grade)]
    pool = near if len(near) >= k else same_group if len(same_group) >= k else seeds
    return rng.sample(pool, k)


def show_turns(turns: list[dict]) -> str:
    return "\n".join(f"학생: {t['user']}\n더메이커: {t['pibo']}" for t in turns)


def make_slot(i: int, rng: random.Random) -> dict:
    topic = cg.ALL_TOPICS[i % len(cg.ALL_TOPICS)]
    lo, hi = cg.TOPIC_GRADES.get(topic, (0, 11))
    grade = rng.choice(cg.GRADES[lo:hi + 1])
    group = cg.TOPIC_GROUP[topic]
    openers = [o for o in cg.OPENERS if o != "부탁이나 도움 요청"] if group == "도구와 헷갈리는 말" else cg.OPENERS
    note = cg.TOPIC_NOTES.get(topic) or cg.TOPIC_NOTES.get(group)
    return {"id": f"q{i:05d}", "topic": topic, "group": group, "grade": grade, "persona": rng.choice(cg.PERSONAS),
            "opener": rng.choice(openers), "note": note, "turns": rng.randint(3, 6)}


def make_prompt(slot: dict, seeds: list[dict], rng: random.Random, shots: int) -> str:
    sh = pick_shots(rng, seeds, slot["group"], slot["grade"], shots)
    note = f"- 이 주제에서 지킬 것: {slot['note']}\n" if slot["note"] else ""
    return PROMPT.format(rules=RULES, shots="\n\n".join(f"[본보기 {n}] ({s['grade']}, {s['topic']})\n{show_turns(s['turns'])}"
                                                     for n, s in enumerate(sh, 1)),
                         grade=slot["grade"], topic=slot["topic"], persona=slot["persona"], opener=slot["opener"],
                         note=note, turns=slot["turns"])


def check(turns, seen_first: set, reply_count: Counter, seed_texts: list[str]) -> str | None:
    """걸러야 할 까닭 (괜찮으면 None). 부수 효과 없음 (seen·count는 호출한 쪽이 갱신)."""
    if not isinstance(turns, list) or not 3 <= len(turns) <= 6:
        return "주고받기 수"
    if any(not isinstance(t, dict) for t in turns):
        return "모양"
    row = {"turns": turns}
    if why := cg.problem(row):
        return why
    for t in turns:
        user, pibo = t["user"].strip(), t["pibo"].strip()
        if len(user) > 160 or len(pibo) < 5:
            return "길이"
        if ADULT_STRICT.search(user):
            return "학생 말이 어른 말투"
        if why := cg.reply_problem(pibo):
            return why
        if cg.SELF_NAE.search(pibo):
            return "내"
        if len(pibo) > 12 and overlap(pibo, user) > 0.6:
            return "학생 말 되받아 읊기"
    if len({same(t["pibo"]) for t in turns}) < len(turns):
        return "같은 대답이 한 대화에 되풀이"
    if same(turns[0]["user"]) in seen_first:
        return "같은 첫 말"
    if any(reply_count[same(t["pibo"])] >= 3 for t in turns):
        return "같은 대답이 3번 넘음"
    first = turns[0]["user"]
    if any(overlap(first, s) > 0.8 for s in seed_texts[:0]):    # 자리: 본보기와 너무 비슷한 것 (아래 gen에서 따로)
        return "본보기와 같음"
    return None


def parse(text: str):
    from teacher import first_json
    obj = first_json(text)
    return obj.get("turns") if isinstance(obj, dict) else None


def selftest() -> None:
    seeds = load_seeds()
    ok, why = 0, Counter()
    seen: set = set(); count: Counter = Counter()
    for s in seeds:
        reason = check(s["turns"], set(), Counter(), [])
        if reason:
            why[reason] += 1
        else:
            ok += 1
    print(f"Claude 대화 {len(seeds)}개 중 통과 {ok} ({100 * ok / len(seeds):.1f}%), 걸린 까닭: {dict(why)}")


def dry_run() -> None:
    rng = random.Random(0)
    seeds = load_seeds()
    for i in (0, 53):
        print(make_prompt(make_slot(i, rng), seeds, rng, 3))
        print("=" * 60)


def gen(n: int, out: Path, seed: int, batch: int, shots: int) -> None:
    from teacher import ask, load_teacher

    rng = random.Random(seed)
    seeds = load_seeds()
    seed_firsts = [s["turns"][0]["user"] for s in seeds]
    llm = load_teacher()
    done: list[dict] = []
    seen_first: set = set()
    reply_count: Counter = Counter()
    why: Counter = Counter()
    i = 0
    while len(done) < n and i < n * 4:
        slots = [make_slot(i + j, rng) for j in range(batch)]
        i += batch
        answers = ask(llm, [make_prompt(s, seeds, rng, shots) for s in slots], 1400, think=False)
        for s, a in zip(slots, answers, strict=True):
            turns = parse(a)
            reason = check(turns, seen_first, reply_count, [])
            if not reason and any(overlap(turns[0]["user"], f) > 0.8 for f in seed_firsts):
                reason = "본보기와 같음"
            if reason:
                why[reason] += 1
                continue
            seen_first.add(same(turns[0]["user"]))
            reply_count.update(same(t["pibo"]) for t in turns)
            done.append({"kind": "chat", "source": "qwen-kid", "topic": s["topic"], "id": s["id"], "grade": s["grade"],
                         "messages": to_messages([{"user": t["user"].strip(), "pibo": t["pibo"].strip()} for t in turns])})
        print(f"{len(done)}/{n}개 (시도 {i}), 걸린 까닭: {dict(why.most_common(6))}", flush=True)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in done), encoding="utf-8")


def rows_to_text(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        lines.append(f"## {r['id']}")
        for m in r["messages"]:
            if m["role"] == "user":
                lines.append(m["content"])
            elif m["role"] == "assistant":
                lines.append("= " + m["content"])
        lines.append("")
    return "\n".join(lines)


def audit(n: int, out: Path, seed: int) -> None:
    rows = [json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip()]
    sample = random.Random(seed).sample(rows, min(n, len(rows)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rows_to_text(sample), encoding="utf-8")
    print(f"{out}: {len(sample)}개 (전체 {len(rows)}개 중). Claude가 읽고 통과율을 정해 주세요")


def export(start: int, n: int, out: Path) -> None:
    rows = [json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip()]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rows_to_text(rows[start:start + n]), encoding="utf-8")
    print(f"{out}: {len(rows[start:start + n])}개")


def import_fixed(paths: list[Path]) -> None:
    """고친 묶음(## id, 학생 말, = 더메이커 말)을 읽어 data/qwen-kid-fixed.jsonl로 (claude_gen 검사를 다시 거침)."""
    meta = {json.loads(x)["id"]: json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip()}
    rows, bad = [], Counter()
    for path in paths:
        for block in path.read_text(encoding="utf-8").split("## ")[1:]:
            lines = [x.strip() for x in block.strip().splitlines() if x.strip()]
            sid, body = lines[0], lines[1:]
            if len(body) % 2 or any(not b.startswith("= ") for b in body[1::2]):
                bad["모양"] += 1
                continue
            turns = [{"user": u, "pibo": p[2:].strip()} for u, p in zip(body[0::2], body[1::2], strict=True)]
            reason = check(turns, set(), Counter(), [])
            if reason or sid not in meta:
                bad[reason or "없는 id"] += 1
                continue
            rows.append({"kind": "chat", "source": "qwen-kid-fixed", "topic": meta[sid]["topic"], "id": sid,
                         "grade": meta[sid].get("grade"), "messages": to_messages(turns)})
    old = [json.loads(x) for x in FIXED.read_text(encoding="utf-8").splitlines()] if FIXED.exists() else []
    merged = {r["id"]: r for r in old + rows}
    FIXED.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged.values()), encoding="utf-8")
    print(f"{FIXED}: {len(merged)}개 (이번 {len(rows)}개, 뺀 것 {dict(bad)})")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("dry-run")
    g = sub.add_parser("gen")
    g.add_argument("--n", type=int, default=25000)
    g.add_argument("--out", type=Path, default=OUT)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--batch", type=int, default=128)
    g.add_argument("--shots", type=int, default=4)
    a = sub.add_parser("audit")
    a.add_argument("--n", type=int, default=200)
    a.add_argument("--out", type=Path, default=Path("samples/qwen-audit.txt"))
    a.add_argument("--seed", type=int, default=1)
    e = sub.add_parser("export")
    e.add_argument("--start", type=int, default=0)
    e.add_argument("--n", type=int, default=80)
    e.add_argument("--out", type=Path, required=True)
    im = sub.add_parser("import-fixed")
    im.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args()
    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "dry-run":
        dry_run()
    elif args.cmd == "gen":
        gen(args.n, args.out, args.seed, args.batch, args.shots)
    elif args.cmd == "audit":
        audit(args.n, args.out, args.seed)
    elif args.cmd == "export":
        export(args.start, args.n, args.out)
    else:
        import_fixed(args.paths)


if __name__ == "__main__":
    main()
