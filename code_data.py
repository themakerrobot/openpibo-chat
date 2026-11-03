"""코딩 도우미 학습 대화: 블록·함수 설명(code_docs.py)마다 선생 모델이 학생 질문과 대답을 만들고, 대답 속 코드를
그 기기의 함수 목록과 대조하고, 선생이 다시 검사해 좋은 것만 남깁니다 (prompts_code.py).

    python code_data.py --out data/code.jsonl --heldout data/heldout-code.jsonl
    python code_data.py --out data/try-code.jsonl --heldout data/try-heldout-code.jsonl --limit 30   # 먼저 조금
    python code_data.py --dropped grounded > samples/drop-code.txt

1. 만들기 (생각 켬): 설명 하나와 그 기기의 블록 목록을 보고 문답 7쌍. 글자 검사(반말, 이모지, 길이, 코드 문법,
   없는 모듈·클래스·메서드, 블록 상자에 없는 「블록」)에 걸린 문답은 뺍니다.
2. 검사 (생각 켬): 설명과 맞는지·쉬운지 4점 미만이거나 코드가 틀리면 뺍니다.
3. 설명마다 남은 문답이 4쌍 이상이면 하나를 채점용(--heldout)으로 뗍니다. 같은 설명의 두 문답을 이어 붙인
   두 번 주고받기 대화도 일부 만듭니다. 파이브레인에 없는 움직이기를 물으면 없다고 하는 대화를 더합니다.

판단과 선생의 답 원문은 --log에 남기고 다시 돌리면 이미 한 것은 건너뜁니다 (규칙을 고쳐도 원문으로 다시 판단).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

from prompts_code import check_prompt, decide_check, decide_gen, dump, gen_prompt, passes
from tools import system_code

UNITS = Path("data/code-units.jsonl")
API = Path("data/code-api.json")
CHUNK = 1000
STAGES = {"gen": 4096, "check": 3072}
#: 두 번 주고받기로 이어 붙일 설명의 몫
CHAIN_SHARE = 0.3

#: 파이브레인에 없는 움직이기를 물을 때 (모터가 없음)
NO_MOTION_Q = ["파이브레인 춤추게 하려면 어떻게 해?", "파이브레인 팔 움직이는 블록 어디 있어?",
               "motion.set_motion 파이브레인에서 써도 돼?", "파이브레인 앞으로 걷게 하고 싶어",
               "모터 각도 바꾸는 블록 알려 줘", "파이브레인 고개 끄덕이게 하는 코드 짜 줘",
               "움직임 카테고리가 안 보여", "파이브레인으로 손 흔들게 할 수 있어?"]
NO_MOTION_A = ["파이브레인에는 모터가 없어서 움직이는 블록이나 Motion 함수가 없어요. 대신 「색상 불빛 바꾸기」 블록으로 "
               "불빛을 바꾸거나 화면에 글자를 띄워서 표현해 볼 수 있어요.",
               "파이브레인은 몸을 움직이는 모터가 없어서 움직이기는 할 수 없어요. 「RGB 불빛 바꾸기」나 「화면 표시하기」 "
               "블록으로 재미있게 표현해 볼까요?"]


def key_of(u: dict) -> str:
    return hashlib.md5(dump(u).encode()).hexdigest()[:12]


def pairs_key(pairs) -> str:
    return hashlib.md5(dump(pairs).encode()).hexdigest()[:12]


def load_log(path: Path, units: dict, api: dict) -> dict:
    done = {}
    if path.exists():
        for row in path.read_text().splitlines():
            try:
                r = json.loads(row)
            except json.JSONDecodeError:
                continue
            u = units.get(r.get("id"))
            if u is not None and key_of(u) == r.get("key"):
                done[r["id"], r["stage"]] = r
    for (i, stage), r in list(done.items()):
        if stage == "gen":
            r["pairs"], r["why"] = decide_gen(r["raw"], units[i], api[units[i]["device"]])
    for (i, stage), r in list(done.items()):
        if stage == "check":
            g = done.get((i, "gen"))
            if g is None or not g["pairs"] or r.get("for") != pairs_key(g["pairs"]):
                del done[i, stage]
            else:
                r["marks"] = decide_check(r["raw"], len(g["pairs"]))
    return done


def run(llm_ask, units: dict, api: dict, log_path: Path, timing: dict | None = None) -> dict:
    """llm_ask(글 목록, 최대 토큰) → 답 목록 (생각 켬)."""
    done = load_log(log_path, units, api)
    for stage, tokens in STAGES.items():
        todo = [i for i in units if (i, stage) not in done and (stage == "gen" or done[i, "gen"]["pairs"])]
        print(f"{stage}: {len(todo)}개 할 차례", flush=True)
        for start in range(0, len(todo), CHUNK):
            part = todo[start:start + CHUNK]
            prompts = [gen_prompt(units[i], api[units[i]["device"]]) if stage == "gen" else
                       check_prompt(units[i], done[i, "gen"]["pairs"], api[units[i]["device"]]) for i in part]
            t = time.time()
            answers = llm_ask(prompts, tokens)
            if timing is not None:
                n, sec = timing.get(stage, [0, 0.0])
                timing[stage] = [n + len(part), sec + time.time() - t]
            with log_path.open("a") as log:
                for i, text in zip(part, answers, strict=True):
                    r = {"id": i, "key": key_of(units[i]), "stage": stage, "raw": text}
                    if stage == "gen":
                        r["pairs"], r["why"] = decide_gen(text, units[i], api[units[i]["device"]])
                    else:
                        r["for"] = pairs_key(done[i, "gen"]["pairs"])
                        r["marks"] = decide_check(text, len(done[i, "gen"]["pairs"]))
                    done[i, stage] = r
                    log.write(dump(r) + "\n")
            print(f"{stage} {start + len(part)}/{len(todo)}", flush=True)
    return done


def example(u: dict, turns: list[dict], k) -> dict:
    msgs = [{"role": "system", "content": system_code(u["device"])}]
    for p in turns:
        msgs += [{"role": "user", "content": p["q"]}, {"role": "assistant", "content": p["a"]}]
    return {"kind": "chat", "source": "code", "device": u["device"], "unit": u["id"], "id": f"{u['id']}#{k}",
            "topic": u.get("category") or u.get("class"), "messages": msgs}


def results(units: dict, done: dict) -> tuple[list[dict], list[dict], Counter, Counter]:
    """(학습 대화, 채점용 대화, 기기마다 남은 문답 수, 뺀 까닭마다 수)."""
    train, held, kept, why = [], [], Counter(), Counter()
    rng = random.Random(0)
    for i, u in units.items():
        g, c = done.get((i, "gen")), done.get((i, "check"))
        if g is None:
            why["아직 안 함"] += 1
            continue
        why.update(g["why"])
        if not g["pairs"]:
            continue
        if c is None:
            why["아직 검사 안 함"] += len(g["pairs"])
            continue
        good = []
        for p, m in zip(g["pairs"], c["marks"], strict=True):
            if reason := passes(m):
                why[reason] += 1
            else:
                good.append(p)
        if len(good) >= 4:
            held.append(example(u, [good[-1]], "h"))
            good = good[:-1]
        train += [example(u, [p], k) for k, p in enumerate(good)]
        if len(good) >= 2 and rng.random() < CHAIN_SHARE:
            a, b = rng.sample(good, 2)
            train.append(example(u, [a, b], "c"))
        kept[u["device"]] += len(good)
    devices = {u["device"] for u in units.values()}
    if "파이브레인" in devices:                       # 파이브레인에 없는 움직이기
        fake = {"device": "파이브레인", "id": "파이브레인:none:motion", "category": "없는 기능"}
        for k, q in enumerate(NO_MOTION_Q):
            ex = example(fake, [{"q": q, "a": NO_MOTION_A[k % len(NO_MOTION_A)]}], k)
            (held if k >= len(NO_MOTION_Q) - 2 else train).append(ex)
    return train, held, kept, why


def dropped(units: dict, done: dict, reason: str) -> str:
    lines = []
    for i, u in units.items():
        g, c = done.get((i, "gen")) or {}, done.get((i, "check")) or {}
        for p, m in zip(g.get("pairs") or [], c.get("marks") or [], strict=False):
            why = passes(m)
            if why and why.startswith(reason):
                lines += [f"--- {i} {why}", f"학생: {p['q']}", f"더메이커: {p['a']}", f"점수: {dump(m)}"]
    return "\n".join(lines) + f"\n\n{reason}: {sum(x.startswith('---') for x in lines)}개\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--units", default=str(UNITS))
    p.add_argument("--api", default=str(API))
    p.add_argument("--out", help="학습 대화")
    p.add_argument("--heldout", help="채점용 대화")
    p.add_argument("--log", default="data/code-log.jsonl")
    p.add_argument("--limit", type=int, help="기기마다 앞에서 이만큼만 (먼저 해 보기)")
    p.add_argument("--dropped", metavar="까닭", help="이 까닭으로 빠진 문답을 보여 주고 끝 (예: grounded)")
    p.add_argument("--teacher", help="정해서 쓸 선생 모델")
    args = p.parse_args()
    if not Path(args.units).exists():
        raise SystemExit(f"{args.units}가 없어요: 먼저 code_docs.py를 돌리세요")
    rows = [json.loads(x) for x in Path(args.units).read_text().splitlines() if x.strip()]
    if args.limit:                                       # 기기마다 블록과 함수를 반씩
        picked = []
        for dev in dict.fromkeys(r["device"] for r in rows):
            for kind in ("block", "func"):
                picked += [r for r in rows if r["device"] == dev and r["kind"] == kind][:max(1, args.limit // 2)]
        rows = picked
    units = {r["id"]: r for r in rows}
    api = json.loads(Path(args.api).read_text())
    log_path = Path(args.log)
    if args.dropped:
        print(dropped(units, load_log(log_path, units, api), args.dropped))
        return
    if not (args.out and args.heldout):
        p.error("--out과 --heldout이 필요해요")
    llm, load_s = None, 0.0

    def llm_ask(prompts, tokens):
        nonlocal llm, load_s
        from teacher import ask, load_teacher

        if llm is None:
            t = time.time()
            llm = load_teacher(args.teacher)
            load_s = time.time() - t
        return ask(llm, prompts, tokens)

    timing: dict = {}
    done = run(llm_ask, units, api, log_path, timing)
    for stage, (n, sec) in timing.items():
        print(f"시간: {stage} {n}개 {max(sec - (load_s if stage == 'gen' else 0), 0):.0f}초", flush=True)
    train, held, kept, why = results(units, done)
    print("남은 문답: " + ", ".join(f"{k} {v}" for k, v in kept.items()), flush=True)
    print("뺀 까닭: " + (", ".join(f"{k} {v}" for k, v in why.most_common(12)) or "없음"), flush=True)
    for path, rows_ in ((args.out, train), (args.heldout, held)):
        Path(path).write_text("".join(dump(e) + "\n" for e in rows_))
        print(f"{path}: {len(rows_)}개", flush=True)


if __name__ == "__main__":
    main()
