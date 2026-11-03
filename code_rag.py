"""코딩 도우미 RAG: 학생 질문으로 블록·함수 설명(data/code-units.jsonl)을 찾아 대화 앞에 붙입니다.

찾기는 모델 없이 BM25 낱말 점수입니다. 한국어는 글자 두 개씩(조사가 붙어도 맞게), 영어·코드는 낱말과
밑줄·점으로 나눈 조각으로 셉니다. 같은 기기의 설명 안에서만 찾습니다.

    python code_rag.py --eval data/heldout-code.jsonl        # 정답 설명이 1등·상위 3개에 든 비율
    python code_rag.py --train data/code.jsonl --out data/code-rag.jsonl   # 학습 대화에 [참고 설명] 붙이기
    python code_rag.py --ask 파이보 "눈을 빨갛게 하려면?"       # 찾아 보기

학습 대화: 정답 설명 + 찾기로 나온 다른 설명들(헷갈리기 쉬운 것)을 섞어 붙이고, 일부(NO_GOLD_SHARE)는 정답을
빼고 붙인 뒤 "참고 설명에 없어서 모르겠다"고 대답하게 합니다 (찾기가 틀렸을 때 지어내지 않게).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

from prompts_code import unit_doc
from tools import system_code

UNITS = Path("data/code-units.jsonl")
#: 대화에 붙이는 설명 수
TOP_K = 3
#: 설명 하나의 최대 글자 수 (노트북에서도 읽을 글이 너무 길어지지 않게)
MAX_REF = 700
#: 정답 설명을 빼고 붙이는 학습 대화의 몫
NO_GOLD_SHARE = 0.1
NO_GOLD_ANSWER = ("참고 설명에서 그 내용을 찾지 못했어요. 블록 이름이나 하고 싶은 일을 조금 더 자세히 말해 줄래요? "
                  "그러면 맞는 블록을 다시 찾아볼게요.")
HEADER = "[참고 설명] 아래 설명에 있는 블록·함수·인자만 써서 대답해. 설명에 없으면 모른다고 해."



#: 학생이 쓰는 말 → 설명에 나오는 낱말 (질문을 넓혀 찾음). 앞 글자만 맞아도 됨 ("빨갛게", "빨간색" → "빨")
SYNONYMS = {
    "빨": "불빛 색상 RGB 빨강", "파랗": "불빛 색상 RGB", "파란": "불빛 색상 RGB", "노랗": "불빛 색상 RGB",
    "노란": "불빛 색상 RGB", "초록": "불빛 색상 RGB", "하얀": "불빛 색상 RGB", "색깔": "불빛 색상", "색": "색상",
    "눈": "눈 불빛", "엘이디": "불빛 led", "LED": "불빛 led", "불": "불빛",
    "춤": "동작 실행 motion dance", "댄스": "동작 실행 motion dance", "움직": "동작 모터 motion",
    "걷": "동작 실행 motion forward", "팔": "동작 모터", "고개": "동작 모터 머리",
    "사진": "사진 찍기 카메라 read", "찍": "사진 찍기 카메라 read", "카메라": "카메라 사진",
    "음악": "오디오 재생", "노래": "오디오 재생", "소리": "오디오 재생 소리", "틀": "오디오 재생",
    "녹음": "녹음 오디오", "말하": "목소리 말하기 tts", "읽어": "목소리 말하기 tts", "목소리": "목소리 말하기",
    "듣": "듣고 글자로 stt", "알아듣": "듣고 글자로 stt", "대답": "대화하기 llm",
    "화면": "화면 oled", "글자": "글자 화면", "그림": "이미지 그리기",
    "사람": "사람 감지 pir", "누가": "사람 감지", "만지": "터치 감지", "터치": "터치 감지",
    "얼굴": "얼굴", "물건": "사물 인식 detect", "사물": "사물 인식 detect", "손": "손",
    "배터리": "배터리", "날씨": "날씨", "뉴스": "뉴스", "검색": "찾아보기 위키피디아",
    "시작": "시작 클릭했을때", "기다": "기다리기 초",
}


def expand(query: str) -> str:
    """질문에 사전의 낱말을 덧붙임 (원래 질문 낱말도 그대로 둠)."""
    return query + " " + " ".join(v for k, v in SYNONYMS.items() if k in query)


def tokens(text: str) -> list[str]:
    """한국어는 글자 두 개씩(한 글자 낱말은 그대로), 영어·코드는 소문자 낱말 (eye_on → eye, on, eye_on)."""
    out = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        low = raw.lower()
        out.append(low)
        parts = [p for p in low.split("_") if p]
        if len(parts) > 1:
            out += parts
    for word in re.findall(r"[가-힣]+", text):
        out += [word] if len(word) == 1 else [word[i:i + 2] for i in range(len(word) - 1)]
    out += re.findall(r"[0-9]+", text)
    return out


def search_text(u: dict) -> str:
    """찾기에 쓰는 글: 이름·카테고리·설명·코드 (이름은 두 번 넣어 무게를 더함)."""
    if u["kind"] == "block":
        name = u.get("name") or u.get("text") or ""
        parts = [name, name, u.get("category") or "", u.get("text") or "", u.get("tooltip") or "",
                 u.get("guide") or "", u.get("note") or "", u.get("code") or ""]
        if u.get("byOption"):
            parts += [f"{k} {v}" for k, v in u["byOption"]["codes"].items()]
        for f in u.get("fields") or []:
            parts += [o[0] if isinstance(o, list) else str(o) for o in (f.get("options") or [])[:14]]
        return " ".join(parts)
    head = f"{u['alias']}.{u['method']} {u['method']} {u['method']} {u.get('instance') or ''} {u['signature']}"
    return head + " " + (u.get("doc") or "")[:600]


class Index:
    """기기마다 BM25 (k1=1.2, b=0.75)."""

    def __init__(self, units: list[dict]):
        self.units = units
        self.by_device: dict[str, list[int]] = {}
        self.docs = [Counter(tokens(search_text(u))) for u in units]
        self.lens = [sum(d.values()) for d in self.docs]
        for i, u in enumerate(units):
            self.by_device.setdefault(u["device"], []).append(i)
        self.df: dict[str, Counter] = {}
        self.avg: dict[str, float] = {}
        for dev, idx in self.by_device.items():
            df = Counter()
            for i in idx:
                df.update(self.docs[i].keys())
            self.df[dev], self.avg[dev] = df, sum(self.lens[i] for i in idx) / max(len(idx), 1)

    def search(self, query: str, device: str, k: int = TOP_K) -> list[dict]:
        idx = self.by_device.get(device, [])
        n, q = len(idx), Counter(tokens(expand(query)))
        scored = []
        for i in idx:
            d, s = self.docs[i], 0.0
            for t, qn in q.items():
                if t not in d:
                    continue
                df = self.df[device][t]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                tf = d[t]
                s += qn * idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * self.lens[i] / self.avg[device]))
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self.units[i] for _, i in scored[:k]]


def load_units(path: Path = UNITS) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def ref_text(u: dict) -> str:
    text = unit_doc(u)
    return text if len(text) <= MAX_REF else text[:MAX_REF].rsplit("\n", 1)[0] + "\n…"


def context(refs: list[dict]) -> str:
    """시스템 글 뒤에 붙일 [참고 설명]."""
    if not refs:
        return HEADER + "\n(찾은 설명 없음)"
    return HEADER + "\n" + "\n\n".join(f"{k}. {ref_text(u)}" for k, u in enumerate(refs, 1))


def system_with(device: str, refs: list[dict]) -> str:
    return system_code(device) + "\n\n" + context(refs)


def first_question(ex: dict) -> str:
    return next(m["content"] for m in ex["messages"] if m["role"] == "user")


def with_refs(ex: dict, index: Index, units: dict, rng: random.Random) -> dict:
    """학습 대화 하나에 [참고 설명]: 정답 + 찾기로 나온 다른 설명들, 순서는 섞음. 가끔 정답을 빼고 모른다고."""
    found = index.search(first_question(ex), ex["device"], TOP_K + 1)
    gold = units.get(ex.get("unit"))
    others = [u for u in found if gold is None or u["id"] != gold["id"]]
    ex = json.loads(json.dumps(ex, ensure_ascii=False))
    if gold is not None and rng.random() < NO_GOLD_SHARE:
        refs = others[:TOP_K]
        msgs = [ex["messages"][0]]
        for m in ex["messages"][1:3]:                 # 첫 주고받기만 남기고 대답을 "모른다"로
            msgs.append(m if m["role"] == "user" else {"role": "assistant", "content": NO_GOLD_ANSWER})
        ex["messages"], ex["no_gold"] = msgs, True
    else:
        refs = ([gold] if gold else []) + others[:TOP_K - (1 if gold else 0)]
        rng.shuffle(refs)
    ex["messages"][0] = {"role": "system", "content": system_with(ex["device"], refs)}
    ex["refs"] = [u["id"] for u in refs]
    return ex


def evaluate(path: Path, index: Index) -> dict:
    """채점 대화의 첫 질문으로 찾았을 때 정답 설명이 1등인 비율, 상위 TOP_K에 든 비율."""
    n = top1 = topk = 0
    for line in path.read_text().splitlines():
        ex = json.loads(line)
        if not ex.get("unit") or ":none:" in ex["unit"]:
            continue
        found = [u["id"] for u in index.search(first_question(ex), ex["device"], TOP_K)]
        n += 1
        top1 += bool(found) and found[0] == ex["unit"]
        topk += ex["unit"] in found
    return {"질문": n, "1등": round(top1 / max(n, 1), 3), f"상위 {TOP_K}개": round(topk / max(n, 1), 3)}


def misses(path: Path, index: Index, by_id: dict) -> str:
    """정답 설명이 상위 TOP_K에 없던 채점 질문: 질문, 정답, 찾은 것 (사전 SYNONYMS를 보강할 때 봄)."""
    lines = []
    for line in path.read_text().splitlines():
        ex = json.loads(line)
        if not ex.get("unit") or ":none:" in ex["unit"]:
            continue
        q = first_question(ex)
        found = index.search(q, ex["device"], TOP_K)
        if ex["unit"] in [u["id"] for u in found]:
            continue
        gold = by_id.get(ex["unit"], {})
        lines += [f"--- {ex['device']} | {q}",
                  f"  정답: {ex['unit'].split(':', 2)[-1]} {gold.get('name') or gold.get('signature') or ''}",
                  "  찾음: " + " / ".join(u["id"].split(":", 2)[-1] for u in found)]
    return "\n".join(lines) + f"\n\n못 찾은 질문: {sum(x.startswith('---') for x in lines)}개\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--units", default=str(UNITS))
    p.add_argument("--eval", metavar="채점 대화")
    p.add_argument("--train", metavar="학습 대화")
    p.add_argument("--out")
    p.add_argument("--ask", nargs=2, metavar=("기기", "질문"))
    p.add_argument("--misses", metavar="채점 대화", help="정답을 못 찾은 질문 목록 (--out에 씀)")
    args = p.parse_args()
    units = load_units(Path(args.units))
    index = Index(units)
    if args.ask:
        for u in index.search(args.ask[1], args.ask[0]):
            print("---", u["id"]); print(ref_text(u))
    if args.misses:
        text = misses(Path(args.misses), index, {u["id"]: u for u in units})
        Path(args.out or "samples/rag-misses.txt").parent.mkdir(exist_ok=True)
        Path(args.out or "samples/rag-misses.txt").write_text(text)
        print(text.strip().splitlines()[-1], "→", args.out or "samples/rag-misses.txt", flush=True)
        return
    if args.eval:
        print("찾기:", json.dumps(evaluate(Path(args.eval), index), ensure_ascii=False), flush=True)
    if args.train:
        if not args.out:
            p.error("--out이 필요해요")
        by_id = {u["id"]: u for u in units}
        rows = []
        for line in Path(args.train).read_text().splitlines():
            ex = json.loads(line)
            rng = random.Random(int(hashlib.md5(line.encode()).hexdigest()[:8], 16))   # 줄마다 늘 같게
            rows.append(with_refs(ex, index, by_id, rng))
        Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        no_gold = sum(bool(r.get("no_gold")) for r in rows)
        print(f"{args.out}: {len(rows)}개 (정답 설명을 뺀 것 {no_gold})", flush=True)


if __name__ == "__main__":
    main()
