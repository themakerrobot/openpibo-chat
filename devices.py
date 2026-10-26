"""여러 기기(파이보, 파이브레인 …)에서 모델 하나를 쓰게 하는 학습 데이터 손질.

모델이 이름과 도구를 외우지 않고 앱이 보낸 것을 따르게 합니다.

- 이름: 일부 예시의 "파이보"를 다른 이름으로 바꾸고 시스템 글도 그 이름으로 (tools.system_for).
  조사(가/이, 는/은, 예요/이에요 …)도 받침에 맞게 바꿉니다.
- 기기 소개: 시스템 글 끝에 "{이름}에 대해: …"를 붙인 예시와, 그걸 보고 대답하는 자기소개
  예시("너 누구야?" → "저는 코딩 로봇 파이브레인이에요"). 소개에 없는 것은 모른다고.
- 도구 목록: 일부 예시에는 기기에 있는 도구만 보여 주고(tools.tool_list), 목록에 없는 일을
  부탁하면 부르지 않고 "그건 제가 할 수 없어요"라고 하는 예시를 더합니다.

파이보 그대로인 예시가 대부분이라, 시스템 글과 도구 목록을 바꾸지 않는 파이보에서는 전과 같습니다.
"""

from __future__ import annotations

import hashlib
import json
import random
import re

from tools import BY_NAME, MOTIONS, has_batchim, system_for

#: 바꿔 쓸 기기 이름 (한국어, 영어). 받침 있는 이름과 없는 이름을 섞습니다
NAMES = [("파이브레인", "PiBrain"), ("코코", "Coco"), ("루미", "Lumi"), ("토리", "Tori"),
         ("다온", "Daon"), ("하늘", "Haneul"), ("보미", "Bomi"), ("누리", "Nuri"),
         ("로빈", "Robin"), ("아리", "Ari"), ("마루", "Maru"), ("포키", "Poki"),
         ("새롬", "Saerom"), ("단비", "Danbi")]
#: 파이보의 소개 (data/extra.txt와 같은 사실)
PIBO_ABOUT = ("인공지능 코딩로봇, 서큘러스에서 만듦, 키 39.5센티미터, 몸무게 2.2킬로그램, 하얀색, "
              "머리·팔·손·다리·발에 모터 10개, 입에 카메라, 눈에 색이 바뀌는 불빛, 가슴에 오엘이디 화면")
KINDS = ["교육용 로봇", "코딩 로봇", "인공지능 코딩 로봇", "학습 도우미 로봇", "교육용 인공지능 키트"]
BODY_COLORS = ["하얀색", "파란색", "노란색", "회색", "검은색", "초록색", "분홍색"]
CANS = ["이야기하기", "춤추기", "눈 색 바꾸기", "물건 찾기", "코딩 공부", "사진 찍기", "노래 틀기",
        "온도 재기", "불빛 켜기", "날씨 알려 주기", "퀴즈 내기", "길 따라 움직이기"]

RENAME_SHARE = 0.25      # 이름을 바꾸는 예시
ABOUT_SHARE = 0.5        # 이름을 바꾼 예시 중 기기 소개를 붙이는 것
PIBO_ABOUT_SHARE = 0.08  # 파이보 예시 중 파이보 소개를 붙이는 것
SUBSET_SHARE = 0.25      # 도구 목록 일부만 보여 주는 예시
UNAVAIL_SHARE = 0.15     # 도구 예시마다 "없는 도구" 거절 예시를 하나 더 만들 확률

#: 파이보 + 조사. 긴 조사부터
PIBO = re.compile(r"파이보(예요|이에요|라고|라는|야|가|는|를|와|랑|로|여|였|지|니|다|나|면)?")
#: 받침 없는 이름의 조사 → 받침 있는 이름의 조사 ("야"는 부를 때 아, 말할 때 이야)
JOSA = {"예요": "이에요", "라고": "이라고", "라는": "이라는", "가": "이", "는": "은", "를": "을",
        "와": "과", "랑": "이랑", "로": "으로", "여": "이여", "였": "이었", "지": "이지",
        "니": "이니", "다": "이다", "나": "이나", "면": "이면"}


def rng_for(line: str, salt: str) -> random.Random:
    """줄마다 늘 같은 결정 (다시 돌려도 같은 데이터)."""
    return random.Random(int(hashlib.md5((salt + line).encode()).hexdigest()[:12], 16))


def josa(name: str, particle: str | None, role: str) -> str:
    if not particle:
        return ""
    if particle == "이에요":
        particle = "예요"
    if not has_batchim(name):
        return particle
    if particle == "야":
        return "아" if role == "user" else "이야"
    if particle == "로" and (ord(name[-1]) - 0xAC00) % 28 == 8:     # ㄹ 받침: 하늘로
        return "로"
    return JOSA.get(particle, particle)


def rename_text(text: str, ko: str, en: str, role: str) -> str:
    text = PIBO.sub(lambda m: ko + josa(ko, m.group(1), role), text)
    return re.sub(r"\bPibo\b", en, text)


def ieyo(word: str) -> str:
    return "이에요" if has_batchim(word) else "예요"


def fake_about(rng: random.Random) -> tuple[str, dict]:
    """(소개 글, 그 내용) — 이름을 바꾼 기기의 지어낸 소개."""
    a = {"kind": rng.choice(KINDS), "color": rng.choice(BODY_COLORS),
         "height": rng.randrange(15, 61), "cans": rng.sample(CANS, rng.randint(3, 4))}
    text = (f"{a['kind']}, 서큘러스에서 만듦, {a['color']}, 키 {a['height']}센티미터, "
            f"할 수 있는 일: {', '.join(a['cans'])}")
    return text, a


def rename(ex: dict, ko: str, en: str, about: str | None) -> dict:
    ex = json.loads(json.dumps(ex, ensure_ascii=False))
    for m in ex["messages"]:
        if m["role"] == "system":
            m["content"] = system_for(ko, about)
        elif m["role"] in ("user", "assistant") and m.get("content"):
            m["content"] = rename_text(m["content"], ko, en, m["role"])
    ex["device"] = ko
    return ex


def needed(ex: dict) -> tuple[set, set]:
    """(이 예시가 부르는 도구들, 부르는 동작들)."""
    names, motions = set(), set()
    for m in ex["messages"]:
        for c in m.get("tool_calls") or []:
            f = c["function"]
            names.add(f["name"])
            if f["name"] == "set_motion":
                motions.add((f.get("arguments") or {}).get("name"))
    return names, motions


def subset(ex: dict, rng: random.Random, drop: str | None = None,
           drop_motion: str | None = None) -> None:
    """도구 목록 일부만 보여 주게 표시 (부르는 도구는 남김, drop·drop_motion은 꼭 뺌)."""
    names, motions = needed(ex)
    keep = sorted(n for n in BY_NAME if (n in names or rng.random() < 0.5) and n != drop)
    if drop_motion and "set_motion" not in keep:
        keep = sorted(keep + ["set_motion"])
    ex["tool_names"] = keep
    if "set_motion" in keep:
        moves = {m for m in MOTIONS if m in motions or rng.random() < 0.4} - {drop_motion}
        if not moves:
            moves = {rng.choice([m for m in MOTIONS if m != drop_motion])}
        ex["motions"] = sorted(moves, key=list(MOTIONS).index)


REFUSE = {
    "set_motion": ["미안해요, 저는 그 동작은 할 수 없어요.", "아쉽지만 그렇게 움직이는 건 제가 못 해요.",
                   "그 동작은 저한테 없어서 할 수 없어요. 다른 걸 해 볼까요?"],
    "eye_on": ["미안해요, 저는 눈 불빛을 바꿀 수 없어요.", "그건 제가 할 수 없는 일이에요."],
    "eye_off": ["미안해요, 저는 눈 불빛을 끌 수 없어요.", "그건 제가 할 수 없는 일이에요."],
    "find_objects": ["미안해요, 저는 카메라로 볼 수가 없어요.", "아쉽지만 앞에 뭐가 있는지 볼 수 없어요."],
    "get_battery": ["배터리가 얼마나 남았는지는 제가 알아볼 수 없어요.", "그건 제가 알아볼 수 없어요."],
    "get_weather": ["미안해요, 저는 날씨를 알아볼 수 없어요.", "날씨는 제가 찾아볼 수 없어요."],
}
REFUSE_EN = ["Sorry, I can't do that.", "I'm sorry, I can't do that one. Want to talk instead?",
             "That's not something I can do, sorry."]


def unavailable(ex: dict, rng: random.Random) -> dict | None:
    """도구 예시 → 그 도구(또는 그 동작)가 없는 기기에서 같은 부탁을 받은 예시 (말로 거절)."""
    msgs = ex["messages"]
    at = next((j for j, m in enumerate(msgs) if m.get("tool_calls")), None)
    if ex["kind"] != "tool" or at is None or len(msgs[at]["tool_calls"]) != 1:
        return None
    call = msgs[at]["tool_calls"][0]["function"]
    out = {k: v for k, v in ex.items() if k not in ("messages", "tool_names", "motions")}
    out.update(kind="chat", tools=True, unavail=True,
               messages=msgs[:at] + [{"role": "assistant", "content": rng.choice(
                   REFUSE_EN if ex.get("lang") == "en" else REFUSE[call["name"]])}])
    if call["name"] == "set_motion" and rng.random() < 0.5:   # 몸은 있는데 그 동작만 없음
        subset(out, rng, drop_motion=(call.get("arguments") or {}).get("name"))
    else:
        subset(out, rng, drop=call["name"])
    return out


def identity(rng: random.Random, tools: bool = True) -> dict:
    """시스템 글의 이름과 소개를 보고 하는 자기소개 (소개에 없는 건 모른다고)."""
    pibo = rng.random() < 0.15
    ko, en = ("파이보", "Pibo") if pibo else rng.choice(NAMES)
    has_about = rng.random() < 0.8
    if pibo:
        about_text, a = (PIBO_ABOUT if has_about else None), None
    else:
        about_text, a = fake_about(rng) if has_about else (None, None)
    call = ko + ("아" if has_batchim(ko) else "야")
    qa = [(rng.choice(["너 누구야?", "넌 누구니?", "이름이 뭐야?", "자기소개 해 봐", f"{call} 너 누구야?"]),
           (f"저는 {a['kind']} {ko}{ieyo(ko)}. 만나서 반가워요." if a else
            f"저는 {ko}{ieyo(ko)}. 만나서 반가워요.")),
          (rng.choice(["누가 만들었어?", "너 어디서 만들었어?", "어느 회사 거야?"]),
           "저는 서큘러스에서 만들었어요.")]
    if not pibo:
        other = rng.choice([n for n, _ in NAMES if n != ko] + ["파이보"])
        qa.append((f"너 {other}{'이야' if has_batchim(other) else '야'}?", f"아니요, 저는 {ko}{ieyo(ko)}."))
    if a:
        qa += [(rng.choice(["너 뭐 할 수 있어?", "무슨 기능 있어?"]),
                f"저는 {', '.join(a['cans'])} 같은 걸 할 수 있어요."),
               ("너 무슨 색이야?", f"저는 {a['color']}이에요."),
               ("너 키가 몇이야?", f"제 키는 {a['height']}센티미터예요."),
               ("너 몸무게 얼마야?", "그건 저도 잘 몰라요.")]
    if rng.random() < 0.25:
        qa += [("What's your name?", f"I'm {en}. Nice to meet you!"),
               ("Who made you?", "I was made by Circulus.")]
        if not pibo:
            qa.append(("Are you Pibo?", f"No, I'm {en}."))
    picked = rng.sample(qa, min(len(qa), rng.choice([1, 2, 2, 3])))
    msgs = [{"role": "system", "content": system_for(ko, about_text)}]
    for q, ans in picked:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
    ex = {"kind": "chat", "tools": tools and rng.random() < 0.5, "identity": True, "messages": msgs}
    if ko != "파이보":
        ex["device"] = ko
    return ex


def vary(lines: list[str], n_identity: int = 3000, seed: int = 7,
         tools: bool = True) -> tuple[list[str], dict]:
    """합친 데이터 → 이름·소개·도구 목록을 바꾼 데이터 (+ 거절, 자기소개 예시), 센 것."""
    out, count = [], {"이름 바꿈": 0, "도구 일부만": 0, "없는 도구 거절": 0, "자기소개": 0}
    for line in lines:
        ex = json.loads(line)
        rng = rng_for(line, "device")
        if rng.random() < RENAME_SHARE:
            ko, en = rng.choice(NAMES)
            about = fake_about(rng)[0] if rng.random() < ABOUT_SHARE else None
            ex = rename(ex, ko, en, about)
            count["이름 바꿈"] += 1
        elif rng.random() < PIBO_ABOUT_SHARE:
            for m in ex["messages"]:
                if m["role"] == "system":
                    m["content"] = system_for("파이보", PIBO_ABOUT)
        if ex.get("tools") and rng.random() < SUBSET_SHARE:
            subset(ex, rng)
            count["도구 일부만"] += 1
        out.append(json.dumps(ex, ensure_ascii=False))
        if ex["kind"] == "tool" and rng.random() < UNAVAIL_SHARE:
            no = unavailable(ex, rng)
            if no:
                out.append(json.dumps(no, ensure_ascii=False))
                count["없는 도구 거절"] += 1
    rng = random.Random(seed)
    for _ in range(n_identity):
        out.append(json.dumps(identity(rng, tools), ensure_ascii=False))
        count["자기소개"] += 1
    return out, count
